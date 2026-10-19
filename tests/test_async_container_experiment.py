"""Tests for the experimental sans-I/O async container client.

These mirror the synchronous retried-create tests in `test_writing.py`
(`LoseResponseOnce`, `FailBeforeSendOnce`, `test_retried_create_adopts_own_node`,
`test_retried_create_conflict_not_adopted`) for the async path
(`AsyncContext` + `AsyncContainer`), plus a happy-path create/write/read test
and direct unit tests of the pure failure-classification functions that both
shells' retry loops delegate to.
"""

import httpx
import numpy
import pytest
import stamina

from tiled.catalog import from_uri
from tiled.client import Context, from_context
from tiled.client._async_array import AsyncArrayClient
from tiled.client._async_container import AsyncContainer
from tiled.client._async_context import AsyncContext
from tiled.client._container_protocol import (
    CreateFailureAction,
    classify_create_failure,
)
from tiled.client.utils import ClientError, request_may_have_reached_server
from tiled.config import Authentication
from tiled.server.app import build_app
from tiled.structures.core import Spec


@pytest.fixture
def app(tmp_path):
    "An in-process app over a file-backed catalog (so data survives restarts)."
    catalog = from_uri(
        f"sqlite:///{tmp_path / 'catalog.db'}",
        writable_storage=str(tmp_path / "data"),
        init_if_not_exists=True,
    )
    return build_app(
        catalog, authentication=Authentication(single_user_api_key="secret")
    )


class AsyncLoseResponseOnce(httpx.AsyncBaseTransport):
    """Simulate a lost response: the first request matching (method,
    url_substring) is forwarded to the wrapped transport -- so the server fully
    processes it -- but a ReadTimeout is raised to the caller. Subsequent
    requests pass through unmodified."""

    def __init__(self, wrapped, method="POST", url_substring="/metadata/"):
        self._wrapped = wrapped
        self._method = method
        self._url_substring = url_substring
        self.tripped = False

    async def handle_async_request(self, request):
        response = await self._wrapped.handle_async_request(request)
        if (
            not self.tripped
            and request.method == self._method
            and self._url_substring in str(request.url)
        ):
            self.tripped = True
            raise httpx.ReadTimeout("Simulated lost response", request=request)
        return response


class AsyncFailBeforeSendOnce(httpx.AsyncBaseTransport):
    """Simulate a connect-phase failure: the first request matching (method,
    url_substring) raises ConnectError WITHOUT being forwarded -- the server
    never sees it. Subsequent requests pass through unmodified."""

    def __init__(self, wrapped, method="POST", url_substring="/metadata/"):
        self._wrapped = wrapped
        self._method = method
        self._url_substring = url_substring
        self.tripped = False

    async def handle_async_request(self, request):
        if (
            not self.tripped
            and request.method == self._method
            and self._url_substring in str(request.url)
        ):
            self.tripped = True
            raise httpx.ConnectError("Simulated connection failure", request=request)
        return await self._wrapped.handle_async_request(request)


@pytest.fixture
def retries_enabled():
    """Enable stamina retries (disabled globally in conftest) without backoff."""
    stamina.set_active(True)
    try:
        with stamina.set_testing(True, attempts=3):
            yield
    finally:
        stamina.set_active(False)


def _wrap_transport(context: AsyncContext, transport_cls, **kwargs):
    transport = transport_cls(context.http_client._transport, **kwargs)
    context.http_client._transport = transport
    return transport


async def test_async_create_and_write_happy_path(app):
    "Create a container, write an array into it, and read everything back."
    original = numpy.arange(10, dtype="float64")
    async with AsyncContext.from_app(app) as context:
        root = await AsyncContainer.from_context(context)
        assert context.server_info is not None  # fetched on __aenter__

        c = await root.create_container("c", metadata={"color": "blue"})
        assert isinstance(c, AsyncContainer)
        assert c.item["id"] == "c"
        assert c.item["attributes"]["metadata"] == {"color": "blue"}

        arr = await c.write_array(original, key="x", metadata={"a": 1})
        assert isinstance(arr, AsyncArrayClient)
        numpy.testing.assert_array_equal(await arr.read(), original)

        # Child lookups dispatch by structure family.
        c2 = await root.get("c")
        assert isinstance(c2, AsyncContainer)
        arr2 = await c2.get("x")
        assert isinstance(arr2, AsyncArrayClient)
        numpy.testing.assert_array_equal(await arr2.read(), original)
        assert arr2.item["attributes"]["metadata"] == {"a": 1}
        with pytest.raises(KeyError):
            await c.get("does_not_exist")

        assert await root.keys() == ["c"]
        assert await c.keys() == ["x"]

    # Confirm with the SYNC client that the async writes persisted.
    with Context.from_app(app) as sync_context:
        sync_client = from_context(sync_context)
        assert list(sync_client) == ["c"]
        assert sync_client["c"].metadata["color"] == "blue"
        numpy.testing.assert_array_equal(sync_client["c"]["x"].read(), original)


@pytest.mark.parametrize(
    "kind, key",
    [("container", "x"), ("array", "x"), ("container", None)],
    ids=["container", "array", "auto-key"],
)
async def test_async_retried_create_adopts_own_node(app, retries_enabled, kind, key):
    """A retried create whose first attempt succeeded server-side (lost
    response) is adopted instead of raising 409 -- also when no key was
    supplied, because the client generates the key before the first attempt so
    all retries share it (no duplicate node under a second server-generated
    key). Metadata with numpy types checks that the match uses the same JSON
    normalization as the create."""
    async with AsyncContext.from_app(app) as context:
        root = await AsyncContainer.from_context(context)
        transport = _wrap_transport(context, AsyncLoseResponseOnce)

        metadata = {"count": numpy.int64(3), "vals": numpy.arange(2)}
        specs = [Spec("SomeSpec")]
        if kind == "container":
            node = await root.create_container(key=key, metadata=metadata, specs=specs)
        else:
            node = await root.write_array(
                numpy.arange(10), key=key, metadata=metadata, specs=specs
            )

        assert transport.tripped  # The simulated lost response did happen
        created_key = node.item["id"]
        if key is not None:
            assert created_key == key
        # One node total: no duplicate from the retry.
        assert await root.keys() == [created_key]
        attributes = node.item["attributes"]
        assert attributes["metadata"]["count"] == 3  # numpy normalized to JSON
        assert attributes["metadata"]["vals"] == [0, 1]
        assert [spec["name"] for spec in attributes["specs"]] == ["SomeSpec"]

        # The adopted node is fully usable.
        if kind == "container":
            await node.create_container(key="child")
            assert await node.keys() == ["child"]
        else:
            numpy.testing.assert_array_equal(await node.read(), numpy.arange(10))


async def test_async_clean_conflict_raises(app, retries_enabled):
    """A 409 on the first attempt -- when no attempt may have reached the
    server -- raises, even though the existing node matches the request."""
    async with AsyncContext.from_app(app) as context:
        root = await AsyncContainer.from_context(context)
        await root.create_container(key="x")
        with pytest.raises(ClientError) as info:
            await root.create_container(key="x")
        assert info.value.response.status_code == httpx.codes.CONFLICT
        assert await root.keys() == ["x"]


@pytest.mark.parametrize(
    "precreate_kind, precreate_kwargs, attempt_kwargs",
    [
        # After a lost response, a 409 for a non-matching node is a genuine
        # key collision and must raise: different structure family, ...
        ("array", {}, {}),
        # ... different specs, ...
        ("container", {"specs": [Spec("SpecA")]}, {"specs": [Spec("SpecB")]}),
        # ... or different metadata.
        ("container", {"metadata": {"a": 1}}, {"metadata": {"a": 2}}),
    ],
    ids=["family-mismatch", "specs-mismatch", "metadata-mismatch"],
)
async def test_async_retried_create_conflict_not_adopted(
    app, retries_enabled, precreate_kind, precreate_kwargs, attempt_kwargs
):
    "409 Conflicts that can not be positively matched to our own create raise."
    async with AsyncContext.from_app(app) as context:
        root = await AsyncContainer.from_context(context)
        if precreate_kind == "container":
            await root.create_container(key="x", **precreate_kwargs)
        else:
            await root.write_array(numpy.arange(5), key="x", **precreate_kwargs)

        transport = _wrap_transport(context, AsyncLoseResponseOnce)
        with pytest.raises(ClientError) as info:
            await root.create_container(key="x", **attempt_kwargs)
        assert info.value.response.status_code == httpx.codes.CONFLICT
        assert transport.tripped
        # The pre-existing node is untouched and no duplicate was created.
        assert await root.keys() == ["x"]


async def test_async_create_retries_unsent_request(app, retries_enabled):
    """A connect-phase failure (request never sent) is retried transparently
    and results in exactly one node, with no adoption involved."""
    async with AsyncContext.from_app(app) as context:
        root = await AsyncContainer.from_context(context)
        transport = _wrap_transport(context, AsyncFailBeforeSendOnce)
        node = await root.create_container(key="x")
        assert transport.tripped
        assert node.item["id"] == "x"
        assert await root.keys() == ["x"]


# Unit tests for the pure failure-classification functions shared by the sync
# and async create retry loops.


def _http_status_error(status_code, cls=httpx.HTTPStatusError):
    "Build a minimal HTTPStatusError (or subclass) with stub request/response."
    request = httpx.Request("POST", "http://example.com/api/v1/metadata/")
    response = httpx.Response(status_code, request=request)
    return cls(f"{status_code}", request=request, response=response)


@pytest.mark.parametrize(
    "exc, expected",
    [
        # Read/write-phase transport errors: the request was sent, so the
        # server may have processed it.
        (httpx.ReadTimeout("x"), True),
        (httpx.ReadError("x"), True),
        (httpx.WriteTimeout("x"), True),
        (httpx.WriteError("x"), True),
        (httpx.RemoteProtocolError("x"), True),
        # Connect-phase transport errors: the request never left the client.
        (httpx.ConnectError("x"), False),
        (httpx.ConnectTimeout("x"), False),
        (httpx.PoolTimeout("x"), False),
        # 5xx: a proxy/server error after possible processing.
        (_http_status_error(500), True),
        (_http_status_error(502), True),
        # ClientError (raised by handle_error) classifies by status too.
        (_http_status_error(504, ClientError), True),
        # 4xx: the server refused without processing.
        (_http_status_error(400), False),
        (_http_status_error(409), False),
        (_http_status_error(429), False),
        # Non-httpx exceptions say nothing about the server.
        (ValueError("x"), False),
    ],
    ids=lambda val: val if isinstance(val, bool) else type(val).__name__,
)
def test_request_may_have_reached_server(exc, expected):
    assert request_may_have_reached_server(exc) is expected


@pytest.mark.parametrize(
    "exc, flag_in, expected_flag, expected_action",
    [
        # Response lost after the request was sent: set the flag, raise into
        # the retry machinery.
        (httpx.ReadTimeout("x"), False, True, CreateFailureAction.RAISE),
        (httpx.WriteError("x"), False, True, CreateFailureAction.RAISE),
        (httpx.RemoteProtocolError("x"), False, True, CreateFailureAction.RAISE),
        # Request never sent: flag unchanged, raise (retry machinery decides).
        (httpx.ConnectError("x"), False, False, CreateFailureAction.RAISE),
        (httpx.ConnectTimeout("x"), False, False, CreateFailureAction.RAISE),
        # 5xx sets the flag but never adopts on the same attempt.
        (_http_status_error(500), False, True, CreateFailureAction.RAISE),
        (_http_status_error(502), False, True, CreateFailureAction.RAISE),
        (_http_status_error(502), True, True, CreateFailureAction.RAISE),
        # A clean 409 (no attempt may have reached the server) raises ...
        (_http_status_error(409), False, False, CreateFailureAction.RAISE),
        # ... but a 409 after a possibly-received attempt offers adoption,
        # including for the ClientError subclass that handle_error raises.
        (_http_status_error(409), True, True, CreateFailureAction.TRY_ADOPT),
        (
            _http_status_error(409, ClientError),
            True,
            True,
            CreateFailureAction.TRY_ADOPT,
        ),
        # Other 4xx never adopt, and never set or clear the flag.
        (_http_status_error(400), True, True, CreateFailureAction.RAISE),
        (_http_status_error(400), False, False, CreateFailureAction.RAISE),
        # Arbitrary exceptions: flag unchanged (in particular, never reset).
        (ValueError("x"), False, False, CreateFailureAction.RAISE),
        (ValueError("x"), True, True, CreateFailureAction.RAISE),
    ],
)
def test_classify_create_failure(exc, flag_in, expected_flag, expected_action):
    decision = classify_create_failure(exc, flag_in)
    assert decision.may_have_reached_server is expected_flag
    assert decision.action is expected_action
