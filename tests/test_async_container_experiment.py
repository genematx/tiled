"""Tests for the experimental async container client.

Exercises `AsyncContext` + `AsyncContainer` (driven by the shared
sans-I/O `requestor` generators, notably `_new_requestor`): the
happy-path create/write/read-back flow, and the retried-create semantics
that mirror the sync tests in `test_writing.py` (adoption of our own
node after a lost response, genuine 409 Conflicts still raising, and
transparent retry of requests that never reached the server).
"""

import httpx
import numpy
import pytest
import stamina

from tiled.catalog import from_uri
from tiled.client._async_container import AsyncContainer
from tiled.client._async_context import AsyncContext
from tiled.client.array import AsyncArrayClient
from tiled.client.utils import ClientError
from tiled.server.app import build_app
from tiled.structures.core import Spec


@pytest.fixture
def app(tmp_path):
    "An in-process Tiled app around a file-backed catalog."
    catalog = from_uri(
        f"sqlite:///{tmp_path / 'catalog.db'}",
        writable_storage=[f"file://localhost{tmp_path / 'data'}"],
        init_if_not_exists=True,
    )
    return build_app(catalog)


def _single_user_api_key(app):
    "Extract the single-user API key from the app settings."
    from tiled.server.settings import get_settings

    return app.dependency_overrides[get_settings]().single_user_api_key


class LoseResponseOnce(httpx.AsyncBaseTransport):
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


class FailBeforeSendOnce(httpx.AsyncBaseTransport):
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
    "Enable stamina retries (disabled globally in conftest) without backoff."
    stamina.set_active(True)
    try:
        with stamina.set_testing(True, attempts=3):
            yield
    finally:
        stamina.set_active(False)


def _wrap_transport(context, transport_cls, **kwargs):
    transport = transport_cls(context.http_client._transport, **kwargs)
    context.http_client._transport = transport
    return transport


async def _count_children(context, path=""):
    "Count the children of a node with one request to the `search` endpoint."
    response = await context.http_client.get(
        f"{context.api_uri}search/{path.lstrip('/')}",
        params={"fields": "count"},
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    return response.json()["meta"]["count"]


async def test_async_create_and_write_array(app):
    "Happy path: create a container, write an array into it, read it back."
    async with AsyncContext.from_app(app, api_key=_single_user_api_key(app)) as context:
        root = await AsyncContainer.from_context(context)
        container = await root.create_container(
            "x", metadata={"color": "blue"}, specs=[Spec("SomeSpec")]
        )
        assert isinstance(container, AsyncContainer)
        assert container.item["id"] == "x"
        assert container.item["attributes"]["metadata"] == {"color": "blue"}
        assert container.item["attributes"]["specs"] == [
            {"name": "SomeSpec", "version": None}
        ]

        original = numpy.arange(24, dtype="float64").reshape((4, 6))
        arr = await container.write_array(original, key="y", metadata={"n": 1})
        assert isinstance(arr, AsyncArrayClient)
        assert numpy.array_equal(await arr.read(), original)

        # Read back fresh clients through child lookups, dispatched by family.
        fetched_container = await root.get("x")
        assert isinstance(fetched_container, AsyncContainer)
        fetched_arr = await fetched_container.get("y")
        assert isinstance(fetched_arr, AsyncArrayClient)
        assert numpy.array_equal(await fetched_arr.read(), original)
        assert fetched_arr.item["attributes"]["metadata"] == {"n": 1}
        with pytest.raises(KeyError):
            await root.get("missing")

        # A chunked write fans out into block writes.
        import dask.array

        chunked = dask.array.from_array(original, chunks=(2, 6))
        arr2 = await container.write_array(chunked, key="z")
        assert numpy.array_equal(await arr2.read(), original)

        assert await _count_children(context) == 1
        assert await _count_children(context, "x") == 2


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
    async with AsyncContext.from_app(app, api_key=_single_user_api_key(app)) as context:
        root = await AsyncContainer.from_context(context)
        transport = _wrap_transport(context, LoseResponseOnce)

        metadata = {"count": numpy.int64(3), "vals": numpy.arange(2)}
        specs = [Spec("SomeSpec")]
        if kind == "container":
            node = await root.create_container(key, metadata=metadata, specs=specs)
        else:
            node = await root.write_array(
                numpy.arange(10), key=key, metadata=metadata, specs=specs
            )

        assert transport.tripped  # The simulated lost response did happen
        created_key = node.item["id"]
        if key is not None:
            assert created_key == key
        # One node total: no duplicate from the retry.
        assert await _count_children(context) == 1
        attrs = node.item["attributes"]
        assert attrs["metadata"]["count"] == 3  # numpy types normalized to JSON
        assert attrs["metadata"]["vals"] == [0, 1]
        assert [spec["name"] for spec in attrs["specs"]] == ["SomeSpec"]

        # The adopted node is fully usable.
        if kind == "container":
            await node.create_container("child")
            fetched = await root.get(created_key)
            assert isinstance(await fetched.get("child"), AsyncContainer)
        else:
            numpy.testing.assert_array_equal(await node.read(), numpy.arange(10))


@pytest.mark.parametrize(
    "lossy, precreate_kind, precreate_kwargs, attempt_kwargs",
    [
        # A clean first-attempt 409 (no lost response) must raise as before.
        (False, "container", {}, {}),
        # After a lost response, a 409 for a non-matching node is a genuine
        # key collision and must raise: different structure family, ...
        (True, "array", {}, {}),
        # ... different specs, ...
        (
            True,
            "container",
            {"specs": [Spec("SpecA")]},
            {"specs": [Spec("SpecB")]},
        ),
        # ... or different metadata.
        (True, "container", {"metadata": {"a": 1}}, {"metadata": {"a": 2}}),
    ],
    ids=["clean-conflict", "family-mismatch", "specs-mismatch", "metadata-mismatch"],
)
async def test_async_retried_create_conflict_not_adopted(
    app, retries_enabled, lossy, precreate_kind, precreate_kwargs, attempt_kwargs
):
    "409 Conflicts that can not be positively matched to our own create raise."
    async with AsyncContext.from_app(app, api_key=_single_user_api_key(app)) as context:
        root = await AsyncContainer.from_context(context)
        if precreate_kind == "container":
            await root.create_container("x", **precreate_kwargs)
        else:
            await root.write_array(numpy.arange(5), key="x", **precreate_kwargs)

        if lossy:
            _wrap_transport(context, LoseResponseOnce)
        with pytest.raises(ClientError) as info:
            await root.create_container("x", **attempt_kwargs)
        assert info.value.response.status_code == httpx.codes.CONFLICT
        # The pre-existing node was not duplicated or replaced.
        assert await _count_children(context) == 1


async def test_async_create_retried_when_never_sent(app, retries_enabled):
    """A connect-phase failure (the request never reached the server) is
    retried transparently: no 409, no adoption, exactly one node."""
    async with AsyncContext.from_app(app, api_key=_single_user_api_key(app)) as context:
        root = await AsyncContainer.from_context(context)
        transport = _wrap_transport(context, FailBeforeSendOnce)

        node = await root.create_container("x", metadata={"a": 1})

        assert transport.tripped
        assert node.item["id"] == "x"
        assert await _count_children(context) == 1
