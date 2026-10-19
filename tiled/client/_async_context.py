"""A minimal asynchronous context for the sans-I/O array client experiment.

This is a deliberately small async counterpart to `tiled.client.context.Context`.
It supports API-key authentication (an `Authorization: Apikey ...` header)
only: no token refresh flows, no HTTP response cache, and no websockets.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Union

import httpx
import stamina
from pydantic import TypeAdapter

from .._version import __version__ as tiled_version
from ..schemas import About
from ..utils import UNSET
from ._request_spec import RequestSpec
from .context import MAX_CONCURRENT_CONNECTIONS
from .utils import (
    DEFAULT_TIMEOUT_PARAMS,
    MSGPACK_MIME_TYPE,
    TILED_RETRY_ATTEMPTS,
    TILED_RETRY_TIMEOUT,
    TiledResponse,
    handle_error,
    should_retry,
)

USER_AGENT = f"python-tiled/{tiled_version}"


class AsyncContext:
    """Wrap an `httpx.AsyncClient` with API-key auth, throttling, and retries.

    Use as an async context manager:

    >>> async with AsyncContext("https://tiled.example.com/api/v1") as context:
    ...     client = await AsyncArrayClient.from_path(context, "my_array")
    """

    def __init__(
        self,
        uri: Union[str, httpx.URL],
        *,
        api_key: Optional[str] = None,
        app: Optional[Any] = None,
        timeout: Optional[httpx.Timeout] = None,
        max_connections: int = MAX_CONCURRENT_CONNECTIONS,
    ) -> None:
        # The uri is expected to reach the root API route.
        uri = httpx.URL(uri)
        path = uri.path
        if not path.endswith("/"):
            path = f"{path}/"
        self.api_uri = uri.copy_with(path=path, query=None)
        headers = {"user-agent": USER_AGENT}
        if api_key is not None:
            headers["Authorization"] = f"Apikey {api_key}"
        if timeout is None:
            timeout = httpx.Timeout(**DEFAULT_TIMEOUT_PARAMS)
        transport: httpx.AsyncBaseTransport
        if app is not None:
            # Dispatch requests to an in-process ASGI app. Note that the app's
            # lifespan is not run by the transport; see `from_app`.
            transport = httpx.ASGITransport(app=app)
        else:
            transport = httpx.AsyncHTTPTransport(
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_connections,
                )
            )
        self.http_client = httpx.AsyncClient(
            transport=transport,
            base_url=self.api_uri,
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )
        # Cap the number of concurrent in-flight requests, mirroring the
        # semaphore throttle in the synchronous `Context`.
        self._semaphore = asyncio.Semaphore(max_connections)
        # Optionally set by `from_app`; entered/exited with this context.
        self._lifespan_context: Optional[Any] = None
        # Fetched from the api root when this context is entered, mirroring
        # the initial request made by the synchronous `Context`.
        self.server_info: Optional[About] = None

    @classmethod
    def from_app(
        cls,
        app: Any,
        *,
        api_key: Any = UNSET,
        uri: str = "http://local-tiled-app/api/v1",
        run_lifespan: bool = True,
    ) -> "AsyncContext":
        """Construct an AsyncContext around an ASGI app. Primarily for testing.

        Unless `run_lifespan` is False, the app's lifespan (startup/shutdown
        events, which e.g. initialize the catalog database) is run when this
        context is entered. Pass `run_lifespan=False` when the app's lifespan
        is already running, e.g. inside a synchronous `Context.from_app`.
        """
        if api_key is UNSET:
            # Extract the single-user API key from the app, as the
            # synchronous `Context.from_app` does.
            from ..server.settings import get_settings

            settings = app.dependency_overrides[get_settings]()
            api_key = settings.single_user_api_key or None
        context = cls(uri, api_key=api_key, app=app)
        if run_lifespan:
            context._lifespan_context = app.router.lifespan_context(app)
        return context

    async def __aenter__(self) -> "AsyncContext":
        if self._lifespan_context is not None:
            await self._lifespan_context.__aenter__()
        # Make an initial request to the api root to get the server_info,
        # as the synchronous `Context` does at construction time. This makes
        # e.g. server-version-dependent request building possible.
        spec = RequestSpec(
            "GET",
            str(self.api_uri),
            headers={
                "Accept": MSGPACK_MIME_TYPE,
                "Cache-Control": "no-cache, no-store",
            },
        )
        response = await self.send(spec)
        self.server_info = TypeAdapter(About).validate_python(response.json())
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.http_client.aclose()
        if self._lifespan_context is not None:
            await self._lifespan_context.__aexit__(exc_type, exc, tb)

    async def _request(self, spec: RequestSpec) -> httpx.Response:
        "Send one request (one attempt, no throttling) and check for errors."
        response = await self.http_client.request(
            spec.method,
            spec.url_path,
            params=spec.params,
            headers=spec.headers,
            content=spec.content,
        )
        # Make `.json()` msgpack-aware, as the synchronous transport does.
        response.__class__ = TiledResponse
        handle_error(response)
        return response

    async def send(self, spec: RequestSpec) -> httpx.Response:
        """Send the request described by `spec`, with throttling and retries.

        This is the single I/O choke point for async clients. The request is
        rebuilt from the spec on each retry attempt, because an already-sent
        `httpx.Request` body cannot be replayed.
        """
        async with self._semaphore:
            async for attempt in stamina.retry_context(
                on=should_retry,
                attempts=TILED_RETRY_ATTEMPTS,
                timeout=TILED_RETRY_TIMEOUT,
            ):
                with attempt:
                    response = await self._request(spec)
            return response

    async def send_once(self, spec: RequestSpec) -> httpx.Response:
        """Send the request described by `spec` once: throttled, but NO retries.

        Most callers want `send`. This exists for multi-exchange flows like
        `AsyncContainer.new` that need attempt-level visibility -- e.g. to
        track whether a failed attempt may have reached the server, and to
        branch on specific status codes between attempts -- and therefore run
        their own `stamina.retry_context` loop around single-attempt sends.
        """
        async with self._semaphore:
            return await self._request(spec)
