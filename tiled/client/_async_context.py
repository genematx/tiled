"""Minimal asynchronous counterpart of `tiled.client.context.Context`.

Holds an `httpx.AsyncClient` plus just enough state to drive requestor
generators (see `tiled.client._requestor`): base URL handling, API-key
authentication via the `Authorization: Apikey ...` header, and an
`asyncio.Semaphore` that throttles concurrent data-fetch requests.

Deliberately out of scope (use the sync `Context` for these): OAuth2
token refresh flows, the HTTP response cache, websockets, and progress
reporting.
"""

import asyncio
import urllib.parse
from typing import Any, Dict, Optional

import httpx
from pydantic import TypeAdapter

from ..schemas import About
from ._requestor import Requestor, TiledRequest, send_requests_async
from .context import API_KEY_AUTH_HEADER_PATTERN, MAX_CONCURRENT_CONNECTIONS, USER_AGENT
from .utils import DEFAULT_TIMEOUT_PARAMS, MSGPACK_MIME_TYPE, response_json


def _server_info_requestor(api_uri: httpx.URL) -> Requestor[Dict[str, Any]]:
    "Requestor generator that fetches the server's `About` document."
    response = yield TiledRequest(
        "GET",
        str(api_uri),
        headers={
            "Accept": MSGPACK_MIME_TYPE,
            "Cache-Control": "no-cache, no-store",
        },
    )
    return response_json(response)


class AsyncContext:
    """Wrap an `httpx.AsyncClient` for use with `send_requests_async`.

    Use as an async context manager:

    >>> async with AsyncContext("https://tiled.example.com/api/v1",
    ...                         api_key="secret") as context:
    ...     client = await AsyncArrayClient.from_context(context, "x")
    """

    def __init__(
        self,
        uri,
        *,
        api_key: Optional[str] = None,
        app=None,
        timeout: Optional[httpx.Timeout] = None,
        max_connections: int = MAX_CONCURRENT_CONNECTIONS,
    ):
        # The uri is expected to reach the root API route, e.g.
        # `https://example.com/api/v1`.
        uri = httpx.URL(uri)
        path = uri.path
        if not path.endswith("/"):
            path = f"{path}/"
        self.api_uri = httpx.URL(
            urllib.parse.urlunsplit((uri.scheme, uri.netloc.decode(), path, {}, ""))
        )
        if timeout is None:
            timeout = httpx.Timeout(**DEFAULT_TIMEOUT_PARAMS)
        headers = {"user-agent": USER_AGENT}
        if api_key is not None:
            headers["Authorization"] = f"Apikey {api_key}"
        if app is None:
            client = httpx.AsyncClient(
                timeout=timeout,
                headers=headers,
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_connections,
                ),
            )
        else:
            # In-process ASGI mode, primarily for testing. Note that
            # `httpx.ASGITransport` does not run the app's lifespan;
            # `__aenter__` starts it manually so that server startup hooks
            # (e.g. catalog database initialization) run.
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url=str(self.api_uri),
                timeout=timeout,
                headers=headers,
                follow_redirects=True,
            )
        self.http_client = client
        self._app = app
        self._lifespan_manager = None
        # Semaphore that caps the number of concurrent data-fetch requests;
        # mirrors `Context._concurrent_request_semaphore`.
        self._semaphore = asyncio.Semaphore(max_connections)
        # Populated by `__aenter__` with one GET to the API root, mirroring
        # the sync `Context`. Client logic shared with the sync client (e.g.
        # server-version backcompat branches) relies on this.
        self.server_info: Optional[About] = None

    @classmethod
    def from_app(
        cls,
        app,
        *,
        api_key: Optional[str] = None,
        uri: str = "http://local-tiled-app/api/v1",
        **kwargs,
    ) -> "AsyncContext":
        "Construct an AsyncContext around an ASGI app. Primarily for testing."
        return cls(uri, api_key=api_key, app=app, **kwargs)

    async def __aenter__(self) -> "AsyncContext":
        if self._app is not None:
            # Run the ASGI lifespan: startup now, shutdown in `aclose`.
            self._lifespan_manager = self._app.router.lifespan_context(self._app)
            await self._lifespan_manager.__aenter__()
        self.server_info = TypeAdapter(About).validate_python(
            await send_requests_async(_server_info_requestor(self.api_uri), self)
        )
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.http_client.aclose()
        if self._lifespan_manager is not None:
            await self._lifespan_manager.__aexit__(None, None, None)
            self._lifespan_manager = None

    @property
    def api_key(self) -> Optional[str]:
        # Extract from header to ensure that there is one "ground truth",
        # as in the sync `Context`.
        header = self.http_client.headers.get("Authorization", "")
        match = API_KEY_AUTH_HEADER_PATTERN.match(header)
        if match is not None:
            return match.group(1)
        return None

    def throttle(self) -> asyncio.Semaphore:
        "Semaphore capping concurrent data-fetch requests (use `async with`)."
        return self._semaphore

    # `retry_context_async(context)` calls these when retries occur.
    # Progress/retry indicators are a sync-client nicety; no-op here.
    def signal_retry(self) -> None:
        pass

    def signal_retry_resolved(self) -> None:
        pass
