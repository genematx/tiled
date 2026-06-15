"""
Adapted from https://raw.githubusercontent.com/obendidi/httpx-cache/main/httpx_cache/transport.py
in accordance with its BSD-3 license
"""
import typing as tp
import weakref

import httpx

from .cache import Cache
from .cache_control import ByteStreamWrapper, CacheControl
from .logger import collect_request, collect_response, log_request, log_response, logger
from .utils import TiledResponse, retry_context


class Transport(httpx.BaseTransport):
    """Custom transport, implementing caching and custom compression encodings.

    Args:
        transport (optional): an existing httpx transport, if no transport
            is given, defaults to an httpx.HTTPTransport with default args.
        cache (optional): cache to use with this transport, defaults to
            httpx_cache.DictCache
        cacheable_methods: methods that are allowed to be cached, defaults to ['GET']
        cacheable_status_codes: status codes that are allowed to be cached,
            defaults to: (200, 203, 300, 301, 308)
    """

    def __init__(
        self,
        *,
        transport: tp.Optional[httpx.BaseTransport] = None,
        cache: tp.Optional[Cache] = None,
        limits: tp.Optional[httpx.Limits] = None,
        cacheable_methods: tp.Tuple[str, ...] = ("GET",),
        cacheable_status_codes: tp.Tuple[int, ...] = (
            httpx.codes.OK,
            httpx.codes.NON_AUTHORITATIVE_INFORMATION,
            httpx.codes.MULTIPLE_CHOICES,
            httpx.codes.MOVED_PERMANENTLY,
            httpx.codes.PERMANENT_REDIRECT,
        ),
        always_cache: bool = False,
    ):
        self.controller = CacheControl(
            cacheable_methods=cacheable_methods,
            cacheable_status_codes=cacheable_status_codes,
            always_cache=always_cache,
        )
        if transport is not None:
            self.transport = transport
        elif limits is not None:
            self.transport = httpx.HTTPTransport(limits=limits)
        else:
            self.transport = httpx.HTTPTransport()
        self.cache = cache

    def close(self) -> None:
        self.transport.close()
        if self.cache is not None:
            self.cache.close()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        # check if request is cacheable
        if (self.cache is not None) and self.controller.is_request_cacheable(request):
            if __debug__:
                logger.debug("Checking cache for: %s", request)
            cached_response = self.cache.get(request)
            if cached_response is not None:
                if self.controller.is_response_fresh(
                    request=request, response=cached_response
                ):
                    if not self.controller.needs_revalidation(
                        request=request, response=cached_response
                    ):
                        if __debug__:
                            logger.debug("Using cached response for: %s", request)
                            log_request(request)
                            collect_request(request)
                        return cached_response
                    if __debug__:
                        logger.debug("Revalidating cached response for: %s", request)
                    request.headers["If-None-Match"] = cached_response.headers["ETag"]
                else:
                    if __debug__:
                        logger.debug("Cached response is stale, deleting: %s", request)
                    self.cache.delete(request)
            else:
                if __debug__:
                    logger.debug(
                        "No valid cached response found in cache for: %s", request
                    )

        # Call original transport
        if __debug__:
            log_request(request)
            collect_request(request)
        response = self.transport.handle_request(request)
        response.__class__ = TiledResponse
        response.request = request
        if __debug__:
            # Log the actual server traffic, not the cached response.
            log_response(response)
            # But, below _collect_ the response with the content in it.

        if self.cache is not None:
            if response.status_code == httpx.codes.NOT_MODIFIED:
                if __debug__:
                    logger.debug(
                        "Server validated as fresh cached entry for: %s", request
                    )
                    collect_response(cached_response)
                return cached_response

            if self.controller.is_response_cacheable(
                request=request, response=response
            ):
                if self.cache.readonly:
                    if __debug__:
                        logger.debug("Cache is read-only; will not store")
                elif not self.cache.write_safe():
                    if __debug__:
                        logger.debug(
                            "Cannot write to cache from another thread; will not store"
                        )
                else:
                    if hasattr(response, "_content"):
                        is_stored = self.cache.set(request=request, response=response)
                        if __debug__:
                            if is_stored:
                                logger.debug("Caching response for: %s", request)
                            else:
                                logger.debug(
                                    "Declined to store large response for: %s", request
                                )
                    else:
                        # Wrap the response with cache callback:
                        def _callback(content: bytes) -> None:
                            is_stored = self.cache.set(
                                request=request, response=response, content=content
                            )
                            if __debug__:
                                if is_stored:
                                    logger.debug("Caching response for: %s", request)
                                else:
                                    logger.debug(
                                        "Declined to store large response for: %s",
                                        request,
                                    )

                        response.stream = ByteStreamWrapper(
                            stream=response.stream, callback=_callback  # type: ignore
                        )
        if __debug__:
            collect_response(response)
        return response


class RetryingTransport(httpx.BaseTransport):
    """Outermost transport that runs every request through tiled's retry loop.

    Centralises retry / cancel / circuit-breaker / spinner behaviour.  With
    this transport installed on a Context's ``http_client``, individual call
    sites do not need ``for attempt in retry_context(...)`` wrappers — every
    HTTP request automatically:

      * blocks on :meth:`Context.wait_for_circuit` if a peer worker is
        currently retrying (no piling onto a struggling server);
      * honours :attr:`Context.cancel_event` for cross-thread cancellation
        (Ctrl-C in the main thread aborts retries on dask worker threads);
      * shows / hides the shared retry indicator through ``Context``'s
        refcount machinery;
      * is retried by stamina on transient ``httpx`` errors and on
        retriable status codes (5xx, 429).

    Holds a ``weakref`` to its owning Context so the transport does not
    prevent garbage collection.  If the Context is GC'd, the transport
    degenerates to a thin pass-through with no retries — safe but unusual.

    Streaming responses are not supported by this transport (tiled itself
    does not use ``http_client.stream`` on data fetches).  A streaming
    request still works, but if the underlying transport returns a 5xx the
    body is read into memory before raising — acceptable because tiled does
    not issue streaming requests on hot paths.
    """

    def __init__(self, inner: httpx.BaseTransport, context_ref: "weakref.ref"):
        self._inner = inner
        self._context_ref = context_ref

    def close(self) -> None:
        self._inner.close()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        ctx = self._context_ref()
        if ctx is None:
            # Owning Context has been garbage-collected; behave as a
            # plain pass-through.  This should not happen during normal
            # use because the Context owns the http_client which owns us.
            return self._inner.handle_request(request)
        # Gate on the circuit breaker before issuing the request so that
        # when one worker is in the middle of a retry storm, peers do not
        # add more load.
        ctx.wait_for_circuit()
        if ctx.cancel_event.is_set():
            # Cancellation was requested while we were waiting (e.g. user
            # hit Ctrl-C in the main thread).  Abort without issuing the
            # request so the calling worker exits its compute loop.
            raise httpx.RequestError(
                "Tiled Context has been cancelled; request aborted.",
                request=request,
            )
        for attempt in retry_context(ctx):
            with attempt:
                response = self._inner.handle_request(request)
                # Convert retriable status codes to exceptions so stamina
                # retries them via the configured ``should_retry`` rule.
                # Non-retriable statuses (1xx / 3xx / 4xx) are returned
                # unchanged for the caller's ``handle_error`` to render.
                if response.status_code == 429 or response.status_code >= 500:
                    # Drain the body so the underlying connection is
                    # released back to the pool before retry sleep.  The
                    # body is then available on the eventual final
                    # exception's ``.response`` for error reporting.
                    response.read()
                    raise httpx.HTTPStatusError(
                        f"{response.status_code} {response.reason_phrase}",
                        request=request,
                        response=response,
                    )
                return response
        # Unreachable: retry_context either yields an attempt that returns
        # above, or raises on exhaustion / non-retriable failure.
        raise AssertionError("retry_context exited without yielding")  # pragma: no cover


# For when we implement an Async client
#
# class AsyncCacheControlTransport(httpx.AsyncBaseTransport):
#     """Async CacheControl transport for httpx_cache.
#
#     Args:
#         transport (optional): an existing httpx async-transport, if no transport
#             is given, defaults to an httpx.AsyncHTTPTransport with default args.
#         cache (optional): cache to use with this transport, defaults to
#             httpx_cache.DictCache
#         cacheable_methods: methods that are allowed to be cached, defaults to ['GET']
#         cacheable_status_codes: status codes that are allowed to be cached,
#             defaults to: (200, 203, 300, 301, 308)
#     """
#
#     def __init__(
#         self,
#         *,
#         transport: tp.Optional[httpx.AsyncBaseTransport] = None,
#         cache: tp.Optional[BaseCache] = None,
#         cacheable_methods: tp.Tuple[str, ...] = ("GET",),
#         cacheable_status_codes: tp.Tuple[int, ...] = (200, 203, 300, 301, 308),
#         always_cache: bool = False,
#     ):
#         self.controller = CacheControl(
#             cacheable_methods=cacheable_methods,
#             cacheable_status_codes=cacheable_status_codes,
#             always_cache=always_cache,
#         )
#         self.transport = transport or httpx.AsyncHTTPTransport()
#         self.cache = cache or DictCache()
#
#     async def aclose(self) -> None:
#         await self.cache.aclose()
#         await self.transport.aclose()
#
#     async def handle_async_request(self, request: httpx.Request) -> TiledResponse:
#         # check if request is cacheable
#         if self.controller.is_request_cacheable(request):
#             logger.debug(f"Checking cache for: %s", request)
#             cached_response = await self.cache.aget(request)
#             if cached_response is not None:
#                 logger.debug(f"Found cached response for: %s", request)
#                 if self.controller.is_response_fresh(
#                     request=request, response=cached_response
#                 ):
#                     setattr(cached_response, "from_cache", True)
#                     return cached_response
#                 else:
#                     logger.debug(f"Cached response is stale, deleting: %s", request)
#                     await self.cache.adelete(request)
#
#         # Request is not in cache, call original transport
#         response = await self.transport.handle_async_request(request)
#
#         if self.controller.is_response_cacheable(request=request, response=response):
#             if hasattr(response, "_content"):
#                 logger.debug(f"Caching response for: %s", request)
#                 await self.cache.aset(request=request, response=response)
#             else:
#                 # Wrap the response with cache callback:
#                 async def _callback(content: bytes) -> None:
#                     logger.debug(f"Caching response for: %s", request)
#                     await self.cache.aset(
#                         request=request, response=response, content=content
#                     )
#
#                 response.stream = ByteStreamWrapper(
#                     stream=response.stream, callback=_callback  # type: ignore
#                 )
#         setattr(response, "from_cache", False)
#         return response
