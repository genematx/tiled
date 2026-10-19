"""Sans-I/O `requestor` pattern for client logic.

Client logic is written once, as a generator that yields descriptions of
HTTP requests (`TiledRequest`) and receives `httpx.Response` objects back
at the `yield`. The generator itself never performs I/O; a *driver*
executes it against either a sync `httpx.Client` (`send_requests`) or an
async `httpx.AsyncClient` (`send_requests_async`).

A requestor may also yield a list of sub-requestors, meaning `run these
to completion, concurrently if possible`; it receives the list of their
results back at the `yield`. The async driver runs them with
`asyncio.gather`; the sync driver runs them sequentially. The semantics
are otherwise identical.

Errors raised by `handle_error` (or by the transport) are thrown into
the generator with `Generator.throw` so that client logic can catch and
translate them; uncaught errors propagate to the caller of the driver.

Retries are driven by stamina inside the driver (`retry_context` /
`retry_context_async`), so the generator only ever sees the *final*
exception of a request: intermediate failed attempts are retried and
swallowed. Flows that must observe every failed attempt (e.g. a create
that tracks whether any attempt may have reached the server) can set
`TiledRequest.on_attempt_error`, a pure callback the driver invokes with
each attempt's exception before the retry policy rules on it.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union

import httpx

from .utils import handle_error, retry_context, retry_context_async, should_retry

T = TypeVar("T")

# A retry predicate, e.g. `should_retry` or `should_retry_if_unprocessed`
# from `tiled.client.utils`. May return a float to request a specific
# backoff delay (used to honor `Retry-After`).
RetryPolicy = Callable[[Exception], Union[bool, float]]


@dataclass(frozen=True)
class TiledRequest:
    """Description of one HTTP request, yielded by a requestor generator.

    The driver builds a fresh `httpx.Request` from these fields on every
    retry attempt, so a request body is never re-sent from a consumed
    stream. The `retry_policy` field selects the retry predicate for this
    request (default `should_retry`; non-idempotent requests may use
    `should_retry_if_unprocessed`).

    The optional `on_attempt_error` callback is invoked by the driver with
    the exception from *each* failed attempt, just before the retry policy
    decides whether to retry. It must be pure bookkeeping (no I/O, no
    raising): typically a closure that mutates the yielding generator's
    local state, because only the final exception is thrown back into the
    generator while intermediate ones are retried and swallowed.
    """

    method: str
    url: str
    params: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    content: Optional[bytes] = None
    retry_policy: RetryPolicy = field(default=should_retry)
    on_attempt_error: Optional[Callable[[Exception], None]] = None


# A requestor yields either a single request or a list of sub-requestors
# to run concurrently. It receives, respectively, the `httpx.Response`
# or the list of the sub-requestors' results back at the `yield`.
Effect = Union[TiledRequest, List["Requestor[Any]"]]
Outcome = Union[httpx.Response, List[Any]]
Requestor = Generator[Effect, Outcome, T]


def _observe(request_spec: TiledRequest, exc: Exception) -> None:
    "Report one failed attempt to the request's `on_attempt_error` callback."
    if request_spec.on_attempt_error is not None:
        request_spec.on_attempt_error(exc)


def _resolve(request_spec: TiledRequest, context) -> httpx.Response:
    "Send one request with blocking I/O, rebuilding it on each retry attempt."
    with context.throttle():
        for attempt in retry_context(context, on=request_spec.retry_policy):
            with attempt:
                request = context.http_client.build_request(
                    request_spec.method,
                    request_spec.url,
                    params=request_spec.params,
                    headers=request_spec.headers,
                    content=request_spec.content,
                )
                try:
                    response = handle_error(context.http_client.send(request))
                except Exception as exc:
                    _observe(request_spec, exc)
                    raise
    return response


async def _resolve_async(request_spec: TiledRequest, context) -> httpx.Response:
    "Send one request with async I/O, rebuilding it on each retry attempt."
    async with context.throttle():
        async for attempt in retry_context_async(context, on=request_spec.retry_policy):
            with attempt:
                request = context.http_client.build_request(
                    request_spec.method,
                    request_spec.url,
                    params=request_spec.params,
                    headers=request_spec.headers,
                    content=request_spec.content,
                )
                try:
                    response = handle_error(await context.http_client.send(request))
                except Exception as exc:
                    _observe(request_spec, exc)
                    raise
    return response


def send_requests(gen: Requestor[T], context) -> T:
    """Drive a requestor generator to completion using blocking I/O.

    Parameters
    ----------
    gen : Requestor
        The generator encapsulating the client logic.
    context : Context
        Supplies the `http_client`, the `throttle()` semaphore, and retry
        signaling (via `retry_context`).
    """
    outcome: Optional[Outcome] = None
    exception: Optional[Exception] = None
    while True:
        try:
            if exception is not None:
                effect = gen.throw(exception)
            else:
                effect = gen.send(outcome)
        except StopIteration as stop:
            return stop.value
        outcome, exception = None, None
        try:
            if isinstance(effect, TiledRequest):
                outcome = _resolve(effect, context)
            else:
                # A list of sub-requestors: the sync driver runs them
                # sequentially (the async driver runs them concurrently).
                outcome = [send_requests(sub, context) for sub in effect]
        except Exception as exc:
            exception = exc


async def send_requests_async(gen: Requestor[T], context) -> T:
    """Drive a requestor generator to completion using async I/O.

    The async counterpart of `send_requests`; lists of sub-requestors are
    run concurrently with `asyncio.gather`.

    Parameters
    ----------
    gen : Requestor
        The generator encapsulating the client logic.
    context : AsyncContext
        Supplies the `http_client`, the `throttle()` semaphore, and retry
        signaling (via `retry_context_async`).
    """
    outcome: Optional[Outcome] = None
    exception: Optional[Exception] = None
    while True:
        try:
            if exception is not None:
                effect = gen.throw(exception)
            else:
                effect = gen.send(outcome)
        except StopIteration as stop:
            return stop.value
        outcome, exception = None, None
        try:
            if isinstance(effect, TiledRequest):
                outcome = await _resolve_async(effect, context)
            else:
                outcome = await asyncio.gather(
                    *(send_requests_async(sub, context) for sub in effect)
                )
        except Exception as exc:
            exception = exc
