import builtins
import logging
import os
import sys
import uuid
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import asdict
from pathlib import Path
import threading
from threading import Lock
from typing import Optional, Union
from urllib.parse import parse_qs, urlparse
from weakref import WeakValueDictionary

import httpx
import msgpack
import stamina
import stamina.instrumentation

from ..structures.core import Spec
from ..type_aliases import Chunks
from ..utils import path_from_uri

# Stamina ships a default retry-logging hook that emits
# "stamina.retry_scheduled" at WARNING.  Tiled handles its own retry
# messages via the _LoggingAttempt wrapper (routed to tiled.client at
# DEBUG), so we strip that built-in hook — but only it, and only on first
# use of a retry context, to avoid evicting third-party hooks registered
# before tiled is imported and to avoid any global side effect at import.
_stamina_log_hook_stripped = False
_stamina_log_hook_lock = threading.Lock()


def _strip_stamina_default_log_hook():
    """Idempotently remove stamina's built-in retry logging hook.

    Leaves any other hooks (third-party metrics, custom logging, etc.)
    intact.  Safe to call from any thread; safe to call repeatedly.
    """
    global _stamina_log_hook_stripped
    if _stamina_log_hook_stripped:
        return
    with _stamina_log_hook_lock:
        if _stamina_log_hook_stripped:
            return
        try:
            current = stamina.instrumentation.get_on_retry_hooks()
            # Stamina's default is the closure returned by init_logging();
            # match by qualname so we don't reach into private modules.
            kept = tuple(
                h
                for h in current
                if not getattr(h, "__qualname__", "").startswith("init_logging")
            )
            if len(kept) != len(current):
                stamina.instrumentation.set_on_retry_hooks(kept)
        except Exception:
            # If stamina's instrumentation API changes shape, fail open
            # rather than break retries.
            pass
        _stamina_log_hook_stripped = True

MSGPACK_MIME_TYPE = "application/x-msgpack"


def raise_for_status(response) -> None:
    """
    Raise the `httpx.HTTPStatusError` if one occurred. Include correlation ID.
    """
    # This is adapted from the method httpx.Response.raise_for_status, modified to
    # remove the generic link to HTTP status documentation and include the
    # correlation ID.
    request = response._request
    if request is None:
        raise RuntimeError(
            "Cannot call `raise_for_status` as the request "
            "instance has not been set on this response."
        )

    if response.is_success:
        return response

    # correlation ID may be missing if request didn't make it to the server
    correlation_id = response.headers.get("x-tiled-request-id", None)

    if response.has_redirect_location:
        message = (
            "{error_type} '{0.status_code} {0.reason_phrase}' for url '{0.url}'\n"
            "Redirect location: '{0.headers[location]}'\n"
            "For more information, server admin can search server logs for "
            "correlation ID {correlation_id}."
        )
    else:
        message = (
            "{error_type} '{0.status_code} {0.reason_phrase}' for url '{0.url}'\n"
            "For more information, server admin can search server logs for "
            "correlation ID {correlation_id}."
        )

    status_class = response.status_code // 100
    error_types = {
        1: "Informational response",
        3: "Redirect response",
        4: "Client error",
        5: "Server error",
    }
    error_type = error_types.get(status_class, "Invalid status code")
    message = message.format(
        response, error_type=error_type, correlation_id=correlation_id
    )
    raise httpx.HTTPStatusError(message, request=request, response=response)


def handle_error(response):
    if not response.is_error:
        return response
    try:
        raise_for_status(response)
    except httpx.RequestError:
        raise  # Nothing to add in this case; just raise it.
    except httpx.HTTPStatusError as exc:
        if response.status_code == httpx.codes.GONE:
            detail = response.reason_phrase
            raise KeyError(f"Unable to open object: likely a broken link. {detail}")
        elif response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            # Let 429 propagate as-is so stamina can respect Retry-After.
            raise
        elif response.status_code < httpx.codes.INTERNAL_SERVER_ERROR:
            # Include more detail that httpx does by default.
            if response.headers.get("Content-Type") == "application/json":
                detail = response.json().get("detail", "")
            else:
                # This can happen when we get an error from a proxy,
                # such as a 502, which serves an HTML error page.
                # Use the stock "reason phrase" for the error code
                # instead of dumping HTML into the terminal.
                detail = response.reason_phrase
            message = f"{exc.response.status_code}: " f"{detail} " f"{exc.request.url}"
            raise ClientError(message, exc.request, exc.response) from exc
        else:
            raise


class ClientError(httpx.HTTPStatusError):
    def __init__(self, message, request, response):
        super().__init__(message=message, request=request, response=response)


def should_retry(exception: Exception) -> "bool | float":
    if isinstance(exception, httpx.HTTPStatusError):
        if exception.response.status_code == 429:
            # Respect Retry-After header from load balancer / server.
            # Return the delay as a float so stamina uses it as the wait time.
            retry_after = exception.response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass
            # No Retry-After header; retry with default backoff.
            return True
        return exception.response.status_code >= 500

    # do not retry for unsupported protocol errors, eg "htps://"
    if isinstance(exception, httpx.UnsupportedProtocol):
        return False

    # do not retry for local protocol errors, eg. carriage return in header value
    if isinstance(exception, httpx.LocalProtocolError):
        return False

    # Otherwise retry on all httpx errors.
    return isinstance(exception, httpx.HTTPError)


# Expose the timeout and max attempts as configurable via env vars. The rest of
# the parameters (wait_initial, wait_jitter, etc.) are intentionally not
# included here, for simplicity and to make it more difficult to configure
# clients to load the server too aggressively.
TILED_RETRY_ATTEMPTS = int(os.getenv("TILED_RETRY_ATTEMPTS", "10"))
TILED_RETRY_TIMEOUT = float(os.getenv("TILED_RETRY_TIMEOUT", "45.0"))

TILED_DEVICE_FLOW_ATTEMPTS = int(os.getenv("TILED_DEVICE_FLOW_ATTEMPTS", "100"))

_retry_logger = logging.getLogger("tiled.client")


class _LoggingAttempt:
    """Wraps a stamina ``Attempt`` to log retries to the ``tiled.client`` logger.

    Logging happens at DEBUG level so messages are suppressed by default and
    only appear when the user calls :func:`tiled.client.logger.show_logs`.

    Using a wrapper here — rather than a global stamina ``RetryHook`` — means
    tiled's retry logging is completely isolated: retries triggered by *other*
    libraries that also use stamina are not affected, and tiled's own retries
    are not routed through the global stamina hook machinery.

    The wrapper signals the retry indicator at most once per retry loop
    (the ``loop_state`` dict is shared across all attempts in a loop) so
    the indicator refcount stays balanced regardless of attempt count.
    """

    __slots__ = ("_attempt", "_context", "_standalone", "_loop_state")

    def __init__(self, attempt, context=None, standalone=None, loop_state=None):
        self._attempt = attempt
        self._context = context
        self._standalone = standalone
        # Shared across all wrappers in this retry loop:
        # {"signaled": bool}
        self._loop_state = loop_state if loop_state is not None else {"signaled": False}

    @property
    def num(self):
        return self._attempt.num

    def __enter__(self):
        return self._attempt.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If the Context has been asked to cancel (e.g. main thread caught
        # Ctrl-C while this worker was sleeping inside stamina), refuse the
        # next retry so the worker exits its retry loop immediately instead
        # of sleeping again.  Worker threads never receive SIGINT, so they
        # rely on this cross-thread flag for cancellation.
        if (
            self._context is not None
            and exc_val is not None
            and self._context.cancel_event.is_set()
        ):
            # Returning False from __exit__ lets the exception propagate
            # out of the `with attempt:` block and out of the retry loop.
            self._attempt.__exit__(None, None, None)
            return False
        result = self._attempt.__exit__(exc_type, exc_val, exc_tb)
        # stamina returns True from __exit__ when it suppresses the exception
        # and schedules a retry.  Tenacity always returns True on exception
        # (it swallows and decides at the next ``next(it)`` whether to retry
        # or re-raise), so we never see the not-retried case here — that path
        # is detected in ``retry_context`` when ``next(it)`` raises.
        if exc_val is not None and result:
            _retry_logger.debug(
                "Retry %d scheduled in %.2fs due to %r",
                self._attempt.num,
                self._attempt.next_wait,
                exc_val,
            )
            # Show the indicator exactly once per loop, on the first
            # scheduled retry.  retry_context's finally hides it exactly
            # once, keeping the Context's retry refcount balanced.
            if not self._loop_state["signaled"]:
                self._loop_state["signaled"] = True
                if self._context is not None:
                    self._context.signal_retry()
                elif self._standalone is not None:
                    self._standalone.show()
        return result


def retry_context(context=None):
    "Iterable that yields a context manager per retry attempt"
    _strip_stamina_default_log_hook()
    # When no context is provided (e.g. from_any_uri probing the server before
    # a Context object exists), we still want to show the retry indicator.
    # Use a standalone indicator owned by this generator so cleanup is guaranteed.
    standalone = StandaloneRetryIndicator() if context is None else None
    it = iter(stamina.retry_context(
        on=should_retry,
        attempts=TILED_RETRY_ATTEMPTS,
        timeout=TILED_RETRY_TIMEOUT,
    ))
    loop_state = {"signaled": False, "failed": False}
    try:
        while True:
            try:
                attempt = next(it)
            except StopIteration:
                break
            except BaseException:
                # Two paths land here:
                #   1. Tenacity raises the original exception when retries
                #      are exhausted (or when the attempt's exception was
                #      not retried at all).
                #   2. ``KeyboardInterrupt`` raised during stamina's
                #      inter-attempt sleep — tenacity would normally swallow
                #      it; re-raising here makes Ctrl-C cancel retries
                #      immediately.
                # In both cases the loop has failed permanently and peers
                # should be notified via ``request_cancel`` (in the finally).
                loop_state["failed"] = True
                raise
            yield _LoggingAttempt(
                attempt,
                context=context,
                standalone=standalone,
                loop_state=loop_state,
            )
    finally:
        # Balance the show with exactly one hide, only if we showed.
        if loop_state["signaled"]:
            if standalone is not None:
                standalone.reset()
            elif context is not None:
                context.signal_retry_resolved()
        # If this loop failed permanently (stamina chose not to retry, or
        # exhausted retries), tell the rest of the Context's workers to
        # stop.  Peers blocked on the circuit gate are released; peers
        # already in their own retry loop observe ``cancel_event`` on the
        # next attempt and bail.  Rationale: if one chunk is broken or
        # the server is down, there is no point hammering it with the
        # remaining workers.
        if loop_state["failed"] and context is not None:
            context.request_cancel()


def should_poll_for_tokens(exception: Exception) -> bool:
    # Retry on transient network errors during device flow polling.
    if isinstance(exception, (httpx.ConnectError, httpx.TimeoutException)):
        return True
    return False


def polling_retry_context(timeout: float):
    _strip_stamina_default_log_hook()
    for attempt in stamina.retry_context(
        on=should_poll_for_tokens,
        attempts=TILED_DEVICE_FLOW_ATTEMPTS,
        timeout=timeout,
    ):
        yield _LoggingAttempt(attempt)


class TiledResponse(httpx.Response):
    def json(self):
        if self.headers["Content-Type"] == MSGPACK_MIME_TYPE:
            return msgpack.unpackb(
                self.content,
                timestamp=3,  # Decode msgpack Timestamp as datetime.datetime object.
            )
        return super().json()


class UnknownStructureFamily(KeyError):
    pass


def export_util(file, format, get, link, params):
    """
    Download client data in some format and write to a file.

    This is used by the export method on clients. It intended for internal use.

    Parameters
    ----------
    file: str, Path, or buffer
        Filepath or writeable buffer.
    format : str, optional
        If format is None and `file` is a filepath, the format is inferred
        from the name, like 'table.csv' implies format="text/csv". The format
        may be given as a file extension ("csv") or a media type ("text/csv").
    get : callable
        Client's internal GET method
    link: str
        URL to download full data
    params : dict
        Additional parameters for the request, which may be used to subselect
        or slice, for example.
    """

    # The server accpets a media type like "text/csv" or a file extension like
    # "csv" (no dot) as a "format".
    if "format" in params:
        raise ValueError("params may not include 'format'. Use the format parameter.")
    if isinstance(format, str) and format.startswith("."):
        format = format[1:]  # e.g. ".csv" -> "csv"
    if isinstance(file, (str, Path)):
        # Infer that `file` is a filepath.
        if format is None:
            format = ".".join(
                suffix[1:] for suffix in Path(file).suffixes
            )  # e.g. "csv"
        for attempt in retry_context():
            with attempt:
                content = handle_error(
                    get(
                        link,
                        params={
                            **parse_qs(urlparse(link).query),
                            "format": format,
                            **params,
                        },
                    )
                ).read()
        with open(file, "wb") as buffer:
            buffer.write(content)
    else:
        # Infer that `file` is a writeable buffer.
        if format is None:
            # We have no filepath to infer to format from.
            raise ValueError("format must be specified when file is writeable buffer")
        for attempt in retry_context():
            with attempt:
                content = handle_error(
                    get(
                        link,
                        params={
                            **parse_qs(urlparse(link).query),
                            "format": format,
                            **params,
                        },
                    )
                ).read()
        file.write(content)


def client_for_item(
    context, structure_clients, item, structure=None, include_data_sources=False
):
    """
    Create an instance of the appropriate client class for an item.

    This is intended primarily for internal use and use by subclasses.
    """
    # The server can use specs to tell us that this is not just *any*
    # node/array/dataframe/etc. but that is matches a certain specification
    # for which there may be a special client available.
    # Check each spec in order for a matching structure client. Use the first
    # one we find. If we find no structure client for any spec, fall back on
    # the default for this structure family.
    specs = item["attributes"].get("specs", []) or []
    for spec in specs:
        class_ = structure_clients.get(spec["name"])
        if class_ is not None:
            break
    else:
        structure_family = item["attributes"]["structure_family"]
        try:
            class_ = structure_clients[structure_family]
        except KeyError:
            raise UnknownStructureFamily(structure_family) from None

    return class_(
        context=context,
        item=item,
        structure_clients=structure_clients,
        structure=structure,
        include_data_sources=include_data_sources,
    )


# These timeouts are really high, but in practice we find that
# ~100 MB chunks over very slow home Internet connections
# can bump into lower timeouts.
DEFAULT_TIMEOUT_PARAMS = {
    "connect": 5.0,
    "read": 30.0,
    "write": 30.0,
    "pool": 5.0,
}


def params_from_slice(slice):
    "Generate URL query param ?slice=... from Python slice object."
    params = {}
    if (slice is not None) and (slice is not ...):
        if isinstance(slice, (int, builtins.slice)):
            slice = [slice]
        slices = []
        for dim in slice:
            if isinstance(dim, builtins.slice):
                # slice(10, 50) -> "10:50"
                # slice(None, 50) -> ":50"
                # slice(10, None) -> "10:"
                # slice(None, None) -> ":"
                if (dim.step is not None) and dim.step != 1:
                    raise ValueError(
                        "Slices with a 'step' other than 1 are not supported."
                    )
                slices.append(
                    (
                        (str(dim.start) if dim.start else "")
                        + ":"
                        + (str(dim.stop) if dim.stop else "")
                    )
                )
            else:
                slices.append(str(int(dim)))
        params["slice"] = ",".join(slices)
    return params


class SerializableLock:
    """A Serializable per-process Lock

    Vendored from dask.utils because it is used parts of tiled that do not
    otherwise have a dask dependency.

    This wraps a normal ``threading.Lock`` object and satisfies the same
    interface.  However, this lock can also be serialized and sent to different
    processes.  It will not block concurrent operations between processes (for
    this you should look at ``multiprocessing.Lock`` or ``locket.lock_file``
    but will consistently deserialize into the same lock.
    So if we make a lock in one process::
        lock = SerializableLock()
    And then send it over to another process multiple times::
        bytes = pickle.dumps(lock)
        a = pickle.loads(bytes)
        b = pickle.loads(bytes)
    Then the deserialized objects will operate as though they were the same
    lock, and collide as appropriate.
    This is useful for consistently protecting resources on a per-process
    level.
    The creation of locks is itself not threadsafe.
    """

    _locks = WeakValueDictionary()
    token: Hashable
    lock: Lock

    def __init__(self, token=None):
        self.token = token or str(uuid.uuid4())
        if self.token in SerializableLock._locks:
            self.lock = SerializableLock._locks[self.token]
        else:
            self.lock = Lock()
            SerializableLock._locks[self.token] = self.lock

    def acquire(self, *args, **kwargs):
        return self.lock.acquire(*args, **kwargs)

    def release(self, *args, **kwargs):
        return self.lock.release(*args, **kwargs)

    def __enter__(self):
        self.lock.__enter__()

    def __exit__(self, *args):
        self.lock.__exit__(*args)

    def locked(self):
        return self.lock.locked()

    def __getstate__(self):
        return self.token

    def __setstate__(self, token):
        self.__init__(token)

    def __str__(self):
        return f"<{self.__class__.__name__}: {self.token}>"

    __repr__ = __str__


def get_asset_filepaths(node):
    """
    Given a node, return a list of filepaths of the data backing it.
    """
    filepaths = []
    for data_source in node.data_sources() or []:
        for asset in data_source.assets:
            # If, in the future, there are nodes with s3:// or other
            # schemes, path_from_uri will raise an exception here
            # because it cannot provide a filepath.
            filepaths.append(path_from_uri(asset.data_uri))
    return filepaths


def chunks_repr(chunks: Chunks) -> str:
    """A human-friendly representation of the chunks spec

    Avoids printing long line of repeated values when representing chunks
    for large arrays.
    """
    result = "("
    for dim in chunks:
        if len(dim) <= 5:
            # Short dimensions, e.g. (1, 1, 1)
            result += str(tuple(dim)) + ", "
        elif len(set(dim)) == 1:
            # All chunk sizes are the same, e.g. (1, 1, ..., 1)
            result += f"({dim[0]}, {dim[0]}, ..., {dim[0]}), "
        elif len(set(dim[:-1])) == 1:
            # All chunk sizes but the last are the same, e.g. (1, 1, ..., 1, 3)
            result += f"({dim[0]}, {dim[0]}, ..., {dim[0]}, {dim[-1]}), "
        else:
            # Mixed chunk sizes, e.g. (1, 2, 3, 4, 5)
            result += "variable, "
    result = result.rstrip(", ") + ")"
    return result


def normalize_specs(
    specs: Optional[Union[list[str], list[Spec], str]]
) -> Optional[list[dict[str, str]]]:
    "Represent a list of Spec objects or strings as a list of dicts"

    if specs is None:
        return None
    normalized_specs = []
    for spec in specs:
        if isinstance(spec, str):
            spec = Spec(spec)
        normalized_specs.append(asdict(spec))
    return normalized_specs


def slices_to_dask_chunks(slice_dict, shape):
    """Convert a dictionary mapping into Dask-style chunk representation

    For example, a dictionary in the form {index_tuple: list[NDSlice]} is
    converted into a tuple of tuples, one per axis.
    """

    # Collect chunk sizes per axis, keyed by axis index
    ndim = len(next(iter(slice_dict.keys())))
    axis_chunks = [defaultdict(int) for _ in range(ndim)]
    for idx, slc in slice_dict.items():
        shp = slc.shape_after_slice(shape)
        for ax in range(ndim):
            axis_chunks[ax][idx[ax]] = shp[ax]

    # Convert to ordered tuples (sorted by chunk index)
    dask_chunks = tuple(
        tuple(size for _, size in sorted(axis_dict.items()))
        for axis_dict in axis_chunks
    )

    return dask_chunks


def is_interactive():
    """Return True when running in an interactive Python session (REPL, IPython, Jupyter)."""
    import importlib.util

    if importlib.util.find_spec("IPython"):
        # IPython is installed
        from IPython import get_ipython

        if get_ipython():
            return True  # This Python process is an IPython process

    return hasattr(sys, "ps1")


def is_jupyter():
    """Return True when running inside a Jupyter notebook kernel."""
    import importlib.util

    if importlib.util.find_spec("IPython"):
        # IPython is installed
        from IPython import get_ipython

        if ip := get_ipython():
            if "ZMQInteractiveShell" in type(ip).__name__:
                return True

    return False


class ProgressState:
    """Holds progress bar state and retry indicator for fetch methods.

    Fetch methods call ``advance()`` after each successful request and
    ``show_retrying()`` / ``hide_retrying()`` around retries.
    """

    __slots__ = ("_progress", "_task_id", "_spinner", "_live", "_retrying", "_lock")

    def __init__(self, progress, task_id, spinner):
        self._progress = progress
        self._task_id = task_id
        self._spinner = spinner
        self._live = None
        self._retrying = False
        # Serialises ``advance`` so the bar is never advanced past ``total``
        # when several worker threads race.  Overshoot would otherwise show
        # confusing values like "11/10" in the M-of-N column.
        self._lock = threading.Lock()

    def advance(self):
        """Advance the progress bar by one completed fetch.

        Clamped at ``total`` to guard against an over-count in the
        caller's fetch_count estimate (e.g. when a composite client read
        recurses into a nested fetch that wasn't included in the total).
        """
        with self._lock:
            task = self._progress.tasks[self._task_id]
            if task.total is None or task.completed < task.total:
                self._progress.advance(self._task_id)

    def show_retrying(self):
        """Show a spinner below the progress bar indicating a retry is in progress."""
        if self._retrying:
            return
        self._retrying = True
        if self._live is not None:
            from rich.console import Group

            self._live.update(Group(self._progress, self._spinner))

    def hide_retrying(self):
        """Remove the retry spinner, restoring the progress bar only."""
        if not self._retrying:
            return
        self._retrying = False
        if self._live is not None:
            self._live.update(self._progress)


def _run_on_jupyter_main_thread(callback):
    """Schedule ``callback`` on the running ipykernel's IOLoop.

    Deprecated: kept only for backwards compatibility with any third-party
    callers.  The internal progress-bar and retry-indicator code now mutates
    ipywidgets directly from worker threads, because during synchronous cell
    execution the kernel's IOLoop is blocked — callbacks queued on it would
    only run after the cell completes, by which time the widget update is no
    longer useful.

    Falls back to a direct call (best-effort) if ipykernel internals are
    unavailable.
    """
    if threading.current_thread() is threading.main_thread():
        callback()
        return
    try:
        from IPython import get_ipython

        shell = get_ipython()
        io_loop = getattr(getattr(shell, "kernel", None), "io_loop", None)
        if io_loop is not None:
            io_loop.add_callback(callback)
            return
    except Exception:
        pass
    # Last resort — direct call, may be unsafe but better than silently dropping.
    callback()


class StandaloneRetryIndicator:
    """Shows a spinner on stderr while retries are in progress.

    On a TTY or in a Jupyter notebook: starts a Rich ``Live`` spinner on first
    ``show()`` and stops it on ``reset()``.  The ``Live`` instance runs for the
    full retry duration — created once, never flickered — because the
    connection-retry path never interleaves with interactive prompts (auth
    prompts come only after a successful connection).

    On non-TTY stderr (CI, pipes): writes a plain "Retrying…" line once.

    Thread-safe: ``show()`` / ``reset()`` may be called from any thread.
    Terminal output goes through Rich's thread-safe ``Live``; Jupyter widget
    creation/teardown is routed onto the kernel's IOLoop so the ipywidgets
    Comm send always originates on the kernel main thread.
    """

    def __init__(self):
        self._showing = False
        self._live = None
        self._lock = threading.Lock()

    @staticmethod
    def _stderr_is_tty():
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    @staticmethod
    def _should_render_spinner():
        """True when we should render a Rich Live spinner on stderr.

        ``sys.stderr.isatty()`` alone is not enough: IPython replaces
        ``sys.stderr`` with an ``OutStream`` (and in some terminal IDEs the
        original ``stderr`` is wrapped) whose ``isatty()`` returns ``False``
        even though the user is in a fully interactive session capable of
        rendering ANSI escapes.  In those cases we still want a spinner —
        Rich does the right thing if we pass ``force_terminal=True``.
        """
        if StandaloneRetryIndicator._stderr_is_tty():
            return True
        # IPython terminal / Jupyter-kernel-via-console / any other REPL.
        return is_interactive()

    def show(self):
        """Start the spinner (or print plain text) on first call; no-op thereafter.

        Safe to call from worker threads.
        """
        with self._lock:
            if self._showing:
                return
            self._showing = True

        if is_jupyter():
            # Mirror the reasoning in _JupyterProgressState: route widget
            # creation directly from the calling thread.  Scheduling on the
            # kernel's IOLoop is tempting but harmful — during synchronous
            # cell execution the IOLoop is blocked, so a queued create()
            # would only run after the cell completes, by which time the
            # retry is either resolved or has failed.  In practice
            # ipykernel serialises the Comm send so direct mutation from
            # worker threads is safe.
            try:
                import ipywidgets as widgets
                from IPython.display import display

                label = widgets.Label(value="⟳ Retrying…")
                display(label)
                # Re-check: reset() may have fired between show()'s entry
                # and the display() call.  If so, close the widget
                # immediately — otherwise it stays visible forever.
                with self._lock:
                    if not self._showing:
                        orphan = True
                    else:
                        self._live = label
                        orphan = False
                if orphan:
                    try:
                        label.close()
                    except Exception:
                        pass
            except ImportError:
                sys.stderr.write("Retrying…\n")
                sys.stderr.flush()
        elif self._should_render_spinner():
            from rich.console import Console
            from rich.live import Live
            from rich.spinner import Spinner

            # force_terminal so the spinner renders even though IPython (and
            # some IDE-embedded terminals) replace stderr with a non-TTY
            # stream.  This mirrors what _tracking_progress_terminal does.
            console = Console(stderr=True, force_terminal=True, highlight=False)
            spinner = Spinner("dots", text="[yellow]Retrying…[/yellow]")
            live = Live(spinner, console=console, transient=True)
            live.start()
            # Re-check: a concurrent reset() may have flipped _showing to
            # False between our show() entry and live.start() returning.
            # If so, stop the Live now — otherwise it renders forever
            # because reset() saw self._live == None and bailed.
            with self._lock:
                if not self._showing:
                    orphan = True
                else:
                    self._live = live
                    orphan = False
            if orphan:
                try:
                    live.stop()
                except Exception:
                    pass
        else:
            sys.stderr.write("Retrying…\n")
            sys.stderr.flush()

    def reset(self):
        """Stop the spinner (or no-op on non-TTY).

        Safe to call from worker threads.
        """
        with self._lock:
            if not self._showing:
                return
            self._showing = False
            live = self._live
            self._live = None

        if live is None:
            return
        if is_jupyter():
            # Direct close on the calling thread — same rationale as show():
            # the kernel IOLoop may be blocked by synchronous cell exec.
            try:
                live.close()
            except Exception:
                pass
        else:
            try:
                live.stop()
            except Exception:
                pass
