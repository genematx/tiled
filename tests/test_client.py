import logging
import threading
from pathlib import Path

import httpx
import numpy
import pandas
import pytest
import stamina
import yaml
from pydantic import ValidationError
from starlette.status import HTTP_400_BAD_REQUEST

from tiled.adapters.array import ArrayAdapter
from tiled.adapters.dataframe import DataFrameAdapter
from tiled.adapters.mapping import MapAdapter
from tiled.client import Context, from_context, from_profile, record_history
from tiled.client.logger import hide_logs, show_logs
from tiled.client.utils import retry_context
from tiled.profiles import load_profiles, paths
from tiled.queries import Key
from tiled.server.app import build_app

from .utils import fail_with_status_code

tree = MapAdapter({})


def test_configurable_timeout():
    with Context.from_app(build_app(tree), timeout=httpx.Timeout(17)) as context:
        assert context.http_client.timeout.connect == 17
        assert context.http_client.timeout.read == 17


def test_configurable_max_connections():
    "max_connections is reflected in the semaphore on the Context."
    with Context.from_app(build_app(tree), max_connections=3) as context:
        assert context.max_connections == 3


def test_client_version_check(caplog):
    with Context.from_app(build_app(tree)) as context:
        client = from_context(context)

        # Too-old user agent should generate a 400.
        context.http_client.headers["user-agent"] = "python-tiled/0.1.0a77"
        with fail_with_status_code(HTTP_400_BAD_REQUEST):
            list(client)

        # Gibberish user agent should generate a warning and log entry.
        context.http_client.headers["user-agent"] = "python-tiled/gibberish"
        caplog.set_level(logging.WARNING)
        with pytest.warns(UserWarning, match=r"gibberish"):
            list(client)

        _, LOG_LEVEL, LOG_MESSAGE = range(3)
        logged_warnings = tuple(
            entry[LOG_MESSAGE]
            for entry in caplog.record_tuples
            if entry[LOG_LEVEL] == logging.WARNING
        )
        assert len(logged_warnings) > 0
        assert any("gibberish" in message for message in logged_warnings)


def test_direct(tmpdir):
    profile_content = {
        "test": {
            "structure_clients": "dask",
            "direct": {
                "trees": [
                    {"path": "/", "tree": "tiled.examples.generated_minimal:tree"}
                ]
            },
        }
    }
    with open(tmpdir / "example.yml", "w") as file:
        file.write(yaml.dump(profile_content))
    profile_dir = Path(tmpdir)
    try:
        paths.append(profile_dir)
        load_profiles.cache_clear()
        from_profile("test")
    finally:
        paths.remove(profile_dir)


def test_direct_config_error(tmpdir):
    profile_content = {
        "test": {
            "direct": {
                # Intentional config mistake!
                # Value of trees must be a list.
                "trees": {"path": "/", "tree": "tiled.examples.generated_minimal:tree"}
            }
        }
    }
    with open(tmpdir / "example.yml", "w") as file:
        file.write(yaml.dump(profile_content))
    profile_dir = Path(tmpdir)
    try:
        paths.append(profile_dir)
        load_profiles.cache_clear()
        with pytest.raises(ValidationError):
            from_profile("test")
    finally:
        paths.remove(profile_dir)


def test_jump_down_tree():
    tree = MapAdapter({}, metadata={"number": 1})
    for number, letter in enumerate(list("abcde"), start=2):
        tree = MapAdapter({letter: tree}, metadata={"number": number})
    with Context.from_app(build_app(tree)) as context:
        client = from_context(context)
    assert (
        client["e"]["d"]["c"]["b"]["a"].metadata["number"]
        == client["e", "d", "c", "b", "a"].metadata["number"]
        == 1
    )
    assert (
        client["e"]["d"]["c"]["b"].metadata["number"]
        == client["e", "d", "c", "b"].metadata["number"]
        == 2
    )
    assert (
        client["e"]["d"]["c"].metadata["number"]
        == client["e", "d", "c"].metadata["number"]
        == 3
    )
    assert (
        client["e"]["d"].metadata["number"] == client["e", "d"].metadata["number"] == 4
    )

    assert client["e"]["d", "c", "b"]["a"].metadata["number"] == 1
    assert client["e"]["d", "c", "b", "a"].metadata["number"] == 1
    assert client["e", "d", "c", "b"]["a"].metadata["number"] == 1
    assert (
        client.search(Key("number") == 5)["e", "d", "c", "b", "a"].metadata["number"]
        == 1
    )
    assert (
        client["e"].search(Key("number") == 4)["d", "c", "b", "a"].metadata["number"]
        == 1
    )

    # Check that a reasonable KeyError is raised.
    # Notice that we do not binary search to find _exactly_ where the problem is.
    with pytest.raises(KeyError) as exc_info:
        client["e", "d", "c", "b"]["X"]
    assert exc_info.value.args[0] == "X"
    with pytest.raises(KeyError) as exc_info:
        client["e", "d", "c", "b", "X"]
    assert exc_info.value.args[0] == ("e", "d", "c", "b", "X")
    with pytest.raises(KeyError) as exc_info:
        client["e", "d", "X", "b", "a"]
    assert exc_info.value.args[0] == ("e", "d", "X", "b", "a")

    # Check that jumping raises if a key along the path is not in the search
    # resuts.
    with pytest.raises(KeyError) as exc_info:
        client.search(Key("number") == 4)["e"]
    assert exc_info.value.args[0] == "e"
    with pytest.raises(KeyError) as exc_info:
        client.search(Key("number") == 4)["e", "d", "c", "b", "a"]
    assert exc_info.value.args[0] == "e"
    with pytest.raises(KeyError) as exc_info:
        client["e"].search(Key("number") == 3)["d"]
    assert exc_info.value.args[0] == "d"
    with pytest.raises(KeyError) as exc_info:
        client["e"].search(Key("number") == 3)["d", "c", "b", "a"]
    assert exc_info.value.args[0] == "d"

    with record_history() as h:
        client["e", "d", "c", "b", "a"]
    assert len(h.requests) == 1

    with record_history() as h:
        client["e"]["d"]["c"]["b"]["a"]
    assert len(h.requests) == 5


def test_no_stamina_retry_scheduled_messages(caplog):
    """stamina's default 'stamina.retry_scheduled' WARNING must never appear.

    Tiled strips the default stamina hook on first use of its retry path,
    so no global retry noise is emitted regardless of show_logs() /
    hide_logs() state.
    """
    from tiled.client.utils import retry_context

    stamina.set_active(True)
    try:
        n = 0
        with caplog.at_level(logging.DEBUG, logger="stamina"):
            for attempt in retry_context():
                with attempt:
                    n += 1
                    if n < 2:
                        raise httpx.ConnectError("transient error")

        stamina_messages = [r for r in caplog.records if r.name == "stamina"]
        assert (
            len(stamina_messages) == 0
        ), "stamina's default hook should be disabled — no 'stamina.retry_scheduled' messages"
    finally:
        stamina.set_active(False)


@pytest.mark.parametrize("logs_enabled", [True, False], ids=["show_logs", "hide_logs"])
def test_tiled_retry_logging(caplog, logs_enabled):
    """Tiled retry messages appear on tiled.client after show_logs(), silent otherwise."""
    if logs_enabled:
        show_logs()
    else:
        hide_logs()

    stamina.set_active(True)
    try:
        n = 0
        # Capture at DEBUG globally to detect any leakage in the hide_logs case.
        with caplog.at_level(logging.DEBUG):
            for attempt in retry_context():
                with attempt:
                    n += 1
                    if n < 2:
                        raise httpx.ReadTimeout("simulated timeout")

        tiled_messages = [r for r in caplog.records if r.name == "tiled.client"]
        if logs_enabled:
            assert len(tiled_messages) >= 1
            assert all(r.levelno == logging.DEBUG for r in tiled_messages)
        else:
            assert len(tiled_messages) == 0
    finally:
        stamina.set_active(False)
        hide_logs()


class TrackingSemaphore:
    "Drop-in replacement for `threading.Semaphore` that also records peak concurrent holders"

    def __init__(self, value):
        self._sem = threading.Semaphore(value)
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def acquire(self, *args, **kwargs):
        self._sem.acquire(*args, **kwargs)
        with self._lock:
            self.current += 1
            if self.current > self.peak:
                self.peak = self.current

    def release(self):
        with self._lock:
            self.current -= 1
        self._sem.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()


def test_semaphore_limits_concurrent_array_fetches():
    """When dask computes a chunked array the semaphore must cap concurrent fetches.

    We use max_connections=2 and an array split across 10 chunks so that dask
    would fire all requests at once without the semaphore.
    """
    MAX_CONNECTIONS = 2

    import dask.array as da

    arr = da.zeros((10, 300, 400), chunks=(1, 300, 400), dtype="float32")
    tree = MapAdapter({"data": ArrayAdapter.from_array(arr)})
    app = build_app(tree)

    with Context.from_app(app, max_connections=MAX_CONNECTIONS) as context:
        sem = TrackingSemaphore(MAX_CONNECTIONS)
        context._concurrent_request_semaphore = sem

        client = from_context(context, structure_clients="dask")["data"]
        client.read().compute()

    assert sem.peak <= MAX_CONNECTIONS
    # Sanity: 10 chunks > MAX_CONNECTIONS, so the cap had something to constrain.
    assert sem.peak > 0


def test_semaphore_limits_concurrent_partition_fetches():
    "When dask computes a partitioned dataframe the semaphore must cap concurrent fetches"

    MAX_CONNECTIONS = 2
    N_PARTITIONS = 8  # well above the cap

    df = pandas.DataFrame({"x": numpy.arange(N_PARTITIONS * 10, dtype="float64")})
    tree = MapAdapter(
        {"data": DataFrameAdapter.from_pandas(df, npartitions=N_PARTITIONS)}
    )
    app = build_app(tree)

    with Context.from_app(app, max_connections=MAX_CONNECTIONS) as context:
        sem = TrackingSemaphore(MAX_CONNECTIONS)
        context._concurrent_request_semaphore = sem

        client = from_context(context, structure_clients="dask")["data"]
        client.read().compute()

    assert sem.peak <= MAX_CONNECTIONS
    assert sem.peak > 0


# --- Progress bar tests ---


@pytest.mark.parametrize(
    "show_progress, total, expect_state",
    [
        (True, 5, True),  # normal: state allocated
        (False, 10, False),  # show_progress=False → no-op
        (True, 1, False),  # total<=1 → no-op even with show_progress
    ],
    ids=["active", "show_progress_false", "total<=1"],
)
def test_tracking_progress_state_lifecycle(show_progress, total, expect_state):
    """tracking_progress sets _progress_state during the context (if and
    only if a bar is actually rendered) and clears it on exit."""
    from unittest.mock import patch

    tree = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree)

    with Context.from_app(app, show_progress=show_progress) as context:
        with patch("tiled.client.utils.is_interactive", return_value=True):
            with context.tracking_progress(total=total):
                if expect_state:
                    assert context.progress_state is not None
                else:
                    assert context.progress_state is None
        assert context.progress_state is None


def test_tracking_progress_nesting_defers_to_outer():
    """Nested tracking_progress reuses the outer state — no nested bars."""
    from unittest.mock import patch

    tree = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree)

    with Context.from_app(app, show_progress=True) as context:
        with patch("tiled.client.utils.is_interactive", return_value=True):
            with context.tracking_progress(total=10):
                outer = context.progress_state
                assert outer is not None
                with context.tracking_progress(total=5):
                    assert context.progress_state is outer
            assert context.progress_state is None


@pytest.mark.parametrize(
    "env_value, explicit, expected",
    [
        (None, None, True),  # default: env unset, no explicit → True
        ("1", None, True),
        ("0", None, False),
        ("false", None, False),
        ("no", None, False),
        ("0", True, True),  # explicit overrides env
        ("1", False, False),
    ],
)
def test_show_progress_resolution(monkeypatch, env_value, explicit, expected):
    """show_progress is set from the explicit kwarg if given, else from
    TILED_SHOW_PROGRESS, else defaults to True.  Recognised falsy values are
    "0", "false", "no" (case-insensitive)."""
    tree_local = MapAdapter({})
    app = build_app(tree_local)
    if env_value is None:
        monkeypatch.delenv("TILED_SHOW_PROGRESS", raising=False)
    else:
        monkeypatch.setenv("TILED_SHOW_PROGRESS", env_value)
    kwargs = {} if explicit is None else {"show_progress": explicit}
    with Context.from_app(app, **kwargs) as context:
        assert context.show_progress is expected


# --- Retry indicator tests ---


def test_progress_state_show_hide_retrying():
    """ProgressState.show_retrying/hide_retrying toggle state and update Live."""
    from unittest.mock import patch

    tree = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree)

    with Context.from_app(app, show_progress=True) as context:
        with patch("tiled.client.utils.is_interactive", return_value=True):
            with context.tracking_progress(total=5) as state:
                assert state._retrying is False

                # Show retrying
                state.show_retrying()
                assert state._retrying is True

                # Calling again is a no-op
                state.show_retrying()
                assert state._retrying is True

                # Hide retrying
                state.hide_retrying()
                assert state._retrying is False

                # Calling hide again is a no-op
                state.hide_retrying()
                assert state._retrying is False


def test_standalone_retry_indicator_show_hide():
    """StandaloneRetryIndicator: plain text on non-TTY, Live spinner on TTY."""
    import sys
    from io import StringIO
    from unittest.mock import MagicMock, patch

    from tiled.client.utils import StandaloneRetryIndicator

    # Non-TTY: plain text written once, reset() is a no-op
    indicator = StandaloneRetryIndicator()
    assert indicator._showing is False

    fake_stderr = StringIO()
    with patch.object(sys, "stderr", fake_stderr):
        indicator.show()
        assert indicator._showing is True
        assert "Retrying" in fake_stderr.getvalue()

        # Second show() is a no-op
        fake_stderr.truncate(0)
        fake_stderr.seek(0)
        indicator.show()
        assert fake_stderr.getvalue() == ""

        indicator.reset()
        assert indicator._showing is False

        # reset() again is a no-op
        indicator.reset()
        assert indicator._showing is False

    # TTY: show() starts a Live spinner; reset() stops it
    mock_live = MagicMock()
    indicator2 = StandaloneRetryIndicator()
    with patch("rich.live.Live", return_value=mock_live):
        tty_stderr = MagicMock()
        tty_stderr.isatty = lambda: True
        with patch.object(sys, "stderr", tty_stderr):
            indicator2.show()
            assert indicator2._live is mock_live
            mock_live.start.assert_called_once()

            # Second show() is a no-op
            indicator2.show()
            mock_live.start.assert_called_once()

            indicator2.reset()
            assert indicator2._live is None
            mock_live.stop.assert_called_once()

            # reset() again is a no-op
            indicator2.reset()
            mock_live.stop.assert_called_once()


@pytest.mark.parametrize("show_progress", [True, False])
def test_signal_retry_uses_standalone_indicator(show_progress):
    """signal_retry creates a standalone indicator regardless of show_progress
    when no progress bar is active."""
    import sys
    from io import StringIO
    from unittest.mock import patch

    tree = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree)

    with Context.from_app(app, show_progress=show_progress) as context:
        assert context.retry_indicator is None

        fake_stderr = StringIO()
        with patch.object(sys, "stderr", fake_stderr):
            context.signal_retry()
            assert context.retry_indicator is not None
            assert context.retry_indicator._showing is True
            assert "Retrying" in fake_stderr.getvalue()

            context.signal_retry_resolved()
            assert context.retry_indicator is None


def test_retry_context_without_context_shows_indicator():
    """retry_context() with no Context still shows a standalone retry
    indicator (used by from_any_uri before a Context exists)."""
    import sys
    from io import StringIO
    from unittest.mock import patch

    from tiled.client.utils import retry_context

    fake_stderr = StringIO()
    with patch.object(sys, "stderr", fake_stderr):
        try:
            stamina.set_active(True)
            call_count = 0
            for attempt in retry_context():
                with attempt:
                    call_count += 1
                    if call_count < 3:
                        raise httpx.ConnectError("test")
            assert "Retrying" in fake_stderr.getvalue()
        finally:
            stamina.set_active(False)


@pytest.mark.parametrize("with_progress_state", [True, False])
def test_signal_retry_refcounted(with_progress_state):
    """signal_retry / signal_retry_resolved keep the indicator visible until
    the last of any number of concurrent retries resolves.  Verified for
    both the progress-bar path and the standalone path.  Extra
    signal_retry_resolved() calls are no-ops."""
    from unittest.mock import MagicMock

    tree = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree)

    with Context.from_app(app, show_progress=True) as context:
        mock_state = MagicMock() if with_progress_state else None
        if mock_state is not None:
            context._progress_state = mock_state

        # Extra resolved with no active retry — no-op.
        context.signal_retry_resolved()
        context.signal_retry_resolved()

        context.signal_retry()  # A: 0→1, shows
        ind_after_A = context._retry_indicator
        context.signal_retry()  # B: 1→2, no-op
        if mock_state is None:
            assert context._retry_indicator is ind_after_A
            assert ind_after_A is not None and ind_after_A._showing is True
        else:
            assert mock_state.show_retrying.call_count == 1
            assert mock_state.hide_retrying.call_count == 0

        context.signal_retry_resolved()  # A done: 2→1, still showing
        if mock_state is None:
            assert (
                context._retry_indicator is ind_after_A
                and ind_after_A._showing is True
            )
        else:
            assert mock_state.hide_retrying.call_count == 0

        context.signal_retry_resolved()  # B done: 1→0, hide
        if mock_state is None:
            assert context._retry_indicator is None
            assert ind_after_A._showing is False
        else:
            assert mock_state.hide_retrying.call_count == 1

        # Extra resolved past zero — no-op, no crash.
        context.signal_retry_resolved()

        if mock_state is not None:
            context._progress_state = None


def test_retry_loop_balances_signal_calls():
    """Each retry loop must call signal_retry exactly once (on the first
    retry, so the user sees feedback during the wait) and
    signal_retry_resolved exactly once (on loop exit).  Otherwise the
    indicator refcount leaks and the spinner stays visible after success.
    """
    from unittest.mock import MagicMock

    from tiled.client.utils import retry_context

    tree = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree)

    with Context.from_app(app, show_progress=True) as context:
        mock_state = MagicMock()
        context._progress_state = mock_state
        try:
            stamina.set_active(True)
            n = 0
            for attempt in retry_context(context):
                with attempt:
                    n += 1
                    if n < 4:  # 3 retries scheduled
                        raise httpx.ConnectError("transient")
        finally:
            stamina.set_active(False)
            context._progress_state = None

        assert n == 4
        assert mock_state.show_retrying.call_count == 1, (
            f"signal_retry must fire exactly once per loop, "
            f"got {mock_state.show_retrying.call_count}"
        )
        assert mock_state.hide_retrying.call_count == 1
        assert context._retry_count == 0, (
            f"refcount leaked: {context._retry_count} after a successful loop"
        )


def test_keyboard_interrupt_cancels_retries_and_cleans_up():
    """KeyboardInterrupt during inter-retry sleep propagates immediately,
    aborts further retries, and lets the indicator be cleaned up."""
    from unittest.mock import MagicMock, patch

    from tiled.client.utils import retry_context

    tree = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree)

    attempts_made = []

    def fake_sleep(_):
        raise KeyboardInterrupt("simulated Ctrl-C")

    with Context.from_app(app, show_progress=True) as context:
        mock_state = MagicMock()
        context._progress_state = mock_state

        try:
            stamina.set_active(True)
            with patch("time.sleep", fake_sleep):
                with pytest.raises(KeyboardInterrupt):
                    for attempt in retry_context(context):
                        with attempt:
                            attempts_made.append(attempt.num)
                            raise httpx.ConnectError("refused")
        finally:
            stamina.set_active(False)
            context._progress_state = None

        # Only one attempt — Ctrl-C during the sleep before attempt 2.
        assert len(attempts_made) == 1, (
            f"expected 1 attempt, got {len(attempts_made)}"
        )
        mock_state.hide_retrying.assert_called()


@pytest.mark.parametrize(
    "exc_factory, expected",
    [
        (
            lambda: httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", "http://example.com/test"),
                response=httpx.Response(
                    429,
                    headers={"Retry-After": "2.5"},
                    request=httpx.Request("GET", "http://example.com/test"),
                ),
            ),
            2.5,
        ),
        (
            lambda: httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", "http://example.com/test"),
                response=httpx.Response(
                    429, request=httpx.Request("GET", "http://example.com/test")
                ),
            ),
            True,
        ),
        (
            lambda: httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", "http://example.com/test"),
                response=httpx.Response(
                    403, request=httpx.Request("GET", "http://example.com/test")
                ),
            ),
            False,
        ),
        (lambda: httpx.UnsupportedProtocol("'htps://'."), False),
        (lambda: httpx.LocalProtocolError("Illegal header value"), False),
    ],
    ids=[
        "429-retry-after",
        "429-no-header",
        "403-no-retry",
        "unsupported-protocol",
        "local-protocol-error",
    ],
)
def test_should_retry(exc_factory, expected):
    """should_retry returns Retry-After float, True, or False depending on
    the exception kind."""
    from tiled.client.utils import should_retry

    assert should_retry(exc_factory()) == expected


def test_handle_error_lets_429_propagate():
    """handle_error does not convert 429 into ClientError — lets it propagate for retry."""
    from tiled.client.utils import handle_error

    request = httpx.Request("GET", "http://example.com/test")
    response = httpx.Response(429, headers={"Retry-After": "5"}, request=request)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        handle_error(response)
    # It should be a plain HTTPStatusError, not a ClientError
    from tiled.client.utils import ClientError

    assert not isinstance(exc_info.value, ClientError)
    assert exc_info.value.response.status_code == 429


def test_retry_context_logs_429_retry(caplog):
    """429 retries are logged at DEBUG on tiled.client logger."""
    show_logs()
    stamina.set_active(True)
    try:
        attempts_made = 0
        with caplog.at_level(logging.DEBUG, logger="tiled.client"):
            for attempt in retry_context():
                with attempt:
                    attempts_made += 1
                    if attempts_made < 2:
                        request = httpx.Request("GET", "http://example.com/data")
                        response = httpx.Response(
                            429,
                            headers={"Retry-After": "0"},
                            request=request,
                        )
                        raise httpx.HTTPStatusError(
                            "Too Many Requests", request=request, response=response
                        )

        tiled_messages = [r for r in caplog.records if r.name == "tiled.client"]
        assert len(tiled_messages) >= 1
        assert "Retry" in tiled_messages[0].message
    finally:
        stamina.set_active(False)
        hide_logs()


def test_429_retry_with_real_server():
    """Integration test: client retries on 429 from a mock transport."""
    from tiled.client.utils import handle_error

    call_count = {"n": 0}

    class ThrottlingTransport(httpx.BaseTransport):
        def handle_request(self, request):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return httpx.Response(
                    429,
                    headers={"Retry-After": "0"},
                )
            return httpx.Response(200, json={"status": "ok"})

    stamina.set_active(True)
    try:
        with httpx.Client(transport=ThrottlingTransport()) as client:
            for attempt in retry_context():
                with attempt:
                    response = client.get("http://testserver/data")
                    handle_error(response)

        assert call_count["n"] == 3
        assert response.json() == {"status": "ok"}
    finally:
        stamina.set_active(False)


# --- Multi-threaded retry indicator + cancellation tests ---
#
# These exercise the realistic dask worker scenario, where retry attempts run
# on a non-main thread.


def test_signal_retry_from_worker_thread_is_visible():
    import sys
    from io import StringIO
    from unittest.mock import patch

    tree_local = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree_local)

    with Context.from_app(app, show_progress=False) as context:
        fake_stderr = StringIO()

        def worker():
            with patch.object(sys, "stderr", fake_stderr):
                context.signal_retry()

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "worker thread did not finish"

        # The retry indicator should have been created AND marked as showing,
        # and *something* (text or a Live spinner start) should have been emitted.
        assert (
            context.retry_indicator is not None
        ), "signal_retry from a worker thread did not create an indicator"
        assert (
            context.retry_indicator._showing is True
        ), "signal_retry from a worker thread did not mark indicator as showing"


def test_small_fetch_worker_retry_is_visible():
    """When tracking_progress is a no-op (total<=1, non-interactive), a
    retry running in a dask worker thread must still produce visible
    feedback via the standalone indicator, and signal_retry called from
    that worker must drive show() to completion."""
    from unittest.mock import patch

    tree_local = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree_local)

    with Context.from_app(app, show_progress=True) as context:
        with context.tracking_progress(total=1):
            # tracking_progress(total=1) is a no-op; _progress_state stays None.
            assert context.progress_state is None

            shown = []
            original_show = (
                __import__(
                    "tiled.client.utils", fromlist=["StandaloneRetryIndicator"]
                ).StandaloneRetryIndicator.show
            )

            def tracking_show(self):
                shown.append(threading.current_thread())
                original_show(self)

            attempts = []
            errors = []

            def worker():
                try:
                    stamina.set_active(True)
                    for attempt in retry_context(context):
                        with attempt:
                            attempts.append(1)
                            if len(attempts) < 3:
                                raise httpx.ConnectError("simulated transient")
                except Exception as e:
                    errors.append(e)
                finally:
                    stamina.set_active(False)

            with patch(
                "tiled.client.utils.StandaloneRetryIndicator.show", tracking_show
            ):
                t = threading.Thread(target=worker)
                t.start()
                t.join(timeout=10.0)

            assert not t.is_alive(), "worker did not finish"
            assert not errors, f"worker raised: {errors}"
            assert len(attempts) == 3, f"expected 3 attempts, got {len(attempts)}"
            assert shown, "StandaloneRetryIndicator.show was not called from worker"
            assert shown[0] is t, "show() ran on a different thread than the worker"


def test_cancel_event_terminates_worker_retry_loop():
    """A Ctrl-C must be able to terminate retries running in dask
    worker threads.  SIGINT is delivered only to the main thread, so the
    Context must expose a cancellation Event that worker retry loops check
    before sleeping again.  Once set, a worker should exit the retry loop
    promptly instead of running through TILED_RETRY_ATTEMPTS.
    """
    tree_local = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree_local)

    with Context.from_app(app) as context:
        # The Context must expose a cancellation Event accessible to workers.
        assert hasattr(
            context, "cancel_event"
        ), "Context must expose a cancel_event for worker-thread Ctrl-C handling"
        assert isinstance(context.cancel_event, threading.Event)
        assert not context.cancel_event.is_set()

        attempts_made = []
        worker_finished = threading.Event()
        worker_exception = []

        def worker():
            try:
                stamina.set_active(True)
                for attempt in retry_context(context):
                    with attempt:
                        attempts_made.append(len(attempts_made) + 1)
                        raise httpx.ConnectError("refused")
            except Exception as e:
                worker_exception.append(e)
            finally:
                stamina.set_active(False)
                worker_finished.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # Let the worker make at least one attempt and enter stamina's sleep.
        import time as _time

        _time.sleep(0.5)
        assert len(attempts_made) >= 1

        # Simulate Ctrl-C cancellation from the main thread.
        context.cancel_event.set()

        # Worker should exit promptly — well before all 10 attempts complete.
        assert worker_finished.wait(timeout=3.0), (
            f"worker did not terminate after cancel_event set; "
            f"attempts_made={attempts_made}"
        )
        # And it should not have run through all attempts.
        from tiled.client.utils import TILED_RETRY_ATTEMPTS

        assert len(attempts_made) < TILED_RETRY_ATTEMPTS, (
            f"cancel_event did not abort worker retries: "
            f"made {len(attempts_made)}/{TILED_RETRY_ATTEMPTS} attempts"
        )


def test_standalone_retry_indicator_show_reset_race_terminal():
    """Race: reset() runs between live.start() and storing self._live.

    Without protection, the Rich Live spinner would keep rendering forever
    because reset() saw self._live == None and bailed out.  The indicator
    must detect the racing reset after creation and tear down immediately.
    """
    import sys
    from unittest.mock import MagicMock, patch

    from tiled.client.utils import StandaloneRetryIndicator

    indicator = StandaloneRetryIndicator()
    mock_live = MagicMock()

    # Simulate the race: reset() fires during live.start().
    def racing_start():
        indicator.reset()

    mock_live.start.side_effect = racing_start

    tty_stderr = MagicMock()
    tty_stderr.isatty = lambda: True
    with patch("rich.live.Live", return_value=mock_live), patch.object(
        sys, "stderr", tty_stderr
    ):
        indicator.show()

    # The Live spinner must have been stopped — otherwise it leaks and
    # renders forever.
    mock_live.stop.assert_called_once()
    assert indicator._showing is False
    assert indicator._live is None


def test_standalone_retry_indicator_jupyter_creates_widget_synchronously():
    """In Jupyter, StandaloneRetryIndicator.show() must create and display
    the ipywidget directly from the calling thread.

    Scheduling the create on the kernel IOLoop (e.g. via
    ``IOLoop.add_callback``) silently breaks single-fetch retry feedback:
    during synchronous cell execution the kernel IOLoop is blocked, so a
    queued create callback only runs after the cell completes — by which
    time the retry has either resolved or failed and the indicator is
    useless.
    """
    import sys
    from unittest.mock import MagicMock, patch

    from tiled.client.utils import StandaloneRetryIndicator

    fake_label = MagicMock()
    fake_widgets = MagicMock()
    fake_widgets.Label.return_value = fake_label
    fake_ipython_display = MagicMock()

    indicator = StandaloneRetryIndicator()

    with patch.dict(
        sys.modules,
        {"ipywidgets": fake_widgets, "IPython.display": fake_ipython_display},
    ), patch("tiled.client.utils.is_jupyter", return_value=True):
        indicator.show()

    # display() must have been called synchronously during show() — not
    # deferred onto some queue that never runs.
    fake_ipython_display.display.assert_called_once_with(fake_label)
    assert indicator._showing is True
    assert indicator._live is fake_label


def test_circuit_breaker_blocks_new_fetches_while_any_worker_retries():
    """When one worker is actively retrying (signal_retry has fired),
    other workers calling ``wait_for_circuit()`` must block until the
    retry resolves.  Workers already past ``wait_for_circuit()`` (i.e.
    already in-flight) are NOT interrupted.
    """
    import time as _time

    tree_local = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree_local)

    with Context.from_app(app) as context:
        assert context._circuit_closed_event.is_set(), "circuit closed by default"

        # Worker A enters a retry — opens the circuit.
        context.signal_retry()
        assert not context._circuit_closed_event.is_set()

        # Worker B tries to start a new fetch; it must block.
        unblocked_at = []

        def peer():
            context.wait_for_circuit(poll_interval=0.05)
            unblocked_at.append(_time.monotonic())

        t = threading.Thread(target=peer)
        t.start()
        _time.sleep(0.2)
        assert t.is_alive(), "peer should still be blocked on circuit gate"

        # Worker A's retry resolves — closes the circuit.
        context.signal_retry_resolved()
        assert context._circuit_closed_event.is_set()

        t.join(timeout=2.0)
        assert not t.is_alive(), "peer did not unblock after retry resolved"
        assert len(unblocked_at) == 1


def test_circuit_breaker_releases_on_cancel():
    """``request_cancel()`` must unblock workers waiting on the circuit
    gate so a Ctrl-C doesn't leave dask workers parked forever.
    """
    import time as _time

    tree_local = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree_local)

    with Context.from_app(app) as context:
        context.signal_retry()  # open circuit
        unblocked = threading.Event()

        def peer():
            context.wait_for_circuit(poll_interval=0.05)
            unblocked.set()

        t = threading.Thread(target=peer)
        t.start()
        _time.sleep(0.2)
        assert not unblocked.is_set()

        context.request_cancel()
        assert unblocked.wait(timeout=2.0), (
            "request_cancel did not release workers blocked on circuit gate"
        )

        # Cleanup: balance the signal_retry so the next test starts clean.
        context.reset_cancel()
        context.signal_retry_resolved()


def test_retry_exhaustion_aborts_peer_retry_loops():
    """When one worker exhausts retries (or hits a non-retriable error),
    ``retry_context`` must set the Context's ``cancel_event`` so peer
    workers stuck in their own retry loops bail out promptly instead of
    each running through their own full attempt budget.
    """
    import tiled.client.utils as cu

    tree_local = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree_local)

    orig_attempts = cu.TILED_RETRY_ATTEMPTS
    orig_timeout = cu.TILED_RETRY_TIMEOUT
    cu.TILED_RETRY_ATTEMPTS = 3
    cu.TILED_RETRY_TIMEOUT = 1.0

    try:
        with Context.from_app(app) as context:
            stamina.set_active(True)
            try:
                # Failing worker — runs to exhaustion synchronously.
                with pytest.raises(httpx.ConnectError):
                    for attempt in retry_context(context):
                        with attempt:
                            raise httpx.ConnectError("permanent")

                # Peer worker's cancel_event must be set now.
                assert context.cancel_event.is_set(), (
                    "exhausted retry_context did not request_cancel; peers "
                    "would each run through their own attempt budget"
                )
            finally:
                stamina.set_active(False)
    finally:
        cu.TILED_RETRY_ATTEMPTS = orig_attempts
        cu.TILED_RETRY_TIMEOUT = orig_timeout


def test_stale_cancel_event_does_not_starve_fresh_fetch_of_retries():
    """A prior failed fetch leaves ``cancel_event`` set.  A subsequent
    user-initiated single-fetch (no ``tracking_progress`` wrapper) must
    still get its full retry budget — the stale flag must be cleared
    before the new retry loop begins.
    """
    import tiled.client.utils as cu

    tree_local = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree_local)

    orig_attempts = cu.TILED_RETRY_ATTEMPTS
    orig_timeout = cu.TILED_RETRY_TIMEOUT
    cu.TILED_RETRY_ATTEMPTS = 3
    cu.TILED_RETRY_TIMEOUT = 1.0

    try:
        with Context.from_app(app) as context:
            stamina.set_active(True)
            try:
                # Step 1: exhaust a retry loop to leave cancel_event set
                # (the documented post-exhaustion behaviour).
                with pytest.raises(httpx.ConnectError):
                    for attempt in retry_context(context):
                        with attempt:
                            raise httpx.ConnectError("permanent")
                assert context.cancel_event.is_set()
                assert context._retry_count == 0

                # Step 2: a single-fetch entry point calls wait_for_circuit
                # (mirroring array._get_block / dataframe._get_partition).
                # It must clear the stale flag so the new retry loop sees
                # a fresh cancel state.
                context.wait_for_circuit(poll_interval=0.05)
                assert not context.cancel_event.is_set(), (
                    "wait_for_circuit did not clear stale cancel_event; "
                    "the next fetch will get zero retries"
                )

                # Step 3: confirm a transient failure actually retries.
                calls = {"n": 0}
                for attempt in retry_context(context):
                    with attempt:
                        calls["n"] += 1
                        if calls["n"] < 2:
                            raise httpx.ConnectError("transient")
                assert calls["n"] == 2, (
                    f"fresh fetch did not retry: only {calls['n']} attempt(s)"
                )
            finally:
                stamina.set_active(False)
                context.reset_cancel()
    finally:
        cu.TILED_RETRY_ATTEMPTS = orig_attempts
        cu.TILED_RETRY_TIMEOUT = orig_timeout


def test_progress_advance_clamped_at_total():
    # --- Terminal ProgressState ---
    from rich.console import Console
    from rich.progress import BarColumn, MofNCompleteColumn, Progress

    from tiled.client.utils import ProgressState

    console = Console(stderr=True, force_terminal=True)
    progress = Progress(BarColumn(), MofNCompleteColumn(), console=console)
    task_id = progress.add_task("test", total=3)
    state = ProgressState(progress, task_id, spinner=None)
    for _ in range(5):
        state.advance()
    assert progress.tasks[task_id].completed == 3, (
        f"terminal advance overshot: completed="
        f"{progress.tasks[task_id].completed}"
    )

    # --- Jupyter shim ---
    tree_local = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree_local)

    import sys
    from unittest.mock import MagicMock, patch

    fake_widgets = MagicMock()
    fake_widgets.IntProgress.return_value = MagicMock(value=0)
    fake_widgets.HTML.return_value = MagicMock(value="")
    fake_widgets.HBox.return_value = MagicMock()
    fake_widgets.Layout.return_value = MagicMock()

    with Context.from_app(app, show_progress=True) as context:
        with patch.dict(sys.modules, {"ipywidgets": fake_widgets,
                                       "IPython.display": MagicMock()}), patch(
            "tiled.client.utils.is_jupyter", return_value=True
        ), patch("tiled.client.utils.is_interactive", return_value=True):
            with context.tracking_progress(total=3) as jstate:
                for _ in range(5):
                    jstate.advance()
                assert jstate._completed == 3, (
                    f"jupyter advance overshot: completed={jstate._completed}"
                )


def test_terminal_progress_does_not_top_up_on_exception():
    """When the fetch raises, the progress bar must NOT be quietly filled
    to 100%.  That would mislead the user into thinking the fetch
    succeeded right before the exception surfaces.
    """
    from unittest.mock import patch

    tree_local = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree_local)

    captured = {"state": None}

    with Context.from_app(app, show_progress=True) as context:
        with patch("tiled.client.utils.is_jupyter", return_value=False), patch(
            "tiled.client.utils.is_interactive", return_value=True
        ):
            with pytest.raises(RuntimeError, match="boom"):
                with context.tracking_progress(total=5) as state:
                    captured["state"] = state
                    state.advance()
                    state.advance()
                    raise RuntimeError("boom")

    state = captured["state"]
    assert state is not None
    # Two real advances happened.  The top-up must NOT have filled the
    # bar to total=5 after the exception.
    completed = state._progress.tasks[state._task_id].completed
    assert completed == 2, (
        f"top-up ran on exception path: completed={completed}, expected 2"
    )


def test_standalone_retry_indicator_show_reset_race_jupyter():
    """Race in Jupyter path: reset() runs between display(label) and the
    lock-protected ``self._live = label`` assignment in ``show()``.

    The widget would otherwise be displayed and never closed.  ``show()``
    must re-check ``_showing`` under lock after ``display()`` and close
    the widget if a concurrent reset happened.
    """
    import sys
    from unittest.mock import MagicMock, patch

    from tiled.client.utils import StandaloneRetryIndicator

    indicator = StandaloneRetryIndicator()

    fake_label = MagicMock()
    fake_widgets = MagicMock()
    fake_widgets.Label.return_value = fake_label
    fake_ipython_display = MagicMock()

    # Simulate the race by having display() itself call reset() before it
    # returns — mimicking another thread that flips _showing to False
    # between widget creation and the orphan check.
    def racing_display(*args, **kwargs):
        indicator.reset()

    fake_ipython_display.display.side_effect = racing_display

    with patch.dict(
        sys.modules,
        {"ipywidgets": fake_widgets, "IPython.display": fake_ipython_display},
    ), patch("tiled.client.utils.is_jupyter", return_value=True):
        indicator.show()

    # The widget must have been closed; otherwise it stays visible forever.
    fake_label.close.assert_called_once()
    assert indicator._live is None
    assert indicator._showing is False


def test_tracking_progress_cancel_event_lifecycle():
    """The outermost tracking_progress clears any stale cancel flag from a
    previously interrupted fetch; a nested entry must not clear a flag set
    by its outer caller.
    """
    tree_local = MapAdapter({"data": ArrayAdapter.from_array(numpy.zeros((10,)))})
    app = build_app(tree_local)

    with Context.from_app(app, show_progress=True) as context:
        # Outermost entry clears stale state.
        context.request_cancel()
        assert context.cancel_event.is_set()
        with context.tracking_progress(total=5):
            assert not context.cancel_event.is_set(), (
                "outermost tracking_progress did not clear stale cancel flag"
            )
            # Inner caller sets cancel; nested entry must not clobber it.
            context.request_cancel()
            with context.tracking_progress(total=1):
                assert context.cancel_event.is_set(), (
                    "nested tracking_progress cleared an outer cancel signal"
                )
            assert context.cancel_event.is_set()


def test_stamina_default_log_hook_stripped_lazily():
    """Stamina's built-in retry log hook must be removed before tiled's
    retry path runs, but third-party hooks registered later must survive.
    """
    import stamina.instrumentation as si

    from tiled.client import utils as client_utils

    sentinel_calls = []

    def sentinel_hook(details):
        sentinel_calls.append(details)

    try:
        from stamina.instrumentation._logging import init_logging
        si.set_on_retry_hooks([init_logging(), sentinel_hook])
    except Exception:
        si.set_on_retry_hooks(list(si.get_on_retry_hooks()) + [sentinel_hook])

    # Reset the lazy guard so we exercise the strip path even if a prior
    # test already triggered it.
    client_utils._stamina_log_hook_stripped = False
    client_utils._strip_stamina_default_log_hook()

    remaining = si.get_on_retry_hooks()
    assert sentinel_hook in remaining, (
        "third-party stamina hook was evicted by tiled"
    )
    assert not any(
        getattr(h, "__qualname__", "").startswith("init_logging") for h in remaining
    ), "stamina default log hook was not stripped"


def test_stamina_hook_strip_is_lazy_not_at_import():
    """Importing tiled.client.utils must not mutate stamina's global hook
    list — stripping happens lazily on first retry entry.
    """
    import importlib
    import sys

    import stamina.instrumentation as si

    sentinel = lambda details: None  # noqa: E731
    si.set_on_retry_hooks([sentinel])
    # Force a reimport of tiled.client.utils.
    for name in list(sys.modules):
        if name.startswith("tiled.client.utils"):
            del sys.modules[name]
    importlib.import_module("tiled.client.utils")
    # Sentinel must still be present after import.
    assert sentinel in si.get_on_retry_hooks(), (
        "tiled.client.utils import wiped stamina hooks; strip must be lazy"
    )
