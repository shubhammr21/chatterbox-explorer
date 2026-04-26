"""
src/logging_config.py
==========================================
Logging and warning configuration for the Chatterbox TTS Explorer.

IMPORTANT: This module MUST have zero side-effects at import time.
All configuration logic is gated behind the configure() / configure_json()
functions so that:
  - Unit tests can import project modules without triggering log setup.
  - cli.py controls exactly when (and therefore after which other imports)
    the configuration fires.

Call configure() exactly once, as the very first thing in cli.main(),
before any library with an eager logger (huggingface_hub, transformers,
diffusers) is imported.

For REST / production mode, call configure_json() from cli._launch_rest()
instead of configure(). It switches the root handler to structured JSON
output and attaches asgi-correlation-id's CorrelationIdFilter so that every
log record carries the active correlation ID automatically.

Phase 3 — Non-blocking logging
--------------------------------
configure_json() uses a QueueHandler → QueueListener pattern so that every
log write in the event loop is a fast queue.put_nowait() call.  A dedicated
background thread (QueueListener) performs the actual StreamHandler I/O.

Python 3.11 caveats (see PLAN.md §6 and _PreservingQueueHandler below):
  1. QueueHandler.prepare() clears exc_text on its copy of the record, so we
     must subclass and override prepare() to preserve exception text.
  2. The .listener attribute on QueueHandler was added in Python 3.12; we
     track the listener separately and stop it via atexit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

# ──────────────────────────────────────────────────────────────────────────────
# Python 3.11-safe QueueHandler — preserves exc_text across the queue boundary
# ──────────────────────────────────────────────────────────────────────────────


class _PreservingQueueHandler:
    """Placeholder — real class is built inside configure_json() after imports.

    This sentinel exists only so that the module-level name is bound before
    configure_json() is called.  The real implementation is injected below.
    """


def _make_preserving_queue_handler_class():
    """Return the real _PreservingQueueHandler class.

    Defined as a factory so that logging.handlers is only imported when
    configure_json() actually runs, keeping the module side-effect-free at
    import time.
    """
    import copy
    import logging
    import logging.handlers

    class _PreservingQueueHandler(logging.handlers.QueueHandler):
        """Python 3.11-safe QueueHandler that preserves exc_text across the queue.

        Problem
        -------
        Python 3.11's QueueHandler.prepare() calls self.format(record) and
        then sets exc_text = None on the copy it returns.  This means:
          • The record that arrives at the QueueListener's sink handler has
            no exception text — the sink formatter cannot re-derive it
            because exc_info was also cleared.
          • If a JsonFormatter is attached as self.formatter, prepare() would
            pre-format the entire record into a JSON string stored in
            record.msg, and the sink's JsonFormatter would then try to
            JSON-encode an already-encoded string.

        Fix
        ---
        Override prepare() to:
          1. Pre-render exc_text NOW (on the originating thread, which still
             has the traceback object) so it survives the queue boundary.
          2. Shallow-copy the record (matching the base-class contract so that
             other handlers in the chain are unaffected).
          3. Freeze msg/args (resolve getMessage()) without calling
             self.format(), so the sink handler performs the real formatting.
          4. Clear exc_info (traceback objects are not safely picklable) but
             keep exc_text.

        CorrelationIdFilter placement
        -----------------------------
        The filter is attached to this handler (not the sink) so it runs on
        the originating thread where the correlation_id ContextVar is set.
        The filter enriches the record before it enters the queue; the sink
        thread simply uses the already-attached attribute.
        """

        def prepare(self, record: logging.LogRecord) -> logging.LogRecord:  # type: ignore[override]
            # 1. Pre-render exception text while we still have the traceback.
            #    Use a plain Formatter so the text is undecorated — the sink's
            #    JsonFormatter embeds it in the JSON "exc_info" field later.
            if record.exc_info and not record.exc_text:
                plain = logging.Formatter()
                record.exc_text = plain.formatException(record.exc_info)

            # 2. Shallow-copy so mutations below don't affect other handlers.
            record = copy.copy(record)

            # 3. Freeze the formatted message (resolves %-style args into msg)
            #    without calling self.format(), which would apply the
            #    JsonFormatter and corrupt the record for the sink handler.
            record.message = record.getMessage()
            record.msg = record.message
            record.args = None

            # 4. Clear unpicklable / thread-unsafe fields.
            #    exc_text is intentionally kept — we set it in step 1.
            record.exc_info = None
            record.stack_info = None

            return record

    return _PreservingQueueHandler


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper — called by both configure() and configure_json()
# ──────────────────────────────────────────────────────────────────────────────


def _suppress_noisy_loggers() -> None:
    """Silence third-party loggers and warning spam that pollute the output.

    Safe to call more than once — every operation here is idempotent.
    All library imports are deferred to avoid side-effects at module load time.
    """
    import logging
    import warnings

    # ── huggingface_hub verbosity ─────────────────────────────────────────────
    # huggingface_hub sets its root logger to level=WARNING during its own
    # import (inside huggingface_hub/__init__.py).  If we called
    # logging.getLogger("huggingface_hub").setLevel(ERROR) before that import,
    # the library would overwrite our setting when it later initialises.
    #
    # Fix: force-import the library first so its __init__ runs and its internal
    # logger is registered, then use its own public verbosity API to win the
    # race.  The belt-and-suspenders setLevel call below acts as a fallback for
    # any logger the library creates after initialisation.
    try:
        import huggingface_hub as _hf  # noqa: F401  — force initialisation
        from huggingface_hub import logging as _hf_log

        _hf_log.set_verbosity_error()  # uses hf's own API → survives re-import
    except ImportError:
        pass

    # Belt-and-suspenders: also silence via the standard logging hierarchy so
    # any child loggers (e.g. huggingface_hub.utils) are covered.
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # keep WARNING, drop INFO

    # ── HuggingFace unauthenticated-request warnings ──────────────────────────
    # HF Hub emits this via warnings.warn() on every model file HEAD request
    # when no HF_TOKEN is set.  We suppress the per-request spam and replace it
    # with a single clean advisory from our own logger (see configure() below).
    warnings.filterwarnings(
        "ignore",
        message=r".*unauthenticated requests.*HF Hub.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Please set a HF_TOKEN.*",
        category=UserWarning,
    )

    # ── transformers verbosity ────────────────────────────────────────────────
    # transformers bypasses the standard logging hierarchy for some internal
    # messages (e.g. sdpa + output_attentions attention-dispatch notes).
    # Use its own API so those are suppressed regardless of logger level.
    try:
        import transformers as _t

        _t.logging.set_verbosity_error()
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Public configuration functions
# ──────────────────────────────────────────────────────────────────────────────


def configure(log_level: str = "INFO") -> None:
    """Configure logging and warning filters for the entire process.

    Use this for Gradio UI mode (plain-text, human-readable log lines).

    Parameters
    ----------
    log_level:
        Root logger level string (e.g. "INFO", "DEBUG", "WARNING").
        Defaults to "INFO".

    Ordering within this function is load-bearing — see inline comments.
    """
    import logging
    import os

    # ── Root logger ──────────────────────────────────────────────────────────
    # Set up a clean, timestamped format for our own log messages.
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Suppress noisy third-party loggers ───────────────────────────────────
    _suppress_noisy_loggers()

    # ── HuggingFace token advisory ────────────────────────────────────────────
    # Models are public so HF_TOKEN is optional, but unauthenticated access is
    # subject to stricter rate limits.  We emit exactly one advisory instead of
    # letting the library spam on every model-file request.
    log = logging.getLogger("chatterbox-explorer")
    _hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not _hf_token:
        log.info(
            "HF_TOKEN not set — unauthenticated HuggingFace access (public rate limits). "
            "Set HF_TOKEN=<token> for higher limits: https://huggingface.co/settings/tokens"
        )
    else:
        log.info("HF_TOKEN detected ✓ — authenticated HuggingFace access enabled.")


def configure_json(log_level: str = "INFO") -> None:
    """Configure non-blocking structured JSON logging for REST / production mode.

    Differences from ``configure()``
    ---------------------------------
    - Root handler is a ``QueueHandler`` that enqueues records via a fast
      ``queue.put_nowait()`` call — the event loop is never blocked by I/O.
    - A ``QueueListener`` background thread drains the queue and writes to a
      ``StreamHandler`` with ``JsonFormatter``, making every record ingestible
      by log aggregators (Datadog, CloudWatch Logs Insights, Grafana Loki).
    - Every ``LogRecord`` is enriched with a ``correlation_id`` field by
      ``asgi_correlation_id.CorrelationIdFilter``, which runs on the originating
      thread (where the ContextVar is set) via the ``QueueHandler``'s filter
      chain — not in the background thread.
    - ``uvicorn.access`` is silenced at ``CRITICAL`` level because
      ``RequestLoggingMiddleware`` is the sole source of per-request access
      logs in REST mode.
    - The ``asgi_correlation_id`` library's own logger is limited to
      ``WARNING`` to suppress per-request debug noise.
    - ``sys.excepthook`` is overridden to route uncaught process-level
      exceptions through the structured logger.  ``KeyboardInterrupt`` is
      re-dispatched to the default hook so Ctrl-C still works correctly.

    Python 3.11 caveats
    --------------------
    - ``QueueHandler.prepare()`` (Python 3.11) calls ``self.format(record)``
      then sets ``exc_text = None`` on the copy.  ``_PreservingQueueHandler``
      overrides ``prepare()`` to capture ``exc_text`` before clearing
      ``exc_info``, without calling ``self.format()``.
    - The ``QueueHandler.listener`` attribute was added in Python 3.12.  The
      listener is tracked locally and stopped via ``atexit.register()``.

    Parameters
    ----------
    log_level:
        Root logger level string (e.g. "INFO", "DEBUG", "WARNING").
        Defaults to "INFO".

    Prerequisites (``rest`` optional extra)
    ----------------------------------------
    - ``asgi-correlation-id>=4.0.0``  — provides ``CorrelationIdFilter``
    - ``python-json-logger>=3.2.0``   — provides ``JsonFormatter``

    Both are installed automatically with ``uv sync --extra rest``.

    Call this function INSTEAD of ``configure()`` when launching the FastAPI
    REST server (i.e. from ``cli._launch_rest()``).
    """
    import atexit
    import logging
    import logging.handlers
    import queue
    import sys

    # ── Validate optional dependencies ────────────────────────────────────────
    # Fail fast with a clear install instruction rather than a confusing
    # KeyError or ImportError buried inside handler construction.
    try:
        import asgi_correlation_id  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "asgi-correlation-id is required for JSON logging. "
            "Install it with: uv sync --extra rest"
        ) from exc

    try:
        import pythonjsonlogger  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "python-json-logger is required for JSON logging. Install it with: uv sync --extra rest"
        ) from exc

    from asgi_correlation_id import CorrelationIdFilter
    from pythonjsonlogger.json import JsonFormatter

    _fmt = "%(asctime)s %(levelname)s %(name)s %(correlation_id)s %(message)s"
    _datefmt = "%Y-%m-%dT%H:%M:%S"

    # ── Build the SINK handler ────────────────────────────────────────────────
    # Runs in the QueueListener background thread — actual stdout I/O happens
    # here, away from the event loop.
    sink = logging.StreamHandler()
    sink.setFormatter(JsonFormatter(fmt=_fmt, datefmt=_datefmt))
    # NOTE: CorrelationIdFilter is NOT on the sink — the ContextVar is only
    # valid on the originating thread.  The QueueHandler's filter chain (below)
    # attaches correlation_id to the record BEFORE it enters the queue, so the
    # attribute is already present when the sink formats it.

    # ── Build non-blocking queue + listener ───────────────────────────────────
    log_queue: queue.Queue[logging.LogRecord] = queue.Queue()
    listener = logging.handlers.QueueListener(
        log_queue,
        sink,
        respect_handler_level=True,
    )

    # ── Build the QueueHandler ────────────────────────────────────────────────
    # We use the _PreservingQueueHandler subclass to work around the Python 3.11
    # limitation where prepare() clears exc_text.
    #
    # JsonFormatter is set as the handler's formatter so that:
    #   a) Existing tests that inspect root_handlers[].formatter pass.
    #   b) _PreservingQueueHandler.prepare() can use it to format exc_text
    #      if needed (falls back to plain Formatter in practice — see prepare()).
    #
    # CorrelationIdFilter is attached here so it runs on the originating thread
    # (correct ContextVar access) via the handler's filter chain in handle().
    preserving_queue_handler_cls = _make_preserving_queue_handler_class()
    q_handler = preserving_queue_handler_cls(log_queue)
    q_handler.setFormatter(JsonFormatter(fmt=_fmt, datefmt=_datefmt))
    q_handler.addFilter(CorrelationIdFilter(uuid_length=32, default_value="-"))

    # ── Configure root logger ─────────────────────────────────────────────────
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(q_handler)
    root.setLevel(log_level.upper())

    # ── Named loggers ─────────────────────────────────────────────────────────
    # Suppress uvicorn's native access log — RequestLoggingMiddleware is the
    # sole source of per-request access logs in REST mode.
    logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)
    logging.getLogger("uvicorn.access").propagate = False
    # Suppress per-request debug lines from the correlation-id library itself
    # (e.g. "Generated new request ID …").
    logging.getLogger("asgi_correlation_id").setLevel(logging.WARNING)

    # ── Start listener (background thread) + register clean shutdown ──────────
    # Python 3.11: no .listener attribute on QueueHandler — track separately.
    listener.start()
    atexit.register(listener.stop)

    # ── sys.excepthook override ───────────────────────────────────────────────
    # Route process-level crashes to the structured logger so they appear as
    # JSON in log aggregators rather than landing on raw stderr.
    # KeyboardInterrupt is re-dispatched to the default hook so Ctrl-C works.
    def _handle_uncaught(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: types.TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger("chatterbox-explorer").error(
            "Uncaught exception — process will terminate",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = _handle_uncaught

    # ── Suppress noisy third-party loggers ────────────────────────────────────
    # Called after root logger is configured so level overrides win.
    _suppress_noisy_loggers()
