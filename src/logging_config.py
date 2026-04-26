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
"""

from __future__ import annotations

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


def configure() -> None:
    """Configure logging and warning filters for the entire process.

    Use this for Gradio UI mode (plain-text, human-readable log lines).

    Ordering within this function is load-bearing — see inline comments.
    """
    import logging
    import os

    # ── Root logger ──────────────────────────────────────────────────────────
    # Set up a clean, timestamped format for our own log messages.
    logging.basicConfig(
        level=logging.INFO,
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


def configure_json() -> None:
    """Configure structured JSON logging for the REST / production mode.

    Differences from ``configure()``
    ---------------------------------
    - Root handler emits one-line JSON objects via ``python-json-logger``,
      making every record ingestible by log aggregators (Datadog, CloudWatch
      Logs Insights, Grafana Loki) without any parsing.
    - Every ``LogRecord`` is enriched with a ``correlation_id`` field by
      ``asgi_correlation_id.CorrelationIdFilter``, which reads the active
      ``correlation_id`` ContextVar (set per-request by
      ``CorrelationIdMiddleware``).  Outside a request context the field is
      set to ``"-"`` so startup / shutdown log lines are always valid JSON.
    - ``uvicorn.access`` is silenced at ``CRITICAL`` level because
      ``RequestLoggingMiddleware`` is the sole source of per-request access
      logs in REST mode.
    - The ``asgi_correlation_id`` library's own logger is limited to
      ``WARNING`` to suppress per-request debug noise.

    Implementation
    --------------
    Uses ``logging.config.dictConfig`` — the idiomatic, fully declarative
    Python logging configuration API.  The ``CorrelationIdFilter`` is declared
    in the ``filters`` section and referenced by the handler so it enriches
    every record that flows through that handler, regardless of which logger
    emitted it.

    Prerequisites (``rest`` optional extra)
    ----------------------------------------
    - ``asgi-correlation-id>=4.0.0``  — provides ``CorrelationIdFilter``
    - ``python-json-logger>=3.2.0``   — provides ``JsonFormatter``

    Both are installed automatically with ``uv sync --extra rest``.

    Call this function INSTEAD of ``configure()`` when launching the FastAPI
    REST server (i.e. from ``cli._launch_rest()``).
    """
    # ── Validate optional dependencies before calling dictConfig ─────────────
    # Fail fast with a clear install instruction rather than a confusing
    # KeyError or ImportError buried inside dictConfig's config machinery.
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

    from logging.config import dictConfig

    dictConfig(
        {
            "version": 1,
            # Keep existing loggers (e.g. already-configured library loggers)
            # rather than resetting them.  _suppress_noisy_loggers() below
            # will then apply the correct levels on top.
            "disable_existing_loggers": False,
            # ── Filters ───────────────────────────────────────────────────
            # CorrelationIdFilter reads the correlation_id ContextVar that
            # CorrelationIdMiddleware sets for every HTTP request.
            #   uuid_length=32  — keep the full 32-char hex UUID in JSON logs
            #                     so grep/aggregators can match on the complete
            #                     ID without truncation.
            #   default_value="-" — emitted for log lines outside a request
            #                       context (startup, shutdown, background tasks)
            #                       so records remain valid JSON.
            "filters": {
                "correlation_id": {
                    "()": "asgi_correlation_id.CorrelationIdFilter",
                    "uuid_length": 32,
                    "default_value": "-",
                },
            },
            # ── Formatters ────────────────────────────────────────────────
            # %(correlation_id)s is safe here because the CorrelationIdFilter
            # on the handler guarantees the attribute exists on every
            # LogRecord before the formatter runs.
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.json.JsonFormatter",
                    "fmt": ("%(asctime)s %(levelname)s %(name)s %(correlation_id)s %(message)s"),
                    "datefmt": "%Y-%m-%dT%H:%M:%S",
                },
            },
            # ── Handlers ──────────────────────────────────────────────────
            # The filter MUST be declared on the handler (not on a logger)
            # so that it enriches every record that passes through,
            # regardless of which logger emitted it.
            "handlers": {
                "json_stdout": {
                    "class": "logging.StreamHandler",
                    "filters": ["correlation_id"],
                    "formatter": "json",
                },
            },
            # ── Root logger ───────────────────────────────────────────────
            "root": {
                "handlers": ["json_stdout"],
                "level": "INFO",
            },
            # ── Named loggers ─────────────────────────────────────────────
            "loggers": {
                # Suppress uvicorn's native access log — RequestLoggingMiddleware
                # is the sole source of per-request access logs in REST mode.
                "uvicorn.access": {
                    "level": "CRITICAL",
                    "propagate": False,
                },
                # Suppress per-request debug lines from the correlation-id
                # library itself (e.g. "Generated new request ID …").
                "asgi_correlation_id": {
                    "level": "WARNING",
                    "propagate": True,
                },
            },
        }
    )

    # Apply the same noisy third-party logger suppression used by configure().
    # Called after dictConfig so it wins any level set by the config dict.
    _suppress_noisy_loggers()
