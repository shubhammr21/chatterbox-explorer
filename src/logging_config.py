"""
src/logging_config.py
==========================================
Logging and warning configuration for the Chatterbox TTS Explorer.

IMPORTANT: This module MUST have zero side-effects at import time.
All configuration logic is gated behind the configure() function so that:
  - Unit tests can import project modules without triggering log setup.
  - cli.py controls exactly when (and therefore after which other imports)
    the configuration fires.

Call configure() exactly once, as the very first thing in cli.main(),
before any library with an eager logger (huggingface_hub, transformers,
diffusers) is imported.
"""

from __future__ import annotations


def configure() -> None:
    """Configure logging and warning filters for the entire process.

    Ordering within this function is load-bearing — see inline comments.
    """
    import logging
    import os
    import warnings

    # ── Root logger ──────────────────────────────────────────────────────────
    # Set up a clean, timestamped format for our own log messages.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

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
    # with a single clean advisory from our own logger (see below).
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
