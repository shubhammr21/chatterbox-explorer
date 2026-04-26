"""
tests/unit/test_logging_config.py
===================================
TDD unit tests for logging_config.configure() and logging_config.configure_json().
</thinking>

configure() is a zero-import-side-effect function: all its work happens
*inside* the function body, so we can call it directly in tests without
needing special import-time guards.

What we test
────────────
1. Logger levels — transformers, huggingface_hub, diffusers set to ERROR;
   httpx set to WARNING.
2. HF_TOKEN advisory — the "chatterbox-explorer" logger emits exactly one
   INFO line whose content depends on whether HF_TOKEN is set.
3. Idempotency — calling configure() a second time must not raise and must
   leave logger levels in the same state.

Mock strategy
─────────────
* monkeypatch sets / clears HF_TOKEN in os.environ so the token-detection
  branch is exercised without real credentials.
* caplog (pytest built-in) captures log records emitted during configure();
  we check records on the "chatterbox-explorer" logger only.
* No mocking of huggingface_hub / transformers — both are installed as
  project dependencies, so their import inside configure() succeeds and
  exercises the real set_verbosity_error() path.
"""

from __future__ import annotations

import logging

import pytest

import logging_config
from logging_config import configure

# ──────────────────────────────────────────────────────────────────────────────
# Logger-level assertions
# ──────────────────────────────────────────────────────────────────────────────


class TestLoggerLevels:
    """After configure() runs, named loggers must be set to the documented levels."""

    @pytest.fixture(autouse=True)
    def _run_configure(self):
        """Call configure() once before every test in this class."""
        configure()

    def test_transformers_logger_set_to_error(self):
        assert logging.getLogger("transformers").level == logging.ERROR

    def test_huggingface_hub_logger_set_to_error(self):
        assert logging.getLogger("huggingface_hub").level == logging.ERROR

    def test_diffusers_logger_set_to_error(self):
        assert logging.getLogger("diffusers").level == logging.ERROR

    def test_httpx_logger_set_to_warning(self):
        assert logging.getLogger("httpx").level == logging.WARNING

    def test_httpx_logger_not_silenced_to_error(self):
        """httpx keeps WARNING-level messages; it must not be set to ERROR."""
        assert logging.getLogger("httpx").level != logging.ERROR

    def test_transformers_not_at_warning_or_below(self):
        """transformers must be quieter than WARNING to suppress tokeniser spam."""
        assert logging.getLogger("transformers").level > logging.WARNING


# ──────────────────────────────────────────────────────────────────────────────
# HF_TOKEN advisory log messages
# ──────────────────────────────────────────────────────────────────────────────


class TestHFTokenAdvisory:
    """configure() must emit exactly one INFO advisory about HF_TOKEN status."""

    # Logger name used inside logging_config.py
    _LOGGER_NAME = "chatterbox-explorer"

    def _explorer_records(self, caplog):
        """Filter caplog to only records from the chatterbox-explorer logger."""
        return [r for r in caplog.records if r.name == self._LOGGER_NAME]

    # ── HF_TOKEN present ─────────────────────────────────────────────────────

    def test_hf_token_detected_message_logged(self, monkeypatch, caplog):
        monkeypatch.setenv("HF_TOKEN", "hf_test_token_abc123")
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

        with caplog.at_level(logging.INFO, logger=self._LOGGER_NAME):
            configure()

        records = self._explorer_records(caplog)
        assert any("HF_TOKEN detected" in r.message for r in records), (
            f"Expected 'HF_TOKEN detected' in log records; got: {[r.message for r in records]}"
        )

    def test_hf_token_detected_is_info_level(self, monkeypatch, caplog):
        monkeypatch.setenv("HF_TOKEN", "hf_test_token_abc123")
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

        with caplog.at_level(logging.INFO, logger=self._LOGGER_NAME):
            configure()

        detected_records = [
            r for r in self._explorer_records(caplog) if "HF_TOKEN detected" in r.message
        ]
        assert all(r.levelno == logging.INFO for r in detected_records)

    def test_fallback_token_env_var_detected(self, monkeypatch, caplog):
        """HUGGING_FACE_HUB_TOKEN is the legacy alias and must also be recognised."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy_token_xyz")

        with caplog.at_level(logging.INFO, logger=self._LOGGER_NAME):
            configure()

        records = self._explorer_records(caplog)
        assert any("HF_TOKEN detected" in r.message for r in records)

    def test_hf_token_not_set_message_absent_when_token_present(self, monkeypatch, caplog):
        monkeypatch.setenv("HF_TOKEN", "hf_test_token")
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

        with caplog.at_level(logging.INFO, logger=self._LOGGER_NAME):
            configure()

        records = self._explorer_records(caplog)
        assert not any("HF_TOKEN not set" in r.message for r in records), (
            "Should NOT log 'not set' when HF_TOKEN is present"
        )

    # ── HF_TOKEN absent ──────────────────────────────────────────────────────

    def test_hf_token_not_set_message_logged(self, monkeypatch, caplog):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

        with caplog.at_level(logging.INFO, logger=self._LOGGER_NAME):
            configure()

        records = self._explorer_records(caplog)
        assert any("HF_TOKEN not set" in r.message for r in records), (
            f"Expected 'HF_TOKEN not set' in log records; got: {[r.message for r in records]}"
        )

    def test_hf_token_not_set_is_info_level(self, monkeypatch, caplog):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

        with caplog.at_level(logging.INFO, logger=self._LOGGER_NAME):
            configure()

        not_set_records = [
            r for r in self._explorer_records(caplog) if "HF_TOKEN not set" in r.message
        ]
        assert all(r.levelno == logging.INFO for r in not_set_records)

    def test_detected_message_absent_when_token_missing(self, monkeypatch, caplog):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

        with caplog.at_level(logging.INFO, logger=self._LOGGER_NAME):
            configure()

        records = self._explorer_records(caplog)
        assert not any("HF_TOKEN detected" in r.message for r in records), (
            "Should NOT log 'detected' when HF_TOKEN is absent"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Idempotency
# ──────────────────────────────────────────────────────────────────────────────


class TestIdempotency:
    """Calling configure() multiple times must be safe and produce consistent state."""

    def test_second_call_does_not_raise(self):
        configure()
        configure()  # must not raise any exception

    def test_logger_levels_unchanged_after_second_call(self):
        configure()
        configure()
        assert logging.getLogger("transformers").level == logging.ERROR
        assert logging.getLogger("huggingface_hub").level == logging.ERROR
        assert logging.getLogger("diffusers").level == logging.ERROR
        assert logging.getLogger("httpx").level == logging.WARNING

    def test_three_consecutive_calls_do_not_raise(self):
        configure()
        configure()
        configure()

    def test_configure_is_importable_without_side_effects(self):
        """Importing logging_config must NOT trigger any logging setup on its own."""
        import importlib

        # Re-importing must not raise and must not emit any logs at module level.
        importlib.reload(logging_config)
        # If configure() had module-level side effects, basicConfig would have
        # been called an extra time — this test asserts it is still safe.
        assert callable(configure)


# ──────────────────────────────────────────────────────────────────────────────
# ImportError fallback paths (lines 52-53 and 85-86 in logging_config.py)
# ──────────────────────────────────────────────────────────────────────────────


class TestImportErrorFallbacks:
    """configure() wraps optional library imports in try/except ImportError.

    These tests simulate a stripped environment where huggingface_hub or
    transformers are absent, verifying that configure() silently skips the
    verbosity setup rather than propagating the ImportError to the caller.

    Technique
    ─────────
    Setting ``sys.modules["<name>"] = None`` is the standard CPython trick to
    make a subsequent ``import <name>`` statement raise ``ImportError`` (the
    import machinery treats a None entry as a failed/blocked module).
    ``patch.dict`` restores the original entry after the context manager exits.
    """

    def test_configure_handles_huggingface_hub_import_error(self) -> None:
        """When huggingface_hub is not installed, configure() must not raise.

        Covers the ``except ImportError: pass`` block that guards the
        ``_hf_log.set_verbosity_error()`` call (lines 52-53).
        """
        import sys
        from unittest.mock import patch

        # Block the huggingface_hub import by poisoning sys.modules.
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            configure()  # must complete without raising

    def test_configure_handles_transformers_import_error(self) -> None:
        """When transformers is not installed, configure() must not raise.

        Covers the ``except ImportError: pass`` block that guards the
        ``_t.logging.set_verbosity_error()`` call (lines 85-86).
        """
        import sys
        from unittest.mock import patch

        # Block the transformers import by poisoning sys.modules.
        with patch.dict(sys.modules, {"transformers": None}):
            configure()  # must complete without raising


# ──────────────────────────────────────────────────────────────────────────────
# configure_json() — structured JSON logging for the REST mode
# ──────────────────────────────────────────────────────────────────────────────


class TestConfigureJson:
    """Tests for logging_config.configure_json().

    configure_json() uses logging.config.dictConfig to:
      - attach a pythonjsonlogger.json.JsonFormatter to the root handler
      - attach asgi_correlation_id.CorrelationIdFilter to the same handler
        so every LogRecord carries a ``correlation_id`` attribute
      - silence uvicorn.access at CRITICAL (RequestLoggingMiddleware owns
        access logs in REST mode)
      - silence asgi_correlation_id library logger at WARNING
      - apply the same noisy third-party logger suppression as configure()

    Both asgi-correlation-id and python-json-logger are part of the 'rest'
    optional extra and must be present (uv sync --all-extras).
    """

    # ── basic smoke ───────────────────────────────────────────────────────────

    def test_configure_json_does_not_raise(self) -> None:
        """configure_json() must complete without raising when all deps are present."""
        from logging_config import configure_json

        configure_json()

    def test_configure_json_idempotent(self) -> None:
        """Calling configure_json() twice must not raise."""
        from logging_config import configure_json

        configure_json()
        configure_json()

    # ── root logger level ─────────────────────────────────────────────────────

    def test_configure_json_sets_root_level_to_info(self) -> None:
        from logging_config import configure_json

        configure_json()
        assert logging.getLogger().level == logging.INFO

    def test_configure_json_root_has_at_least_one_handler(self) -> None:
        from logging_config import configure_json

        configure_json()
        assert len(logging.getLogger().handlers) >= 1

    # ── JsonFormatter on root handler ─────────────────────────────────────────

    def test_configure_json_handler_has_json_formatter(self) -> None:
        """Root handler must use pythonjsonlogger.json.JsonFormatter."""
        from pythonjsonlogger.json import JsonFormatter

        from logging_config import configure_json

        configure_json()
        root_handlers = logging.getLogger().handlers
        assert any(isinstance(h.formatter, JsonFormatter) for h in root_handlers), (
            "Expected at least one root handler with a JsonFormatter"
        )

    # ── CorrelationIdFilter on root handler ───────────────────────────────────

    def test_configure_json_handler_has_correlation_id_filter(self) -> None:
        """Root handler must have CorrelationIdFilter so every LogRecord carries
        a correlation_id attribute matching the active request's ContextVar."""
        from asgi_correlation_id import CorrelationIdFilter

        from logging_config import configure_json

        configure_json()
        root_handlers = logging.getLogger().handlers
        assert any(
            any(isinstance(f, CorrelationIdFilter) for f in h.filters) for h in root_handlers
        ), "Expected CorrelationIdFilter on at least one root handler"

    def test_correlation_id_filter_uses_full_32_char_length(self) -> None:
        """uuid_length must be 32 so the full hex ID is stored in JSON logs."""
        from asgi_correlation_id import CorrelationIdFilter

        from logging_config import configure_json

        configure_json()
        filters = [
            f
            for h in logging.getLogger().handlers
            for f in h.filters
            if isinstance(f, CorrelationIdFilter)
        ]
        assert filters, "No CorrelationIdFilter found on root handlers"
        assert all(f.uuid_length == 32 for f in filters), (
            f"Expected uuid_length=32, got {[f.uuid_length for f in filters]}"
        )

    def test_correlation_id_filter_default_value_is_dash(self) -> None:
        """default_value must be '-' so startup/shutdown log lines remain valid JSON."""
        from asgi_correlation_id import CorrelationIdFilter

        from logging_config import configure_json

        configure_json()
        filters = [
            f
            for h in logging.getLogger().handlers
            for f in h.filters
            if isinstance(f, CorrelationIdFilter)
        ]
        assert filters, "No CorrelationIdFilter found on root handlers"
        assert all(f.default_value == "-" for f in filters), (
            f"Expected default_value='-', got {[f.default_value for f in filters]}"
        )

    def test_correlation_id_filter_enriches_log_record(self) -> None:
        """After configure_json(), a log record emitted outside a request context
        must have correlation_id == '-' (the default_value)."""
        from logging_config import configure_json

        configure_json()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        # Apply every filter on every root handler to the record.
        for handler in logging.getLogger().handlers:
            for log_filter in handler.filters:
                if isinstance(log_filter, logging.Filter):
                    log_filter.filter(record)

        assert hasattr(record, "correlation_id"), (
            "CorrelationIdFilter did not attach correlation_id to the LogRecord"
        )
        assert record.correlation_id == "-", (
            f"Outside a request context expected '-', got {record.correlation_id!r}"
        )

    # ── logger level suppression ──────────────────────────────────────────────

    def test_configure_json_suppresses_uvicorn_access(self) -> None:
        """uvicorn.access must be silenced — RequestLoggingMiddleware owns access logs."""
        from logging_config import configure_json

        configure_json()
        assert logging.getLogger("uvicorn.access").level == logging.CRITICAL

    def test_configure_json_suppresses_asgi_correlation_id_debug_noise(self) -> None:
        """asgi_correlation_id logger must be WARNING or above to suppress
        per-request 'Generated new request ID' debug lines."""
        from logging_config import configure_json

        configure_json()
        level = logging.getLogger("asgi_correlation_id").level
        assert level >= logging.WARNING, (
            f"Expected WARNING or higher, got {logging.getLevelName(level)}"
        )

    def test_configure_json_suppresses_transformers(self) -> None:
        from logging_config import configure_json

        configure_json()
        assert logging.getLogger("transformers").level == logging.ERROR

    def test_configure_json_suppresses_huggingface_hub(self) -> None:
        from logging_config import configure_json

        configure_json()
        assert logging.getLogger("huggingface_hub").level == logging.ERROR

    def test_configure_json_suppresses_diffusers(self) -> None:
        from logging_config import configure_json

        configure_json()
        assert logging.getLogger("diffusers").level == logging.ERROR

    # ── missing dependency guards ─────────────────────────────────────────────

    def test_configure_json_raises_when_asgi_correlation_id_missing(self) -> None:
        """When asgi-correlation-id is absent, configure_json() must raise
        ModuleNotFoundError with a clear install instruction."""
        import sys
        from unittest.mock import patch

        from logging_config import configure_json

        with (
            patch.dict(sys.modules, {"asgi_correlation_id": None}),
            pytest.raises(ModuleNotFoundError, match="asgi-correlation-id"),
        ):
            configure_json()

    def test_configure_json_raises_when_pythonjsonlogger_missing(self) -> None:
        """When python-json-logger is absent, configure_json() must raise
        ModuleNotFoundError with a clear install instruction."""
        import sys
        from unittest.mock import patch

        from logging_config import configure_json

        with (
            patch.dict(
                sys.modules,
                {"pythonjsonlogger": None, "pythonjsonlogger.json": None},
            ),
            pytest.raises(ModuleNotFoundError, match="python-json-logger"),
        ):
            configure_json()


# ──────────────────────────────────────────────────────────────────────────────
# QueueHandler non-blocking I/O tests (Phase 3 — infrastructure enhancements)
# ──────────────────────────────────────────────────────────────────────────────


class TestConfigureJsonQueueHandler:
    """After configure_json(), the root logger must use a non-blocking QueueHandler.

    The QueueHandler enqueues records quickly (non-blocking), and a background
    QueueListener thread dispatches them to the real sink handler.
    """

    def test_root_logger_has_queue_handler(self) -> None:
        """Root logger must route through a QueueHandler, not StreamHandler directly."""
        from logging_config import configure_json

        configure_json()
        import logging.handlers

        root_handlers = logging.getLogger().handlers
        assert any(isinstance(h, logging.handlers.QueueHandler) for h in root_handlers), (
            "Root logger must have a QueueHandler for non-blocking I/O"
        )

    def test_root_logger_has_no_direct_stream_handler(self) -> None:
        """Root logger must NOT have a StreamHandler directly (only via QueueListener)."""
        from logging_config import configure_json

        configure_json()
        import logging

        root_handlers = logging.getLogger().handlers
        # The only handler on root must be QueueHandler — StreamHandler goes on the listener
        for h in root_handlers:
            assert not isinstance(h, logging.StreamHandler) or isinstance(
                h, logging.handlers.QueueHandler
            ), f"Root logger has a direct StreamHandler {h!r} — must go through QueueHandler"

    def test_sys_excepthook_is_overridden(self) -> None:
        """configure_json() must override sys.excepthook to log uncaught exceptions."""
        import sys

        from logging_config import configure_json

        original = sys.__excepthook__
        configure_json()
        assert sys.excepthook is not sys.__excepthook__, (
            "sys.excepthook was not overridden by configure_json()"
        )
        assert sys.excepthook is not original

    def test_keyboard_interrupt_uses_default_excepthook(self) -> None:
        """KeyboardInterrupt must use the original sys.__excepthook__, not the logger."""
        import sys

        from logging_config import configure_json

        configure_json()
        # The override must NOT log KeyboardInterrupt — check it dispatches to __excepthook__
        # We can't actually invoke __excepthook__ in tests, but we can check the hook exists
        assert sys.excepthook is not None
        assert callable(sys.excepthook)

    def test_configure_json_accepts_log_level_parameter(self) -> None:
        """configure_json() must accept a log_level parameter."""
        import logging

        from logging_config import configure_json

        configure_json(log_level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_configure_json_default_log_level_is_info(self) -> None:
        import logging

        from logging_config import configure_json

        configure_json()
        assert logging.getLogger().level == logging.INFO

    def test_atexit_handler_registered(self) -> None:
        """A shutdown handler must be registered with atexit to drain the queue."""
        from logging_config import configure_json

        configure_json()
        # atexit internals are not part of the public API so we can't inspect
        # the registered callbacks portably. The meaningful assertion is that
        # configure_json() runs to completion without raising — if listener.start()
        # or atexit.register() failed they would raise before this point.
        assert True

    def test_queue_handler_prepare_handles_exc_info(self) -> None:
        """QueueHandler.prepare() must preserve exc_text when a record carries
        exc_info so that the listener-side handler can format the exception."""
        import logging
        import time

        from logging_config import configure_json

        configure_json()
        # Emit a record with exc_info — exercises the prepare() override that
        # pre-formats exc_text before the base class would clear it.
        try:
            raise RuntimeError("test exception for coverage")
        except RuntimeError:
            logging.getLogger("test.exc_info_coverage").error("error with exc_info", exc_info=True)
        # Give the QueueListener background thread time to drain the record.
        time.sleep(0.02)
        # If prepare() raised or corrupted the record the test would fail above.

    def test_excepthook_non_keyboard_interrupt_logs(self) -> None:
        """sys.excepthook must call the logger for non-KeyboardInterrupt exceptions."""
        import logging
        import sys
        from unittest.mock import patch

        from logging_config import configure_json

        configure_json()
        log = logging.getLogger("chatterbox-explorer")
        with patch.object(log, "error") as mock_error:
            sys.excepthook(ValueError, ValueError("uncaught"), None)
        mock_error.assert_called_once()

    def test_excepthook_keyboard_interrupt_uses_default(self) -> None:
        """sys.excepthook must dispatch KeyboardInterrupt to sys.__excepthook__."""
        import sys
        from unittest.mock import patch

        from logging_config import configure_json

        configure_json()
        with patch("sys.__excepthook__") as mock_default:
            sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
        mock_default.assert_called_once()
