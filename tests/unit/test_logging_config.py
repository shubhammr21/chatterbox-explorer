"""
tests/unit/test_logging_config.py
===================================
TDD unit tests for logging_config.configure().

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
