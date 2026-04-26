"""
tests/unit/adapters/test_watermark_detector.py
===============================================
TDD unit tests for PerThWatermarkDetector.

Both `librosa` and `perth` are deferred-imported inside detect(), so they
may or may not be installed in the test environment.  All tests use
patch.dict(sys.modules, ...) to inject controlled fakes, ensuring the suite
runs without either package present and without touching real audio files.

Mock strategy
─────────────
* librosa → sys.modules['librosa'] replaced with a MagicMock whose
  .load(path, sr=None) returns (fake_audio, fake_sr).

* perth  → sys.modules['perth'] replaced with a MagicMock whose
  .PerthImplicitWatermarker() returns a watermarker mock whose
  .get_watermark(audio, sample_rate=sr) returns the configured score.

Both fakes are active only inside the `with patch.dict(...)` block, so
the real sys.modules entries (if any) are always restored afterwards.
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from adapters.secondary.watermark import PerThWatermarkDetector

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers / fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _make_librosa_mock(audio=None, sr: int = 22_050) -> MagicMock:
    """Return a fake librosa module whose .load() yields (audio, sr)."""
    if audio is None:
        audio = MagicMock(name="fake_audio_array")
    m = MagicMock(name="librosa_module")
    m.load.return_value = (audio, sr)
    return m


def _make_perth_mock(score: float = 0.85) -> tuple[MagicMock, MagicMock]:
    """Return (fake_perth_module, fake_watermarker_instance).

    fake_perth_module.PerthImplicitWatermarker() returns fake_watermarker.
    fake_watermarker.get_watermark(...) returns *score*.
    """
    fake_watermarker = MagicMock(name="watermarker_instance")
    fake_watermarker.get_watermark.return_value = score

    fake_perth = MagicMock(name="perth_module")
    fake_perth.PerthImplicitWatermarker.return_value = fake_watermarker

    return fake_perth, fake_watermarker


# ──────────────────────────────────────────────────────────────────────────────
# is_available
# ──────────────────────────────────────────────────────────────────────────────


class TestIsAvailable:
    def test_returns_true_when_initialised_with_available_true(self):
        detector = PerThWatermarkDetector(available=True)
        assert detector.is_available() is True

    def test_returns_false_when_initialised_with_available_false(self):
        detector = PerThWatermarkDetector(available=False)
        assert detector.is_available() is False

    def test_default_is_available_true(self):
        """Default constructor arg must be available=True (matches class signature)."""
        detector = PerThWatermarkDetector()
        assert detector.is_available() is True


# ──────────────────────────────────────────────────────────────────────────────
# detect — unavailable path (early return, no external calls)
# ──────────────────────────────────────────────────────────────────────────────


class TestDetectUnavailable:
    """When available=False, detect() must short-circuit and return 0.0
    without importing or calling librosa or perth at all."""

    def test_returns_zero_float(self):
        detector = PerThWatermarkDetector(available=False)
        assert detector.detect("/any/path.wav") == 0.0

    def test_does_not_call_librosa_load(self):
        detector = PerThWatermarkDetector(available=False)
        mock_librosa = _make_librosa_mock()
        mock_perth, _ = _make_perth_mock()
        with patch.dict(sys.modules, {"librosa": mock_librosa, "perth": mock_perth}):
            detector.detect("/any/path.wav")
        mock_librosa.load.assert_not_called()

    def test_does_not_instantiate_perth_watermarker(self):
        detector = PerThWatermarkDetector(available=False)
        mock_librosa = _make_librosa_mock()
        mock_perth, _ = _make_perth_mock()
        with patch.dict(sys.modules, {"librosa": mock_librosa, "perth": mock_perth}):
            detector.detect("/any/path.wav")
        mock_perth.PerthImplicitWatermarker.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# detect — available path (happy path)
# ──────────────────────────────────────────────────────────────────────────────


class TestDetectAvailableHappyPath:
    """When available=True and no exception occurs, detect() must call
    librosa.load, instantiate PerthImplicitWatermarker, call get_watermark,
    and return the score as a Python float."""

    @pytest.fixture
    def mocks(self):
        """Pre-wired librosa + perth mocks with score=0.85."""
        fake_audio = MagicMock(name="audio_array")
        fake_sr = 22_050
        fake_score = 0.85

        mock_librosa = _make_librosa_mock(audio=fake_audio, sr=fake_sr)
        mock_perth, mock_watermarker = _make_perth_mock(score=fake_score)

        return {
            "librosa": mock_librosa,
            "perth": mock_perth,
            "watermarker": mock_watermarker,
            "audio": fake_audio,
            "sr": fake_sr,
            "score": fake_score,
        }

    def test_calls_librosa_load_with_path_and_sr_none(self, mocks):
        detector = PerThWatermarkDetector(available=True)
        with patch.dict(sys.modules, {"librosa": mocks["librosa"], "perth": mocks["perth"]}):
            detector.detect("/tmp/output.wav")
        mocks["librosa"].load.assert_called_once_with("/tmp/output.wav", sr=None)

    def test_instantiates_perth_watermarker(self, mocks):
        detector = PerThWatermarkDetector(available=True)
        with patch.dict(sys.modules, {"librosa": mocks["librosa"], "perth": mocks["perth"]}):
            detector.detect("/tmp/output.wav")
        mocks["perth"].PerthImplicitWatermarker.assert_called_once()

    def test_calls_get_watermark_with_audio_and_sample_rate(self, mocks):
        detector = PerThWatermarkDetector(available=True)
        with patch.dict(sys.modules, {"librosa": mocks["librosa"], "perth": mocks["perth"]}):
            detector.detect("/tmp/output.wav")
        mocks["watermarker"].get_watermark.assert_called_once_with(
            mocks["audio"],
            sample_rate=mocks["sr"],
        )

    def test_returns_score_as_python_float(self, mocks):
        detector = PerThWatermarkDetector(available=True)
        with patch.dict(sys.modules, {"librosa": mocks["librosa"], "perth": mocks["perth"]}):
            score = detector.detect("/tmp/output.wav")
        assert isinstance(score, float)
        assert score == mocks["score"]

    def test_sample_rate_from_librosa_is_forwarded_to_get_watermark(self):
        """The sr returned by librosa.load must be passed verbatim to get_watermark."""
        specific_sr = 44_100
        mock_librosa = _make_librosa_mock(sr=specific_sr)
        mock_perth, mock_watermarker = _make_perth_mock(score=0.5)

        detector = PerThWatermarkDetector(available=True)
        with patch.dict(sys.modules, {"librosa": mock_librosa, "perth": mock_perth}):
            detector.detect("/tmp/audio.wav")

        _, call_kwargs = mock_watermarker.get_watermark.call_args
        assert call_kwargs["sample_rate"] == specific_sr

    def test_high_score_returned_correctly(self):
        mock_librosa = _make_librosa_mock()
        mock_perth, _ = _make_perth_mock(score=1.0)
        detector = PerThWatermarkDetector(available=True)
        with patch.dict(sys.modules, {"librosa": mock_librosa, "perth": mock_perth}):
            assert detector.detect("/tmp/audio.wav") == 1.0

    def test_zero_score_returned_correctly(self):
        mock_librosa = _make_librosa_mock()
        mock_perth, _ = _make_perth_mock(score=0.0)
        detector = PerThWatermarkDetector(available=True)
        with patch.dict(sys.modules, {"librosa": mock_librosa, "perth": mock_perth}):
            assert detector.detect("/tmp/audio.wav") == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# detect — exception path
# ──────────────────────────────────────────────────────────────────────────────


class TestDetectAvailableExceptionPath:
    """Any exception raised inside the try block must be caught, logged as a
    WARNING, and the method must return 0.0 (never re-raise)."""

    def test_returns_zero_on_librosa_load_exception(self):
        mock_librosa = MagicMock(name="librosa_module")
        mock_librosa.load.side_effect = RuntimeError("file not found")
        mock_perth, _ = _make_perth_mock()

        detector = PerThWatermarkDetector(available=True)
        with patch.dict(sys.modules, {"librosa": mock_librosa, "perth": mock_perth}):
            score = detector.detect("/nonexistent/audio.wav")

        assert score == 0.0

    def test_returns_zero_on_perth_watermarker_exception(self):
        mock_librosa = _make_librosa_mock()
        mock_perth, mock_watermarker = _make_perth_mock()
        mock_watermarker.get_watermark.side_effect = ValueError("model weights missing")

        detector = PerThWatermarkDetector(available=True)
        with patch.dict(sys.modules, {"librosa": mock_librosa, "perth": mock_perth}):
            score = detector.detect("/tmp/audio.wav")

        assert score == 0.0

    def test_returns_zero_on_perth_instantiation_exception(self):
        mock_librosa = _make_librosa_mock()
        mock_perth = MagicMock(name="perth_module")
        mock_perth.PerthImplicitWatermarker.side_effect = OSError("shared library not found")

        detector = PerThWatermarkDetector(available=True)
        with patch.dict(sys.modules, {"librosa": mock_librosa, "perth": mock_perth}):
            score = detector.detect("/tmp/audio.wav")

        assert score == 0.0

    def test_logs_warning_on_exception(self, caplog):
        mock_librosa = MagicMock(name="librosa_module")
        mock_librosa.load.side_effect = RuntimeError("boom")
        mock_perth, _ = _make_perth_mock()

        detector = PerThWatermarkDetector(available=True)
        with (
            patch.dict(sys.modules, {"librosa": mock_librosa, "perth": mock_perth}),
            caplog.at_level(logging.WARNING, logger="adapters.secondary.watermark"),
        ):
            detector.detect("/tmp/audio.wav")

        assert len(caplog.records) >= 1
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Watermark detection failed" in msg for msg in warning_messages)

    def test_does_not_reraise_exception(self):
        """Exception inside detect() must be swallowed — never propagated."""
        mock_librosa = MagicMock(name="librosa_module")
        mock_librosa.load.side_effect = MemoryError("OOM")
        mock_perth, _ = _make_perth_mock()

        detector = PerThWatermarkDetector(available=True)
        # Must not raise MemoryError (or any other exception).
        with patch.dict(sys.modules, {"librosa": mock_librosa, "perth": mock_perth}):
            result = detector.detect("/tmp/audio.wav")

        assert result == 0.0
