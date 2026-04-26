"""
tests/unit/services/test_watermark_service.py
===============================================
Unit tests for WatermarkService.

Covers:
- verdict logic: detected / not_detected / inconclusive / unavailable
- delegation to IWatermarkDetector
- boundary values (0.0, 0.1, 0.5, 0.9, 1.0)
"""
from __future__ import annotations

import pytest

from chatterbox_explorer.domain.models import WatermarkResult
from chatterbox_explorer.services.watermark import WatermarkService


# ──────────────────────────────────────────────────────────────────────────────
# Verdict logic
# ──────────────────────────────────────────────────────────────────────────────

class TestWatermarkServiceVerdicts:
    def test_detect_returns_detected_for_high_score(self, mock_watermark_detector):
        """score = 1.0 → verdict 'detected'."""
        mock_watermark_detector.detect.return_value = 1.0
        svc = WatermarkService(mock_watermark_detector)
        result = svc.detect("/tmp/audio.wav")

        assert result.verdict == "detected"
        assert result.score == 1.0
        assert result.available is True

    def test_detect_returns_detected_at_boundary_09(self, mock_watermark_detector):
        """score = 0.9 is exactly at the 'detected' threshold — must be 'detected'."""
        mock_watermark_detector.detect.return_value = 0.9
        svc = WatermarkService(mock_watermark_detector)
        result = svc.detect("/tmp/audio.wav")

        assert result.verdict == "detected"

    def test_detect_returns_not_detected_for_low_score(self, mock_watermark_detector):
        """score = 0.0 → verdict 'not_detected'."""
        mock_watermark_detector.detect.return_value = 0.0
        svc = WatermarkService(mock_watermark_detector)
        result = svc.detect("/tmp/audio.wav")

        assert result.verdict == "not_detected"
        assert result.score == 0.0
        assert result.available is True

    def test_detect_returns_not_detected_at_boundary_01(self, mock_watermark_detector):
        """score = 0.1 is exactly at the 'not_detected' threshold."""
        mock_watermark_detector.detect.return_value = 0.1
        svc = WatermarkService(mock_watermark_detector)
        result = svc.detect("/tmp/audio.wav")

        assert result.verdict == "not_detected"

    def test_detect_returns_inconclusive_for_mid_score(self, mock_watermark_detector):
        """score = 0.5 is ambiguous → verdict 'inconclusive'."""
        mock_watermark_detector.detect.return_value = 0.5
        svc = WatermarkService(mock_watermark_detector)
        result = svc.detect("/tmp/audio.wav")

        assert result.verdict == "inconclusive"
        assert result.score == 0.5
        assert result.available is True

    def test_detect_returns_inconclusive_just_above_lower_bound(
        self, mock_watermark_detector
    ):
        """score = 0.11 (just above 0.1) → 'inconclusive'."""
        mock_watermark_detector.detect.return_value = 0.11
        svc = WatermarkService(mock_watermark_detector)
        result = svc.detect("/tmp/audio.wav")

        assert result.verdict == "inconclusive"

    def test_detect_returns_inconclusive_just_below_upper_bound(
        self, mock_watermark_detector
    ):
        """score = 0.89 (just below 0.9) → 'inconclusive'."""
        mock_watermark_detector.detect.return_value = 0.89
        svc = WatermarkService(mock_watermark_detector)
        result = svc.detect("/tmp/audio.wav")

        assert result.verdict == "inconclusive"

    def test_detect_returns_unavailable_when_not_available(
        self, mock_watermark_detector
    ):
        """When detector.is_available() is False, verdict must be 'unavailable'."""
        mock_watermark_detector.is_available.return_value = False
        svc = WatermarkService(mock_watermark_detector)
        result = svc.detect("/tmp/audio.wav")

        assert result.verdict == "unavailable"
        assert result.available is False
        assert result.score == 0.0

    def test_detect_unavailable_does_not_call_detector(self, mock_watermark_detector):
        """When unavailable, detect() on the underlying detector must NOT be called."""
        mock_watermark_detector.is_available.return_value = False
        svc = WatermarkService(mock_watermark_detector)
        svc.detect("/tmp/audio.wav")

        mock_watermark_detector.detect.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# Return type and field correctness
# ──────────────────────────────────────────────────────────────────────────────

class TestWatermarkServiceReturnType:
    def test_detect_returns_watermark_result_instance(self, mock_watermark_detector):
        svc = WatermarkService(mock_watermark_detector)
        result = svc.detect("/tmp/audio.wav")
        assert isinstance(result, WatermarkResult)

    def test_detect_result_has_non_empty_message(self, mock_watermark_detector):
        """Every verdict must carry a human-readable message."""
        for score, _ in [(1.0, "detected"), (0.5, "inconclusive"), (0.0, "not_detected")]:
            mock_watermark_detector.detect.return_value = score
            svc = WatermarkService(mock_watermark_detector)
            result = svc.detect("/tmp/audio.wav")
            assert result.message, f"message must not be empty for score={score}"

    def test_detect_unavailable_result_has_message(self, mock_watermark_detector):
        mock_watermark_detector.is_available.return_value = False
        svc = WatermarkService(mock_watermark_detector)
        result = svc.detect("/tmp/audio.wav")
        assert result.message

    def test_detect_available_flag_true_when_detector_available(
        self, mock_watermark_detector
    ):
        mock_watermark_detector.is_available.return_value = True
        mock_watermark_detector.detect.return_value = 0.95
        svc = WatermarkService(mock_watermark_detector)
        result = svc.detect("/tmp/audio.wav")
        assert result.available is True


# ──────────────────────────────────────────────────────────────────────────────
# Delegation to detector
# ──────────────────────────────────────────────────────────────────────────────

class TestWatermarkServiceDelegation:
    def test_detect_delegates_to_detector(self, mock_watermark_detector):
        """detector.detect() must be called with the exact path passed in."""
        mock_watermark_detector.detect.return_value = 0.95
        svc = WatermarkService(mock_watermark_detector)
        svc.detect("/tmp/audio.wav")

        mock_watermark_detector.detect.assert_called_once_with("/tmp/audio.wav")

    def test_detect_checks_availability_before_detecting(self, mock_watermark_detector):
        """is_available() must be called on every detect() invocation."""
        svc = WatermarkService(mock_watermark_detector)
        svc.detect("/tmp/audio.wav")

        mock_watermark_detector.is_available.assert_called_once()

    def test_detect_passes_through_score_from_detector(self, mock_watermark_detector):
        """The score in the result must match exactly what the detector returned."""
        mock_watermark_detector.detect.return_value = 0.73
        svc = WatermarkService(mock_watermark_detector)
        result = svc.detect("/tmp/audio.wav")

        assert result.score == pytest.approx(0.73)
