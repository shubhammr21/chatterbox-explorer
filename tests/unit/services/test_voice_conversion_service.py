"""
tests/unit/services/test_voice_conversion_service.py
======================================================
Unit tests for VoiceConversionService.

All model I/O is faked via the fixtures in tests/conftest.py — no real
Chatterbox weights are loaded.  torch IS required at test-time (via the
mock_wav_tensor fixture chain) but is NOT imported by the service module
under test.
"""

from __future__ import annotations

import pytest

from domain.exceptions import MissingSourceAudioError, MissingTargetVoiceError
from domain.models import AudioResult, VoiceConversionRequest
from services.voice_conversion import VoiceConversionService

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_service(mock_model_repo, mock_preprocessor) -> VoiceConversionService:
    return VoiceConversionService(mock_model_repo, mock_preprocessor)


def _valid_request(
    source: str = "/tmp/source.wav",
    target: str = "/tmp/target.wav",
) -> VoiceConversionRequest:
    return VoiceConversionRequest(
        source_audio_path=source,
        target_voice_path=target,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────────────────


class TestVoiceConversionServiceValidation:
    def test_convert_raises_on_empty_source(self, mock_model_repo, mock_preprocessor):
        svc = _make_service(mock_model_repo, mock_preprocessor)
        req = VoiceConversionRequest(
            source_audio_path="",
            target_voice_path="/tmp/target.wav",
        )
        with pytest.raises(MissingSourceAudioError):
            svc.convert(req)

    def test_convert_raises_on_none_source(self, mock_model_repo, mock_preprocessor):
        svc = _make_service(mock_model_repo, mock_preprocessor)
        req = VoiceConversionRequest(
            source_audio_path=None,
            target_voice_path="/tmp/target.wav",
        )
        with pytest.raises(MissingSourceAudioError):
            svc.convert(req)

    def test_convert_raises_on_empty_target(self, mock_model_repo, mock_preprocessor):
        svc = _make_service(mock_model_repo, mock_preprocessor)
        req = VoiceConversionRequest(
            source_audio_path="/tmp/source.wav",
            target_voice_path="",
        )
        with pytest.raises(MissingTargetVoiceError):
            svc.convert(req)

    def test_convert_raises_on_none_target(self, mock_model_repo, mock_preprocessor):
        svc = _make_service(mock_model_repo, mock_preprocessor)
        req = VoiceConversionRequest(
            source_audio_path="/tmp/source.wav",
            target_voice_path=None,
        )
        with pytest.raises(MissingTargetVoiceError):
            svc.convert(req)

    def test_convert_raises_on_both_empty(self, mock_model_repo, mock_preprocessor):
        svc = _make_service(mock_model_repo, mock_preprocessor)
        req = VoiceConversionRequest(
            source_audio_path="",
            target_voice_path="",
        )
        # source is validated first, so MissingSourceAudioError is raised
        with pytest.raises(MissingSourceAudioError):
            svc.convert(req)


# ──────────────────────────────────────────────────────────────────────────────
# Model / preprocessor interaction
# ──────────────────────────────────────────────────────────────────────────────


class TestVoiceConversionServiceModelInteraction:
    def test_convert_uses_vc_model_key(self, mock_model_repo, mock_preprocessor):
        """Service must request the 'vc' model, not any other key."""
        svc = _make_service(mock_model_repo, mock_preprocessor)
        svc.convert(_valid_request())
        mock_model_repo.get_model.assert_called_with("vc")

    def test_convert_calls_model_generate(self, mock_model_repo, mock_preprocessor, mock_model):
        svc = _make_service(mock_model_repo, mock_preprocessor)
        svc.convert(_valid_request())
        mock_model.generate.assert_called_once()

    def test_convert_passes_source_as_audio_kwarg(
        self, mock_model_repo, mock_preprocessor, mock_model
    ):
        svc = _make_service(mock_model_repo, mock_preprocessor)
        svc.convert(_valid_request(source="/tmp/source.wav"))

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["audio"] == "/tmp/source.wav"

    def test_convert_passes_target_as_target_voice_path_kwarg(
        self, mock_model_repo, mock_preprocessor, mock_model
    ):
        svc = _make_service(mock_model_repo, mock_preprocessor)
        svc.convert(_valid_request(target="/tmp/target.wav"))

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["target_voice_path"] == "/tmp/target.wav"

    def test_convert_preprocesses_target_voice(self, mock_model_repo, mock_preprocessor):
        """Target voice path must go through the preprocessor before the model."""
        svc = _make_service(mock_model_repo, mock_preprocessor)
        svc.convert(_valid_request(target="/tmp/target.wav"))
        mock_preprocessor.preprocess.assert_called_once_with("/tmp/target.wav")

    def test_convert_passes_preprocessed_target_to_model(
        self, mock_model_repo, mock_preprocessor, mock_model
    ):
        """The preprocessed path (not the raw one) must reach the model."""
        mock_preprocessor.preprocess.side_effect = lambda p: p + ".preprocessed"
        svc = _make_service(mock_model_repo, mock_preprocessor)
        svc.convert(_valid_request(target="/tmp/target.wav"))

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["target_voice_path"] == "/tmp/target.wav.preprocessed"


# ──────────────────────────────────────────────────────────────────────────────
# Output contract
# ──────────────────────────────────────────────────────────────────────────────


class TestVoiceConversionServiceOutput:
    def test_convert_returns_audio_result(self, mock_model_repo, mock_preprocessor):
        svc = _make_service(mock_model_repo, mock_preprocessor)
        result = svc.convert(_valid_request())
        assert isinstance(result, AudioResult)

    def test_convert_returns_correct_sample_rate(self, mock_model_repo, mock_preprocessor):
        svc = _make_service(mock_model_repo, mock_preprocessor)
        result = svc.convert(_valid_request())
        assert result.sample_rate == 24000

    def test_convert_returns_correct_samples_shape(self, mock_model_repo, mock_preprocessor):
        """Mock tensor is (1, 24000); after squeeze it must be (24000,)."""
        svc = _make_service(mock_model_repo, mock_preprocessor)
        result = svc.convert(_valid_request())
        assert result.samples.shape == (24000,)

    def test_convert_returns_float32_samples(self, mock_model_repo, mock_preprocessor):
        import numpy as np

        svc = _make_service(mock_model_repo, mock_preprocessor)
        result = svc.convert(_valid_request())
        assert result.samples.dtype == np.float32
