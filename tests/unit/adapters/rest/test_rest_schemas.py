"""
tests/unit/adapters/rest/test_rest_schemas.py
=============================================
Unit tests for Pydantic request/response schemas and the
audio_result_to_wav_bytes() helper in
src/adapters/inbound/rest/schemas.py.

No FastAPI app or HTTP client is needed here — schemas are pure
Python/Pydantic objects exercised in isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from domain.types import ModelKey

import io

import numpy as np
from pydantic import ValidationError
import pytest

from adapters.inbound.rest.schemas import (
    HealthResponse,
    MemoryStatsResponse,
    ModelStatusResponse,
    MultilingualRequestSchema,
    TTSRequestSchema,
    TurboRequestSchema,
    WatermarkResponse,
    audio_result_to_wav_bytes,
)
from domain.models import (
    AudioResult,
    MemoryStats,
    ModelStatus,
    WatermarkResult,
)

# ─────────────────────────────────────────────────────────────────────────────
# TTSRequestSchema
# ─────────────────────────────────────────────────────────────────────────────


class TestTTSRequestSchema:
    """Tests for POST /api/v1/tts/generate request body schema."""

    # ── defaults ─────────────────────────────────────────────────────────────

    def test_default_exaggeration(self):
        assert TTSRequestSchema(text="hello").exaggeration == 0.5

    def test_default_cfg_weight(self):
        assert TTSRequestSchema(text="hello").cfg_weight == 0.5

    def test_default_temperature(self):
        assert TTSRequestSchema(text="hello").temperature == 0.8

    def test_default_rep_penalty(self):
        assert TTSRequestSchema(text="hello").rep_penalty == 1.2

    def test_default_min_p(self):
        assert TTSRequestSchema(text="hello").min_p == 0.05

    def test_default_top_p(self):
        assert TTSRequestSchema(text="hello").top_p == 1.0

    def test_default_seed(self):
        assert TTSRequestSchema(text="hello").seed == 0

    # ── to_domain ─────────────────────────────────────────────────────────────

    def test_to_domain_returns_tts_request_instance(self):
        from domain.models import TTSRequest

        req = TTSRequestSchema(text="hello world").to_domain()
        assert isinstance(req, TTSRequest)

    def test_to_domain_maps_text(self):
        req = TTSRequestSchema(text="synthesise this").to_domain()
        assert req.text == "synthesise this"

    def test_to_domain_ref_audio_path_is_none(self):
        """REST v1 never exposes ref_audio_path — must always be None."""
        req = TTSRequestSchema(text="hi").to_domain()
        assert req.ref_audio_path is None

    def test_to_domain_streaming_is_false(self):
        """REST v1 is one-shot only — streaming must always be False."""
        req = TTSRequestSchema(text="hi").to_domain()
        assert req.streaming is False

    def test_to_domain_maps_all_numeric_fields(self):
        schema = TTSRequestSchema(
            text="test",
            exaggeration=0.7,
            cfg_weight=0.3,
            temperature=1.0,
            rep_penalty=1.5,
            min_p=0.1,
            top_p=0.9,
            seed=42,
        )
        req = schema.to_domain()
        assert req.exaggeration == 0.7
        assert req.cfg_weight == 0.3
        assert req.temperature == 1.0
        assert req.rep_penalty == 1.5
        assert req.min_p == 0.1
        assert req.top_p == 0.9
        assert req.seed == 42

    # ── validation errors ─────────────────────────────────────────────────────

    def test_missing_text_raises_validation_error(self):
        with pytest.raises(ValidationError):
            TTSRequestSchema()  # type: ignore[call-arg]

    def test_empty_text_raises_validation_error(self):
        """min_length=1 must reject empty strings."""
        with pytest.raises(ValidationError):
            TTSRequestSchema(text="")

    def test_exaggeration_above_max_raises_validation_error(self):
        """le=1.0 must reject values greater than 1.0."""
        with pytest.raises(ValidationError):
            TTSRequestSchema(text="hello", exaggeration=1.01)

    def test_exaggeration_below_min_raises_validation_error(self):
        """ge=0.0 must reject negative values."""
        with pytest.raises(ValidationError):
            TTSRequestSchema(text="hello", exaggeration=-0.1)

    def test_exaggeration_at_max_boundary_is_valid(self):
        schema = TTSRequestSchema(text="hello", exaggeration=1.0)
        assert schema.exaggeration == 1.0

    def test_exaggeration_at_min_boundary_is_valid(self):
        schema = TTSRequestSchema(text="hello", exaggeration=0.0)
        assert schema.exaggeration == 0.0

    def test_seed_negative_raises_validation_error(self):
        """ge=0 must reject negative seeds."""
        with pytest.raises(ValidationError):
            TTSRequestSchema(text="hello", seed=-1)

    def test_temperature_below_min_raises_validation_error(self):
        """ge=0.05 must reject temperatures below the minimum."""
        with pytest.raises(ValidationError):
            TTSRequestSchema(text="hello", temperature=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# TurboRequestSchema
# ─────────────────────────────────────────────────────────────────────────────


class TestTurboRequestSchema:
    """Tests for POST /api/v1/turbo/generate request body schema."""

    # ── defaults ─────────────────────────────────────────────────────────────

    def test_default_temperature(self):
        assert TurboRequestSchema(text="hello").temperature == 0.8

    def test_default_top_k(self):
        assert TurboRequestSchema(text="hello").top_k == 1000

    def test_default_top_p(self):
        assert TurboRequestSchema(text="hello").top_p == 0.95

    def test_default_rep_penalty(self):
        assert TurboRequestSchema(text="hello").rep_penalty == 1.2

    def test_default_min_p(self):
        assert TurboRequestSchema(text="hello").min_p == 0.0

    def test_default_norm_loudness(self):
        assert TurboRequestSchema(text="hello").norm_loudness is True

    def test_default_seed(self):
        assert TurboRequestSchema(text="hello").seed == 0

    # ── to_domain ─────────────────────────────────────────────────────────────

    def test_to_domain_returns_turbo_tts_request_instance(self):
        from domain.models import TurboTTSRequest

        req = TurboRequestSchema(text="hello world").to_domain()
        assert isinstance(req, TurboTTSRequest)

    def test_to_domain_maps_text(self):
        req = TurboRequestSchema(text="turbo test").to_domain()
        assert req.text == "turbo test"

    def test_to_domain_ref_audio_path_is_none(self):
        req = TurboRequestSchema(text="hi").to_domain()
        assert req.ref_audio_path is None

    def test_to_domain_streaming_is_false(self):
        req = TurboRequestSchema(text="hi").to_domain()
        assert req.streaming is False

    def test_to_domain_maps_all_numeric_fields(self):
        schema = TurboRequestSchema(
            text="test",
            temperature=1.5,
            top_k=500,
            top_p=0.8,
            rep_penalty=1.8,
            min_p=0.1,
            norm_loudness=False,
            seed=99,
        )
        req = schema.to_domain()
        assert req.temperature == 1.5
        assert req.top_k == 500
        assert req.top_p == 0.8
        assert req.rep_penalty == 1.8
        assert req.min_p == 0.1
        assert req.norm_loudness is False
        assert req.seed == 99

    # ── validation errors ─────────────────────────────────────────────────────

    def test_missing_text_raises_validation_error(self):
        with pytest.raises(ValidationError):
            TurboRequestSchema()  # type: ignore[call-arg]

    def test_empty_text_raises_validation_error(self):
        with pytest.raises(ValidationError):
            TurboRequestSchema(text="")

    def test_top_k_below_min_raises_validation_error(self):
        """ge=1 must reject top_k=0."""
        with pytest.raises(ValidationError):
            TurboRequestSchema(text="hello", top_k=0)

    def test_top_k_negative_raises_validation_error(self):
        with pytest.raises(ValidationError):
            TurboRequestSchema(text="hello", top_k=-5)

    def test_top_k_at_min_boundary_is_valid(self):
        schema = TurboRequestSchema(text="hello", top_k=1)
        assert schema.top_k == 1


# ─────────────────────────────────────────────────────────────────────────────
# MultilingualRequestSchema
# ─────────────────────────────────────────────────────────────────────────────


class TestMultilingualRequestSchema:
    """Tests for POST /api/v1/multilingual/generate request body schema."""

    # ── defaults ─────────────────────────────────────────────────────────────

    def test_default_language(self):
        assert MultilingualRequestSchema(text="hello").language == "en"

    def test_default_exaggeration(self):
        assert MultilingualRequestSchema(text="hello").exaggeration == 0.5

    def test_default_cfg_weight(self):
        assert MultilingualRequestSchema(text="hello").cfg_weight == 0.5

    def test_default_temperature(self):
        assert MultilingualRequestSchema(text="hello").temperature == 0.8

    def test_default_rep_penalty_is_2_not_1_2(self):
        """Multilingual uses 2.0 (higher than Standard's 1.2) to suppress artefacts."""
        assert MultilingualRequestSchema(text="hello").rep_penalty == 2.0

    def test_default_min_p(self):
        assert MultilingualRequestSchema(text="hello").min_p == 0.05

    def test_default_top_p(self):
        assert MultilingualRequestSchema(text="hello").top_p == 1.0

    def test_default_seed(self):
        assert MultilingualRequestSchema(text="hello").seed == 0

    # ── to_domain ─────────────────────────────────────────────────────────────

    def test_to_domain_returns_multilingual_tts_request_instance(self):
        from domain.models import MultilingualTTSRequest

        req = MultilingualRequestSchema(text="Bonjour", language="fr").to_domain()
        assert isinstance(req, MultilingualTTSRequest)

    def test_to_domain_maps_text(self):
        req = MultilingualRequestSchema(text="Hola", language="es").to_domain()
        assert req.text == "Hola"

    def test_to_domain_maps_language(self):
        req = MultilingualRequestSchema(text="Bonjour", language="fr").to_domain()
        assert req.language == "fr"

    def test_to_domain_ref_audio_path_is_none(self):
        req = MultilingualRequestSchema(text="hi").to_domain()
        assert req.ref_audio_path is None

    def test_to_domain_streaming_is_false(self):
        req = MultilingualRequestSchema(text="hi").to_domain()
        assert req.streaming is False

    def test_to_domain_maps_all_numeric_fields(self):
        schema = MultilingualRequestSchema(
            text="test",
            language="de",
            exaggeration=0.6,
            cfg_weight=0.4,
            temperature=1.2,
            rep_penalty=2.0,
            min_p=0.08,
            top_p=0.95,
            seed=7,
        )
        req = schema.to_domain()
        assert req.language == "de"
        assert req.exaggeration == 0.6
        assert req.cfg_weight == 0.4
        assert req.temperature == 1.2
        assert req.rep_penalty == 2.0
        assert req.min_p == 0.08
        assert req.top_p == 0.95
        assert req.seed == 7

    # ── validation errors ─────────────────────────────────────────────────────

    def test_missing_text_raises_validation_error(self):
        with pytest.raises(ValidationError):
            MultilingualRequestSchema()  # type: ignore[call-arg]

    def test_empty_text_raises_validation_error(self):
        with pytest.raises(ValidationError):
            MultilingualRequestSchema(text="")


# ─────────────────────────────────────────────────────────────────────────────
# ModelStatusResponse
# ─────────────────────────────────────────────────────────────────────────────


class TestModelStatusResponse:
    """Tests for GET /api/v1/models/status response item schema."""

    def _make_domain_status(self, **overrides: object) -> ModelStatus:
        key: ModelKey = cast("ModelKey", overrides.get("key", "tts"))
        display_name: str = str(overrides.get("display_name", "Standard TTS"))
        class_name: str = str(overrides.get("class_name", "ChatterboxTTS"))
        description: str = str(overrides.get("description", "Standard TTS model"))
        params: str = str(overrides.get("params", "500M"))
        size_gb: float = float(str(overrides.get("size_gb", 1.4)))
        in_memory: bool = bool(overrides.get("in_memory", True))
        on_disk: bool = bool(overrides.get("on_disk", True))
        return ModelStatus(
            key=key,
            display_name=display_name,
            class_name=class_name,
            description=description,
            params=params,
            size_gb=size_gb,
            in_memory=in_memory,
            on_disk=on_disk,
        )

    def test_from_domain_maps_key(self):
        resp = ModelStatusResponse.from_domain(self._make_domain_status(key="turbo"))
        assert resp.key == "turbo"

    def test_from_domain_maps_display_name(self):
        resp = ModelStatusResponse.from_domain(self._make_domain_status(display_name="Turbo TTS"))
        assert resp.display_name == "Turbo TTS"

    def test_from_domain_maps_class_name(self):
        resp = ModelStatusResponse.from_domain(
            self._make_domain_status(class_name="ChatterboxTurboTTS")
        )
        assert resp.class_name == "ChatterboxTurboTTS"

    def test_from_domain_maps_description(self):
        resp = ModelStatusResponse.from_domain(self._make_domain_status(description="A fast model"))
        assert resp.description == "A fast model"

    def test_from_domain_maps_params(self):
        resp = ModelStatusResponse.from_domain(self._make_domain_status(params="350M"))
        assert resp.params == "350M"

    def test_from_domain_maps_size_gb(self):
        resp = ModelStatusResponse.from_domain(self._make_domain_status(size_gb=2.8))
        assert resp.size_gb == 2.8

    def test_from_domain_maps_in_memory_true(self):
        resp = ModelStatusResponse.from_domain(self._make_domain_status(in_memory=True))
        assert resp.in_memory is True

    def test_from_domain_maps_in_memory_false(self):
        resp = ModelStatusResponse.from_domain(self._make_domain_status(in_memory=False))
        assert resp.in_memory is False

    def test_from_domain_maps_on_disk_true(self):
        resp = ModelStatusResponse.from_domain(self._make_domain_status(on_disk=True))
        assert resp.on_disk is True

    def test_from_domain_maps_on_disk_false(self):
        resp = ModelStatusResponse.from_domain(self._make_domain_status(on_disk=False))
        assert resp.on_disk is False

    def test_full_round_trip(self):
        status = self._make_domain_status()
        resp = ModelStatusResponse.from_domain(status)
        assert resp.key == status.key
        assert resp.display_name == status.display_name
        assert resp.class_name == status.class_name
        assert resp.description == status.description
        assert resp.params == status.params
        assert resp.size_gb == status.size_gb
        assert resp.in_memory == status.in_memory
        assert resp.on_disk == status.on_disk


# ─────────────────────────────────────────────────────────────────────────────
# MemoryStatsResponse
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryStatsResponse:
    """Tests for GET /api/v1/models/memory response schema."""

    def test_from_domain_with_none_device_fields(self):
        stats = MemoryStats(
            sys_total_gb=16.0,
            sys_used_gb=8.0,
            sys_avail_gb=8.0,
            sys_percent=50.0,
            proc_rss_gb=2.0,
            device_name="CPU",
            device_driver_gb=None,
            device_max_gb=None,
        )
        resp = MemoryStatsResponse.from_domain(stats)
        assert resp.sys_total_gb == 16.0
        assert resp.sys_used_gb == 8.0
        assert resp.sys_avail_gb == 8.0
        assert resp.sys_percent == 50.0
        assert resp.proc_rss_gb == 2.0
        assert resp.device_name == "CPU"
        assert resp.device_driver_gb is None
        assert resp.device_max_gb is None

    def test_from_domain_with_gpu_device_fields(self):
        stats = MemoryStats(
            sys_total_gb=64.0,
            sys_used_gb=32.0,
            sys_avail_gb=32.0,
            sys_percent=50.0,
            proc_rss_gb=4.0,
            device_name="NVIDIA A100",
            device_driver_gb=40.0,
            device_max_gb=80.0,
        )
        resp = MemoryStatsResponse.from_domain(stats)
        assert resp.device_name == "NVIDIA A100"
        assert resp.device_driver_gb == 40.0
        assert resp.device_max_gb == 80.0

    def test_from_domain_maps_all_sys_fields(self):
        stats = MemoryStats(
            sys_total_gb=32.0,
            sys_used_gb=12.5,
            sys_avail_gb=19.5,
            sys_percent=39.1,
            proc_rss_gb=1.8,
        )
        resp = MemoryStatsResponse.from_domain(stats)
        assert resp.sys_total_gb == 32.0
        assert resp.sys_used_gb == 12.5
        assert resp.sys_avail_gb == 19.5
        assert resp.sys_percent == 39.1
        assert resp.proc_rss_gb == 1.8


# ─────────────────────────────────────────────────────────────────────────────
# WatermarkResponse
# ─────────────────────────────────────────────────────────────────────────────


class TestWatermarkResponse:
    """Tests for POST /api/v1/watermark/detect response schema."""

    def test_from_domain_maps_score(self):
        result = WatermarkResult(score=0.95, verdict="detected", message="Found", available=True)
        assert WatermarkResponse.from_domain(result).score == 0.95

    def test_from_domain_maps_verdict(self):
        result = WatermarkResult(
            score=0.05, verdict="not_detected", message="Not found", available=True
        )
        assert WatermarkResponse.from_domain(result).verdict == "not_detected"

    def test_from_domain_maps_message(self):
        result = WatermarkResult(
            score=0.5, verdict="inconclusive", message="Ambiguous signal", available=True
        )
        assert WatermarkResponse.from_domain(result).message == "Ambiguous signal"

    def test_from_domain_maps_available_true(self):
        result = WatermarkResult(score=0.9, verdict="detected", message="Found", available=True)
        assert WatermarkResponse.from_domain(result).available is True

    def test_from_domain_maps_available_false(self):
        result = WatermarkResult(
            score=0.0, verdict="unavailable", message="Not installed", available=False
        )
        assert WatermarkResponse.from_domain(result).available is False

    def test_full_round_trip_detected(self):
        result = WatermarkResult(
            score=0.95,
            verdict="detected",
            message="Watermark detected",
            available=True,
        )
        resp = WatermarkResponse.from_domain(result)
        assert resp.score == 0.95
        assert resp.verdict == "detected"
        assert resp.message == "Watermark detected"
        assert resp.available is True

    def test_full_round_trip_unavailable(self):
        result = WatermarkResult(
            score=0.0,
            verdict="unavailable",
            message="Library not installed",
            available=False,
        )
        resp = WatermarkResponse.from_domain(result)
        assert resp.score == 0.0
        assert resp.verdict == "unavailable"
        assert resp.available is False


# ─────────────────────────────────────────────────────────────────────────────
# HealthResponse
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthResponse:
    """Tests for GET /api/v1/health response schema."""

    def test_has_status_field(self):
        resp = HealthResponse(status="ok", device="cpu")
        assert resp.status == "ok"

    def test_has_device_field(self):
        resp = HealthResponse(status="ok", device="cpu")
        assert resp.device == "cpu"

    def test_device_can_be_mps(self):
        resp = HealthResponse(status="ok", device="mps")
        assert resp.device == "mps"

    def test_device_can_be_cuda(self):
        resp = HealthResponse(status="ok", device="cuda")
        assert resp.device == "cuda"

    def test_missing_status_raises_validation_error(self):
        with pytest.raises(ValidationError):
            HealthResponse.model_validate({"device": "cpu"})  # missing 'status'

    def test_missing_device_raises_validation_error(self):
        with pytest.raises(ValidationError):
            HealthResponse.model_validate({"status": "ok"})  # missing 'device'


# ─────────────────────────────────────────────────────────────────────────────
# audio_result_to_wav_bytes()
# ─────────────────────────────────────────────────────────────────────────────


class TestAudioResultToWavBytes:
    """Tests for the float32 → 16-bit WAV encoder helper."""

    def _make_audio(
        self,
        n_samples: int = 24000,
        sample_rate: int = 24000,
        value: float = 0.0,
    ) -> AudioResult:
        samples = np.full(n_samples, value, dtype=np.float32)
        return AudioResult(sample_rate=sample_rate, samples=samples)

    # ── basic contract ────────────────────────────────────────────────────────

    def test_returns_bytes(self):
        result = self._make_audio()
        assert isinstance(audio_result_to_wav_bytes(result), bytes)

    def test_returns_riff_wav_header(self):
        """Output must be a valid WAV file — RIFF header at offset 0."""
        wav = audio_result_to_wav_bytes(self._make_audio())
        assert wav[:4] == b"RIFF"

    def test_contains_wave_marker(self):
        """Bytes 8-12 of a valid WAV file are b'WAVE'."""
        wav = audio_result_to_wav_bytes(self._make_audio())
        assert wav[8:12] == b"WAVE"

    def test_output_longer_than_header(self):
        """PCM data follows the 44-byte header — output must be > 44 bytes."""
        wav = audio_result_to_wav_bytes(self._make_audio(n_samples=100))
        assert len(wav) > 44

    # ── silence ───────────────────────────────────────────────────────────────

    def test_handles_all_zero_samples(self):
        """All-zero (silence) array must produce a valid WAV without raising."""
        result = AudioResult(
            sample_rate=24000,
            samples=np.zeros(24000, dtype=np.float32),
        )
        wav = audio_result_to_wav_bytes(result)
        assert wav[:4] == b"RIFF"

    def test_silence_has_expected_data_length(self):
        """1 second @ 24 kHz, 16-bit mono → 48 000 bytes of PCM data."""
        n_samples = 24000
        result = AudioResult(
            sample_rate=24000,
            samples=np.zeros(n_samples, dtype=np.float32),
        )
        wav = audio_result_to_wav_bytes(result)
        # WAV file size field (little-endian uint32 at bytes 4-7) encodes
        # total size minus the first 8 bytes (RIFF + size field itself).
        import struct

        reported_size = struct.unpack_from("<I", wav, 4)[0]
        # reported_size = total_file_size - 8
        assert reported_size == len(wav) - 8

    # ── peak normalisation ────────────────────────────────────────────────────

    def test_peak_above_1_does_not_raise(self):
        """Samples with peak > 1.0 must be normalised, not raise an exception."""
        samples = np.ones(24000, dtype=np.float32) * 2.0
        result = AudioResult(sample_rate=24000, samples=samples)
        wav = audio_result_to_wav_bytes(result)
        assert wav[:4] == b"RIFF"

    def test_peak_above_1_produces_valid_wav(self):
        """After peak normalisation the output must still be parseable by scipy."""
        from scipy.io import wavfile

        samples = np.ones(24000, dtype=np.float32) * 5.0
        result = AudioResult(sample_rate=24000, samples=samples)
        wav = audio_result_to_wav_bytes(result)

        sr, data = wavfile.read(io.BytesIO(wav))
        assert sr == 24000
        assert data.dtype == np.int16

    def test_peak_at_1_does_not_clip_aggressively(self):
        """Samples with peak exactly 1.0 (within range) should not trigger scaling."""
        samples = np.ones(24000, dtype=np.float32) * 1.0
        result = AudioResult(sample_rate=24000, samples=samples)
        wav = audio_result_to_wav_bytes(result)
        assert wav[:4] == b"RIFF"

    # ── round-trip via scipy ──────────────────────────────────────────────────

    def test_round_trip_sample_rate(self):
        """Sample rate encoded in the WAV header must match AudioResult.sample_rate."""
        from scipy.io import wavfile

        result = AudioResult(
            sample_rate=44100,
            samples=np.zeros(44100, dtype=np.float32),
        )
        wav = audio_result_to_wav_bytes(result)
        sr, _ = wavfile.read(io.BytesIO(wav))
        assert sr == 44100

    def test_round_trip_sample_count(self):
        """Number of int16 samples in the WAV file must equal the source count."""
        from scipy.io import wavfile

        n = 12000
        result = AudioResult(
            sample_rate=24000,
            samples=np.zeros(n, dtype=np.float32),
        )
        wav = audio_result_to_wav_bytes(result)
        _, data = wavfile.read(io.BytesIO(wav))
        assert len(data) == n

    def test_output_is_16bit_pcm(self):
        """audio_result_to_wav_bytes always encodes as int16 (16-bit PCM)."""
        from scipy.io import wavfile

        result = AudioResult(
            sample_rate=24000,
            samples=np.zeros(24000, dtype=np.float32),
        )
        wav = audio_result_to_wav_bytes(result)
        _, data = wavfile.read(io.BytesIO(wav))
        assert data.dtype == np.int16

    # ── import guard ──────────────────────────────────────────────────────────

    def test_scipy_wavfile_is_importable(self):
        """scipy.io.wavfile must be available in the project venv."""
        from scipy.io import wavfile

        assert wavfile is not None

    def test_numpy_is_importable(self):
        """numpy must be available in the project venv."""
        import numpy as _np

        assert _np is not None


# ──────────────────────────────────────────────────────────────────────────────
# audio_delta_to_pcm_bytes()
# ──────────────────────────────────────────────────────────────────────────────


class TestAudioDeltaToPcmBytes:
    """audio_delta_to_pcm_bytes() converts float32 delta samples to raw int16 PCM bytes."""

    def test_returns_bytes(self) -> None:
        from adapters.inbound.rest.schemas import audio_delta_to_pcm_bytes

        result = audio_delta_to_pcm_bytes(np.zeros(100, dtype=np.float32))
        assert isinstance(result, bytes)

    def test_byte_count_is_double_sample_count(self) -> None:
        """int16 = 2 bytes per sample."""
        from adapters.inbound.rest.schemas import audio_delta_to_pcm_bytes

        result = audio_delta_to_pcm_bytes(np.zeros(100, dtype=np.float32))
        assert len(result) == 200  # 100 samples x 2 bytes

    def test_silence_produces_zero_bytes(self) -> None:
        from adapters.inbound.rest.schemas import audio_delta_to_pcm_bytes

        result = audio_delta_to_pcm_bytes(np.zeros(10, dtype=np.float32))
        assert all(b == 0 for b in result)

    def test_peak_normalisation_prevents_clipping(self) -> None:
        """Samples > 1.0 must be normalised before int16 conversion."""
        from adapters.inbound.rest.schemas import audio_delta_to_pcm_bytes

        samples = np.array([2.0, -4.0, 1.0], dtype=np.float32)
        result = audio_delta_to_pcm_bytes(samples)
        # Peak = 4.0, normalised: [0.5, -1.0, 0.25]
        # int16: [16383, -32767, 8191]  (approx)
        values = np.frombuffer(result, dtype=np.int16)
        assert values[1] <= -32000  # -1.0 → near -32767

    def test_empty_array_returns_empty_bytes(self) -> None:
        from adapters.inbound.rest.schemas import audio_delta_to_pcm_bytes

        result = audio_delta_to_pcm_bytes(np.array([], dtype=np.float32))
        assert result == b""


# ──────────────────────────────────────────────────────────────────────────────
# ErrorDetail and ErrorResponse
# ──────────────────────────────────────────────────────────────────────────────


class TestErrorResponse:
    """ErrorDetail and ErrorResponse — consistent error envelope."""

    def test_error_detail_default_path_is_empty_list(self) -> None:
        from adapters.inbound.rest.schemas import ErrorDetail

        e = ErrorDetail(message="bad input")
        assert e.path == []

    def test_error_detail_with_path(self) -> None:
        from adapters.inbound.rest.schemas import ErrorDetail

        e = ErrorDetail(message="required", path=["body", "text"])
        assert e.path == ["body", "text"]

    def test_error_detail_serializes_to_dict(self) -> None:
        from adapters.inbound.rest.schemas import ErrorDetail

        e = ErrorDetail(message="x", path=["a", 0])
        d = e.model_dump()
        assert d == {"message": "x", "path": ["a", 0]}

    def test_error_response_errors_field(self) -> None:
        from adapters.inbound.rest.schemas import ErrorDetail, ErrorResponse

        r = ErrorResponse(errors=[ErrorDetail(message="fail")])
        assert len(r.errors) == 1
        assert r.errors[0].message == "fail"

    def test_error_response_serializes(self) -> None:
        from adapters.inbound.rest.schemas import ErrorDetail, ErrorResponse

        r = ErrorResponse(errors=[ErrorDetail(message="x")])
        d = r.model_dump()
        assert "errors" in d
        assert d["errors"][0]["message"] == "x"

    def test_error_response_multiple_errors(self) -> None:
        from adapters.inbound.rest.schemas import ErrorDetail, ErrorResponse

        r = ErrorResponse(
            errors=[
                ErrorDetail(message="field a", path=["body", "a"]),
                ErrorDetail(message="field b", path=["body", "b"]),
            ]
        )
        assert len(r.errors) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Schema contract tests  (InboundSchema / OutboundSchema)
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaContracts:
    """InboundSchema and OutboundSchema contract enforcement tests.
    Written before the base classes exist (RED phase).
    """

    # ── InboundSchema ─────────────────────────────────────────────────────────

    def test_inbound_schema_is_abstract(self) -> None:
        """Direct instantiation of InboundSchema must raise TypeError."""
        from adapters.inbound.rest.schemas import InboundSchema

        with pytest.raises(TypeError):
            InboundSchema()  # type: ignore[abstract]

    def test_tts_request_schema_is_inbound(self) -> None:
        from adapters.inbound.rest.schemas import InboundSchema, TTSRequestSchema

        assert issubclass(TTSRequestSchema, InboundSchema)

    def test_turbo_request_schema_is_inbound(self) -> None:
        from adapters.inbound.rest.schemas import InboundSchema, TurboRequestSchema

        assert issubclass(TurboRequestSchema, InboundSchema)

    def test_multilingual_request_schema_is_inbound(self) -> None:
        from adapters.inbound.rest.schemas import InboundSchema, MultilingualRequestSchema

        assert issubclass(MultilingualRequestSchema, InboundSchema)

    def test_inbound_schema_missing_to_domain_raises(self) -> None:
        """A subclass that forgets to_domain() must raise TypeError on instantiation."""
        from adapters.inbound.rest.schemas import InboundSchema
        from domain.models import TTSRequest

        class _Broken(InboundSchema[TTSRequest]):
            text: str
            # intentionally missing to_domain()

        with pytest.raises(TypeError):
            _Broken(text="hello")

    def test_inbound_schema_to_domain_returns_correct_type(self) -> None:
        from adapters.inbound.rest.schemas import TTSRequestSchema
        from domain.models import TTSRequest

        schema = TTSRequestSchema(text="Hello")
        result = schema.to_domain()
        assert isinstance(result, TTSRequest)

    # ── OutboundSchema ────────────────────────────────────────────────────────

    def test_outbound_schema_is_abstract(self) -> None:
        """Direct instantiation of OutboundSchema must raise TypeError."""
        from adapters.inbound.rest.schemas import OutboundSchema

        with pytest.raises(TypeError):
            OutboundSchema()  # type: ignore[abstract]

    def test_model_status_response_is_outbound(self) -> None:
        from adapters.inbound.rest.schemas import ModelStatusResponse, OutboundSchema

        assert issubclass(ModelStatusResponse, OutboundSchema)

    def test_memory_stats_response_is_outbound(self) -> None:
        from adapters.inbound.rest.schemas import MemoryStatsResponse, OutboundSchema

        assert issubclass(MemoryStatsResponse, OutboundSchema)

    def test_watermark_response_is_outbound(self) -> None:
        from adapters.inbound.rest.schemas import OutboundSchema, WatermarkResponse

        assert issubclass(WatermarkResponse, OutboundSchema)

    def test_outbound_schema_missing_from_domain_raises(self) -> None:
        """A subclass that forgets from_domain() must raise TypeError on instantiation."""
        from adapters.inbound.rest.schemas import OutboundSchema
        from domain.models import ModelStatus

        class _Broken(OutboundSchema[ModelStatus]):
            key: str
            # intentionally missing from_domain()

        from typing import cast

        with pytest.raises(TypeError):
            _Broken(key=cast("ModelKey", "tts"))

    def test_outbound_schema_from_domain_returns_correct_type(self) -> None:
        from typing import cast

        from adapters.inbound.rest.schemas import ModelStatusResponse
        from domain.models import ModelStatus

        status = ModelStatus(
            key=cast("ModelKey", "tts"),
            display_name="Standard TTS",
            class_name="ChatterboxTTS",
            description="desc",
            params="500M",
            size_gb=1.4,
            in_memory=False,
            on_disk=True,
        )
        resp = ModelStatusResponse.from_domain(status)
        assert isinstance(resp, ModelStatusResponse)
        assert resp.key == "tts"

    # ── Neutral schemas (no domain counterpart) ───────────────────────────────

    def test_health_response_is_plain_base_model(self) -> None:
        """HealthResponse has no domain counterpart — stays plain BaseModel."""
        from pydantic import BaseModel

        from adapters.inbound.rest.schemas import HealthResponse, InboundSchema, OutboundSchema

        assert issubclass(HealthResponse, BaseModel)
        assert not issubclass(HealthResponse, InboundSchema)
        assert not issubclass(HealthResponse, OutboundSchema)
