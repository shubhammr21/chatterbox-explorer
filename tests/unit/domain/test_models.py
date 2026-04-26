"""
tests/unit/domain/test_models.py
=================================
TDD — RED phase: tests for chatterbox_explorer.domain.models

Run before implementation exists → all tests must FAIL initially.
Run after implementation → all tests must PASS.

Rules:
- No torch, gradio, chatterbox, psutil, or huggingface_hub imports.
- Pure dataclass behaviour only.
"""
from __future__ import annotations

import numpy as np
import pytest

from chatterbox_explorer.domain.models import (
    AppConfig,
    AudioResult,
    MemoryStats,
    ModelStatus,
    MultilingualTTSRequest,
    TTSRequest,
    TurboTTSRequest,
    VoiceConversionRequest,
    WatermarkResult,
)


# ──────────────────────────────────────────────────────────────────────────────
# TTSRequest
# ──────────────────────────────────────────────────────────────────────────────

class TestTTSRequest:
    def test_defaults(self):
        req = TTSRequest(text="Hello world")
        assert req.text == "Hello world"
        assert req.ref_audio_path is None
        assert req.exaggeration == pytest.approx(0.5)
        assert req.cfg_weight == pytest.approx(0.5)
        assert req.temperature == pytest.approx(0.8)
        assert req.rep_penalty == pytest.approx(1.2)
        assert req.min_p == pytest.approx(0.05)
        assert req.top_p == pytest.approx(1.0)
        assert req.seed == 0
        assert req.streaming is False

    def test_custom_values(self):
        req = TTSRequest(
            text="Test",
            ref_audio_path="/tmp/ref.wav",
            exaggeration=0.75,
            cfg_weight=0.3,
            temperature=1.2,
            rep_penalty=1.4,
            min_p=0.02,
            top_p=0.95,
            seed=42,
            streaming=True,
        )
        assert req.ref_audio_path == "/tmp/ref.wav"
        assert req.exaggeration == pytest.approx(0.75)
        assert req.cfg_weight == pytest.approx(0.3)
        assert req.temperature == pytest.approx(1.2)
        assert req.rep_penalty == pytest.approx(1.4)
        assert req.min_p == pytest.approx(0.02)
        assert req.top_p == pytest.approx(0.95)
        assert req.seed == 42
        assert req.streaming is True

    def test_ref_audio_path_none_by_default(self):
        req = TTSRequest(text="No ref")
        assert req.ref_audio_path is None

    def test_streaming_false_by_default(self):
        req = TTSRequest(text="Stream check")
        assert req.streaming is False

    def test_seed_zero_by_default(self):
        req = TTSRequest(text="Seed check")
        assert req.seed == 0

    def test_is_dataclass_instance(self):
        import dataclasses
        req = TTSRequest(text="Dataclass check")
        assert dataclasses.is_dataclass(req)


# ──────────────────────────────────────────────────────────────────────────────
# TurboTTSRequest
# ──────────────────────────────────────────────────────────────────────────────

class TestTurboTTSRequest:
    def test_defaults(self):
        req = TurboTTSRequest(text="Turbo!")
        assert req.text == "Turbo!"
        assert req.ref_audio_path is None
        assert req.temperature == pytest.approx(0.8)
        assert req.top_k == 1000
        assert req.top_p == pytest.approx(0.95)
        assert req.rep_penalty == pytest.approx(1.2)
        assert req.min_p == pytest.approx(0.0)
        assert req.norm_loudness is True
        assert req.seed == 0
        assert req.streaming is False

    def test_norm_loudness_true_by_default(self):
        req = TurboTTSRequest(text="Norm check")
        assert req.norm_loudness is True

    def test_top_k_default(self):
        req = TurboTTSRequest(text="TopK")
        assert req.top_k == 1000

    def test_custom_values(self):
        req = TurboTTSRequest(
            text="Custom turbo",
            ref_audio_path="/tmp/voice.flac",
            temperature=1.05,
            top_k=200,
            top_p=0.90,
            rep_penalty=1.3,
            min_p=0.0,
            norm_loudness=False,
            seed=7,
            streaming=True,
        )
        assert req.ref_audio_path == "/tmp/voice.flac"
        assert req.temperature == pytest.approx(1.05)
        assert req.top_k == 200
        assert req.top_p == pytest.approx(0.90)
        assert req.rep_penalty == pytest.approx(1.3)
        assert req.norm_loudness is False
        assert req.seed == 7
        assert req.streaming is True

    def test_is_dataclass_instance(self):
        import dataclasses
        req = TurboTTSRequest(text="Dataclass check")
        assert dataclasses.is_dataclass(req)


# ──────────────────────────────────────────────────────────────────────────────
# MultilingualTTSRequest
# ──────────────────────────────────────────────────────────────────────────────

class TestMultilingualTTSRequest:
    def test_defaults(self):
        req = MultilingualTTSRequest(text="Bonjour")
        assert req.text == "Bonjour"
        assert req.language == "en"
        assert req.ref_audio_path is None
        assert req.exaggeration == pytest.approx(0.5)
        assert req.cfg_weight == pytest.approx(0.5)
        assert req.temperature == pytest.approx(0.8)
        assert req.rep_penalty == pytest.approx(2.0)
        assert req.min_p == pytest.approx(0.05)
        assert req.top_p == pytest.approx(1.0)
        assert req.seed == 0
        assert req.streaming is False

    def test_language_default_is_en(self):
        req = MultilingualTTSRequest(text="Language check")
        assert req.language == "en"

    def test_rep_penalty_default_is_2_0(self):
        """Multilingual uses rep_penalty=2.0 (higher than Standard TTS default)."""
        req = MultilingualTTSRequest(text="Penalty check")
        assert req.rep_penalty == pytest.approx(2.0)

    def test_custom_language(self):
        req = MultilingualTTSRequest(text="Hallo", language="de")
        assert req.language == "de"

    def test_is_dataclass_instance(self):
        import dataclasses
        req = MultilingualTTSRequest(text="ML dataclass")
        assert dataclasses.is_dataclass(req)


# ──────────────────────────────────────────────────────────────────────────────
# VoiceConversionRequest
# ──────────────────────────────────────────────────────────────────────────────

class TestVoiceConversionRequest:
    def test_requires_both_paths(self):
        req = VoiceConversionRequest(
            source_audio_path="/tmp/source.wav",
            target_voice_path="/tmp/target.wav",
        )
        assert req.source_audio_path == "/tmp/source.wav"
        assert req.target_voice_path == "/tmp/target.wav"

    def test_no_optional_fields(self):
        """VoiceConversionRequest has no optional parameters — both paths are mandatory."""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(VoiceConversionRequest)}
        assert fields == {"source_audio_path", "target_voice_path"}

    def test_is_dataclass_instance(self):
        import dataclasses
        req = VoiceConversionRequest(
            source_audio_path="/a.wav",
            target_voice_path="/b.wav",
        )
        assert dataclasses.is_dataclass(req)


# ──────────────────────────────────────────────────────────────────────────────
# AudioResult
# ──────────────────────────────────────────────────────────────────────────────

class TestAudioResult:
    def test_duration_s_basic(self):
        samples = np.zeros(24000, dtype=np.float32)
        result = AudioResult(sample_rate=24000, samples=samples)
        assert result.duration_s == pytest.approx(1.0)

    def test_duration_s_half_second(self):
        samples = np.zeros(12000, dtype=np.float32)
        result = AudioResult(sample_rate=24000, samples=samples)
        assert result.duration_s == pytest.approx(0.5)

    def test_duration_s_zero_samples(self):
        samples = np.zeros(0, dtype=np.float32)
        result = AudioResult(sample_rate=24000, samples=samples)
        assert result.duration_s == pytest.approx(0.0)

    def test_duration_s_zero_sample_rate_returns_zero(self):
        """Guard against division by zero when sample_rate is 0."""
        samples = np.ones(100, dtype=np.float32)
        result = AudioResult(sample_rate=0, samples=samples)
        assert result.duration_s == pytest.approx(0.0)

    def test_samples_are_numpy_array(self):
        samples = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        result = AudioResult(sample_rate=16000, samples=samples)
        assert isinstance(result.samples, np.ndarray)

    def test_sample_rate_stored(self):
        samples = np.zeros(8000, dtype=np.float32)
        result = AudioResult(sample_rate=8000, samples=samples)
        assert result.sample_rate == 8000

    def test_duration_s_44100_hz(self):
        samples = np.zeros(44100, dtype=np.float32)
        result = AudioResult(sample_rate=44100, samples=samples)
        assert result.duration_s == pytest.approx(1.0)

    def test_is_dataclass_instance(self):
        import dataclasses
        result = AudioResult(sample_rate=16000, samples=np.zeros(100, dtype=np.float32))
        assert dataclasses.is_dataclass(result)


# ──────────────────────────────────────────────────────────────────────────────
# ModelStatus
# ──────────────────────────────────────────────────────────────────────────────

class TestModelStatus:
    def test_all_fields_stored(self):
        status = ModelStatus(
            key="tts",
            display_name="Standard TTS",
            class_name="ChatterboxTTS",
            description="English voice cloning",
            params="500M",
            size_gb=1.4,
            in_memory=False,
            on_disk=True,
        )
        assert status.key == "tts"
        assert status.display_name == "Standard TTS"
        assert status.class_name == "ChatterboxTTS"
        assert status.description == "English voice cloning"
        assert status.params == "500M"
        assert status.size_gb == pytest.approx(1.4)
        assert status.in_memory is False
        assert status.on_disk is True

    def test_is_dataclass_instance(self):
        import dataclasses
        s = ModelStatus(
            key="vc", display_name="VC", class_name="ChatterboxVC",
            description="", params="—", size_gb=0.4,
            in_memory=False, on_disk=False,
        )
        assert dataclasses.is_dataclass(s)


# ──────────────────────────────────────────────────────────────────────────────
# MemoryStats
# ──────────────────────────────────────────────────────────────────────────────

class TestMemoryStats:
    def test_required_fields(self):
        stats = MemoryStats(
            sys_total_gb=16.0,
            sys_used_gb=8.0,
            sys_avail_gb=8.0,
            sys_percent=50.0,
            proc_rss_gb=1.2,
        )
        assert stats.sys_total_gb == pytest.approx(16.0)
        assert stats.sys_used_gb == pytest.approx(8.0)
        assert stats.sys_avail_gb == pytest.approx(8.0)
        assert stats.sys_percent == pytest.approx(50.0)
        assert stats.proc_rss_gb == pytest.approx(1.2)

    def test_device_name_defaults_to_cpu(self):
        stats = MemoryStats(
            sys_total_gb=8.0, sys_used_gb=4.0,
            sys_avail_gb=4.0, sys_percent=50.0,
            proc_rss_gb=0.5,
        )
        assert stats.device_name == "CPU"

    def test_optional_device_fields_default_to_none(self):
        stats = MemoryStats(
            sys_total_gb=8.0, sys_used_gb=4.0,
            sys_avail_gb=4.0, sys_percent=50.0,
            proc_rss_gb=0.5,
        )
        assert stats.device_driver_gb is None
        assert stats.device_max_gb is None

    def test_device_fields_can_be_set(self):
        stats = MemoryStats(
            sys_total_gb=32.0, sys_used_gb=10.0,
            sys_avail_gb=22.0, sys_percent=31.25,
            proc_rss_gb=2.0,
            device_name="MPS",
            device_driver_gb=4.5,
            device_max_gb=16.0,
        )
        assert stats.device_name == "MPS"
        assert stats.device_driver_gb == pytest.approx(4.5)
        assert stats.device_max_gb == pytest.approx(16.0)

    def test_is_dataclass_instance(self):
        import dataclasses
        stats = MemoryStats(
            sys_total_gb=8.0, sys_used_gb=4.0,
            sys_avail_gb=4.0, sys_percent=50.0,
            proc_rss_gb=0.5,
        )
        assert dataclasses.is_dataclass(stats)


# ──────────────────────────────────────────────────────────────────────────────
# WatermarkResult
# ──────────────────────────────────────────────────────────────────────────────

class TestWatermarkResult:
    @pytest.mark.parametrize("verdict", [
        "detected", "not_detected", "inconclusive", "unavailable",
    ])
    def test_valid_verdicts(self, verdict: str):
        result = WatermarkResult(
            score=0.9,
            verdict=verdict,
            message="Test message",
            available=True,
        )
        assert result.verdict == verdict

    def test_all_fields_stored(self):
        result = WatermarkResult(
            score=0.85,
            verdict="detected",
            message="Watermark detected with high confidence.",
            available=True,
        )
        assert result.score == pytest.approx(0.85)
        assert result.verdict == "detected"
        assert result.message == "Watermark detected with high confidence."
        assert result.available is True

    def test_unavailable_watermark(self):
        result = WatermarkResult(
            score=0.0,
            verdict="unavailable",
            message="Watermark detection not available.",
            available=False,
        )
        assert result.available is False
        assert result.verdict == "unavailable"

    def test_is_dataclass_instance(self):
        import dataclasses
        result = WatermarkResult(
            score=0.0, verdict="unavailable",
            message="", available=False,
        )
        assert dataclasses.is_dataclass(result)


# ──────────────────────────────────────────────────────────────────────────────
# AppConfig
# ──────────────────────────────────────────────────────────────────────────────

class TestAppConfig:
    def test_fields(self):
        cfg = AppConfig(device="cuda", watermark_available=True)
        assert cfg.device == "cuda"
        assert cfg.watermark_available is True

    def test_cpu_device(self):
        cfg = AppConfig(device="cpu", watermark_available=False)
        assert cfg.device == "cpu"
        assert cfg.watermark_available is False

    def test_mps_device(self):
        cfg = AppConfig(device="mps", watermark_available=False)
        assert cfg.device == "mps"

    def test_is_dataclass_instance(self):
        import dataclasses
        cfg = AppConfig(device="cpu", watermark_available=False)
        assert dataclasses.is_dataclass(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-model: no framework contamination
# ──────────────────────────────────────────────────────────────────────────────

def test_domain_models_module_has_no_torch_import():
    """Ensure models.py never imports torch at module level."""
    import importlib
    import sys

    # Remove cached module if already imported so we can inspect fresh
    mod_name = "chatterbox_explorer.domain.models"
    mod = sys.modules.get(mod_name)
    if mod is None:
        mod = importlib.import_module(mod_name)

    # torch must NOT appear in the module's globals
    assert "torch" not in vars(mod), (
        "models.py must not import torch — domain layer must be framework-free"
    )


def test_domain_models_module_has_no_gradio_import():
    """Ensure models.py never imports gradio at module level."""
    import importlib
    import sys

    mod_name = "chatterbox_explorer.domain.models"
    mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
    assert "gradio" not in vars(mod), (
        "models.py must not import gradio — domain layer must be framework-free"
    )
