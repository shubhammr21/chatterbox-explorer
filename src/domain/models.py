"""
src/domain/models.py
=====================
Pure domain models — pydantic BaseModel, zero framework dependencies.

Allowed imports at runtime: pydantic, numpy, domain.types, and stdlib.
Pydantic needs numpy available at runtime for arbitrary_types_allowed,
and needs domain.types Literal definitions available at runtime so that
Literal fields (DeviceType, LanguageCode, etc.) are validated at
construction time.

Forbidden at runtime: torch, gradio, chatterbox, psutil, huggingface_hub
— any of these appearing outside TYPE_CHECKING is an architecture violation.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

from domain.types import DeviceType, LanguageCode, ModelKey, WatermarkVerdict

# ──────────────────────────────────────────────────────────────────────────────
# TTS request value-objects
# ──────────────────────────────────────────────────────────────────────────────


class TTSRequest(BaseModel):
    """Input value-object for the Standard TTS service.

    Defaults mirror the '🎯 Default' preset and the Gradio slider defaults
    in app.py so callers that don't customise anything get sensible output.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    ref_audio_path: str | None = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    rep_penalty: float = 1.2
    min_p: float = 0.05
    top_p: float = 1.0
    seed: int = 0
    streaming: bool = False


class TurboTTSRequest(BaseModel):
    """Input value-object for the Turbo TTS service.

    Turbo does NOT support exaggeration or cfg_weight — those params are absent.
    norm_loudness normalises the reference audio to -27 LUFS before conditioning.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    ref_audio_path: str | None = None
    temperature: float = 0.8
    top_k: int = 1000
    top_p: float = 0.95
    rep_penalty: float = 1.2
    min_p: float = 0.0
    norm_loudness: bool = True
    seed: int = 0
    streaming: bool = False


class MultilingualTTSRequest(BaseModel):
    """Input value-object for the Multilingual TTS service (23 languages).

    Uses the same parameter set as TTSRequest but adds a `language` code and
    ships with a higher default rep_penalty (2.0) which empirically reduces
    artefacts on non-English languages.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    language: LanguageCode = "en"
    ref_audio_path: str | None = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    rep_penalty: float = 2.0
    min_p: float = 0.05
    top_p: float = 1.0
    seed: int = 0
    streaming: bool = False


class VoiceConversionRequest(BaseModel):
    """Input value-object for the Voice Conversion service.

    Both paths are mandatory — VC converts source audio to the target voice
    without any text prompt.
    """

    model_config = ConfigDict(frozen=True)

    source_audio_path: str | None
    target_voice_path: str | None


# ──────────────────────────────────────────────────────────────────────────────
# Output value-objects
# ──────────────────────────────────────────────────────────────────────────────


class AudioResult(BaseModel):
    """Holds the raw audio output produced by any TTS or VC service.

    `samples` must be a 1-D float32 NumPy array of shape (N,).
    `sample_rate` is in Hz (typically 24 000 for Chatterbox models).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    sample_rate: int
    samples: np.ndarray  # float32, shape (N,)

    @computed_field
    @property
    def duration_s(self) -> float:
        """Duration in seconds.  Returns 0.0 if sample_rate is 0 (guard)."""
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / self.sample_rate


# ──────────────────────────────────────────────────────────────────────────────
# Model management value-objects
# ──────────────────────────────────────────────────────────────────────────────


class ModelStatus(BaseModel):
    """Snapshot of a single model's identity and load state.

    Populated by IModelManagerService.get_all_status() and consumed by the
    Gradio model-manager tab renderer.
    """

    model_config = ConfigDict(frozen=True)

    key: ModelKey  # "tts" | "turbo" | "multilingual" | "vc"
    display_name: str  # e.g. "Standard TTS"
    class_name: str  # e.g. "ChatterboxTTS"
    description: str
    params: str  # human-readable, e.g. "500M" or "—"
    size_gb: float
    in_memory: bool
    on_disk: bool


class MemoryStats(BaseModel):
    """System + device memory snapshot returned by IMemoryMonitor.get_stats().

    Device fields (device_driver_gb, device_max_gb) are None on CPU-only
    systems where no GPU / MPS memory tracking is available.
    """

    model_config = ConfigDict(frozen=True)

    sys_total_gb: float
    sys_used_gb: float
    sys_avail_gb: float
    sys_percent: float
    proc_rss_gb: float
    device_name: str = "CPU"
    device_driver_gb: float | None = None
    device_max_gb: float | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Watermark value-object
# ──────────────────────────────────────────────────────────────────────────────


class WatermarkResult(BaseModel):
    """Result of a watermark-detection pass on a generated audio clip.

    verdict is one of:
        "detected"      — watermark found above threshold
        "not_detected"  — no watermark signal found
        "inconclusive"  — score is in the ambiguous range
        "unavailable"   — detection library not installed / failed to initialise
    """

    model_config = ConfigDict(frozen=True)

    score: float
    verdict: WatermarkVerdict
    message: str
    available: bool


# ──────────────────────────────────────────────────────────────────────────────
# Application configuration
# ──────────────────────────────────────────────────────────────────────────────


class AppConfig(BaseModel):
    """Immutable runtime configuration resolved once at bootstrap.

    Passed into services and adapters via dependency injection so that no
    module reads environment variables or detects hardware on its own.
    """

    model_config = ConfigDict(frozen=True)

    device: DeviceType
    watermark_available: bool
