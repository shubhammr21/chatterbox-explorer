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

Inheritance hierarchy
---------------------
DomainModel                     frozen base — eliminates repeated model_config
├── BaseTTSRequest              text · ref_audio_path · seed · streaming
│   ├── StandardSamplingRequest exaggeration · cfg_weight · temperature ·
│   │   │                       rep_penalty=1.2 · min_p · top_p
│   │   ├── TTSRequest          Standard model — inherits everything
│   │   └── MultilingualTTSRequest  + language · rep_penalty override → 2.0
│   └── TurboTTSRequest         temperature · top_k · top_p=0.95 ·
│                               rep_penalty · min_p=0.0 · norm_loudness
├── VoiceConversionRequest      source_audio_path · target_voice_path
├── AudioResult                 sample_rate · samples (np.ndarray) · duration_s
├── ModelStatus                 key · display_name · class_name · …
├── MemoryStats                 sys_* · proc_rss_gb · device_*
├── WatermarkResult             score · verdict · message · available
└── AppConfig                   device · watermark_available
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

from domain.types import DeviceType, LanguageCode, ModelKey, WatermarkVerdict

# ──────────────────────────────────────────────────────────────────────────────
# Root base — shared frozen configuration
# ──────────────────────────────────────────────────────────────────────────────


class DomainModel(BaseModel):
    """Shared frozen base for all domain value-objects.

    Declaring ``frozen=True`` here eliminates the repeated
    ``model_config = ConfigDict(frozen=True)`` from every concrete class.
    All domain objects are immutable by design — use ``model_copy(update=…)``
    to create modified variants.
    """

    model_config = ConfigDict(frozen=True)


# ──────────────────────────────────────────────────────────────────────────────
# TTS request base classes
# ──────────────────────────────────────────────────────────────────────────────


class BaseTTSRequest(DomainModel):
    """Fields common to every TTS synthesis request.

    All three TTS services (Standard, Turbo, Multilingual) accept exactly
    these four fields — no service-specific params belong here.
    """

    text: str
    ref_audio_path: str | None = None
    seed: int = 0
    streaming: bool = False


class StandardSamplingRequest(BaseTTSRequest):
    """Adds the CFG / exaggeration sampling parameters shared by Standard
    and Multilingual TTS.

    Turbo uses a different decoder (MeanFlow, 1-step) and does not expose
    these controls, so it inherits from ``BaseTTSRequest`` directly instead.

    Default ``rep_penalty`` is 1.2 — Multilingual overrides it to 2.0 which
    empirically reduces artefacts on non-English outputs.
    """

    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    rep_penalty: float = 1.2
    min_p: float = 0.05
    top_p: float = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Concrete TTS request value-objects
# ──────────────────────────────────────────────────────────────────────────────


class TTSRequest(StandardSamplingRequest):
    """Input value-object for the Standard TTS service.

    Defaults mirror the '🎯 Default' preset and the Gradio slider defaults
    so callers that don't customise anything get sensible output.  Every
    field is inherited from ``StandardSamplingRequest`` / ``BaseTTSRequest``.
    """


class TurboTTSRequest(BaseTTSRequest):
    """Input value-object for the Turbo TTS service.

    Turbo does NOT support exaggeration or cfg_weight — those params are
    absent.  It uses a 1-step MeanFlow decoder which accepts ``top_k`` and
    ``norm_loudness`` instead.  ``norm_loudness`` normalises the reference
    audio to -27 LUFS before conditioning.

    Note the differing defaults vs Standard: ``top_p=0.95``, ``min_p=0.0``.
    """

    temperature: float = 0.8
    top_k: int = 1000
    top_p: float = 0.95
    rep_penalty: float = 1.2
    min_p: float = 0.0
    norm_loudness: bool = True


class MultilingualTTSRequest(StandardSamplingRequest):
    """Input value-object for the Multilingual TTS service (23 languages).

    Inherits the full CFG / exaggeration parameter set from
    ``StandardSamplingRequest`` and adds a ``language`` code.

    ``rep_penalty`` is overridden to 2.0 — a higher value than the Standard
    default (1.2) which empirically reduces repetition artefacts on
    non-English languages.
    """

    language: LanguageCode = "en"
    rep_penalty: float = 2.0  # override StandardSamplingRequest default (1.2)


# ──────────────────────────────────────────────────────────────────────────────
# Voice Conversion request value-object
# ──────────────────────────────────────────────────────────────────────────────


class VoiceConversionRequest(DomainModel):
    """Input value-object for the Voice Conversion service.

    VC converts source audio to sound like the target voice — no text prompt
    is involved.  Both audio paths are required.
    """

    source_audio_path: str | None
    target_voice_path: str | None


# ──────────────────────────────────────────────────────────────────────────────
# Output value-objects
# ──────────────────────────────────────────────────────────────────────────────


class AudioResult(DomainModel):
    """Holds the raw audio output produced by any TTS or VC service.

    ``samples`` must be a 1-D float32 NumPy array of shape ``(N,)``.
    ``sample_rate`` is in Hz (typically 24 000 for Chatterbox models).

    ``arbitrary_types_allowed=True`` is required because pydantic cannot
    natively validate ``numpy.ndarray``.  ``frozen=True`` is re-declared
    explicitly because a child ``model_config`` replaces the parent's
    config entirely in pydantic v2 — the ``frozen`` key would otherwise
    be lost.
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


class ModelStatus(DomainModel):
    """Snapshot of a single model's identity and load state.

    Populated by IModelManagerService.get_all_status() and consumed by the
    Gradio model-manager tab renderer.
    """

    key: ModelKey  # "tts" | "turbo" | "multilingual" | "vc"
    display_name: str  # e.g. "Standard TTS"
    class_name: str  # e.g. "ChatterboxTTS"
    description: str
    params: str  # human-readable, e.g. "500M" or "—"
    size_gb: float
    in_memory: bool
    on_disk: bool


class MemoryStats(DomainModel):
    """System + device memory snapshot returned by IMemoryMonitor.get_stats().

    Device fields (device_driver_gb, device_max_gb) are None on CPU-only
    systems where no GPU / MPS memory tracking is available.
    """

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


class WatermarkResult(DomainModel):
    """Result of a watermark-detection pass on a generated audio clip.

    verdict is one of:
        "detected"      — watermark found above threshold
        "not_detected"  — no watermark signal found
        "inconclusive"  — score is in the ambiguous range
        "unavailable"   — detection library not installed / failed to initialise
    """

    score: float
    verdict: WatermarkVerdict
    message: str
    available: bool


# ──────────────────────────────────────────────────────────────────────────────
# Application configuration
# ──────────────────────────────────────────────────────────────────────────────


class AppConfig(DomainModel):
    """Immutable runtime configuration resolved once at bootstrap.

    Passed into services and adapters via dependency injection so that no
    module reads environment variables or detects hardware on its own.
    """

    device: DeviceType
    watermark_available: bool
