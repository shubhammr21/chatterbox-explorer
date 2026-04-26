"""
src/adapters/inbound/rest/schemas.py
======================================
Pydantic request and response schemas for the FastAPI REST adapter.

Architecture contract
---------------------
- Schemas live in the inbound adapter layer — NOT in the domain.
- Each request schema owns a ``to_domain()`` method that converts the
  validated Pydantic model into the corresponding domain dataclass.
  This keeps the translation responsibility inside the adapter boundary.
- Response schemas serialise domain value-objects (ModelStatus, MemoryStats,
  WatermarkResult) into JSON-serialisable Pydantic models.
- The ``audio_result_to_wav_bytes()`` helper encodes an AudioResult into a
  raw 16-bit WAV byte string for HTTP audio/wav responses.

Forbidden imports (architecture rule)
--------------------------------------
- torch, torchaudio, gradio, chatterbox  — no framework deps in schemas
- No direct imports of secondary/outbound adapter classes

Allowed imports
---------------
- pydantic (BaseModel, Field)
- stdlib  (io, typing)
- domain models (under TYPE_CHECKING to keep runtime deps minimal)
- numpy (for audio encoding only, inside the helper function)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from domain.models import (
        AudioResult,
        MemoryStats,
        ModelStatus,
        MultilingualTTSRequest,
        TTSRequest,
        TurboTTSRequest,
        WatermarkResult,
    )
    from domain.types import LanguageCode


# ──────────────────────────────────────────────────────────────────────────────
# Request schemas
# ──────────────────────────────────────────────────────────────────────────────


class TTSRequestSchema(BaseModel):
    """Request body for POST /api/v1/tts/generate.

    Mirrors the fields of :class:`~domain.models.TTSRequest` with the
    exception of ``streaming`` (not exposed in v1 — one-shot only) and
    ``ref_audio_path`` (path strings from remote clients are meaningless
    server-side; voice cloning via file upload is deferred to v2).
    """

    text: str = Field(..., min_length=1, description="Text to synthesise.")
    exaggeration: float = Field(0.5, ge=0.0, le=1.0, description="Emotion exaggeration level.")
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0, description="Classifier-free guidance weight.")
    temperature: float = Field(0.8, ge=0.05, le=5.0, description="Sampling temperature.")
    rep_penalty: float = Field(1.2, ge=1.0, le=2.0, description="Repetition penalty.")
    min_p: float = Field(0.05, ge=0.0, le=1.0, description="Min-p sampling threshold.")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p (nucleus) sampling cutoff.")
    seed: int = Field(0, ge=0, description="RNG seed; 0 means non-deterministic.")

    def to_domain(self) -> TTSRequest:
        """Convert to the :class:`~domain.models.TTSRequest` domain dataclass."""
        from domain.models import TTSRequest

        return TTSRequest(
            text=self.text,
            ref_audio_path=None,
            exaggeration=self.exaggeration,
            cfg_weight=self.cfg_weight,
            temperature=self.temperature,
            rep_penalty=self.rep_penalty,
            min_p=self.min_p,
            top_p=self.top_p,
            seed=self.seed,
            streaming=False,
        )


class TurboRequestSchema(BaseModel):
    """Request body for POST /api/v1/turbo/generate.

    Mirrors :class:`~domain.models.TurboTTSRequest`.  Turbo does not
    support exaggeration or cfg_weight; it adds top_k and norm_loudness.
    """

    text: str = Field(..., min_length=1, description="Text to synthesise.")
    temperature: float = Field(0.8, ge=0.05, le=5.0, description="Sampling temperature.")
    top_k: int = Field(1000, ge=1, description="Top-k sampling candidates.")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p (nucleus) sampling cutoff.")
    rep_penalty: float = Field(1.2, ge=1.0, le=2.0, description="Repetition penalty.")
    min_p: float = Field(0.0, ge=0.0, le=1.0, description="Min-p sampling threshold.")
    norm_loudness: bool = Field(
        True, description="Normalise reference audio to -27 LUFS before conditioning."
    )
    seed: int = Field(0, ge=0, description="RNG seed; 0 means non-deterministic.")

    def to_domain(self) -> TurboTTSRequest:
        """Convert to the :class:`~domain.models.TurboTTSRequest` domain dataclass."""
        from domain.models import TurboTTSRequest

        return TurboTTSRequest(
            text=self.text,
            ref_audio_path=None,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            rep_penalty=self.rep_penalty,
            min_p=self.min_p,
            norm_loudness=self.norm_loudness,
            seed=self.seed,
            streaming=False,
        )


class MultilingualRequestSchema(BaseModel):
    """Request body for POST /api/v1/multilingual/generate.

    Mirrors :class:`~domain.models.MultilingualTTSRequest`.  Adds a
    ``language`` field (ISO 639-1 code or ``"<code> - <name>"`` label).
    Default ``rep_penalty`` is 2.0 (higher than Standard) to suppress
    artefacts on non-English languages.
    """

    text: str = Field(..., min_length=1, description="Text to synthesise.")
    language: str = Field("en", description="Target language — ISO 639-1 code, e.g. 'fr'.")
    exaggeration: float = Field(0.5, ge=0.0, le=1.0, description="Emotion exaggeration level.")
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0, description="Classifier-free guidance weight.")
    temperature: float = Field(0.8, ge=0.05, le=5.0, description="Sampling temperature.")
    rep_penalty: float = Field(2.0, ge=1.0, le=2.0, description="Repetition penalty.")
    min_p: float = Field(0.05, ge=0.0, le=1.0, description="Min-p sampling threshold.")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p (nucleus) sampling cutoff.")
    seed: int = Field(0, ge=0, description="RNG seed; 0 means non-deterministic.")

    def to_domain(self) -> MultilingualTTSRequest:
        """Convert to the :class:`~domain.models.MultilingualTTSRequest` domain dataclass."""
        from domain.models import MultilingualTTSRequest

        return MultilingualTTSRequest(
            text=self.text,
            language=cast("LanguageCode", self.language),
            ref_audio_path=None,
            exaggeration=self.exaggeration,
            cfg_weight=self.cfg_weight,
            temperature=self.temperature,
            rep_penalty=self.rep_penalty,
            min_p=self.min_p,
            top_p=self.top_p,
            seed=self.seed,
            streaming=False,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Response schemas
# ──────────────────────────────────────────────────────────────────────────────


class ModelStatusResponse(BaseModel):
    """JSON response item for GET /api/v1/models/status."""

    key: str
    display_name: str
    class_name: str
    description: str
    params: str
    size_gb: float
    in_memory: bool
    on_disk: bool

    @classmethod
    def from_domain(cls, status: ModelStatus) -> ModelStatusResponse:
        """Build from a :class:`~domain.models.ModelStatus` domain object."""
        return cls(
            key=status.key,
            display_name=status.display_name,
            class_name=status.class_name,
            description=status.description,
            params=status.params,
            size_gb=status.size_gb,
            in_memory=status.in_memory,
            on_disk=status.on_disk,
        )


class MemoryStatsResponse(BaseModel):
    """JSON response for GET /api/v1/models/memory."""

    sys_total_gb: float
    sys_used_gb: float
    sys_avail_gb: float
    sys_percent: float
    proc_rss_gb: float
    device_name: str
    device_driver_gb: float | None
    device_max_gb: float | None

    @classmethod
    def from_domain(cls, stats: MemoryStats) -> MemoryStatsResponse:
        """Build from a :class:`~domain.models.MemoryStats` domain object."""
        return cls(
            sys_total_gb=stats.sys_total_gb,
            sys_used_gb=stats.sys_used_gb,
            sys_avail_gb=stats.sys_avail_gb,
            sys_percent=stats.sys_percent,
            proc_rss_gb=stats.proc_rss_gb,
            device_name=stats.device_name,
            device_driver_gb=stats.device_driver_gb,
            device_max_gb=stats.device_max_gb,
        )


class WatermarkResponse(BaseModel):
    """JSON response for POST /api/v1/watermark/detect."""

    score: float
    verdict: str
    message: str
    available: bool

    @classmethod
    def from_domain(cls, result: WatermarkResult) -> WatermarkResponse:
        """Build from a :class:`~domain.models.WatermarkResult` domain object."""
        return cls(
            score=result.score,
            verdict=result.verdict,
            message=result.message,
            available=result.available,
        )


class MessageResponse(BaseModel):
    """Generic JSON response carrying a single human-readable message string.

    Used by model management endpoints (load, unload) that return a status
    description rather than a structured object.
    """

    message: str


class HealthResponse(BaseModel):
    """JSON response for GET /api/v1/health."""

    status: str
    device: str


# ──────────────────────────────────────────────────────────────────────────────
# Audio encoding helper
# ──────────────────────────────────────────────────────────────────────────────


def audio_result_to_wav_bytes(result: AudioResult) -> bytes:
    """Encode a float32 :class:`~domain.models.AudioResult` as 16-bit WAV bytes.

    Uses ``scipy.io.wavfile`` — lightweight WAV encoding with no torchaudio
    dependency in the REST adapter layer.  scipy is a transitive dependency of
    chatterbox-tts, so it is always available in the project venv.

    Normalisation:
        If the absolute peak of ``result.samples`` exceeds 1.0 the array is
        divided by the peak before int16 conversion, preventing clipping.
        Samples already within [-1.0, 1.0] are clipped (not scaled) to guard
        against floating-point rounding artefacts at the edges.

    Args:
        result: The audio output from any TTS or VC domain service.

    Returns:
        A bytes object containing a valid WAV file (RIFF/PCM 16-bit, mono).
    """
    import io

    import numpy as np
    from scipy.io import wavfile

    arr = result.samples.astype(np.float32)
    peak = float(np.abs(arr).max())
    if peak > 1.0:
        arr = arr / peak

    int16_arr = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, result.sample_rate, int16_arr)
    return buf.getvalue()
