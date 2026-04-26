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
- stdlib  (io, typing, abc)
- domain models (runtime imports for Generic bases; AudioResult under TYPE_CHECKING)
- numpy (for audio encoding only, inside the helper function)
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Self, cast

from pydantic import BaseModel, Field

from domain.models import (
    MemoryStats,
    ModelStatus,
    MultilingualTTSRequest,
    TTSRequest,
    TurboTTSRequest,
    WatermarkResult,
)

if TYPE_CHECKING:
    import numpy as np

    from domain.models import AudioResult
    from domain.types import LanguageCode

# ──────────────────────────────────────────────────────────────────────────────
# Abstract base classes  (PEP 695 type-parameter syntax — Python 3.12+)
# ──────────────────────────────────────────────────────────────────────────────


class InboundSchema[DomainT](BaseModel, abc.ABC):
    """Base for schemas that translate HTTP input → domain objects.

    Subclasses must implement ``to_domain()`` returning the parametrised
    domain type.  Direct instantiation is prevented by ``abc.ABC``.

    Example::

        class TTSRequestSchema(InboundSchema[TTSRequest]):
            text: str

            def to_domain(self) -> TTSRequest:
                return TTSRequest(text=self.text)
    """

    @abc.abstractmethod
    def to_domain(self) -> DomainT:
        """Translate validated HTTP payload → domain object."""
        ...


class OutboundSchema[DomainT](BaseModel, abc.ABC):
    """Base for schemas that translate domain objects → HTTP response payloads.

    Subclasses must implement ``from_domain()`` as a classmethod.
    Direct instantiation is prevented by ``abc.ABC``.

    Example::

        class ModelStatusResponse(OutboundSchema[ModelStatus]):
            key: str

            @classmethod
            def from_domain(cls, domain: ModelStatus) -> Self:
                return cls(key=domain.key)
    """

    @classmethod
    @abc.abstractmethod
    def from_domain(cls, domain: DomainT) -> Self:
        """Translate domain object → serializable response schema."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Request schemas
# ──────────────────────────────────────────────────────────────────────────────


class TTSRequestSchema(InboundSchema[TTSRequest]):
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


class TurboRequestSchema(InboundSchema[TurboTTSRequest]):
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


class MultilingualRequestSchema(InboundSchema[MultilingualTTSRequest]):
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


class ModelStatusResponse(OutboundSchema[ModelStatus]):
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
    def from_domain(cls, domain: ModelStatus) -> Self:
        """Build from a :class:`~domain.models.ModelStatus` domain object."""
        return cls(
            key=domain.key,
            display_name=domain.display_name,
            class_name=domain.class_name,
            description=domain.description,
            params=domain.params,
            size_gb=domain.size_gb,
            in_memory=domain.in_memory,
            on_disk=domain.on_disk,
        )


class MemoryStatsResponse(OutboundSchema[MemoryStats]):
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
    def from_domain(cls, domain: MemoryStats) -> Self:
        """Build from a :class:`~domain.models.MemoryStats` domain object."""
        return cls(
            sys_total_gb=domain.sys_total_gb,
            sys_used_gb=domain.sys_used_gb,
            sys_avail_gb=domain.sys_avail_gb,
            sys_percent=domain.sys_percent,
            proc_rss_gb=domain.proc_rss_gb,
            device_name=domain.device_name,
            device_driver_gb=domain.device_driver_gb,
            device_max_gb=domain.device_max_gb,
        )


class WatermarkResponse(OutboundSchema[WatermarkResult]):
    """JSON response for POST /api/v1/watermark/detect."""

    score: float
    verdict: str
    message: str
    available: bool

    @classmethod
    def from_domain(cls, domain: WatermarkResult) -> Self:
        """Build from a :class:`~domain.models.WatermarkResult` domain object."""
        return cls(
            score=domain.score,
            verdict=domain.verdict,
            message=domain.message,
            available=domain.available,
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


def audio_delta_to_pcm_bytes(delta_samples: np.ndarray) -> bytes:
    """Convert a float32 delta ndarray to raw int16 PCM bytes for streaming.

    Does NOT produce a WAV header — raw PCM only.
    Each HTTP chunk from a streaming endpoint is the output of this function.

    Peak normalisation is applied when samples exceed [-1.0, 1.0] to prevent
    clipping.  In-range samples are clipped (not scaled) to guard against
    floating-point rounding artefacts at the boundaries.

    Args:
        delta_samples: float32 ndarray slice representing NEW samples only
            (the delta since the previous yield from generate_stream()).

    Returns:
        Raw bytes — 2 bytes per sample (int16, little-endian, mono).
        Returns b"" for empty input.
    """
    import numpy as _np

    if len(delta_samples) == 0:
        return b""

    arr = delta_samples.astype(_np.float32)
    peak = float(_np.abs(arr).max())
    if peak > 1.0:
        arr = arr / peak

    int16_arr = _np.clip(arr * 32767.0, -32768, 32767).astype(_np.int16)
    return int16_arr.tobytes()


# ──────────────────────────────────────────────────────────────────────────────
# Consistent error envelope
# ──────────────────────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    """Single error item in an error response."""

    message: str
    path: list[str | int] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Consistent error envelope for all REST error responses.

    Every error response (422, 503, 500, etc.) must use this shape so
    clients only need to parse one structure.
    """

    errors: list[ErrorDetail]
