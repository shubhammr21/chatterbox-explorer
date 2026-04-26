"""
src/domain/exceptions.py
==========================
Typed domain exception hierarchy for the Chatterbox TTS Explorer.

Design rules
------------
1. ZERO imports beyond the Python standard library.
   No fastapi, no starlette, no pydantic — not even as TYPE_CHECKING imports.
   This module must be safe to import at any point in the startup sequence,
   in any mode (UI or REST), without optional extras installed.

2. NO HTTP status codes on any exception class.
   HTTP mapping is the REST adapter's responsibility and lives exclusively in
   ``adapters/inbound/rest/exception_handlers.py``.  Domain exceptions carry
   domain meaning, not transport meaning.

3. Every public exception inherits from ``ChatterboxError``.
   Callers can do ``except ChatterboxError`` to catch all intentional domain
   failures without accidentally catching Python programming errors
   (``AttributeError``, ``TypeError``, ``NameError``, etc.).

4. Services catch ONLY exceptions they are specifically prepared to handle.
   ``except Exception`` or ``except BaseException`` is FORBIDDEN in all service
   code.  Unexpected exceptions propagate naturally and surface as 500-level
   errors at the adapter boundary.

Exception hierarchy
-------------------
ChatterboxError                       root — catch-all for all domain failures
├── TTSInputError                     TTS business-rule violations (expected)
│   ├── EmptyTextError                text is empty or whitespace-only
│   └── ReferenceTooShortError        reference audio below minimum duration
├── VoiceConversionInputError         VC business-rule violations (expected)
│   ├── MissingSourceAudioError       source_audio_path is absent or empty
│   └── MissingTargetVoiceError       target_voice_path is absent or empty
└── ModelError                        model lifecycle / inference failures
    ├── ModelNotLoadedError            model has not been initialised
    ├── ModelLoadError                 failure while loading / downloading
    └── InferenceError                 failure during model.generate() call
"""

from __future__ import annotations

__all__ = (
    "ChatterboxError",
    "EmptyTextError",
    "InferenceError",
    "MissingSourceAudioError",
    "MissingTargetVoiceError",
    "ModelError",
    "ModelLoadError",
    "ModelNotLoadedError",
    "ReferenceTooShortError",
    "TTSInputError",
    "VoiceConversionInputError",
)


# ──────────────────────────────────────────────────────────────────────────────
# Root
# ──────────────────────────────────────────────────────────────────────────────


class ChatterboxError(Exception):
    """Root exception for all intentional Chatterbox domain failures.

    Catching ``ChatterboxError`` gives callers a single, unambiguous handle
    on every expected failure mode without risking swallowing programming
    errors (which propagate as bare ``Exception`` subclasses that do NOT
    inherit from this class).
    """


# ──────────────────────────────────────────────────────────────────────────────
# TTS input errors  (business-rule violations from the caller)
# ──────────────────────────────────────────────────────────────────────────────


class TTSInputError(ChatterboxError):
    """Base class for TTS input that violates a business rule.

    These are *expected* failure modes — the caller provided data that the
    service cannot process.  They map to 4xx HTTP responses at the adapter
    layer, but the service itself knows nothing about HTTP.
    """


class EmptyTextError(TTSInputError):
    """The text provided for synthesis is empty or contains only whitespace.

    Args:
        text: The offending text value (stored for diagnostic use by callers).
    """

    def __init__(self, text: str = "") -> None:
        self.text: str = text
        super().__init__("Text input is empty or whitespace-only.")


class ReferenceTooShortError(TTSInputError):
    """The reference audio clip is shorter than the minimum required duration.

    The Turbo TTS model asserts that the reference audio must be longer than
    a minimum duration (default 5 seconds) before it will condition on it.
    This exception surfaces that constraint as a typed domain failure.

    Args:
        minimum_sec: The minimum required duration in seconds (default 5.0).
    """

    def __init__(self, *, minimum_sec: float = 5.0) -> None:
        self.minimum_sec: float = minimum_sec
        super().__init__(f"Reference audio must be longer than {minimum_sec:.0f} seconds.")


# ──────────────────────────────────────────────────────────────────────────────
# Voice Conversion input errors  (independent branch — not TTS errors)
# ──────────────────────────────────────────────────────────────────────────────


class VoiceConversionInputError(ChatterboxError):
    """Base class for Voice Conversion input that violates a business rule.

    Intentionally a sibling of ``TTSInputError``, not a subclass, so that
    callers can distinguish between TTS and VC failures at the type level.
    """


class MissingSourceAudioError(VoiceConversionInputError):
    """The source audio path required for voice conversion is absent or empty.

    Voice conversion needs a source audio file whose speech content is
    preserved.  Without it the operation cannot proceed.
    """

    def __init__(self) -> None:
        super().__init__("source_audio_path is required for voice conversion but was not provided.")


class MissingTargetVoiceError(VoiceConversionInputError):
    """The target voice path required for voice conversion is absent or empty.

    Voice conversion needs a target voice reference file whose timbre is
    applied to the source audio.  Without it the operation cannot proceed.
    """

    def __init__(self) -> None:
        super().__init__("target_voice_path is required for voice conversion but was not provided.")


# ──────────────────────────────────────────────────────────────────────────────
# Model errors  (infrastructure / lifecycle failures — unexpected)
# ──────────────────────────────────────────────────────────────────────────────


class ModelError(ChatterboxError):
    """Base class for model lifecycle and inference failures.

    Unlike input errors, model errors are *unexpected* — the service did not
    do anything wrong, but the underlying infrastructure (GPU, disk, model
    weights) failed.  They map to 5xx HTTP responses at the adapter layer.

    Intentionally a sibling of both ``TTSInputError`` and
    ``VoiceConversionInputError`` so that callers can distinguish between
    caller-side and server-side failures at the type level.
    """


class ModelNotLoadedError(ModelError):
    """The requested model has not been initialised into memory.

    Raised when a service method requires a model that is not yet loaded,
    rather than triggering a lazy load that would stall the caller.

    Args:
        model_key: The registry key of the model that is not loaded
            (e.g. ``"tts"``, ``"turbo"``).  Stored for diagnostic use.
    """

    def __init__(self, model_key: str = "") -> None:
        self.model_key: str = model_key
        msg = (
            f"Model '{model_key}' is not loaded into memory."
            if model_key
            else "The requested model is not loaded into memory."
        )
        super().__init__(msg)


class ModelLoadError(ModelError):
    """A failure occurred while loading or downloading a model.

    Raised when ``IModelRepository.get_model()`` or a download operation
    fails for an infrastructure reason (OOM, network error, corrupt weights,
    insufficient disk space, etc.).

    Args:
        model_key: The registry key of the model that failed to load.
        message:   A human-readable description of the underlying failure.
    """

    def __init__(self, *, model_key: str = "", message: str = "") -> None:
        self.model_key: str = model_key
        if model_key and message:
            detail = f"Failed to load model '{model_key}': {message}"
        elif model_key:
            detail = f"Failed to load model '{model_key}'."
        elif message:
            detail = f"Model load failed: {message}"
        else:
            detail = "Model load failed."
        super().__init__(detail)


class InferenceError(ModelError):
    """A failure occurred during model inference (model.generate() call).

    Raised when the model raises an unexpected error during synthesis or
    conversion that cannot be attributed to caller input.  Programming
    errors inside the model (``AttributeError``, ``RuntimeError`` from CUDA,
    etc.) should be wrapped in this exception rather than propagated as-is,
    so that the adapter layer can log and surface them uniformly.

    Args:
        message: A human-readable description of the inference failure.
    """

    def __init__(self, message: str = "") -> None:
        super().__init__(f"Inference failed: {message}" if message else "Inference failed.")
