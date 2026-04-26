"""
src/chatterbox_explorer/services/voice_conversion.py
======================================================
VoiceConversionService — pure domain service, zero framework dependencies.

Allowed imports: stdlib + numpy + ports/domain models.
Forbidden: torch, gradio, chatterbox — any of these appearing here is an
architecture violation.
"""

from __future__ import annotations

import numpy as np

from domain.models import AudioResult, VoiceConversionRequest
from ports.output import IAudioPreprocessor, IModelRepository


class VoiceConversionService:
    """Convert source audio to sound like a target voice.

    Wraps the ChatterboxVC model behind the IModelRepository port so that
    the service itself never touches torch, file-system paths, or device
    management directly.

    Args:
        model_repo:   Repository that owns model lifecycle (load / unload).
        preprocessor: Normalises raw audio paths before they reach the model.
    """

    def __init__(
        self,
        model_repo: IModelRepository,
        preprocessor: IAudioPreprocessor,
    ) -> None:
        self._repo = model_repo
        self._prep = preprocessor

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def convert(self, request: VoiceConversionRequest) -> AudioResult:
        """Convert *request.source_audio_path* to the timbre of
        *request.target_voice_path*.

        Args:
            request: A :class:`VoiceConversionRequest` value-object holding
                     both audio paths.

        Returns:
            An :class:`AudioResult` with raw float32 samples and the model's
            native sample rate.

        Raises:
            ValueError: if either audio path is ``None`` or empty.
            RuntimeError: if the model or infrastructure layer fails.
        """
        self._validate(request)

        model = self._repo.get_model("vc")
        target_path = self._prep.preprocess(request.target_voice_path)

        wav = model.generate(
            audio=request.source_audio_path,
            target_voice_path=target_path,
        )

        samples = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        return AudioResult(sample_rate=model.sr, samples=samples)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(request: VoiceConversionRequest) -> None:
        """Raise ValueError for any missing/empty path."""
        if not request.source_audio_path:
            raise ValueError("source_audio_path is required for voice conversion.")
        if not request.target_voice_path:
            raise ValueError("target_voice_path is required for voice conversion.")
