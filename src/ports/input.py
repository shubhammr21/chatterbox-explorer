"""
src/ports/input.py
===================
Input-side (primary) port interfaces — the ABCs that the application core
exposes to its driving adapters (Gradio UI, CLI, tests).

Forbidden imports: torch, gradio, chatterbox, psutil — any of these here is
an architecture violation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from domain.types import ModelKey
from domain.models import (
    AudioResult,
    MemoryStats,
    ModelStatus,
    MultilingualTTSRequest,
    TTSRequest,
    TurboTTSRequest,
    VoiceConversionRequest,
    WatermarkResult,
)


class ITTSService(ABC):
    """Driving port for the Standard TTS capability."""

    @abstractmethod
    def generate(self, request: TTSRequest) -> AudioResult:
        """Generate a complete audio clip from *request* in one shot.

        Raises:
            ValueError: if request.text is empty or whitespace-only.
            RuntimeError: if the underlying model/infrastructure fails.
        """
        ...

    @abstractmethod
    def generate_stream(self, request: TTSRequest) -> Iterator[AudioResult]:
        """Generate audio sentence-by-sentence, yielding cumulative results.

        Each yielded :class:`AudioResult` contains *all* audio produced so far
        (i.e. results are cumulative, not differential).

        Raises:
            ValueError: if request.text is empty or whitespace-only.
            RuntimeError: if the underlying model/infrastructure fails.
        """
        ...


class ITurboTTSService(ABC):
    """Driving port for the Turbo TTS capability."""

    @abstractmethod
    def generate(self, request: TurboTTSRequest) -> AudioResult:
        """Generate a complete audio clip from *request* in one shot.

        Raises:
            ValueError: if request.text is empty, or if the reference audio
                        is shorter than 5 seconds (model assertion).
            RuntimeError: if the underlying model/infrastructure fails.
        """
        ...

    @abstractmethod
    def generate_stream(self, request: TurboTTSRequest) -> Iterator[AudioResult]:
        """Generate audio sentence-by-sentence, yielding cumulative results.

        Raises:
            ValueError: if request.text is empty, or reference audio too short.
            RuntimeError: if the underlying model/infrastructure fails.
        """
        ...


class IMultilingualTTSService(ABC):
    """Driving port for the Multilingual TTS capability (23 languages)."""

    @abstractmethod
    def generate(self, request: MultilingualTTSRequest) -> AudioResult:
        """Generate a complete audio clip from *request* in one shot.

        Raises:
            ValueError: if request.text is empty or whitespace-only.
            RuntimeError: if the underlying model/infrastructure fails.
        """
        ...

    @abstractmethod
    def generate_stream(self, request: MultilingualTTSRequest) -> Iterator[AudioResult]:
        """Generate audio sentence-by-sentence, yielding cumulative results.

        Raises:
            ValueError: if request.text is empty or whitespace-only.
            RuntimeError: if the underlying model/infrastructure fails.
        """
        ...


class IVoiceConversionService(ABC):
    """Driving port for the Voice Conversion capability."""

    @abstractmethod
    def convert(self, request: VoiceConversionRequest) -> AudioResult:
        """Convert *request.source_audio_path* to sound like *request.target_voice_path*.

        Raises:
            ValueError: if either audio path is empty/None.
            RuntimeError: if the underlying model/infrastructure fails.
        """
        ...


class IModelManagerService(ABC):
    """Driving port for model lifecycle management (load / unload / download)."""

    @abstractmethod
    def load(self, key: ModelKey) -> str:
        """Load the model identified by *key* into memory.

        Returns:
            A human-readable status message (already loaded, success, etc.).

        Raises:
            RuntimeError: if the model fails to load.
        """
        ...

    @abstractmethod
    def unload(self, key: ModelKey) -> str:
        """Unload the model identified by *key* from memory.

        Returns:
            A human-readable status message (not loaded, success, etc.).
        """
        ...

    @abstractmethod
    def download(self, key: ModelKey) -> Iterator[str]:
        """Download the model identified by *key*, yielding progress lines."""
        ...

    @abstractmethod
    def get_all_status(self) -> list[ModelStatus]:
        """Return a :class:`ModelStatus` snapshot for every known model."""
        ...

    @abstractmethod
    def get_memory_stats(self) -> MemoryStats:
        """Return current system + device memory statistics."""
        ...


class IWatermarkService(ABC):
    """Driving port for PerTh watermark detection."""

    @abstractmethod
    def detect(self, audio_path: str) -> WatermarkResult:
        """Detect whether *audio_path* carries a Chatterbox watermark.

        Returns:
            A :class:`WatermarkResult` whose ``verdict`` is one of
            ``"detected"``, ``"not_detected"``, ``"inconclusive"``, or
            ``"unavailable"``.
        """
        ...
