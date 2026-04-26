"""
src/chatterbox_explorer/ports/output.py
=========================================
Output-port ABCs: interfaces the domain/services use to reach infrastructure.

Hexagonal rule: service code depends ONLY on these abstractions.
Adapters (infra layer) provide the concrete implementations.

Allowed imports: abc, typing, domain models.
Forbidden: torch, gradio, chatterbox, psutil, huggingface_hub.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from chatterbox_explorer.domain.models import MemoryStats


class IModelRepository(ABC):
    """Gateway to model storage — loading, unloading, downloading, metadata.

    Implementations are responsible for caching models in memory, writing to
    disk, and streaming download progress.  The service layer never touches
    files, devices, or framework-specific objects directly.
    """

    @abstractmethod
    def get_model(self, key: str) -> Any:
        """Return the live model object for *key*, loading it if necessary.

        Raises:
            RuntimeError: if the model cannot be loaded (missing weights,
                device OOM, etc.).
        """
        ...

    @abstractmethod
    def is_loaded(self, key: str) -> bool:
        """Return True if the model for *key* is currently in memory."""
        ...

    @abstractmethod
    def is_cached_on_disk(self, key: str) -> bool:
        """Return True if the model weights are present in the local cache."""
        ...

    @abstractmethod
    def unload(self, key: str) -> None:
        """Remove the model for *key* from memory (does NOT delete disk cache).

        No-op if the model is not currently loaded.
        """
        ...

    @abstractmethod
    def download(self, key: str) -> Iterator[str]:
        """Download weights for *key* and yield human-readable progress lines.

        Each yielded string is suitable for streaming to a UI log widget.
        """
        ...

    @abstractmethod
    def get_all_keys(self) -> list[str]:
        """Return every model key known to this repository."""
        ...

    @abstractmethod
    def get_display_name(self, key: str) -> str:
        """Return a human-readable label for *key* (e.g. 'Standard TTS')."""
        ...

    @abstractmethod
    def get_model_metadata(self, key: str) -> dict:
        """Return a metadata dict for *key*.

        Expected keys (all optional but recommended):
            size_gb    (float)  — weight file size on disk
            params     (str)    — parameter count label, e.g. '500M'
            description (str)  — one-line description
            class_name (str)   — Python class name of the model, e.g. 'ChatterboxTTS'
        """
        ...


class IAudioPreprocessor(ABC):
    """Normalises a raw audio path before it is fed to a model.

    Typical implementations resample to 16 kHz, trim silence, and write
    a temporary WAV file.  The contract: given *None*, return *None*.
    """

    @abstractmethod
    def preprocess(self, path: str | None) -> str | None:
        """Preprocess the audio at *path* and return a path to the result.

        Args:
            path: Absolute or relative path to the source audio file,
                  or *None* if no reference audio was provided.

        Returns:
            Path to the preprocessed file, or *None* if input was *None*.
        """
        ...


class IMemoryMonitor(ABC):
    """Snapshot of current system (and optional device) memory usage."""

    @abstractmethod
    def get_stats(self) -> MemoryStats:
        """Return a fresh :class:`~chatterbox_explorer.domain.models.MemoryStats`
        snapshot.  Implementations must not cache values between calls.
        """
        ...


class IWatermarkDetector(ABC):
    """Thin wrapper around the PerTh (or compatible) watermark detector.

    Separated from the service so that the unavailable-library case is
    handled at the adapter level; the service just calls ``is_available()``.
    """

    @abstractmethod
    def detect(self, audio_path: str) -> float:
        """Run watermark detection on the audio at *audio_path*.

        Returns:
            A confidence score in [0.0, 1.0] where 1.0 means the watermark
            is definitely present and 0.0 means it is definitely absent.

        Raises:
            RuntimeError: if detection fails for reasons other than the
                library being unavailable (use :meth:`is_available` for that).
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the underlying detection library is usable.

        When False, :meth:`detect` must *not* be called; the service layer
        will surface an 'unavailable' verdict instead.
        """
        ...
