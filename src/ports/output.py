"""
src/ports/output.py
====================
Output-port ABCs: interfaces the domain/services use to reach infrastructure.

Hexagonal rule: service code depends ONLY on these abstractions.
Adapters (infra layer) provide the concrete implementations.

Allowed imports: abc, typing, domain models and types.
Forbidden: torch, gradio, chatterbox, psutil, huggingface_hub.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from domain.models import MemoryStats
from domain.types import ModelKey, ModelMetadata


class IModelRepository(ABC):
    """Gateway to model storage — loading, unloading, downloading, metadata.

    Implementations are responsible for caching models in memory, writing to
    disk, and streaming download progress.  The service layer never touches
    files, devices, or framework-specific objects directly.
    """

    @abstractmethod
    def get_model(self, key: ModelKey) -> Any:
        """Return the live model object for *key*, loading it if necessary.

        Raises:
            RuntimeError: if the model cannot be loaded (missing weights,
                device OOM, etc.).
        """
        ...

    @abstractmethod
    def is_loaded(self, key: ModelKey) -> bool:
        """Return True if the model for *key* is currently in memory."""
        ...

    @abstractmethod
    def is_cached_on_disk(self, key: ModelKey) -> bool:
        """Return True if the model weights are present in the local cache."""
        ...

    @abstractmethod
    def unload(self, key: ModelKey) -> None:
        """Remove the model for *key* from memory (does NOT delete disk cache).

        No-op if the model is not currently loaded.
        """
        ...

    @abstractmethod
    def download(self, key: ModelKey) -> Iterator[str]:
        """Download weights for *key* and yield human-readable progress lines.

        Each yielded string is suitable for streaming to a UI log widget.
        """
        ...

    @abstractmethod
    def get_all_keys(self) -> list[ModelKey]:
        """Return every model key known to this repository."""
        ...

    @abstractmethod
    def get_display_name(self, key: ModelKey) -> str:
        """Return a human-readable label for *key* (e.g. 'Standard TTS')."""
        ...

    @abstractmethod
    def get_model_metadata(self, key: ModelKey) -> ModelMetadata:
        """Return the typed metadata record for *key*.

        Keys and value types are defined by :class:`~domain.types.ModelMetadata`.
        Every field marked ``Required`` in that TypedDict is guaranteed to be
        present; ``NotRequired`` fields depend on the ``dl_mode`` value.
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
