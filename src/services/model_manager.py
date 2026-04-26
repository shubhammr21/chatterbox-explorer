"""
src/chatterbox_explorer/services/model_manager.py
===================================================
ModelManagerService — pure domain service for model lifecycle management.

Architecture rules enforced here:
    - ZERO imports from torch, gradio, chatterbox, psutil, huggingface_hub
    - All infrastructure access is mediated through IModelRepository / IMemoryMonitor
    - Returns human-readable strings (not framework-specific errors/warnings)
    - Raises RuntimeError for infrastructure failures
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.models import ModelStatus
from ports.input import IModelManagerService

if TYPE_CHECKING:
    from collections.abc import Iterator

    from domain.models import MemoryStats
    from domain.types import ModelKey
    from ports.output import IMemoryMonitor, IModelRepository


class ModelManagerService(IModelManagerService):
    """Orchestrates loading, unloading, downloading and status reporting
    for all Chatterbox models.

    This service is intentionally free of framework specifics — it knows
    nothing about Gradio, PyTorch devices, or HuggingFace Hub internals.
    All of those concerns live in the concrete IModelRepository adapter.

    Args:
        model_repo:     Gateway to model weights storage and memory management.
        memory_monitor: Provides system / device memory snapshots.
    """

    def __init__(
        self,
        model_repo: IModelRepository,
        memory_monitor: IMemoryMonitor,
    ) -> None:
        self._repo = model_repo
        self._monitor = memory_monitor

    # ──────────────────────────────────────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────────────────────────────────────

    def load(self, key: ModelKey) -> str:
        """Load the model identified by *key* into memory.

        If the model is already in memory this is a no-op and an informational
        message is returned without re-loading.

        Args:
            key: Model identifier (e.g. ``"tts"``, ``"turbo"``).

        Returns:
            A human-readable status string describing the outcome.

        Raises:
            RuntimeError: if the repository raises while loading the model.
        """
        display = self._repo.get_display_name(key)

        if self._repo.is_loaded(key):
            return f"{display} is already loaded in memory."

        try:
            self._repo.get_model(key)
        except RuntimeError:
            # Re-raise repo RuntimeErrors as-is so callers can handle them.
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to load model '{key}' ({display}): {exc}") from exc

        return f"{display} loaded successfully."

    # ──────────────────────────────────────────────────────────────────────────
    # Unload
    # ──────────────────────────────────────────────────────────────────────────

    def unload(self, key: ModelKey) -> str:
        """Unload the model identified by *key* from memory.

        If the model is not currently in memory this is a no-op and an
        informational message is returned without error.

        Args:
            key: Model identifier (e.g. ``"tts"``, ``"turbo"``).

        Returns:
            A human-readable status string describing the outcome.
        """
        display = self._repo.get_display_name(key)

        if not self._repo.is_loaded(key):
            return f"{display} is not currently loaded in memory."

        self._repo.unload(key)
        return f"{display} unloaded from memory."

    # ──────────────────────────────────────────────────────────────────────────
    # Download
    # ──────────────────────────────────────────────────────────────────────────

    def download(self, key: ModelKey) -> Iterator[str]:
        """Download model weights for *key*, streaming progress lines.

        Each yielded string is a human-readable progress message suitable for
        display in a log widget (e.g. "Downloading shard 1/4 … 25%").

        Args:
            key: Model identifier (e.g. ``"tts"``, ``"turbo"``).

        Yields:
            Progress / status strings from the underlying repository.
        """
        yield from self._repo.download(key)

    # ──────────────────────────────────────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────────────────────────────────────

    def get_all_status(self) -> list[ModelStatus]:
        """Return a :class:`~chatterbox_explorer.domain.models.ModelStatus`
        snapshot for every model known to the repository.

        The status is assembled from:
            - ``repo.get_all_keys()``      — ordered list of model keys
            - ``repo.get_display_name()``  — human-readable label
            - ``repo.get_model_metadata()``— size, params, description, class
            - ``repo.is_loaded()``         — live memory flag
            - ``repo.is_cached_on_disk()`` — disk-cache flag

        Returns:
            List of :class:`ModelStatus` instances in the same order as
            ``repo.get_all_keys()``.
        """
        statuses: list[ModelStatus] = []

        for key in self._repo.get_all_keys():
            meta = self._repo.get_model_metadata(key)
            statuses.append(
                ModelStatus(
                    key=key,
                    display_name=self._repo.get_display_name(key),
                    class_name=meta.get("class_name", ""),
                    description=meta.get("description", ""),
                    params=meta.get("params", "—"),
                    size_gb=float(meta.get("size_gb", 0.0)),
                    in_memory=self._repo.is_loaded(key),
                    on_disk=self._repo.is_cached_on_disk(key),
                )
            )

        return statuses

    # ──────────────────────────────────────────────────────────────────────────
    # Memory
    # ──────────────────────────────────────────────────────────────────────────

    def get_memory_stats(self) -> MemoryStats:
        """Return a fresh system + device memory snapshot.

        Delegates entirely to :class:`~chatterbox_explorer.ports.output.IMemoryMonitor`;
        values are *not* cached between calls.

        Returns:
            A :class:`~chatterbox_explorer.domain.models.MemoryStats` instance.
        """
        return self._monitor.get_stats()
