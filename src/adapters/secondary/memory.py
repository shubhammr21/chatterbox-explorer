"""System and device memory monitoring."""
import logging

from domain.types import DeviceType
import time
from typing import ClassVar

from domain.models import MemoryStats
from ports.output import IMemoryMonitor

log = logging.getLogger(__name__)


class PsutilMemoryMonitor(IMemoryMonitor):
    """Reports system RAM and device (MPS/CUDA) memory usage.

    Uses a short TTL cache (1.5 s) to avoid hammering psutil on every
    Gradio render tick (e.g. on each keypress in a live-update component).
    Within the TTL window the same MemoryStats snapshot is returned; after
    it expires the next call performs a real stat read.

    MPS notes (Apple Silicon unified memory):
        - driver_allocated_memory() includes the allocator pool — matches
          what Activity Monitor reports as "GPU" memory.
        - current_allocated_memory() excludes the pool — useful for leak
          detection only, not for showing users "how full is the GPU".
        - recommended_max_memory() ≈ 75 % of physical RAM — use as the
          practical capacity ceiling, not total physical RAM.
        - Never compute total − free on macOS; always use total − available.
          The "free" figure is almost always < 200 MB due to kernel caching.
    """

    _TTL: ClassVar[float] = 1.5  # seconds between real stat reads

    def __init__(self, device: DeviceType) -> None:
        """
        Args:
            device: Compute device string — ``"cpu"``, ``"cuda"``, or ``"mps"``.
                    Determines which device-memory fields are populated.
        """
        self._device = device
        # Cache entry: (timestamp_monotonic, MemoryStats) or None when empty.
        self._cache: tuple[float, MemoryStats] | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # IMemoryMonitor
    # ──────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> MemoryStats:
        """Return a system + device memory snapshot, respecting the TTL cache.

        On cache hit (last read < 1.5 s ago) the previous snapshot is returned
        without any system calls.  On cache miss a fresh snapshot is built from
        psutil and torch device APIs.

        Returns:
            A :class:`~chatterbox_explorer.domain.models.MemoryStats` instance
            populated with the values available on the current platform.
        """
        import psutil
        import torch

        now = time.monotonic()
        if self._cache is not None and now - self._cache[0] < self._TTL:
            return self._cache[1]

        # ── System RAM ────────────────────────────────────────────────────────
        vm = psutil.virtual_memory()
        rss = psutil.Process().memory_info().rss

        stats = MemoryStats(
            sys_total_gb=round(vm.total / 1024 ** 3, 1),
            sys_used_gb=round(vm.used / 1024 ** 3, 1),
            sys_avail_gb=round(vm.available / 1024 ** 3, 1),
            sys_percent=vm.percent,          # (total − available) / total × 100
            proc_rss_gb=round(rss / 1024 ** 3, 2),
        )

        # ── Device memory ─────────────────────────────────────────────────────
        if self._device == "mps" and torch.backends.mps.is_available():
            # driver_allocated includes the allocator pool — matches Activity
            # Monitor and is the right figure to show end users.
            stats.device_name = "Apple Silicon MPS"
            stats.device_driver_gb = round(
                torch.mps.driver_allocated_memory() / 1024 ** 3, 2
            )
            # recommended_max_memory ≈ 75 % of physical RAM — use as ceiling.
            stats.device_max_gb = round(
                torch.mps.recommended_max_memory() / 1024 ** 3, 1
            )

        elif self._device == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            stats.device_name = props.name
            stats.device_driver_gb = round(
                torch.cuda.memory_allocated() / 1024 ** 3, 2
            )
            stats.device_max_gb = round(
                props.total_memory / 1024 ** 3, 1
            )

        # ── Update cache ──────────────────────────────────────────────────────
        self._cache = (now, stats)
        return stats
