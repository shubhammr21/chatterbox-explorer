"""
tests/unit/adapters/test_memory_monitor.py
==========================================
TDD unit tests for PsutilMemoryMonitor.

All external I/O is mocked — no real psutil or torch calls occur.

Mock strategy
─────────────
* psutil.virtual_memory   → patch() directly (psutil is installed)
* psutil.Process          → patch() directly; return_value chains .memory_info().rss
* torch                   → patch.dict(sys.modules, {'torch': mock_torch})
  torch is imported lazily inside get_stats(); replacing sys.modules['torch']
  gives us a disposable mock for MPS / CUDA memory queries.
* time.monotonic          → patch() directly for TTL cache tests
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from adapters.secondary.memory import PsutilMemoryMonitor
from domain.models import MemoryStats

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def vm_mock():
    """A fake psutil.virtual_memory() return value representing a 16 GB system."""
    m = MagicMock()
    m.total = 16 * 1024**3  # 16 GB
    m.used = 8 * 1024**3  # 8 GB
    m.available = 8 * 1024**3
    m.percent = 50.0
    return m


@pytest.fixture
def proc_mock():
    """A fake psutil.Process() instance with 2 GB RSS."""
    m = MagicMock()
    m.memory_info.return_value.rss = 2 * 1024**3  # 2 GB
    return m


@pytest.fixture
def cpu_torch_mock():
    """A torch mock with both MPS and CUDA reported as unavailable."""
    t = MagicMock()
    t.backends.mps.is_available.return_value = False
    t.cuda.is_available.return_value = False
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Helper context-manager factory
# ──────────────────────────────────────────────────────────────────────────────


def _cpu_patches(vm, proc, torch_mock):
    """Return a combined patch stack for a CPU get_stats() call."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with (
            patch("psutil.virtual_memory", return_value=vm),
            patch("psutil.Process", return_value=proc),
            patch.dict(sys.modules, {"torch": torch_mock}),
        ):
            yield

    return _ctx()


# ──────────────────────────────────────────────────────────────────────────────
# Return type
# ──────────────────────────────────────────────────────────────────────────────


class TestGetStatsReturnType:
    def test_returns_memory_stats_instance(self, vm_mock, proc_mock, cpu_torch_mock):
        monitor = PsutilMemoryMonitor(device="cpu")
        with _cpu_patches(vm_mock, proc_mock, cpu_torch_mock):
            stats = monitor.get_stats()
        assert isinstance(stats, MemoryStats)


# ──────────────────────────────────────────────────────────────────────────────
# CPU device — device fields are None
# ──────────────────────────────────────────────────────────────────────────────


class TestGetStatsCpuDevice:
    """On a CPU-only system no device-memory fields should be populated."""

    def test_device_driver_gb_is_none(self, vm_mock, proc_mock, cpu_torch_mock):
        monitor = PsutilMemoryMonitor(device="cpu")
        with _cpu_patches(vm_mock, proc_mock, cpu_torch_mock):
            stats = monitor.get_stats()
        assert stats.device_driver_gb is None

    def test_device_max_gb_is_none(self, vm_mock, proc_mock, cpu_torch_mock):
        monitor = PsutilMemoryMonitor(device="cpu")
        with _cpu_patches(vm_mock, proc_mock, cpu_torch_mock):
            stats = monitor.get_stats()
        assert stats.device_max_gb is None


# ──────────────────────────────────────────────────────────────────────────────
# System RAM fields
# ──────────────────────────────────────────────────────────────────────────────


class TestGetStatsSystemFields:
    """All five system-level fields must be computed correctly from the mocks."""

    def test_sys_total_gb(self, vm_mock, proc_mock, cpu_torch_mock):
        monitor = PsutilMemoryMonitor(device="cpu")
        with _cpu_patches(vm_mock, proc_mock, cpu_torch_mock):
            stats = monitor.get_stats()
        # 16 * 1024**3 / 1024**3 = 16.0, rounded to 1 decimal place
        assert stats.sys_total_gb == 16.0

    def test_sys_used_gb(self, vm_mock, proc_mock, cpu_torch_mock):
        monitor = PsutilMemoryMonitor(device="cpu")
        with _cpu_patches(vm_mock, proc_mock, cpu_torch_mock):
            stats = monitor.get_stats()
        assert stats.sys_used_gb == 8.0

    def test_sys_avail_gb(self, vm_mock, proc_mock, cpu_torch_mock):
        monitor = PsutilMemoryMonitor(device="cpu")
        with _cpu_patches(vm_mock, proc_mock, cpu_torch_mock):
            stats = monitor.get_stats()
        assert stats.sys_avail_gb == 8.0

    def test_sys_percent(self, vm_mock, proc_mock, cpu_torch_mock):
        monitor = PsutilMemoryMonitor(device="cpu")
        with _cpu_patches(vm_mock, proc_mock, cpu_torch_mock):
            stats = monitor.get_stats()
        assert stats.sys_percent == 50.0

    def test_proc_rss_gb(self, vm_mock, proc_mock, cpu_torch_mock):
        monitor = PsutilMemoryMonitor(device="cpu")
        with _cpu_patches(vm_mock, proc_mock, cpu_torch_mock):
            stats = monitor.get_stats()
        # 2 * 1024**3 / 1024**3 = 2.0, rounded to 2 decimal places
        assert stats.proc_rss_gb == 2.0


# ──────────────────────────────────────────────────────────────────────────────
# MPS device — device fields are populated
# ──────────────────────────────────────────────────────────────────────────────


class TestGetStatsMpsDevice:
    """When device='mps' and MPS is available, device_driver_gb and
    device_max_gb must be derived from torch.mps APIs."""

    @pytest.fixture
    def mps_torch_mock(self):
        t = MagicMock()
        t.backends.mps.is_available.return_value = True
        # driver_allocated_memory → 2 GB, recommended_max_memory → 8 GB
        t.mps.driver_allocated_memory.return_value = 2 * 1024**3
        t.mps.recommended_max_memory.return_value = 8 * 1024**3
        return t

    def test_device_driver_gb_populated(self, vm_mock, proc_mock, mps_torch_mock):
        monitor = PsutilMemoryMonitor(device="mps")
        with (
            patch("psutil.virtual_memory", return_value=vm_mock),
            patch("psutil.Process", return_value=proc_mock),
            patch.dict(sys.modules, {"torch": mps_torch_mock}),
        ):
            stats = monitor.get_stats()
        # 2 GB driver allocation
        assert stats.device_driver_gb == 2.0

    def test_device_max_gb_populated(self, vm_mock, proc_mock, mps_torch_mock):
        monitor = PsutilMemoryMonitor(device="mps")
        with (
            patch("psutil.virtual_memory", return_value=vm_mock),
            patch("psutil.Process", return_value=proc_mock),
            patch.dict(sys.modules, {"torch": mps_torch_mock}),
        ):
            stats = monitor.get_stats()
        # 8 GB recommended max
        assert stats.device_max_gb == 8.0

    def test_device_name_set_to_apple_silicon_mps(self, vm_mock, proc_mock, mps_torch_mock):
        monitor = PsutilMemoryMonitor(device="mps")
        with (
            patch("psutil.virtual_memory", return_value=vm_mock),
            patch("psutil.Process", return_value=proc_mock),
            patch.dict(sys.modules, {"torch": mps_torch_mock}),
        ):
            stats = monitor.get_stats()
        assert stats.device_name == "Apple Silicon MPS"

    def test_mps_unavailable_leaves_device_fields_none(self, vm_mock, proc_mock):
        """If MPS hardware is absent, device_driver_gb must remain None."""
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        monitor = PsutilMemoryMonitor(device="mps")
        with (
            patch("psutil.virtual_memory", return_value=vm_mock),
            patch("psutil.Process", return_value=proc_mock),
            patch.dict(sys.modules, {"torch": mock_torch}),
        ):
            stats = monitor.get_stats()
        assert stats.device_driver_gb is None
        assert stats.device_max_gb is None


# ──────────────────────────────────────────────────────────────────────────────
# TTL cache behaviour
# ──────────────────────────────────────────────────────────────────────────────


class TestTTLCache:
    """get_stats() caches the snapshot for _TTL seconds (1.5 s by default).

    Within the window the *identical object* is returned; after expiry a fresh
    snapshot is built (psutil is called again).
    """

    def test_second_call_within_ttl_returns_same_object(self, vm_mock, proc_mock, cpu_torch_mock):
        """time.monotonic returns 0.0 then 1.0 — delta 1.0 < 1.5 → cache hit."""
        monitor = PsutilMemoryMonitor(device="cpu")
        with (
            patch("psutil.virtual_memory", return_value=vm_mock),
            patch("psutil.Process", return_value=proc_mock),
            patch.dict(sys.modules, {"torch": cpu_torch_mock}),
            patch("time.monotonic", side_effect=[0.0, 1.0]),
        ):
            stats1 = monitor.get_stats()
            stats2 = monitor.get_stats()
        assert stats1 is stats2

    def test_second_call_within_ttl_does_not_reread_psutil(
        self, vm_mock, proc_mock, cpu_torch_mock
    ):
        """psutil.virtual_memory must be called only once during the TTL window."""
        monitor = PsutilMemoryMonitor(device="cpu")
        with (
            patch("psutil.virtual_memory", return_value=vm_mock) as mock_vm,
            patch("psutil.Process", return_value=proc_mock),
            patch.dict(sys.modules, {"torch": cpu_torch_mock}),
            patch("time.monotonic", side_effect=[0.0, 1.0]),
        ):
            monitor.get_stats()
            monitor.get_stats()
        mock_vm.assert_called_once()

    def test_second_call_after_ttl_returns_different_object(
        self, vm_mock, proc_mock, cpu_torch_mock
    ):
        """time.monotonic returns 0.0 then 2.0 — delta 2.0 >= 1.5 → cache miss."""
        monitor = PsutilMemoryMonitor(device="cpu")
        with (
            patch("psutil.virtual_memory", return_value=vm_mock),
            patch("psutil.Process", return_value=proc_mock),
            patch.dict(sys.modules, {"torch": cpu_torch_mock}),
            patch("time.monotonic", side_effect=[0.0, 2.0]),
        ):
            stats1 = monitor.get_stats()
            stats2 = monitor.get_stats()
        assert stats1 is not stats2

    def test_second_call_after_ttl_rereads_psutil(self, vm_mock, proc_mock, cpu_torch_mock):
        """After the TTL expires psutil.virtual_memory must be called again."""
        monitor = PsutilMemoryMonitor(device="cpu")
        with (
            patch("psutil.virtual_memory", return_value=vm_mock) as mock_vm,
            patch("psutil.Process", return_value=proc_mock),
            patch.dict(sys.modules, {"torch": cpu_torch_mock}),
            patch("time.monotonic", side_effect=[0.0, 2.0]),
        ):
            monitor.get_stats()
            monitor.get_stats()
        assert mock_vm.call_count == 2

    def test_fresh_monitor_has_no_cache(self):
        """A brand-new monitor must have _cache=None (documented invariant)."""
        monitor = PsutilMemoryMonitor(device="cpu")
        assert monitor._cache is None

    def test_cache_is_populated_after_first_call(self, vm_mock, proc_mock, cpu_torch_mock):
        monitor = PsutilMemoryMonitor(device="cpu")
        with _cpu_patches(vm_mock, proc_mock, cpu_torch_mock):
            monitor.get_stats()
        assert monitor._cache is not None


# ──────────────────────────────────────────────────────────────────────────────
# CUDA device — device fields are populated (lines 89-92 branch)
# ──────────────────────────────────────────────────────────────────────────────


class TestGetStatsCudaDevice:
    """When device='cuda' and CUDA is available, device_name, device_driver_gb,
    and device_max_gb must be populated from torch.cuda APIs."""

    @pytest.fixture
    def cuda_torch_mock(self):
        """A torch mock with CUDA reported as available and a fake V100 GPU."""
        t = MagicMock()
        t.cuda.is_available.return_value = True

        props = MagicMock()
        props.name = "Tesla V100"
        props.total_memory = 16 * 1024**3  # 16 GB

        t.cuda.get_device_properties.return_value = props
        t.cuda.memory_allocated.return_value = 4 * 1024**3  # 4 GB currently used
        return t

    def test_device_name_set_to_gpu_name(self, vm_mock, proc_mock, cuda_torch_mock):
        """device_name must match props.name returned by get_device_properties(0)."""
        monitor = PsutilMemoryMonitor(device="cuda")
        with (
            patch("psutil.virtual_memory", return_value=vm_mock),
            patch("psutil.Process", return_value=proc_mock),
            patch.dict(sys.modules, {"torch": cuda_torch_mock}),
        ):
            stats = monitor.get_stats()
        assert stats.device_name == "Tesla V100"

    def test_device_driver_gb_equals_memory_allocated(self, vm_mock, proc_mock, cuda_torch_mock):
        """device_driver_gb must reflect torch.cuda.memory_allocated() in GB."""
        monitor = PsutilMemoryMonitor(device="cuda")
        with (
            patch("psutil.virtual_memory", return_value=vm_mock),
            patch("psutil.Process", return_value=proc_mock),
            patch.dict(sys.modules, {"torch": cuda_torch_mock}),
        ):
            stats = monitor.get_stats()
        # 4 * 1024**3 bytes / 1024**3 = 4.0 GB
        assert stats.device_driver_gb == 4.0

    def test_device_max_gb_equals_total_memory(self, vm_mock, proc_mock, cuda_torch_mock):
        """device_max_gb must reflect props.total_memory in GB."""
        monitor = PsutilMemoryMonitor(device="cuda")
        with (
            patch("psutil.virtual_memory", return_value=vm_mock),
            patch("psutil.Process", return_value=proc_mock),
            patch.dict(sys.modules, {"torch": cuda_torch_mock}),
        ):
            stats = monitor.get_stats()
        # 16 * 1024**3 bytes / 1024**3 = 16.0 GB
        assert stats.device_max_gb == 16.0

    def test_get_device_properties_called_with_device_zero(
        self, vm_mock, proc_mock, cuda_torch_mock
    ):
        """get_device_properties must be queried for device index 0."""
        monitor = PsutilMemoryMonitor(device="cuda")
        with (
            patch("psutil.virtual_memory", return_value=vm_mock),
            patch("psutil.Process", return_value=proc_mock),
            patch.dict(sys.modules, {"torch": cuda_torch_mock}),
        ):
            monitor.get_stats()
        cuda_torch_mock.cuda.get_device_properties.assert_called_once_with(0)
