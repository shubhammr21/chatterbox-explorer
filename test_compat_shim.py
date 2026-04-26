#!/usr/bin/env python3
"""
test_compat_shim.py — Standalone verification suite for compat.py
==================================================================
Exercises every code-path in the sdp_kernel compatibility shim without
requiring pytest or any external test framework.

Run:
    uv run python test_compat_shim.py
    uv run python test_compat_shim.py -v   # verbose (shows DEBUG logs)

Exit code is 0 when every test passes, 1 on any failure.
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
import sys
import traceback
import unittest.mock as mock

# ──────────────────────────────────────────────────────────────────────────────
# Minimal test harness  (no pytest dependency)
# ──────────────────────────────────────────────────────────────────────────────

_RESULTS: list[tuple[str, bool, str]] = []  # (name, passed, detail)
_VERBOSE = False


def _pass(name: str, detail: str = "") -> None:
    _RESULTS.append((name, True, detail))
    print(f"  \033[32mPASS\033[0m  {name}" + (f"  ({detail})" if detail else ""))


def _fail(name: str, detail: str) -> None:
    _RESULTS.append((name, False, detail))
    print(f"  \033[31mFAIL\033[0m  {name}")
    print(f"         {detail}")


def test(name: str) -> Callable:
    """Decorator that wraps a function as a named test case."""

    def decorator(fn: Callable) -> Callable:
        def wrapper() -> None:
            try:
                fn()
                # If fn() doesn't call _pass / _fail itself, treat return as pass
                if not _RESULTS or _RESULTS[-1][0] != name:
                    _pass(name)
            except AssertionError as exc:
                _fail(name, str(exc) or "AssertionError (no message)")
            except Exception as exc:
                _fail(name, f"{type(exc).__name__}: {exc}")
                if _VERBOSE:
                    traceback.print_exc()

        wrapper.__name__ = name
        return wrapper

    return decorator


def _section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 60 - len(title))}")


# ──────────────────────────────────────────────────────────────────────────────
# Environment probe  (printed once, before any tests)
# ──────────────────────────────────────────────────────────────────────────────


def _probe_environment() -> None:
    import torch

    print(f"\n  torch version : {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    mps_backend = getattr(torch.backends, "mps", None)
    mps_ok = mps_backend is not None and torch.backends.mps.is_available()
    print(f"  MPS available : {mps_ok}")
    sdpa_present = hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel")
    print(f"  sdpa_kernel   : {sdpa_present}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Helper to reset patch state between tests
# ──────────────────────────────────────────────────────────────────────────────


def _ensure_patched() -> None:
    """Make sure torch.backends.cuda.sdp_kernel points at our shim."""
    import compat

    compat._patch_torch_backends()


def _ensure_unpatched() -> None:
    """Restore the original (or remove shim) for isolation."""
    import compat

    compat._unpatch_torch_backends()


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Import & auto-patch
# ──────────────────────────────────────────────────────────────────────────────

_section("Section 1 — Import & auto-patch on module load")


@test("import compat does not raise")
def t_import() -> None:
    import compat  # noqa: F401

    _pass("import compat does not raise")


@test("auto-patch: torch.backends.cuda.sdp_kernel is shim after import")
def t_auto_patch() -> None:
    import compat
    import torch.backends.cuda as _cuda

    assert _cuda.sdp_kernel is compat.sdp_kernel, f"Expected shim, got {_cuda.sdp_kernel!r}"
    _pass("auto-patch: torch.backends.cuda.sdp_kernel is shim after import")


@test("auto-patch: original stored as _sdp_kernel_orig")
def t_orig_stored() -> None:
    import torch.backends.cuda as _cuda

    assert hasattr(_cuda, "_sdp_kernel_orig"), (
        "torch.backends.cuda._sdp_kernel_orig not found after patch"
    )
    _pass("auto-patch: original stored as _sdp_kernel_orig")


@test("auto-patch: _sdp_kernel_orig is callable")
def t_orig_callable() -> None:
    import torch.backends.cuda as _cuda

    orig = getattr(_cuda, "_sdp_kernel_orig", None)
    if orig is None:
        # Attribute was absent in this build — acceptable
        _pass("auto-patch: _sdp_kernel_orig is callable  [attr absent in this build — skipped]")
        return
    assert callable(orig), f"_sdp_kernel_orig is not callable: {orig!r}"
    _pass("auto-patch: _sdp_kernel_orig is callable")


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — Context manager basic usage
# ──────────────────────────────────────────────────────────────────────────────

_section("Section 2 — Context manager basic usage")


@test("with sdp_kernel(): body executes (all flags=True)")
def t_cm_all_true() -> None:
    _ensure_patched()
    import compat

    entered = []
    with compat.sdp_kernel(
        enable_flash=True,
        enable_math=True,
        enable_mem_efficient=True,
        enable_cudnn=True,
    ):
        entered.append(True)
    assert entered == [True], "Context manager body was never entered"
    _pass("with sdp_kernel(): body executes (all flags=True)")


@test("with sdp_kernel(): body executes (math-only)")
def t_cm_math_only() -> None:
    _ensure_patched()
    import compat

    entered = []
    with compat.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
        enable_cudnn=False,
    ):
        entered.append(True)
    assert entered == [True], "Context manager body was never entered"
    _pass("with sdp_kernel(): body executes (math-only)")


@test("with sdp_kernel(): body executes with default args")
def t_cm_defaults() -> None:
    _ensure_patched()
    import compat

    entered = []
    with compat.sdp_kernel():
        entered.append(True)
    assert entered == [True]
    _pass("with sdp_kernel(): body executes with default args")


@test("with sdp_kernel(): exceptions inside body propagate cleanly")
def t_cm_exception_propagates() -> None:
    _ensure_patched()
    import compat

    class _Sentinel(Exception):
        pass

    caught = []
    try:
        with compat.sdp_kernel(enable_math=True):
            raise _Sentinel("boom")
    except _Sentinel:
        caught.append(True)

    assert caught == [True], "Exception inside with-block was swallowed"
    _pass("with sdp_kernel(): exceptions inside body propagate cleanly")


@test("monkey-patched path: torch.backends.cuda.sdp_kernel() works")
def t_monkey_patch_path() -> None:
    _ensure_patched()
    import torch

    entered = []
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True):
        entered.append(True)
    assert entered == [True]
    _pass("monkey-patched path: torch.backends.cuda.sdp_kernel() works")


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — _build_backends() parameter mapping
# ──────────────────────────────────────────────────────────────────────────────

_section("Section 3 — _build_backends() parameter mapping")

import torch as _torch  # noqa: E402

_SDPA_AVAILABLE = hasattr(_torch.nn, "attention") and hasattr(_torch.nn.attention, "sdpa_kernel")
_MPS_ONLY: bool = False
if _SDPA_AVAILABLE:
    _mps_b = getattr(_torch.backends, "mps", None)
    _mps_ok = _mps_b is not None and _torch.backends.mps.is_available()
    _MPS_ONLY = _mps_ok and not _torch.cuda.is_available()


@test("_build_backends: MATH only when enable_math=True, rest False")
def t_bb_math_only() -> None:
    if not _SDPA_AVAILABLE:
        _pass("_build_backends: MATH only  [sdpa_kernel unavailable — skipped]")
        return
    import compat
    from torch.nn.attention import SDPBackend

    result = compat._build_backends(False, True, False, False)
    assert SDPBackend.MATH in result, f"MATH missing from {result}"
    assert SDPBackend.FLASH_ATTENTION not in result
    assert SDPBackend.EFFICIENT_ATTENTION not in result
    assert SDPBackend.CUDNN_ATTENTION not in result
    _pass("_build_backends: MATH only when enable_math=True, rest False")


@test("_build_backends: FLASH_ATTENTION present when enable_flash=True (CUDA or MPS)")
def t_bb_flash() -> None:
    if not _SDPA_AVAILABLE:
        _pass("_build_backends: FLASH_ATTENTION  [sdpa_kernel unavailable — skipped]")
        return
    import compat
    from torch.nn.attention import SDPBackend

    if _MPS_ONLY:
        # On MPS-only, flash is dropped and MATH is returned regardless
        result = compat._build_backends(True, True, False, False)
        assert SDPBackend.MATH in result
        assert SDPBackend.FLASH_ATTENTION not in result
        _pass(
            "_build_backends: FLASH_ATTENTION present when enable_flash=True  [MPS-only: correctly dropped]"
        )
    else:
        result = compat._build_backends(True, False, False, False)
        assert SDPBackend.FLASH_ATTENTION in result, f"FLASH_ATTENTION missing from {result}"
        _pass("_build_backends: FLASH_ATTENTION present when enable_flash=True (CUDA or MPS)")


@test("_build_backends: EFFICIENT_ATTENTION present when enable_mem_efficient=True")
def t_bb_efficient() -> None:
    if not _SDPA_AVAILABLE:
        _pass("_build_backends: EFFICIENT_ATTENTION  [sdpa_kernel unavailable — skipped]")
        return
    import compat
    from torch.nn.attention import SDPBackend

    if _MPS_ONLY:
        result = compat._build_backends(False, True, True, False)
        assert SDPBackend.MATH in result
        assert SDPBackend.EFFICIENT_ATTENTION not in result
        _pass("_build_backends: EFFICIENT_ATTENTION present  [MPS-only: correctly dropped]")
    else:
        result = compat._build_backends(False, False, True, False)
        assert SDPBackend.EFFICIENT_ATTENTION in result, (
            f"EFFICIENT_ATTENTION missing from {result}"
        )
        _pass("_build_backends: EFFICIENT_ATTENTION present when enable_mem_efficient=True")


@test("_build_backends: CUDNN_ATTENTION present when enable_cudnn=True")
def t_bb_cudnn() -> None:
    if not _SDPA_AVAILABLE:
        _pass("_build_backends: CUDNN_ATTENTION  [sdpa_kernel unavailable — skipped]")
        return
    import compat
    from torch.nn.attention import SDPBackend

    if _MPS_ONLY:
        result = compat._build_backends(False, True, False, True)
        assert SDPBackend.MATH in result
        assert SDPBackend.CUDNN_ATTENTION not in result
        _pass("_build_backends: CUDNN_ATTENTION present  [MPS-only: correctly dropped]")
    else:
        result = compat._build_backends(False, False, False, True)
        assert SDPBackend.CUDNN_ATTENTION in result, f"CUDNN_ATTENTION missing from {result}"
        _pass("_build_backends: CUDNN_ATTENTION present when enable_cudnn=True")


@test("_build_backends: all four backends present when all flags=True (CUDA/CPU)")
def t_bb_all_cuda() -> None:
    if not _SDPA_AVAILABLE or _MPS_ONLY:
        _pass("_build_backends: all four backends  [skipped — MPS-only or sdpa unavailable]")
        return
    import compat
    from torch.nn.attention import SDPBackend

    result = compat._build_backends(True, True, True, True)
    for backend in (
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.MATH,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
    ):
        assert backend in result, f"{backend} missing from {result}"
    _pass("_build_backends: all four backends present when all flags=True (CUDA/CPU)")


@test("_build_backends: ordering — FLASH before MATH (CUDA/CPU)")
def t_bb_order() -> None:
    if not _SDPA_AVAILABLE or _MPS_ONLY:
        _pass("_build_backends: ordering  [skipped — MPS-only or sdpa unavailable]")
        return
    import compat
    from torch.nn.attention import SDPBackend

    result = compat._build_backends(True, True, False, False)
    flash_idx = result.index(SDPBackend.FLASH_ATTENTION)
    math_idx = result.index(SDPBackend.MATH)
    assert flash_idx < math_idx, (
        f"FLASH_ATTENTION ({flash_idx}) should come before MATH ({math_idx})"
    )
    _pass("_build_backends: ordering — FLASH before MATH (CUDA/CPU)")


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — Error / edge cases
# ──────────────────────────────────────────────────────────────────────────────

_section("Section 4 — Error and edge cases")


@test("_build_backends: ValueError when all four flags=False (CUDA/CPU)")
def t_bb_all_false_raises() -> None:
    if not _SDPA_AVAILABLE:
        _pass("_build_backends: ValueError all-False  [sdpa_kernel unavailable — skipped]")
        return
    import compat

    if _MPS_ONLY:
        # On MPS-only the shim forces MATH — ValueError cannot be triggered
        # through normal flag usage; verify no error is raised instead.
        result = compat._build_backends(False, False, False, False)
        from torch.nn.attention import SDPBackend

        assert SDPBackend.MATH in result
        _pass("_build_backends: ValueError all-False  [MPS-only: MATH forced, no ValueError — OK]")
        return

    raised = False
    try:
        compat._build_backends(False, False, False, False)
    except ValueError:
        raised = True
    assert raised, "_build_backends should raise ValueError when all flags are False"
    _pass("_build_backends: ValueError when all four flags=False (CUDA/CPU)")


@test("sdp_kernel context manager: ValueError propagates from _build_backends")
def t_cm_all_false_raises() -> None:
    if not _SDPA_AVAILABLE or _MPS_ONLY:
        _pass("sdp_kernel ValueError propagation  [skipped]")
        return
    import compat

    raised = False
    try:
        with compat.sdp_kernel(
            enable_flash=False,
            enable_math=False,
            enable_mem_efficient=False,
            enable_cudnn=False,
        ):
            pass
    except ValueError:
        raised = True
    assert raised, "Expected ValueError when all backends disabled"
    _pass("sdp_kernel context manager: ValueError propagates from _build_backends")


@test("_build_backends: MPS-only forces [MATH] regardless of all flags=True")
def t_bb_mps_override() -> None:
    if not _SDPA_AVAILABLE:
        _pass("_build_backends: MPS override  [sdpa_kernel unavailable — skipped]")
        return
    import compat
    from torch.nn.attention import SDPBackend

    # Simulate MPS-only environment by patching _is_mps_only to return True
    with mock.patch.object(compat, "_is_mps_only", return_value=True):
        result = compat._build_backends(True, True, True, True)
    assert result == [SDPBackend.MATH], f"Expected [MATH] on MPS-only, got {result}"
    _pass("_build_backends: MPS-only forces [MATH] regardless of all flags=True")


@test("_build_backends: MPS-only issues UserWarning when enable_math=False")
def t_bb_mps_math_false_warns() -> None:
    if not _SDPA_AVAILABLE:
        _pass("_build_backends: MPS UserWarning  [sdpa_kernel unavailable — skipped]")
        return
    import warnings as _warnings

    import compat
    from torch.nn.attention import SDPBackend

    with (
        mock.patch.object(compat, "_is_mps_only", return_value=True),
        _warnings.catch_warnings(record=True) as w,
    ):
        _warnings.simplefilter("always")
        result = compat._build_backends(True, False, True, True)

    assert result == [SDPBackend.MATH]
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    assert len(user_warnings) >= 1, "Expected at least one UserWarning for enable_math=False on MPS"
    assert (
        "math" in str(user_warnings[0].message).lower()
        or "mps" in str(user_warnings[0].message).lower()
    )
    _pass("_build_backends: MPS-only issues UserWarning when enable_math=False")


# ──────────────────────────────────────────────────────────────────────────────
# Section 5 — Patch / unpatch lifecycle
# ──────────────────────────────────────────────────────────────────────────────

_section("Section 5 — Patch / unpatch lifecycle")


@test("_patch_torch_backends: idempotent (double-patch safe)")
def t_double_patch() -> None:
    import compat
    import torch.backends.cuda as _cuda

    _ensure_patched()
    result = compat._patch_torch_backends()  # second call
    # Should detect already-patched state and return True
    assert result is True
    assert _cuda.sdp_kernel is compat.sdp_kernel
    _pass("_patch_torch_backends: idempotent (double-patch safe)")


@test("_unpatch_torch_backends: restores original symbol")
def t_unpatch_restores() -> None:
    import compat
    import torch.backends.cuda as _cuda

    _ensure_patched()
    orig = getattr(_cuda, "_sdp_kernel_orig", None)
    compat._unpatch_torch_backends()
    after = getattr(_cuda, "sdp_kernel", None)

    if orig is None:
        # Attribute was absent — nothing to restore, just check shim is gone
        assert after is not compat.sdp_kernel or after is None
    else:
        assert after is orig, f"Original not restored; got {after!r}"
    _pass("_unpatch_torch_backends: restores original symbol")


@test("_unpatch_torch_backends: removes _sdp_kernel_orig sentinel")
def t_unpatch_removes_sentinel() -> None:
    import compat
    import torch.backends.cuda as _cuda

    _ensure_patched()
    compat._unpatch_torch_backends()
    assert not hasattr(_cuda, "_sdp_kernel_orig"), (
        "_sdp_kernel_orig should be deleted after unpatch"
    )
    _pass("_unpatch_torch_backends: removes _sdp_kernel_orig sentinel")


@test("_unpatch_torch_backends: returns False when no patch was applied")
def t_unpatch_when_clean() -> None:
    import compat

    _ensure_unpatched()  # make sure it's clean first
    result = compat._unpatch_torch_backends()
    assert result is False, f"Expected False, got {result!r}"
    _pass("_unpatch_torch_backends: returns False when no patch was applied")


@test("patch → unpatch → re-patch cycle is safe")
def t_repatch_cycle() -> None:
    import compat
    import torch.backends.cuda as _cuda

    compat._unpatch_torch_backends()
    compat._patch_torch_backends()
    compat._unpatch_torch_backends()
    compat._patch_torch_backends()  # final state: patched

    assert _cuda.sdp_kernel is compat.sdp_kernel
    _pass("patch → unpatch → re-patch cycle is safe")


# ──────────────────────────────────────────────────────────────────────────────
# Section 6 — No-op behaviour on old PyTorch (simulated)
# ──────────────────────────────────────────────────────────────────────────────

_section("Section 6 — Graceful no-op when sdpa_kernel unavailable")


@test("sdp_kernel no-op: body still executes when _SDPA_KERNEL_AVAILABLE=False")
def t_noop_executes() -> None:
    import compat

    entered = []
    with (
        mock.patch.object(compat, "_SDPA_KERNEL_AVAILABLE", False),
        compat.sdp_kernel(enable_flash=True, enable_math=True),
    ):
        entered.append(True)
    assert entered == [True], "Body should still execute in no-op path"
    _pass("sdp_kernel no-op: body still executes when _SDPA_KERNEL_AVAILABLE=False")


@test("sdp_kernel no-op: exceptions still propagate when _SDPA_KERNEL_AVAILABLE=False")
def t_noop_exception_propagates() -> None:
    import compat

    class _Boom(Exception):
        pass

    caught = []
    with mock.patch.object(compat, "_SDPA_KERNEL_AVAILABLE", False):
        try:
            with compat.sdp_kernel():
                raise _Boom
        except _Boom:
            caught.append(True)
    assert caught == [True]
    _pass("sdp_kernel no-op: exceptions still propagate when _SDPA_KERNEL_AVAILABLE=False")


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 — _is_mps_only() helper
# ──────────────────────────────────────────────────────────────────────────────

_section("Section 7 — _is_mps_only() device detection helper")


@test("_is_mps_only: returns bool")
def t_is_mps_only_type() -> None:
    import compat

    result = compat._is_mps_only()
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    _pass("_is_mps_only: returns bool")


@test("_is_mps_only: False when CUDA is available (simulated)")
def t_is_mps_only_false_on_cuda() -> None:
    import compat

    with mock.patch("torch.cuda.is_available", return_value=True):
        result = compat._is_mps_only()
    assert result is False, "_is_mps_only should be False when CUDA is available"
    _pass("_is_mps_only: False when CUDA is available (simulated)")


@test("_is_mps_only: True when MPS=True and CUDA=False (simulated)")
def t_is_mps_only_true_simulated() -> None:
    import compat

    # We need to mock both cuda.is_available() and mps.is_available()
    with mock.patch("torch.cuda.is_available", return_value=False):
        import torch

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None:
            _pass("_is_mps_only: True when MPS=True  [mps backend absent in build — skipped]")
            return
        with mock.patch.object(mps_backend, "is_available", return_value=True):
            result = compat._is_mps_only()
    assert result is True, "_is_mps_only should be True when MPS=True and CUDA=False"
    _pass("_is_mps_only: True when MPS=True and CUDA=False (simulated)")


# ──────────────────────────────────────────────────────────────────────────────
# Section 8 — _active_device_label()
# ──────────────────────────────────────────────────────────────────────────────

_section("Section 8 — _active_device_label() helper")


@test("_active_device_label: returns a non-empty string")
def t_device_label_str() -> None:
    import compat

    label = compat._active_device_label()
    assert isinstance(label, str) and label, f"Expected non-empty str, got {label!r}"
    _pass("_active_device_label: returns a non-empty string")


@test("_active_device_label: returns 'cpu' when CUDA and MPS unavailable (simulated)")
def t_device_label_cpu() -> None:
    import compat
    import torch

    mps_backend = getattr(torch.backends, "mps", None)
    ctx = mock.patch("torch.cuda.is_available", return_value=False)

    if mps_backend is not None:
        with ctx, mock.patch.object(mps_backend, "is_available", return_value=False):
            label = compat._active_device_label()
    else:
        with ctx:
            label = compat._active_device_label()

    assert label == "cpu", f"Expected 'cpu', got {label!r}"
    _pass("_active_device_label: returns 'cpu' when CUDA and MPS unavailable (simulated)")


# ──────────────────────────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────────────────────────


def _run_all() -> int:
    """Execute every test function and return exit code."""
    test_fns = [
        t_import,
        t_auto_patch,
        t_orig_stored,
        t_orig_callable,
        t_cm_all_true,
        t_cm_math_only,
        t_cm_defaults,
        t_cm_exception_propagates,
        t_monkey_patch_path,
        t_bb_math_only,
        t_bb_flash,
        t_bb_efficient,
        t_bb_cudnn,
        t_bb_all_cuda,
        t_bb_order,
        t_bb_all_false_raises,
        t_cm_all_false_raises,
        t_bb_mps_override,
        t_bb_mps_math_false_warns,
        t_double_patch,
        t_unpatch_restores,
        t_unpatch_removes_sentinel,
        t_unpatch_when_clean,
        t_repatch_cycle,
        t_noop_executes,
        t_noop_exception_propagates,
        t_is_mps_only_type,
        t_is_mps_only_false_on_cuda,
        t_is_mps_only_true_simulated,
        t_device_label_str,
        t_device_label_cpu,
    ]

    for fn in test_fns:
        fn()

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = sum(1 for _, ok, _ in _RESULTS if not ok)
    total = len(_RESULTS)

    print(f"\n{'─' * 64}")
    colour = "\033[32m" if failed == 0 else "\033[31m"
    print(f"{colour}{passed}/{total} passed  ·  {failed} failed\033[0m")
    return 0 if failed == 0 else 1


def _parse_args() -> None:
    global _VERBOSE
    parser = argparse.ArgumentParser(
        description="Verify compat.py shim behaviour",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show DEBUG log output and full tracebacks on failure",
    )
    args = parser.parse_args()
    _VERBOSE = args.verbose
    if _VERBOSE:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s — %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)


if __name__ == "__main__":
    _parse_args()
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║         compat.py — sdp_kernel shim test suite              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    _probe_environment()
    sys.exit(_run_all())
