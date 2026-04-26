"""
tests/unit/test_compat.py
==========================
Unit tests for src/compat.py — the two public migration shim functions.

Tests exercise observable behaviour through the public API only:
    apply_torch_sdp_kernel_migration()
    apply_diffusers_lora_migration()

Both functions are designed to be idempotent (safe to call multiple times),
so tests call them directly and verify the resulting state of the patched
symbols rather than trying to intercept internal implementation details.
"""

from __future__ import annotations

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# apply_torch_sdp_kernel_migration()
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyTorchSdpKernelMigration:
    """Observable behaviour of apply_torch_sdp_kernel_migration()."""

    def test_module_is_importable(self) -> None:
        """compat must import without side-effects or errors."""
        import compat  # noqa: F401

    def test_function_is_callable(self) -> None:
        import compat

        assert callable(compat.apply_torch_sdp_kernel_migration)

    def test_no_exception_on_call(self) -> None:
        """Calling the migration must not raise."""
        import compat

        compat.apply_torch_sdp_kernel_migration()  # must not raise

    def test_patches_torch_backends_cuda_sdp_kernel(self) -> None:
        """After migration torch.backends.cuda.sdp_kernel is our shim."""
        import torch

        import compat

        compat.apply_torch_sdp_kernel_migration()

        if not hasattr(torch.backends.cuda, "sdp_kernel"):
            pytest.skip("torch.backends.cuda.sdp_kernel absent in this build")

        shim = torch.backends.cuda.sdp_kernel
        assert getattr(shim, "_is_migrated_shim", False), (
            "sdp_kernel was not replaced with the migration shim"
        )

    def test_shim_is_a_context_manager(self) -> None:
        """The installed shim must be usable in a ``with`` statement."""
        import torch

        import compat

        compat.apply_torch_sdp_kernel_migration()

        if not hasattr(torch.backends.cuda, "sdp_kernel"):
            pytest.skip("torch.backends.cuda.sdp_kernel absent in this build")

        entered: list[bool] = []
        with torch.backends.cuda.sdp_kernel(enable_math=True):
            entered.append(True)

        assert entered == [True], "Shim body did not execute"

    def test_shim_body_executes_and_exits(self) -> None:
        """Code runs before AND after the body inside the shim."""
        import torch

        import compat

        compat.apply_torch_sdp_kernel_migration()

        if not hasattr(torch.backends.cuda, "sdp_kernel"):
            pytest.skip("torch.backends.cuda.sdp_kernel absent in this build")

        log: list[str] = []
        with torch.backends.cuda.sdp_kernel(enable_math=True):
            log.append("inside")
        log.append("after")

        assert log == ["inside", "after"]

    def test_shim_propagates_exceptions(self) -> None:
        """Exceptions raised inside the shim body must escape unchanged."""
        import torch

        import compat

        compat.apply_torch_sdp_kernel_migration()

        if not hasattr(torch.backends.cuda, "sdp_kernel"):
            pytest.skip("torch.backends.cuda.sdp_kernel absent in this build")

        class _Boom(Exception):
            pass

        with (
            pytest.raises(_Boom, match="intentional"),
            torch.backends.cuda.sdp_kernel(enable_math=True),
        ):
            raise _Boom("intentional")

    def test_shim_accepts_all_four_flags(self) -> None:
        """Shim must accept the same keyword arguments as the old API."""
        import torch

        import compat

        compat.apply_torch_sdp_kernel_migration()

        if not hasattr(torch.backends.cuda, "sdp_kernel"):
            pytest.skip("torch.backends.cuda.sdp_kernel absent in this build")

        # All four legacy flags — must not raise TypeError
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
            enable_cudnn=False,
        ):
            pass

    def test_shim_works_with_default_args(self) -> None:
        """Shim must be callable with no arguments (all defaults)."""
        import torch

        import compat

        compat.apply_torch_sdp_kernel_migration()

        if not hasattr(torch.backends.cuda, "sdp_kernel"):
            pytest.skip("torch.backends.cuda.sdp_kernel absent in this build")

        with torch.backends.cuda.sdp_kernel():
            pass

    def test_idempotent_second_call_leaves_shim_in_place(self) -> None:
        """Calling the migration twice must not replace the shim with itself."""
        import torch

        import compat

        compat.apply_torch_sdp_kernel_migration()

        if not hasattr(torch.backends.cuda, "sdp_kernel"):
            pytest.skip("torch.backends.cuda.sdp_kernel absent in this build")

        first_shim = torch.backends.cuda.sdp_kernel

        compat.apply_torch_sdp_kernel_migration()  # second call

        second_shim = torch.backends.cuda.sdp_kernel
        assert first_shim is second_shim, (
            "Second migration call replaced the already-installed shim"
        )

    def test_shim_sentinel_attribute_is_set(self) -> None:
        """Shim carries _is_migrated_shim=True so idempotency can be detected."""
        import torch

        import compat

        compat.apply_torch_sdp_kernel_migration()

        if not hasattr(torch.backends.cuda, "sdp_kernel"):
            pytest.skip("torch.backends.cuda.sdp_kernel absent in this build")

        assert torch.backends.cuda.sdp_kernel._is_migrated_shim is True


# ─────────────────────────────────────────────────────────────────────────────
# apply_diffusers_lora_migration()
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyDiffusersLoraMigration:
    """Observable behaviour of apply_diffusers_lora_migration()."""

    def test_function_is_callable(self) -> None:
        import compat

        assert callable(compat.apply_diffusers_lora_migration)

    def test_no_exception_on_call(self) -> None:
        """Calling the migration must not raise."""
        import compat

        compat.apply_diffusers_lora_migration()  # must not raise

    def test_replaces_with_nn_linear(self) -> None:
        """After migration LoRACompatibleLinear is torch.nn.Linear."""
        import torch.nn as nn

        import compat

        compat.apply_diffusers_lora_migration()

        try:
            import diffusers.models.lora as lora_mod
        except ImportError:
            pytest.skip("diffusers not installed")

        assert lora_mod.LoRACompatibleLinear is nn.Linear, (
            f"Expected nn.Linear, got {lora_mod.LoRACompatibleLinear!r}"
        )

    def test_idempotent_second_call(self) -> None:
        """Calling migration twice must not raise and must leave nn.Linear."""
        import torch.nn as nn

        import compat

        compat.apply_diffusers_lora_migration()
        compat.apply_diffusers_lora_migration()  # second call — must not raise

        try:
            import diffusers.models.lora as lora_mod
        except ImportError:
            pytest.skip("diffusers not installed")

        assert lora_mod.LoRACompatibleLinear is nn.Linear

    def test_patched_class_behaves_as_linear(self) -> None:
        """The replacement must behave as nn.Linear (instantiatable with in/out features)."""
        import compat

        compat.apply_diffusers_lora_migration()

        try:
            import diffusers.models.lora as lora_mod
        except ImportError:
            pytest.skip("diffusers not installed")

        # Must be instantiatable the same way chatterbox does it
        layer = lora_mod.LoRACompatibleLinear(4, 8)
        assert layer.in_features == 4
        assert layer.out_features == 8

    def test_forward_pass_works(self) -> None:
        """The replacement must produce valid output for a forward pass."""
        import torch

        import compat

        compat.apply_diffusers_lora_migration()

        try:
            import diffusers.models.lora as lora_mod
        except ImportError:
            pytest.skip("diffusers not installed")

        layer = lora_mod.LoRACompatibleLinear(4, 8)
        x = torch.randn(2, 4)
        out = layer(x)
        assert out.shape == (2, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Both migrations together
# ─────────────────────────────────────────────────────────────────────────────


class TestBothMigrationsTogether:
    """Applying both migrations in sequence must leave the system consistent."""

    def test_both_run_without_error(self) -> None:
        import compat

        compat.apply_diffusers_lora_migration()
        compat.apply_torch_sdp_kernel_migration()

    def test_both_idempotent_multiple_calls(self) -> None:
        import compat

        for _ in range(3):
            compat.apply_diffusers_lora_migration()
            compat.apply_torch_sdp_kernel_migration()

    def test_order_does_not_matter(self) -> None:
        """Reverse order must also work without errors."""
        import compat

        compat.apply_torch_sdp_kernel_migration()
        compat.apply_diffusers_lora_migration()
