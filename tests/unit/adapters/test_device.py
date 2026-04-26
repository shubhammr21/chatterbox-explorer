"""
tests/unit/adapters/test_device.py
====================================
TDD unit tests for the device detection and seed management helpers.

    detect_device() — returns the best available compute device string
    set_seed()      — sets global RNG seeds for reproducible generation

Both functions live in:
    chatterbox_explorer.adapters.secondary.device

torch is required at test-time (imported via pytest.importorskip).
numpy and the stdlib random module are also exercised by set_seed tests.
"""
from __future__ import annotations

import random

import pytest

# torch is a hard dependency of chatterbox-tts, so it is always present in the
# project venv.  pytest.importorskip provides a clean skip message if it ever
# isn't (e.g. a stripped CI environment).
torch = pytest.importorskip("torch", reason="torch required for device adapter tests")

from chatterbox_explorer.adapters.secondary.device import detect_device, set_seed


# ──────────────────────────────────────────────────────────────────────────────
# detect_device
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectDevice:
    """Tests for the detect_device() helper."""

    VALID_DEVICES = frozenset({"cuda", "mps", "cpu"})

    def test_detect_device_returns_string(self):
        """detect_device() must return a str, not None or some other type."""
        result = detect_device()
        assert isinstance(result, str), (
            f"Expected str, got {type(result).__name__!r}"
        )

    def test_detect_device_valid_values(self):
        """Return value must be one of the three recognised device strings."""
        result = detect_device()
        assert result in self.VALID_DEVICES, (
            f"detect_device() returned {result!r}; "
            f"expected one of {sorted(self.VALID_DEVICES)}"
        )

    def test_detect_device_not_empty(self):
        """The returned string must not be empty."""
        assert detect_device() != ""

    def test_detect_device_is_deterministic(self):
        """Calling detect_device() twice on the same machine returns the same
        device (hardware does not change between calls).
        """
        assert detect_device() == detect_device()

    def test_detect_device_cuda_requires_cuda_available(self):
        """'cuda' may only be returned when torch.cuda.is_available() is True."""
        result = detect_device()
        if result == "cuda":
            assert torch.cuda.is_available(), (
                "detect_device() returned 'cuda' but torch.cuda.is_available() is False"
            )

    def test_detect_device_mps_requires_mps_available(self):
        """'mps' may only be returned when the MPS backend is present and available."""
        result = detect_device()
        if result == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            assert mps_backend is not None and mps_backend.is_available(), (
                "detect_device() returned 'mps' but MPS is not available"
            )

    def test_detect_device_cpu_is_always_valid_fallback(self):
        """'cpu' is always a valid result — this test documents the invariant."""
        result = detect_device()
        # cpu is valid regardless of which device was chosen
        assert result in self.VALID_DEVICES


# ──────────────────────────────────────────────────────────────────────────────
# set_seed
# ──────────────────────────────────────────────────────────────────────────────

class TestSetSeed:
    """Tests for the set_seed() helper."""

    # ── seed=0 no-op ──────────────────────────────────────────────────────────

    def test_set_seed_zero_is_noop(self):
        """set_seed(0) must not raise and must not alter the torch RNG state."""
        # Record the current state before calling with seed=0.
        state_before = torch.get_rng_state().clone()

        set_seed(0)  # must be a no-op

        state_after = torch.get_rng_state()
        assert torch.equal(state_before, state_after), (
            "set_seed(0) must be a no-op — it should not change the torch RNG state"
        )

    def test_set_seed_zero_does_not_raise(self):
        """set_seed(0) must complete without raising any exception."""
        try:
            set_seed(0)
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"set_seed(0) raised an unexpected exception: {exc}")

    # ── seed != 0 changes torch state ─────────────────────────────────────────

    def test_set_seed_nonzero_sets_torch_seed(self):
        """set_seed(n) with n != 0 must change the torch RNG state.

        We verify by seeding with two different values and confirming the
        torch initial_seed() differs between them.
        """
        set_seed(42)
        seed_42 = torch.initial_seed()

        set_seed(99)
        seed_99 = torch.initial_seed()

        assert seed_42 != seed_99, (
            "set_seed(42) and set_seed(99) produced the same torch initial_seed(); "
            "torch.manual_seed() does not appear to be taking effect"
        )

    def test_set_seed_nonzero_produces_deterministic_torch_output(self):
        """After set_seed(n), torch random output is the same as after
        torch.manual_seed(n) directly — i.e. our wrapper is a thin pass-through.
        """
        # Direct torch seeding as reference.
        torch.manual_seed(7)
        expected = torch.randn(8).tolist()

        # Via set_seed.
        set_seed(7)
        actual = torch.randn(8).tolist()

        assert actual == expected, (
            "set_seed(7) did not produce the same torch random output as "
            "torch.manual_seed(7)"
        )

    # ── reproducibility ───────────────────────────────────────────────────────

    def test_set_seed_reproducible(self):
        """Identical seed → identical torch random sequence on both calls."""
        set_seed(123)
        run1 = torch.randn(16).tolist()

        set_seed(123)
        run2 = torch.randn(16).tolist()

        assert run1 == run2, (
            "set_seed(123) produced different torch random sequences on two calls; "
            "the seed is not being applied correctly"
        )

    def test_set_seed_different_seeds_produce_different_output(self):
        """Different seeds must (with overwhelming probability) produce
        different random sequences — a collision here would indicate the seed
        is being ignored.
        """
        set_seed(1)
        run_1 = torch.randn(32).tolist()

        set_seed(2)
        run_2 = torch.randn(32).tolist()

        assert run_1 != run_2, (
            "set_seed(1) and set_seed(2) produced identical torch random sequences"
        )

    # ── numpy RNG ─────────────────────────────────────────────────────────────

    def test_set_seed_reproducible_numpy(self):
        """The numpy global RNG must also be seeded for reproducibility."""
        import numpy as np

        set_seed(456)
        np_run1 = np.random.rand(8).tolist()

        set_seed(456)
        np_run2 = np.random.rand(8).tolist()

        assert np_run1 == np_run2, (
            "set_seed(456) did not produce the same numpy random sequence on two calls"
        )

    # ── Python stdlib random ──────────────────────────────────────────────────

    def test_set_seed_reproducible_stdlib_random(self):
        """The Python stdlib random module must also be seeded."""
        set_seed(789)
        py_run1 = [random.random() for _ in range(8)]

        set_seed(789)
        py_run2 = [random.random() for _ in range(8)]

        assert py_run1 == py_run2, (
            "set_seed(789) did not produce the same Python random sequence on two calls"
        )

    # ── accepts int-convertible types ─────────────────────────────────────────

    def test_set_seed_accepts_float_seed(self):
        """set_seed coerces its argument to int — float inputs must not raise."""
        try:
            set_seed(42.9)  # int(42.9) == 42
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"set_seed(42.9) raised unexpectedly: {exc}")

    def test_set_seed_float_coercion_is_deterministic(self):
        """set_seed(42.9) and set_seed(42) should behave identically because
        both coerce to int(42).
        """
        set_seed(42)
        ref = torch.randn(4).tolist()

        set_seed(42.9)
        actual = torch.randn(4).tolist()

        assert actual == ref, (
            "set_seed(42.9) and set_seed(42) produced different output — "
            "int() coercion may not be working"
        )
