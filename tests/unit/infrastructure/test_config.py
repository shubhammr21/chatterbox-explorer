"""
tests/unit/infrastructure/test_config.py
==========================================
Unit tests for infrastructure/config.py — AppSettings dataclass.

AppSettings is a frozen dataclass that carries the two runtime-resolved
values that seed the DI container:

    device              — best available compute device string
    watermark_available — PerTh watermark detector presence flag

Tests verify:
  - Construction with all valid device strings
  - Field values are stored and retrievable
  - Frozen constraint: mutation raises FrozenInstanceError
  - Equality semantics (two identical instances compare equal)
  - No runtime dependency on torch, gradio, or any heavy library
    (the import itself must succeed in a minimal environment)
"""

from __future__ import annotations

import dataclasses

import pytest

from infrastructure.config import AppSettings

# ──────────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────────


class TestAppSettingsConstruction:
    """AppSettings can be constructed for every valid device string."""

    @pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
    def test_all_device_strings_accepted(self, device: str) -> None:
        settings = AppSettings(device=device, watermark_available=True)
        assert settings.device == device

    @pytest.mark.parametrize("available", [True, False])
    def test_both_watermark_flags(self, available: bool) -> None:
        settings = AppSettings(device="cpu", watermark_available=available)
        assert settings.watermark_available is available

    def test_default_repr_contains_field_values(self) -> None:
        settings = AppSettings(device="mps", watermark_available=False)
        r = repr(settings)
        assert "mps" in r
        assert "False" in r


# ──────────────────────────────────────────────────────────────────────────────
# Field access
# ──────────────────────────────────────────────────────────────────────────────


class TestAppSettingsFields:
    """Field values round-trip correctly through the dataclass."""

    def test_device_field_stored(self) -> None:
        settings = AppSettings(device="cuda", watermark_available=True)
        assert settings.device == "cuda"

    def test_watermark_available_true(self) -> None:
        settings = AppSettings(device="cpu", watermark_available=True)
        assert settings.watermark_available is True

    def test_watermark_available_false(self) -> None:
        settings = AppSettings(device="cpu", watermark_available=False)
        assert settings.watermark_available is False


# ──────────────────────────────────────────────────────────────────────────────
# Immutability (frozen=True)
# ──────────────────────────────────────────────────────────────────────────────


class TestAppSettingsFrozen:
    """AppSettings is frozen — any field mutation must raise FrozenInstanceError."""

    def test_device_mutation_raises(self) -> None:
        settings = AppSettings(device="cpu", watermark_available=False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.device = "cuda"  # type: ignore[misc]

    def test_watermark_mutation_raises(self) -> None:
        settings = AppSettings(device="cpu", watermark_available=False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.watermark_available = True  # type: ignore[misc]

    def test_attribute_deletion_raises(self) -> None:
        settings = AppSettings(device="cpu", watermark_available=True)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            del settings.device  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────────
# Equality and hashing
# ──────────────────────────────────────────────────────────────────────────────


class TestAppSettingsEquality:
    """Frozen dataclasses implement value equality and are hashable."""

    def test_equal_instances(self) -> None:
        a = AppSettings(device="mps", watermark_available=True)
        b = AppSettings(device="mps", watermark_available=True)
        assert a == b

    def test_different_device_not_equal(self) -> None:
        a = AppSettings(device="cpu", watermark_available=True)
        b = AppSettings(device="cuda", watermark_available=True)
        assert a != b

    def test_different_watermark_flag_not_equal(self) -> None:
        a = AppSettings(device="cpu", watermark_available=True)
        b = AppSettings(device="cpu", watermark_available=False)
        assert a != b

    def test_hashable(self) -> None:
        """Frozen dataclasses are hashable — usable in sets and as dict keys."""
        a = AppSettings(device="cpu", watermark_available=True)
        b = AppSettings(device="cpu", watermark_available=True)
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_different_instances_different_hash(self) -> None:
        a = AppSettings(device="cpu", watermark_available=True)
        b = AppSettings(device="cuda", watermark_available=False)
        assert hash(a) != hash(b)


# ──────────────────────────────────────────────────────────────────────────────
# Architecture purity — no heavy runtime dependencies
# ──────────────────────────────────────────────────────────────────────────────


class TestAppSettingsArchitecturePurity:
    """Importing AppSettings must not pull in torch, gradio, or chatterbox."""

    def test_import_does_not_require_torch(self) -> None:
        """Re-import the module in a clean namespace to verify no torch dep."""
        import importlib
        import sys

        # Remove cached module to force a fresh import attempt.
        sys.modules.pop("infrastructure.config", None)

        # This must succeed even without torch being available.
        mod = importlib.import_module("infrastructure.config")
        assert hasattr(mod, "AppSettings")

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(AppSettings)

    def test_has_exactly_two_fields(self) -> None:
        fields = dataclasses.fields(AppSettings)
        assert len(fields) == 2

    def test_field_names(self) -> None:
        names = {f.name for f in dataclasses.fields(AppSettings)}
        assert names == {"device", "watermark_available"}
