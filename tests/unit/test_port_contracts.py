"""
tests/unit/test_port_contracts.py
====================================
Unit tests for the primary-port (input) ABCs declared in ``ports.input``.

Purpose
───────
``ports/input.py`` contains only module-level class definitions; the module
reaches 0% coverage unless something explicitly imports it.  These tests
force the import, exercise each ABC's contract, and verify the abstract-method
declarations match the documented interfaces.

What we check per ABC
─────────────────────
1. Is a subclass of :class:`abc.ABC` (architecture rule: all driving ports
   must be abstract base classes).
2. Has at least one abstract method (i.e. cannot be used as a concrete class
   without subclassing and implementing every method).
3. Raises :exc:`TypeError` on direct instantiation (Python ABC guarantee).
4. Declares exactly the expected set of abstract method names (guards against
   accidental method renames or additions).

No mocks required — these tests probe pure class-level metadata.
"""

from __future__ import annotations

from abc import ABC, ABCMeta

import pytest

from ports.input import (
    IModelManagerService,
    IMultilingualTTSService,
    ITTSService,
    ITurboTTSService,
    IVoiceConversionService,
    IWatermarkService,
)

# ──────────────────────────────────────────────────────────────────────────────
# Registry: (ABC class, expected abstract method names)
# ──────────────────────────────────────────────────────────────────────────────

_ABC_REGISTRY: list[tuple[ABCMeta, frozenset[str]]] = [
    (ITTSService, frozenset({"generate", "generate_stream"})),
    (ITurboTTSService, frozenset({"generate", "generate_stream"})),
    (IMultilingualTTSService, frozenset({"generate", "generate_stream"})),
    (IVoiceConversionService, frozenset({"convert"})),
    (
        IModelManagerService,
        frozenset({"load", "unload", "download", "get_all_status", "get_memory_stats"}),
    ),
    (IWatermarkService, frozenset({"detect"})),
]

_ABC_IDS = [cls.__name__ for cls, _ in _ABC_REGISTRY]


# ──────────────────────────────────────────────────────────────────────────────
# Parametrized universal checks
# ──────────────────────────────────────────────────────────────────────────────


class TestPortABCContracts:
    """Universal structural invariants that every primary-port ABC must satisfy."""

    @pytest.mark.parametrize("cls,_expected", _ABC_REGISTRY, ids=_ABC_IDS)
    def test_is_subclass_of_abc(self, cls: ABCMeta, _expected: frozenset[str]) -> None:
        """Every driving-port interface must inherit from :class:`abc.ABC`."""
        assert issubclass(cls, ABC), (
            f"{cls.__name__} must be a subclass of abc.ABC — "
            "primary ports are architecture-enforced abstract interfaces"
        )

    @pytest.mark.parametrize("cls,_expected", _ABC_REGISTRY, ids=_ABC_IDS)
    def test_has_at_least_one_abstract_method(
        self, cls: ABCMeta, _expected: frozenset[str]
    ) -> None:
        """Every ABC must declare at least one @abstractmethod so it cannot
        be instantiated without a concrete implementation."""
        assert len(cls.__abstractmethods__) >= 1, (
            f"{cls.__name__}.__abstractmethods__ is empty; "
            "the ABC has no @abstractmethod declarations and can be instantiated directly"
        )

    @pytest.mark.parametrize("cls,_expected", _ABC_REGISTRY, ids=_ABC_IDS)
    def test_direct_instantiation_raises_type_error(
        self, cls: ABCMeta, _expected: frozenset[str]
    ) -> None:
        """Python's ABC machinery must prevent direct construction of any
        primary-port interface."""
        with pytest.raises(TypeError):
            type.__call__(cls)

    @pytest.mark.parametrize("cls,expected_methods", _ABC_REGISTRY, ids=_ABC_IDS)
    def test_abstract_method_names_match_spec(
        self, cls: ABCMeta, expected_methods: frozenset[str]
    ) -> None:
        """The exact set of abstract method names must match the documented
        interface spec — catches accidental renames or missing declarations."""
        actual = cls.__abstractmethods__
        assert actual == expected_methods, (
            f"{cls.__name__}.__abstractmethods__ = {set(actual)!r}\n"
            f"Expected:                              {set(expected_methods)!r}\n"
            "Update this test if the interface contract was intentionally changed."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Per-class smoke checks (import coverage + individual method presence)
# ──────────────────────────────────────────────────────────────────────────────


class TestITTSServiceContract:
    """Spot-checks that ITTSService exposes the standard TTS driving interface."""

    def test_generate_is_abstract(self) -> None:
        assert "generate" in ITTSService.__abstractmethods__

    def test_generate_stream_is_abstract(self) -> None:
        assert "generate_stream" in ITTSService.__abstractmethods__


class TestITurboTTSServiceContract:
    """Spot-checks that ITurboTTSService exposes the turbo TTS driving interface."""

    def test_generate_is_abstract(self) -> None:
        assert "generate" in ITurboTTSService.__abstractmethods__

    def test_generate_stream_is_abstract(self) -> None:
        assert "generate_stream" in ITurboTTSService.__abstractmethods__


class TestIMultilingualTTSServiceContract:
    """Spot-checks that IMultilingualTTSService exposes the multilingual interface."""

    def test_generate_is_abstract(self) -> None:
        assert "generate" in IMultilingualTTSService.__abstractmethods__

    def test_generate_stream_is_abstract(self) -> None:
        assert "generate_stream" in IMultilingualTTSService.__abstractmethods__


class TestIVoiceConversionServiceContract:
    """Spot-checks that IVoiceConversionService exposes the VC driving interface."""

    def test_convert_is_abstract(self) -> None:
        assert "convert" in IVoiceConversionService.__abstractmethods__


class TestIModelManagerServiceContract:
    """Spot-checks that IModelManagerService exposes the full lifecycle interface."""

    def test_load_is_abstract(self) -> None:
        assert "load" in IModelManagerService.__abstractmethods__

    def test_unload_is_abstract(self) -> None:
        assert "unload" in IModelManagerService.__abstractmethods__

    def test_download_is_abstract(self) -> None:
        assert "download" in IModelManagerService.__abstractmethods__

    def test_get_all_status_is_abstract(self) -> None:
        assert "get_all_status" in IModelManagerService.__abstractmethods__

    def test_get_memory_stats_is_abstract(self) -> None:
        assert "get_memory_stats" in IModelManagerService.__abstractmethods__


class TestIWatermarkServiceContract:
    """Spot-checks that IWatermarkService exposes the watermark detection interface."""

    def test_detect_is_abstract(self) -> None:
        assert "detect" in IWatermarkService.__abstractmethods__
