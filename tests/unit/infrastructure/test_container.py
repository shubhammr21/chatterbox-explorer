"""
tests/unit/infrastructure/test_container.py
============================================
Unit tests for infrastructure/container.py — AppContainer declarative DI container.

What is tested
--------------
1. Module is importable and AppContainer is present at module level.
2. AppContainer can be instantiated and configured without raising.
3. Runtime configuration (config.watermark_available) is applied correctly.
4. config.from_pydantic() loads all RestSettings fields into the config tree.
5. Provider overriding works — replacing any provider propagates into every
   downstream service that depends on it.
6. Singleton behaviour — repeated calls to the same provider return the same
   instance within a single container.
7. app_config resolution — the AppConfig domain object is built from the
   configured device and watermark_available values.
8. wiring_config is declared on AppContainer with auto_wire=False.
9. All expected provider names are present on the container class.

What is NOT tested
------------------
- Real hardware detection (detect_device() calls torch; overridden to "cpu" everywhere)
- Real model loading (ChatterboxModelLoader.get_model; uses mock IModelRepository)
- Integration between the container and the Gradio inbound adapter (build_demo
  requires a running Gradio server; integration-level concern)

Mock strategy
-------------
All providers that touch hardware are overridden with dependency_injector's
provider.override() context manager.  Each test receives a fresh AppContainer()
instance so singletons from one test never leak into another.
"""

from __future__ import annotations

import typing
from unittest.mock import MagicMock

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_container(watermark_available: bool = False, device: str = "cpu"):
    """Return a fresh AppContainer configured with safe test defaults.

    ``device`` is always overridden to avoid calling detect_device() which
    would import torch.  ``watermark_available`` defaults to False so that
    PerThWatermarkDetector is constructed in no-op mode (no librosa needed).
    """
    from infrastructure.container import AppContainer
    from ports.output import (
        IAudioPreprocessor,
        IMemoryMonitor,
        IModelRepository,
        IWatermarkDetector,
    )

    container = AppContainer()
    container.config.watermark_available.from_value(watermark_available)

    # Override every hardware-touching provider with safe lightweight mocks.
    container.device.override(device)
    container.model_loader.override(MagicMock(spec=IModelRepository))
    container.audio_preprocessor.override(MagicMock(spec=IAudioPreprocessor))
    container.memory_monitor.override(MagicMock(spec=IMemoryMonitor))
    container.watermark_detector.override(MagicMock(spec=IWatermarkDetector))

    return container


# ──────────────────────────────────────────────────────────────────────────────
# 1. Module import
# ──────────────────────────────────────────────────────────────────────────────


class TestAppContainerImport:
    """Importing infrastructure.container must succeed and expose AppContainer."""

    def test_module_is_importable(self) -> None:
        import importlib

        mod = importlib.import_module("infrastructure.container")
        assert mod is not None

    def test_app_container_class_present(self) -> None:
        from infrastructure.container import AppContainer

        assert AppContainer is not None

    def test_app_container_is_class(self) -> None:
        from infrastructure.container import AppContainer

        assert isinstance(AppContainer, type)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Instantiation
# ──────────────────────────────────────────────────────────────────────────────


class TestAppContainerInstantiation:
    """AppContainer() must instantiate without raising regardless of config order."""

    def test_bare_instantiation_does_not_raise(self) -> None:
        from infrastructure.container import AppContainer

        container = AppContainer()
        assert container is not None

    def test_multiple_instances_are_independent(self) -> None:
        from infrastructure.container import AppContainer

        a = AppContainer()
        b = AppContainer()
        a.config.watermark_available.from_value(True)
        b.config.watermark_available.from_value(False)
        # Neither instance should affect the other.
        assert a is not b


# ──────────────────────────────────────────────────────────────────────────────
# 3. wiring_config declaration
# ──────────────────────────────────────────────────────────────────────────────


class TestAppContainerWiringConfig:
    """AppContainer must declare wiring_config with auto_wire=False.

    wiring_config documents which modules the container is designed to wire.
    auto_wire=False preserves the deferred-import guarantee: routes must be
    imported inside build_rest_app() after compat patches have fired, so
    automatic wiring at Container() instantiation time would be too early.
    The explicit container.wire(modules=[routes_module]) call in
    build_rest_app() does the actual wiring once the module is safely loaded.
    """

    def test_wiring_config_exists_on_container_class(self) -> None:
        """AppContainer must have a wiring_config class attribute."""
        from dependency_injector import containers

        from infrastructure.container import AppContainer

        assert hasattr(AppContainer, "wiring_config"), (
            "AppContainer must declare wiring_config to document wiring intent"
        )
        assert isinstance(AppContainer.wiring_config, containers.WiringConfiguration)

    def test_wiring_config_auto_wire_is_false(self) -> None:
        """auto_wire must be False — wiring is done manually after deferred import."""
        from infrastructure.container import AppContainer

        assert AppContainer.wiring_config.auto_wire is False, (
            "auto_wire must be False to preserve the deferred-import guarantee: "
            "routes are imported inside build_rest_app() after compat patches fire."
        )

    def test_wiring_config_includes_rest_routes_module(self) -> None:
        """wiring_config must reference the REST routes module."""
        from infrastructure.container import AppContainer

        modules = AppContainer.wiring_config.modules or []
        module_strings = [
            m if isinstance(m, str) else getattr(m, "__name__", str(m)) for m in modules
        ]
        assert any("routes" in m for m in module_strings), (
            f"wiring_config.modules must include the REST routes module; got: {module_strings}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Runtime configuration
# ──────────────────────────────────────────────────────────────────────────────


class TestAppContainerConfiguration:
    """config must be configurable via both from_value() and from_pydantic()."""

    def test_configure_watermark_true_does_not_raise(self) -> None:
        from infrastructure.container import AppContainer

        container = AppContainer()
        container.config.watermark_available.from_value(True)  # must not raise

    def test_configure_watermark_false_does_not_raise(self) -> None:
        from infrastructure.container import AppContainer

        container = AppContainer()
        container.config.watermark_available.from_value(False)  # must not raise

    def test_device_override_accepted(self) -> None:
        from infrastructure.container import AppContainer

        container = AppContainer()
        container.device.override("cpu")
        assert container.device() == "cpu"

    @pytest.mark.parametrize("device_str", ["cpu", "mps", "cuda"])
    def test_device_override_all_values(self, device_str: str) -> None:
        from infrastructure.container import AppContainer

        container = AppContainer()
        container.device.override(device_str)
        assert container.device() == device_str


# ──────────────────────────────────────────────────────────────────────────────
# 5. config.from_pydantic() — RestSettings integration
# ──────────────────────────────────────────────────────────────────────────────


class TestAppContainerFromPydantic:
    """container.config.from_pydantic(rest_settings) must populate the config tree.

    This is the official dependency-injector pattern for loading pydantic-settings
    into a Configuration provider.  from_pydantic() calls rest_settings.model_dump()
    and merges the result recursively into the config tree, making every nested key
    available as a typed, validated value.

    runtime-detected values (watermark_available, device) are NOT in RestSettings
    and must still be set via from_value() after from_pydantic().
    """

    def test_from_pydantic_does_not_raise(self) -> None:
        """container.config.from_pydantic(RestSettings()) must not raise."""
        from infrastructure.container import AppContainer
        from infrastructure.settings import RestSettings

        container = AppContainer()
        container.config.from_pydantic(RestSettings())  # must not raise

    def test_from_pydantic_populates_server_host(self) -> None:
        """config.server.host must be accessible after from_pydantic()."""
        from infrastructure.container import AppContainer
        from infrastructure.settings import RestSettings

        container = AppContainer()
        container.config.from_pydantic(RestSettings())
        assert container.config.server.host() == "0.0.0.0"

    def test_from_pydantic_populates_server_port(self) -> None:
        """config.server.port must be accessible after from_pydantic()."""
        from infrastructure.container import AppContainer
        from infrastructure.settings import RestSettings

        container = AppContainer()
        container.config.from_pydantic(RestSettings())
        assert container.config.server.port() == 7860

    def test_from_pydantic_populates_logging_level(self) -> None:
        """config.logging.level must be accessible after from_pydantic()."""
        from infrastructure.container import AppContainer
        from infrastructure.settings import RestSettings

        container = AppContainer()
        container.config.from_pydantic(RestSettings())
        assert container.config.logging.level() == "INFO"

    def test_from_pydantic_populates_environment(self) -> None:
        """config.environment must be accessible after from_pydantic()."""
        from infrastructure.constants import Environment
        from infrastructure.container import AppContainer
        from infrastructure.settings import RestSettings

        container = AppContainer()
        container.config.from_pydantic(RestSettings())
        assert container.config.environment() == Environment.LOCAL

    def test_from_pydantic_then_from_value_override(self) -> None:
        """from_value() called after from_pydantic() must win for that key.

        This is the pattern used in build_rest_app(): load all settings from
        pydantic first, then set runtime-detected values (watermark_available)
        via from_value().  The last write to any config key wins.
        """
        from infrastructure.container import AppContainer
        from infrastructure.settings import RestSettings

        container = AppContainer()
        container.config.from_pydantic(RestSettings())
        container.config.watermark_available.from_value(True)
        assert container.config.watermark_available() is True

    def test_from_pydantic_with_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RestSettings reads env vars; from_pydantic() passes validated values."""
        monkeypatch.setenv("SERVER__PORT", "9090")
        from infrastructure.container import AppContainer
        from infrastructure.settings import RestSettings

        container = AppContainer()
        container.config.from_pydantic(RestSettings())
        assert container.config.server.port() == 9090


# ──────────────────────────────────────────────────────────────────────────────
# 6. Provider overriding propagation
# ──────────────────────────────────────────────────────────────────────────────


class TestAppContainerProviderOverriding:
    """Overriding a provider must propagate into every downstream service."""

    def test_override_model_loader_propagates_to_tts_service(self) -> None:
        container = _make_container()
        mock_repo = container.model_loader()  # already a MagicMock from _make_container

        tts = container.tts_service()

        assert tts._repo is mock_repo

    def test_override_model_loader_propagates_to_turbo_service(self) -> None:
        container = _make_container()
        mock_repo = container.model_loader()

        turbo = container.turbo_service()

        assert turbo._repo is mock_repo

    def test_override_model_loader_propagates_to_multilingual_service(self) -> None:
        container = _make_container()
        mock_repo = container.model_loader()

        mtl = container.multilingual_service()

        assert mtl._repo is mock_repo

    def test_override_model_loader_propagates_to_vc_service(self) -> None:
        container = _make_container()
        mock_repo = container.model_loader()

        vc = container.vc_service()

        assert vc._repo is mock_repo

    def test_override_model_loader_propagates_to_model_manager_service(self) -> None:
        container = _make_container()
        mock_repo = container.model_loader()

        mgr = container.model_manager_service()

        assert mgr._repo is mock_repo

    def test_override_audio_preprocessor_propagates_to_tts_service(self) -> None:
        container = _make_container()
        mock_prep = container.audio_preprocessor()

        tts = container.tts_service()

        assert tts._prep is mock_prep

    def test_override_watermark_detector_propagates_to_watermark_service(self) -> None:
        container = _make_container()
        mock_detector = container.watermark_detector()

        wm = container.watermark_service()

        assert wm._detector is mock_detector

    def test_override_memory_monitor_propagates_to_model_manager_service(self) -> None:
        container = _make_container()
        mock_mem = container.memory_monitor()

        mgr = container.model_manager_service()

        assert mgr._monitor is mock_mem

    def test_context_manager_override_restores_original(self) -> None:
        """After the context manager exits the override must be reset."""
        from infrastructure.container import AppContainer

        container = AppContainer()
        container.device.override("mps")

        with container.device.override("cuda"):
            assert container.device() == "cuda"

        assert container.device() == "mps"


# ──────────────────────────────────────────────────────────────────────────────
# 7. Singleton behaviour
# ──────────────────────────────────────────────────────────────────────────────


class TestAppContainerSingletonBehaviour:
    """Singleton providers must return the same instance on repeated calls."""

    def test_tts_service_same_instance(self) -> None:
        container = _make_container()
        assert container.tts_service() is container.tts_service()

    def test_turbo_service_same_instance(self) -> None:
        container = _make_container()
        assert container.turbo_service() is container.turbo_service()

    def test_multilingual_service_same_instance(self) -> None:
        container = _make_container()
        assert container.multilingual_service() is container.multilingual_service()

    def test_vc_service_same_instance(self) -> None:
        container = _make_container()
        assert container.vc_service() is container.vc_service()

    def test_model_manager_service_same_instance(self) -> None:
        container = _make_container()
        assert container.model_manager_service() is container.model_manager_service()

    def test_watermark_service_same_instance(self) -> None:
        container = _make_container()
        assert container.watermark_service() is container.watermark_service()

    def test_model_loader_same_instance(self) -> None:
        container = _make_container()
        assert container.model_loader() is container.model_loader()

    def test_audio_preprocessor_same_instance(self) -> None:
        container = _make_container()
        assert container.audio_preprocessor() is container.audio_preprocessor()


# ──────────────────────────────────────────────────────────────────────────────
# 8. app_config resolution
# ──────────────────────────────────────────────────────────────────────────────


class TestAppContainerAppConfig:
    """app_config() must build an AppConfig with values from the container config."""

    def test_app_config_device_matches_override(self) -> None:
        container = _make_container(device="mps")
        config = container.app_config()
        assert config.device == "mps"

    def test_app_config_watermark_true(self) -> None:
        container = _make_container(watermark_available=True)
        config = container.app_config()
        assert config.watermark_available is True

    def test_app_config_watermark_false(self) -> None:
        container = _make_container(watermark_available=False)
        config = container.app_config()
        assert config.watermark_available is False

    def test_app_config_is_domain_appconfig(self) -> None:
        from domain.models import AppConfig

        container = _make_container(device="cpu", watermark_available=False)
        config = container.app_config()
        assert isinstance(config, AppConfig)

    def test_app_config_is_pydantic_model(self) -> None:
        from pydantic import BaseModel

        from domain.models import AppConfig

        container = _make_container()
        config = container.app_config()
        assert isinstance(config, BaseModel)
        assert isinstance(config, AppConfig)

    def test_app_config_same_instance_singleton(self) -> None:
        """app_config provider is Singleton — same AppConfig returned each call."""
        container = _make_container()
        assert container.app_config() is container.app_config()

    @pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
    def test_app_config_device_parametrized(self, device: str) -> None:
        container = _make_container(device=device)
        assert container.app_config().device == device


# ──────────────────────────────────────────────────────────────────────────────
# 9. Provider structure
# ──────────────────────────────────────────────────────────────────────────────


class TestAppContainerProviderStructure:
    """Every expected provider must exist on AppContainer as a class attribute."""

    EXPECTED_PROVIDERS: typing.ClassVar[list[str]] = [
        "config",
        "device",
        "seed_setter",
        "model_loader",
        "audio_preprocessor",
        "memory_monitor",
        "watermark_detector",
        "tts_service",
        "turbo_service",
        "multilingual_service",
        "vc_service",
        "model_manager_service",
        "watermark_service",
        "app_config",
    ]

    @pytest.mark.parametrize("provider_name", EXPECTED_PROVIDERS)
    def test_provider_exists(self, provider_name: str) -> None:
        from infrastructure.container import AppContainer

        assert hasattr(AppContainer, provider_name), (
            f"AppContainer is missing expected provider: {provider_name!r}"
        )

    def test_provider_count_matches_expected(self) -> None:
        """Guard against accidental provider additions without test coverage."""
        from dependency_injector import providers as di_providers

        from infrastructure.container import AppContainer

        actual = [
            name
            for name, val in vars(AppContainer).items()
            if isinstance(val, di_providers.Provider) and not name.startswith("__")
        ]
        assert set(actual) == set(self.EXPECTED_PROVIDERS), (
            f"Provider mismatch.\n"
            f"  Expected: {sorted(self.EXPECTED_PROVIDERS)}\n"
            f"  Actual:   {sorted(actual)}"
        )
