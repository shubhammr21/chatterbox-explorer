"""
src/infrastructure/container.py
=================================
AppContainer — declarative dependency-injection container.

Uses the ``dependency-injector`` library (ets-labs/python-dependency-injector).
Replaces the manual object-construction wiring that previously lived in
``bootstrap.py``, making the dependency graph explicit, inspectable, and
overridable in tests.

Import contract (CRITICAL — do NOT relax)
-----------------------------------------
This module is imported INSIDE the body of ``build_app()`` in ``bootstrap.py``,
never at module level.  This guarantees that all compat patches applied in
``cli.main()`` (sdp_kernel, LoRACompatibleLinear, PerTh no-op) have already
fired before any chatterbox, torch, or torchaudio code is touched.

Do NOT add a top-level import of this module anywhere outside build_app().

Provider choices
----------------
Singleton     — used for all adapters and services.  Each object is
                constructed once, cached for the container's lifetime, and
                reused on every subsequent call.  This matches the previous
                manual wiring where all objects were local variables inside
                build_app().

Callable      — NOT used directly; device detection is wrapped in Singleton
                so hardware detection runs exactly once per container instance.

Object        — used for the ``set_seed`` callable itself (not its return
                value).  Injects the function reference into TTS services
                that accept a ``seed_setter: Callable[[int], None]`` arg.

Configuration — holds ``watermark_available`` (bool) supplied at runtime by
                cli.main() and forwarded through build_app().  Must be
                populated via ``container.config.watermark_available.from_value(...)``
                before any provider that depends on it is resolved.

Test overriding
---------------
Any provider can be replaced for testing without modifying source code::

    container = AppContainer()
    container.config.watermark_available.from_value(False)

    with container.model_loader.override(MockModelLoader()):
        svc = container.tts_service()
        # svc._repo is MockModelLoader — real ChatterboxModelLoader never loaded

This is the primary advantage over the previous manual wiring: the entire
infrastructure can be replaced in tests by overriding individual providers,
without instantiating the real heavy adapters.
"""

from __future__ import annotations

from dependency_injector import containers, providers

# ── Outbound adapters ──────────────────────────────────────────────────────────
# All imports here fire only when THIS MODULE is imported, which happens
# only inside build_app() — safely after all compat patches are active.
from adapters.outbound.audio import TorchAudioPreprocessor
from adapters.outbound.device import detect_device, set_seed
from adapters.outbound.memory import PsutilMemoryMonitor
from adapters.outbound.model_loader import ChatterboxModelLoader
from adapters.outbound.watermark import PerThWatermarkDetector

# ── Domain models ──────────────────────────────────────────────────────────────
from domain.models import AppConfig

# ── Domain services ────────────────────────────────────────────────────────────
from services.model_manager import ModelManagerService
from services.tts import MultilingualTTSService, TTSService, TurboTTSService
from services.voice_conversion import VoiceConversionService
from services.watermark import WatermarkService


class AppContainer(containers.DeclarativeContainer):
    """Declarative DI container for the Chatterbox TTS Explorer.

    Wiring order (mirrors the previous build_app() sequence):

        1. ``config``               — runtime values supplied before wiring
        2. ``device``               — compute device string, detected once
        3. Outbound adapters        — model_loader, audio_preprocessor, …
        4. Domain services          — each receives only the port ABCs it needs
        5. ``app_config``           — immutable AppConfig domain value-object

    Wiring configuration
    --------------------
    ``wiring_config`` declares the modules that must be wired so that
    ``@inject`` + ``Depends(Provide[AppContainer.*])`` resolve correctly in
    route handlers.  ``auto_wire=False`` suppresses automatic wiring at
    ``Container()`` instantiation time because ``adapters.inbound.rest.routes``
    is a deferred import — it must be imported inside ``build_rest_app()``
    *after* the compat patches in ``cli.main()`` have fired.  The explicit
    ``container.wire(modules=[routes_module])`` call in ``build_rest_app()``
    does the actual wiring once the module is safely imported.

    Typical usage inside build_rest_app()::

        container = AppContainer()
        container.config.from_pydantic(rest_settings)
        container.config.watermark_available.from_value(watermark_available)
        container.wire(modules=[routes_module])  # deferred — after compat patches

    Typical usage inside build_app() (Gradio UI mode)::

        container = AppContainer()
        container.config.watermark_available.from_value(watermark_available)

        demo = build_demo(
            tts=container.tts_service(),
            turbo=container.turbo_service(),
            mtl=container.multilingual_service(),
            vc=container.vc_service(),
            manager=container.model_manager_service(),
            watermark=container.watermark_service(),
            config=container.app_config(),
        )
    """

    # ── Wiring declaration ─────────────────────────────────────────────────
    # Documents which modules this container is designed to wire.
    # auto_wire=False: wiring is triggered manually inside build_rest_app()
    # after the deferred import of routes_module — never at Container()
    # instantiation time.  This preserves the compat-patch ordering guarantee.
    wiring_config = containers.WiringConfiguration(
        modules=["adapters.inbound.rest.routes"],
        auto_wire=False,
    )

    # ── Runtime configuration ──────────────────────────────────────────────
    # For the REST adapter, populated via:
    #   container.config.from_pydantic(rest_settings)   — all RestSettings fields
    #   container.config.watermark_available.from_value(...)  — runtime-detected
    #   container.config.device is NOT used here; device is a Singleton provider
    #
    # For the Gradio adapter (build_app), only watermark_available is set.
    #
    # Accessing a config key before it is populated raises a dependency_injector
    # error — correct fail-fast behaviour.
    config = providers.Configuration()

    # ── Compute device ─────────────────────────────────────────────────────
    # Singleton wrapping detect_device() so hardware detection runs exactly
    # once per container instance, regardless of how many downstream
    # providers reference this node (model_loader, memory_monitor, app_config).
    device = providers.Singleton(detect_device)

    # ── seed_setter callable ───────────────────────────────────────────────
    # TTS services accept an optional ``seed_setter: Callable[[int], None]``.
    # Object provider wraps the function reference itself — not its return
    # value — so each service receives the callable at construction time.
    seed_setter = providers.Object(set_seed)

    # ── Outbound adapters ──────────────────────────────────────────────────
    # Services depend ONLY on port ABCs (IModelRepository, IAudioPreprocessor,
    # etc.).  The concrete adapter classes are referenced here, in the
    # infrastructure layer, and nowhere else.  This preserves the hexagonal
    # architecture boundary even inside the container definition.

    model_loader = providers.Singleton(
        ChatterboxModelLoader,
        device=device,
    )

    audio_preprocessor = providers.Singleton(
        TorchAudioPreprocessor,
    )

    memory_monitor = providers.Singleton(
        PsutilMemoryMonitor,
        device=device,
    )

    watermark_detector = providers.Singleton(
        PerThWatermarkDetector,
        available=config.watermark_available,
    )

    # ── Domain services ────────────────────────────────────────────────────
    # Constructor arguments are matched by keyword name against each service's
    # __init__ signature.  See services/tts.py, services/voice_conversion.py,
    # services/model_manager.py, and services/watermark.py for the exact
    # parameter names.

    tts_service = providers.Singleton(
        TTSService,
        model_repo=model_loader,
        preprocessor=audio_preprocessor,
        seed_setter=seed_setter,
    )

    turbo_service = providers.Singleton(
        TurboTTSService,
        model_repo=model_loader,
        preprocessor=audio_preprocessor,
        seed_setter=seed_setter,
    )

    multilingual_service = providers.Singleton(
        MultilingualTTSService,
        model_repo=model_loader,
        preprocessor=audio_preprocessor,
        seed_setter=seed_setter,
    )

    vc_service = providers.Singleton(
        VoiceConversionService,
        model_repo=model_loader,
        preprocessor=audio_preprocessor,
    )

    model_manager_service = providers.Singleton(
        ModelManagerService,
        model_repo=model_loader,
        memory_monitor=memory_monitor,
    )

    watermark_service = providers.Singleton(
        WatermarkService,
        detector=watermark_detector,
    )

    # ── Application config value-object ───────────────────────────────────
    # AppConfig is a domain value-object consumed by services and the Gradio
    # inbound adapter.  Singleton ensures the same instance is returned on
    # every call so that every component holding a reference sees a
    # consistent view of runtime state.
    app_config = providers.Singleton(
        AppConfig,
        device=device,
        watermark_available=config.watermark_available,
    )
