"""
src/bootstrap.py
======================================
Dependency-injection root — thin shell that delegates all wiring to AppContainer.

Design constraints
------------------
1. ALL imports are deferred to the body of each ``build_*`` function — nothing is
   imported at module load time except the TYPE_CHECKING guard.

   Reason: cli.main() must apply compat patches (sdp_kernel, LoRACompatibleLinear)
   and the PerTh no-op patch *before* any chatterbox or torch code is imported.
   If this module imported those packages at the top level the patches would be
   too late.  Deferred imports guarantee the correct ordering:

       cli.py              bootstrap.py
       ──────────────      ─────────────────────────────────
       configure()         (nothing at import time)
       apply_patches()
       build_*() ────────► deferred imports fire here — patches already active

2. The function signatures expose only ``watermark_available`` because that is
   the only runtime value that cannot be derived inside the function itself
   (it comes from the PerTh patch check performed in cli.main()).

3. All object construction and wiring lives in ``infrastructure/container.py``.
   This file is intentionally kept minimal so the wiring stays in one
   declarative place and is easy to test via provider overrides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI
    import gradio as gr

    from domain.models import AppConfig


def build_app(watermark_available: bool) -> tuple[gr.Blocks, AppConfig]:
    """Wire all adapters → services → Gradio inbound adapter.

    All imports are deliberately placed inside this function body so that
    compat patches and logging configuration in ``cli.main()`` have already
    executed before any framework code (torch, chatterbox, gradio) is
    imported for the first time.

    Wiring is fully delegated to :class:`~infrastructure.container.AppContainer`.

    Args:
        watermark_available: ``True`` when the full PerTh neural watermarker
            is present; ``False`` when the open-source no-op edition is active.
            Passed in from cli.main() after the PerTh patch check so this
            module never imports ``perth`` itself.

    Returns:
        A ``(demo, config)`` tuple where ``demo`` is the Gradio Blocks instance
        ready for ``demo.launch()`` and ``config`` is the immutable
        :class:`~domain.models.AppConfig`.

    Raises:
        ModuleNotFoundError: if the ``ui`` optional extra (gradio) is not
            installed.  cli.main() checks for this earlier and produces a
            cleaner error, but the guard here makes the failure explicit.
    """
    # Deferred imports — patches are guaranteed active at this point.
    from adapters.inbound.gradio.ui import build_demo
    from infrastructure.container import AppContainer

    # ── Construct and configure the DI container ───────────────────────────
    container = AppContainer()
    container.config.watermark_available.from_value(watermark_available)

    # ── Build inbound adapter — Gradio UI ──────────────────────────────────
    # Resolve each service singleton from the container and pass them into
    # the Gradio adapter.  The adapter never touches the container directly —
    # it only receives the port-ABC instances it declared as parameters.
    demo = build_demo(
        tts=container.tts_service(),
        turbo=container.turbo_service(),
        mtl=container.multilingual_service(),
        vc=container.vc_service(),
        manager=container.model_manager_service(),
        watermark=container.watermark_service(),
        config=container.app_config(),
    )

    return demo, container.app_config()


def build_rest_app(watermark_available: bool) -> FastAPI:
    """Wire all adapters → services → FastAPI inbound adapter.

    All imports are deliberately placed inside this function body so that
    compat patches and logging configuration in ``cli.main()`` have already
    executed before any framework code (torch, chatterbox, fastapi) is
    imported for the first time.

    The container is wired into routes_module via container.wire() — this call
    is intentionally made ONCE per process (wiring modifies the module globally).
    Tests must share one app instance and use app.container.provider.override().

    Args:
        watermark_available: True when the full PerTh watermarker is present.

    Returns:
        FastAPI application instance with app.container set and lifespan wired.
    """
    from contextlib import asynccontextmanager

    import anyio.to_thread
    from fastapi import FastAPI

    from adapters.inbound.rest import routes as routes_module
    from adapters.inbound.rest.middleware import RequestLoggingMiddleware
    from infrastructure.container import AppContainer

    container = AppContainer()
    container.config.watermark_available.from_value(watermark_available)
    container.wire(modules=[routes_module])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── STARTUP ────────────────────────────────────────────────────────
        # Tune AnyIO's thread pool: 1 inference thread + headroom for admin
        # ops (memory stats, status checks, upload I/O).
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 10

        yield  # uvicorn begins accepting connections here

        # ── SHUTDOWN ───────────────────────────────────────────────────────
        container.unwire()

    app = FastAPI(
        title="Chatterbox TTS API",
        version="1.0.0",
        description=(
            "REST API for Chatterbox TTS — Standard, Turbo, Multilingual, "
            "Voice Conversion, Model Management, and Watermark Detection."
        ),
        lifespan=lifespan,
    )

    # Attach container to app — official dependency-injector pattern.
    # Tests access overrides via: with app.container.provider.override(mock): ...
    app.container = container  # type: ignore[attr-defined]

    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(routes_module.router)

    return app
