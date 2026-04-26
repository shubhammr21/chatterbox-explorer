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

REST middleware stack (outermost → innermost, i.e. request arrival order)
--------------------------------------------------------------------------
CorrelationIdMiddleware  — raw ASGI middleware (asgi-correlation-id)
                           Sets the ``correlation_id`` ContextVar and appends
                           ``X-Request-ID`` to the response headers before any
                           other middleware or route handler runs.

RequestLoggingMiddleware — BaseHTTPMiddleware
                           Measures wall-clock timing and emits one structured
                           access-log record per request.  The CorrelationIdFilter
                           on the JSON handler injects ``correlation_id`` into
                           every LogRecord automatically — this middleware never
                           touches the correlation ID directly.

Registration order note
-----------------------
``app.add_middleware()`` prepends each middleware, so the *last* call becomes
the *outermost* wrapper.  We therefore register RequestLoggingMiddleware first
and CorrelationIdMiddleware last so that CorrelationIdMiddleware is outermost
and fires before RequestLoggingMiddleware on every request.
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

    The container is wired into routes_module via ``container.wire()`` — this
    call is intentionally made ONCE per process (wiring modifies the routes
    module globally).  Tests must share one app instance and use
    ``app.container.provider.override()`` for per-test isolation.

    Middleware registration order
    -----------------------------
    ``app.add_middleware()`` prepends each registration, so the last call
    becomes the outermost middleware (first to receive each request).

        add_middleware(RequestLoggingMiddleware)   ← registered first → inner
        add_middleware(CorrelationIdMiddleware)    ← registered last  → outer

    At runtime the request passes through:

        CorrelationIdMiddleware → RequestLoggingMiddleware → route handler

    This guarantees that the ``correlation_id`` ContextVar is set before
    ``RequestLoggingMiddleware.dispatch()`` runs, so the CorrelationIdFilter
    on the log handler can inject it into every access-log record.

    Args:
        watermark_available: True when the full PerTh watermarker is present.

    Returns:
        FastAPI application instance.  ``app.container`` holds the DI
        container for test provider overrides.

    Raises:
        ModuleNotFoundError: if the ``rest`` optional extra (fastapi, uvicorn,
            asgi-correlation-id) is not installed.  cli.main() checks earlier
            and produces a cleaner error, but this guard makes failure explicit.
    """
    from contextlib import asynccontextmanager

    import anyio.to_thread
    from asgi_correlation_id import CorrelationIdMiddleware
    from fastapi import FastAPI

    from adapters.inbound.rest import routes as routes_module
    from adapters.inbound.rest.middleware import RequestLoggingMiddleware
    from infrastructure.container import AppContainer
    from infrastructure.settings import RestSettings

    rest_settings = RestSettings()

    class ChatterboxAPI(FastAPI):
        """FastAPI subclass that carries a typed reference to the DI container.

        Declaring ``container`` as a class-level annotation (with no default)
        means the attribute must be assigned before use, which the type checker
        can verify, while avoiding any runtime overhead beyond a normal
        instance attribute assignment.
        """

        container: AppContainer

    container = AppContainer()
    container.config.watermark_available.from_value(watermark_available)
    container.wire(modules=[routes_module])

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── STARTUP ────────────────────────────────────────────────────────
        # Tune AnyIO's thread pool capacity.
        # 1 thread for GPU/CPU inference (serialised by inference_semaphore)
        # + headroom for admin I/O (disk probes, psutil, upload buffering).
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 10

        yield  # uvicorn begins accepting connections here

        # ── SHUTDOWN ───────────────────────────────────────────────────────
        container.unwire()

    app = ChatterboxAPI(
        title="Chatterbox TTS API",
        version="1.0.0",
        description=(
            "REST API for Chatterbox TTS — Standard, Turbo, Multilingual, "
            "Voice Conversion, Model Management, and Watermark Detection."
        ),
        debug=rest_settings.environment.is_debug,
        lifespan=lifespan,
    )

    # Attach container to the app — official dependency-injector pattern.
    # Tests access provider overrides via:
    #     with app.container.<provider>.override(mock): ...
    app.container = container

    app.include_router(routes_module.router)

    # ── Exception handlers ────────────────────────────────────────────────
    # Registered via add_exception_handler() (not the decorator) because the
    # handlers are defined in a separate module.
    #
    # Starlette resolves handlers by MRO — most-specific type wins.
    # Register more-specific domain types BEFORE the ChatterboxError catch-all.
    # Register against StarletteHTTPException (not fastapi.HTTPException) so
    # that exceptions raised by Starlette internals are also caught.
    #
    # TTSInputError             → 422: TTS input error (not logged — caller's fault)
    # VoiceConversionInputError → 422: VC input error (not logged — caller's fault)
    # ModelError                → 503: infrastructure failure (logged at ERROR)
    # ChatterboxError           → 500: unexpected domain error (logged at ERROR)
    # StarletteHTTPException: delegate to FastAPI default + log ≥500 at ERROR
    # RequestValidationError: delegate to FastAPI default + log at DEBUG
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from adapters.inbound.rest.exception_handlers import (
        chatterbox_error_handler,
        http_exception_handler_with_logging,
        model_error_handler,
        tts_input_error_handler,
        validation_exception_handler_with_logging,
        vc_input_error_handler,
    )
    from domain.exceptions import (
        ChatterboxError,
        ModelError,
        TTSInputError,
        VoiceConversionInputError,
    )

    app.add_exception_handler(TTSInputError, tts_input_error_handler)
    app.add_exception_handler(VoiceConversionInputError, vc_input_error_handler)
    app.add_exception_handler(ModelError, model_error_handler)
    app.add_exception_handler(ChatterboxError, chatterbox_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler_with_logging)
    app.add_exception_handler(RequestValidationError, validation_exception_handler_with_logging)

    # ── Middleware registration (innermost → outermost) ────────────────────
    # add_middleware() prepends, so last registered = outermost at runtime.

    # Inner: timing + structured access log.
    # The CorrelationIdFilter on the JSON handler injects correlation_id
    # into every LogRecord automatically — no manual UUID work here.
    app.add_middleware(RequestLoggingMiddleware)

    # Outer: generates/validates/propagates the correlation ID ContextVar and
    # appends X-Request-ID to every response header.
    # Must be outermost so it fires before RequestLoggingMiddleware.
    app.add_middleware(CorrelationIdMiddleware)

    return app
