"""
src/bootstrap.py
======================================
Dependency-injection root — thin shell that delegates all wiring to AppContainer.

Design constraints
------------------
1. ALL imports are deferred to the body of ``build_app()`` — nothing is
   imported at module load time except the TYPE_CHECKING guard.

   Reason: cli.main() must apply compat patches (sdp_kernel, LoRACompatibleLinear)
   and the PerTh no-op patch *before* any chatterbox or torch code is imported.
   If this module imported those packages at the top level the patches would be
   too late.  Deferred imports guarantee the correct ordering:

       cli.py              bootstrap.py
       ──────────────      ─────────────────────────────────
       configure()         (nothing at import time)
       apply_patches()
       build_app() ──────► deferred imports fire here — patches already active

2. The function signature exposes only ``watermark_available`` because that is
   the only runtime value that cannot be derived inside the function itself
   (it comes from the PerTh patch check performed in cli.main()).

3. All object construction and wiring lives in ``infrastructure/container.py``.
   This file is intentionally kept to ≤ 30 lines of logic so the wiring stays
   in one declarative place and is easy to test via provider overrides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
