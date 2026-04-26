"""
src/bootstrap.py
======================================
Dependency-injection root — creates and wires all secondary adapters,
domain services, and the primary (Gradio) adapter into a single runnable app.

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
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gradio as gr

    from domain.models import AppConfig


def build_app(watermark_available: bool) -> "tuple[gr.Blocks, AppConfig]":
    """Wire all adapters → services → Gradio primary adapter.

    All imports are deliberately placed inside this function body so that
    compat patches and logging configuration in ``cli.main()`` have already
    executed before any framework code (torch, chatterbox, gradio) is
    imported for the first time.

    Wiring order
    ------------
    1. Detect compute device (CUDA > MPS > CPU).
    2. Construct secondary adapters (infra layer).
    3. Construct domain services (inject adapters via port ABCs).
    4. Build AppConfig (immutable runtime snapshot).
    5. Build Gradio UI (primary adapter) by injecting domain services.

    Args:
        watermark_available: ``True`` when the full PerTh neural watermarker
            is present; ``False`` when the open-source no-op edition is active.
            Passed in from cli.main() after the PerTh patch check so this
            module never imports ``perth`` itself.

    Returns:
        A ``(demo, config)`` tuple where ``demo`` is the Gradio Blocks instance
        ready for ``demo.launch()`` and ``config`` is the immutable
        :class:`~chatterbox_explorer.domain.models.AppConfig`.
    """
    # ── Secondary adapters (infrastructure layer) ──────────────────────────
    from adapters.secondary.device import detect_device, set_seed
    from adapters.secondary.model_loader import ChatterboxModelLoader
    from adapters.secondary.audio import TorchAudioPreprocessor
    from adapters.secondary.memory import PsutilMemoryMonitor
    from adapters.secondary.watermark import PerThWatermarkDetector

    # ── Domain services ────────────────────────────────────────────────────
    from services.tts import (
        TTSService,
        TurboTTSService,
        MultilingualTTSService,
    )
    from services.voice_conversion import VoiceConversionService
    from services.model_manager import ModelManagerService
    from services.watermark import WatermarkService

    # ── Domain models ──────────────────────────────────────────────────────
    from domain.models import AppConfig

    # ── Step 1: device detection ───────────────────────────────────────────
    device = detect_device()

    # ── Step 2: secondary adapters ─────────────────────────────────────────
    model_repo   = ChatterboxModelLoader(device)
    preprocessor = TorchAudioPreprocessor()
    mem_monitor  = PsutilMemoryMonitor(device)
    wm_detector  = PerThWatermarkDetector(available=watermark_available)

    # ── Step 3: domain services ────────────────────────────────────────────
    # Each service receives only the port ABCs it needs — never the concrete
    # adapter classes — preserving the hexagonal architecture boundary.
    tts_svc   = TTSService(model_repo, preprocessor, seed_setter=set_seed)
    turbo_svc = TurboTTSService(model_repo, preprocessor, seed_setter=set_seed)
    mtl_svc   = MultilingualTTSService(model_repo, preprocessor, seed_setter=set_seed)
    vc_svc    = VoiceConversionService(model_repo, preprocessor)
    mgr_svc   = ModelManagerService(model_repo, mem_monitor)
    wm_svc    = WatermarkService(wm_detector)

    # ── Step 4: application config ─────────────────────────────────────────
    config = AppConfig(device=device, watermark_available=watermark_available)

    # ── Step 5: primary adapter — Gradio UI ───────────────────────────────
    # Imported last so that all compat patches are active before any
    # Gradio component or chatterbox model code is touched.
    from adapters.primary.gradio.ui import build_demo  # noqa: PLC0415

    demo = build_demo(
        tts=tts_svc,
        turbo=turbo_svc,
        mtl=mtl_svc,
        vc=vc_svc,
        manager=mgr_svc,
        watermark=wm_svc,
        config=config,
    )

    return demo, config
