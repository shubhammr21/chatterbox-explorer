#!/usr/bin/env python3
"""
chatterbox_explorer.cli
=======================
Official entry point for the Chatterbox TTS Explorer package.

Invoked via:
    uv run chatterbox-explorer              # standard launch
    uv run chatterbox-explorer --share       # public Gradio share link
    uv run chatterbox-explorer --mcp         # expose as MCP tool for AI agents
    uv run chatterbox-explorer --port 8080   # custom port
    uv run chatterbox-explorer --no-browser  # skip auto-opening browser

Also executable directly as a module:
    uv run python -m chatterbox_explorer
"""

from __future__ import annotations

import argparse
import atexit
import logging
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chatterbox-explorer",
        description="Chatterbox TTS Explorer — interactive demo for all Chatterbox capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run chatterbox-explorer                 Launch with default settings
  uv run chatterbox-explorer --share         Create a public Gradio share URL
  uv run chatterbox-explorer --mcp           Expose as MCP tool for AI agents
  uv run chatterbox-explorer --port 8080     Listen on a custom port
  uv run chatterbox-explorer --no-browser    Do not auto-open the browser
  uv run chatterbox-explorer --host 127.0.0.1 --port 7861  Bind to localhost only
""",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        metavar="ADDRESS",
        help="Server bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        default=7860,
        type=int,
        metavar="PORT",
        help="Server port (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share URL (tunnelled via Gradio servers)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser on startup",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Launch as an MCP server — exposes TTS as a tool for AI agents",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Package entry point.

    Execution order (order is load-bearing — do not rearrange):
      1. Parse CLI arguments
      2. Apply cross-cutting concerns:
         a. Logging configuration + third-party log suppression
         b. compat.py migrations (sdp_kernel, LoRACompatibleLinear)
         c. PerTh no-op patch (must precede any chatterbox import)
      3. Bootstrap: detect device, wire adapters → services → Gradio UI
      4. Register atexit handler for clean Gradio shutdown
      5. Launch
    """
    args = _parse_args(argv)

    # ── Step 2a: logging + warning suppression ────────────────────────────────
    # Must happen before any library import that calls logging.getLogger or
    # warnings.warn at import time (huggingface_hub, transformers, diffusers).
    from logging_config import configure

    configure()

    log = logging.getLogger("chatterbox-explorer")
    log.info(
        "Starting Chatterbox TTS Explorer on %s:%d%s%s",
        args.host,
        args.port,
        " [share]" if args.share else "",
        " [MCP]" if args.mcp else "",
    )

    # ── Step 2b: compat migrations ────────────────────────────────────────────
    # torch.backends.cuda.sdp_kernel → torch.nn.attention.sdpa_kernel
    # diffusers.LoRACompatibleLinear  → nn.Linear
    # Both must fire before the first lazy chatterbox model import.
    # compat is part of the installed package (src/compat.py) — direct import,
    # no try/except: if this fails, the migrations genuinely cannot run and the
    # caller must know about it rather than silently proceeding without them.
    from compat import (
        apply_diffusers_lora_migration,
        apply_torch_sdp_kernel_migration,
    )

    apply_diffusers_lora_migration()
    apply_torch_sdp_kernel_migration()

    # ── Step 2c: PerTh no-op patch ────────────────────────────────────────────
    # resemble-perth open-source edition ships with PerthImplicitWatermarker=None.
    # Chatterbox models call perth.PerthImplicitWatermarker() in __init__, so we
    # must patch before any chatterbox class is instantiated.
    import perth as _perth_mod

    watermark_available: bool = _perth_mod.PerthImplicitWatermarker is not None

    if not watermark_available:
        log.warning(
            "resemble-perth open-source edition: PerthImplicitWatermarker is None. "
            "Installing no-op fallback — outputs will NOT carry a PerTh AI watermark."
        )

        class _NoOpWatermarker:
            """Passthrough watermarker for the open-source resemble-perth edition."""

            def apply_watermark(
                self,
                audio: object,
                sample_rate: int,  # must match chatterbox keyword arg name
            ) -> object:
                return audio

            def get_watermark(
                self,
                audio: object,
                sample_rate: int,  # must match chatterbox keyword arg name
            ) -> float:
                return 0.0

        _perth_mod.PerthImplicitWatermarker = _NoOpWatermarker
        log.info("No-op PerTh watermarker installed ✓ — models will load correctly.")

    # ── Step 3: bootstrap ─────────────────────────────────────────────────────
    from bootstrap import build_app

    demo, _config = build_app(watermark_available=watermark_available)

    # ── Step 4: clean shutdown handler ────────────────────────────────────────
    # Register demo.close() with atexit BEFORE demo.launch().
    # Python's atexit runs handlers LIFO, so ours fires before resource_tracker,
    # ensuring the multiprocessing.Lock semaphore created by gradio/flagging.py
    # is explicitly released (prevents macOS Python 3.11 UserWarning about leaked
    # semaphore objects).
    atexit.register(demo.close)

    # ── Step 5: launch ────────────────────────────────────────────────────────
    from adapters.inbound.gradio.ui import (
        GRADIO_CSS,
        GRADIO_THEME,
    )

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        mcp_server=args.mcp,
        theme=GRADIO_THEME,
        css=GRADIO_CSS,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
