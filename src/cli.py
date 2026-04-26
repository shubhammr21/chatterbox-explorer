#!/usr/bin/env python3
"""
chatterbox_explorer.cli
=======================
Official entry point for the Chatterbox TTS Explorer package.

Invoked via:
    uv run chatterbox-explorer                        # Gradio UI (default)
    uv run chatterbox-explorer --mode ui              # explicit Gradio UI
    uv run chatterbox-explorer --mode rest            # FastAPI REST server
    uv run chatterbox-explorer --mode ui --share      # public Gradio share link
    uv run chatterbox-explorer --mode ui --mcp        # expose as MCP tool
    uv run chatterbox-explorer --port 8080            # custom port
    uv run chatterbox-explorer --no-browser           # skip auto-opening browser

Also executable directly as a module:
    uv run python -m chatterbox_explorer

Optional extras
---------------
The two delivery adapters are optional extras — install only what you need:

    uv sync --extra ui          # Gradio web UI  (gradio)
    uv sync --extra rest        # REST API       (fastapi + uvicorn)
    uv sync --extra all         # both adapters
    uv sync --group dev         # everything, including dev tooling

Attempting to launch a mode whose extra is not installed will produce a clear
error message with the exact install command to fix it.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chatterbox-explorer",
        description="Chatterbox TTS Explorer — interactive demo for all Chatterbox capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  ui    Launch the Gradio web UI  (requires: uv sync --extra ui)
  rest  Launch the FastAPI REST server (requires: uv sync --extra rest)

Examples:
  uv run chatterbox-explorer                      Launch Gradio UI (default)
  uv run chatterbox-explorer --mode rest          Launch REST API on port 7860
  uv run chatterbox-explorer --mode rest --port 8000
  uv run chatterbox-explorer --mode ui --share    Create a public Gradio share URL
  uv run chatterbox-explorer --mode ui --mcp      Expose as MCP tool for AI agents
  uv run chatterbox-explorer --port 8080          Custom port (any mode)
  uv run chatterbox-explorer --host 127.0.0.1 --port 7861
""",
    )
    parser.add_argument(
        "--mode",
        choices=["ui", "rest"],
        default="ui",
        metavar="MODE",
        help=(
            "Delivery adapter to launch. "
            "'ui' starts the Gradio web interface (default); "
            "'rest' starts the FastAPI REST server via uvicorn. "
            "Each mode requires its own optional extra — see package README."
        ),
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
    # ── ui-mode-only flags ────────────────────────────────────────────────────
    parser.add_argument(
        "--share",
        action="store_true",
        help="[--mode ui only] Create a public Gradio share URL (tunnelled via Gradio servers)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="[--mode ui only] Do not automatically open the browser on startup",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="[--mode ui only] Launch as an MCP server — exposes TTS as a tool for AI agents",
    )
    return parser.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────────────
# Extra availability guards
# ──────────────────────────────────────────────────────────────────────────────


def _require_ui_extra() -> None:
    """Raise SystemExit with a helpful message if the 'ui' extra is not installed."""
    try:
        import gradio  # noqa: F401
    except ModuleNotFoundError:
        print(
            "\n[chatterbox-explorer] ERROR: The Gradio UI extra is not installed.\n"
            "\n"
            "Fix with:\n"
            "    uv sync --extra ui\n"
            "\n"
            "Or install everything:\n"
            "    uv sync --extra all\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _require_rest_extra() -> None:
    """Raise SystemExit with a helpful message if the 'rest' extra is not installed."""
    missing = []
    for pkg in ("fastapi", "uvicorn"):
        try:
            __import__(pkg)
        except ModuleNotFoundError:
            missing.append(pkg)

    if missing:
        print(
            "\n[chatterbox-explorer] ERROR: The REST extra is not installed "
            f"(missing: {', '.join(missing)}).\n"
            "\n"
            "Fix with:\n"
            "    uv sync --extra rest\n"
            "\n"
            "Or install everything:\n"
            "    uv sync --extra all\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Common startup sequence (shared by all modes)
# ──────────────────────────────────────────────────────────────────────────────


def _apply_patches_and_perth() -> bool:
    """Apply compat patches and PerTh no-op fix; return watermark_available flag.

    This function must run BEFORE any chatterbox or torch code is imported.
    All steps are load-order-sensitive — do not rearrange.

    Returns:
        watermark_available: True when the full PerTh neural watermarker is
            present; False when the open-source no-op edition is active.
    """
    log = logging.getLogger("chatterbox-explorer")

    # ── compat migrations ─────────────────────────────────────────────────────
    # torch.backends.cuda.sdp_kernel → torch.nn.attention.sdpa_kernel
    # diffusers.LoRACompatibleLinear  → nn.Linear
    # Both must fire before the first lazy chatterbox model import.
    from compat import (
        apply_diffusers_lora_migration,
        apply_torch_sdp_kernel_migration,
    )

    apply_diffusers_lora_migration()
    apply_torch_sdp_kernel_migration()

    # ── PerTh no-op patch ─────────────────────────────────────────────────────
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

            def apply_watermark(self, audio: object, sample_rate: int) -> object:
                return audio

            def get_watermark(self, audio: object, sample_rate: int) -> float:
                return 0.0

        _perth_mod.PerthImplicitWatermarker = _NoOpWatermarker
        log.info("No-op PerTh watermarker installed ✓ — models will load correctly.")

    return watermark_available


# ──────────────────────────────────────────────────────────────────────────────
# Mode launchers
# ──────────────────────────────────────────────────────────────────────────────


def _launch_ui(args: argparse.Namespace, watermark_available: bool) -> None:
    """Build and launch the Gradio inbound adapter.

    Requires the 'ui' extra (gradio).
    """
    from bootstrap import build_app

    demo, _config = build_app(watermark_available=watermark_available)

    # Register demo.close() with atexit BEFORE demo.launch().
    # Python's atexit runs handlers LIFO, so ours fires before resource_tracker,
    # ensuring the multiprocessing.Lock semaphore created by gradio/flagging.py
    # is explicitly released (prevents macOS Python 3.11 UserWarning about leaked
    # semaphore objects).
    atexit.register(demo.close)

    from adapters.inbound.gradio.ui import GRADIO_CSS, GRADIO_THEME

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        mcp_server=args.mcp,
        theme=GRADIO_THEME,
        css=GRADIO_CSS,
    )


def _launch_rest(args: argparse.Namespace, watermark_available: bool) -> None:
    """Build and launch the FastAPI inbound adapter via uvicorn."""
    import uvicorn

    from bootstrap import build_rest_app
    from logging_config import configure_json

    app = build_rest_app(watermark_available=watermark_available)

    # Switch root logger to JSON for the REST server.
    configure_json()

    log = logging.getLogger("chatterbox-explorer")
    log.info("REST API docs available at http://%s:%d/docs", args.host, args.port)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=None,  # don't overwrite app's log configuration
        access_log=False,  # suppress uvicorn's own access logger — middleware is the source
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Package entry point.

    Execution order (load-order is load-bearing — do not rearrange):
      1. Parse CLI arguments + validate mode-specific flags
      2. Check that the required optional extra is installed
      3. Configure logging + warning suppression
      4. Apply compat patches (sdp_kernel, LoRACompatibleLinear)
      5. Apply PerTh no-op patch (must precede any chatterbox import)
      6. Dispatch to the selected mode launcher
    """
    args = _parse_args(argv)

    # ── Step 2: extra availability check ─────────────────────────────────────
    # Fail fast with a clear install instruction rather than a raw ImportError
    # buried in a stack trace after heavy model-loading has already started.
    if args.mode == "ui":
        _require_ui_extra()
    elif args.mode == "rest":
        _require_rest_extra()

    # ── Step 3: logging + warning suppression ─────────────────────────────────
    # Must happen before any library import that calls logging.getLogger or
    # warnings.warn at import time (huggingface_hub, transformers, diffusers).
    from logging_config import configure

    configure()

    log = logging.getLogger("chatterbox-explorer")
    log.info(
        "Starting Chatterbox TTS Explorer [mode=%s] on %s:%d",
        args.mode,
        args.host,
        args.port,
    )

    # ── Steps 4 + 5: patches ──────────────────────────────────────────────────
    watermark_available = _apply_patches_and_perth()

    # ── Step 6: dispatch ──────────────────────────────────────────────────────
    if args.mode == "ui":
        _launch_ui(args, watermark_available)
    elif args.mode == "rest":
        _launch_rest(args, watermark_available)


if __name__ == "__main__":
    main(sys.argv[1:])
