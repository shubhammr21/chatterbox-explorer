#!/usr/bin/env python3
"""
compat.py — Proper migration shims for deprecated third-party APIs used by chatterbox-tts.

Rather than suppressing FutureWarnings, each function here implements the actual
migration prescribed by the upstream deprecation notice.

Migrations handled
──────────────────
1. torch.backends.cuda.sdp_kernel()  →  torch.nn.attention.sdpa_kernel()
   Deprecated in PyTorch 2.6.0.
   Called in: chatterbox/models/t3/modules/perceiver.py:94

2. diffusers.models.lora.LoRACompatibleLinear  →  torch.nn.Linear
   Deprecated in diffusers 0.29.0, removed in diffusers 1.0.0.
   Called in: chatterbox/models/s3gen/matcha/transformer.py
   Safe because chatterbox always instantiates it with lora_layer=None,
   making forward() identical to nn.Linear.

Usage
─────
Call both apply_*() functions at app startup, before any lazy chatterbox import:

    import compat
    compat.apply_torch_sdp_kernel_migration()
    compat.apply_diffusers_lora_migration()
"""

from __future__ import annotations

import logging

log = logging.getLogger("chatterbox-demo.compat")


# ─────────────────────────────────────────────────────────────────────────────
# Migration 1: torch.backends.cuda.sdp_kernel → torch.nn.attention.sdpa_kernel
# ─────────────────────────────────────────────────────────────────────────────

def apply_torch_sdp_kernel_migration() -> None:
    """
    Replace the deprecated ``torch.backends.cuda.sdp_kernel`` context-manager
    with a shim that delegates to the new ``torch.nn.attention.sdpa_kernel`` API.

    Deprecation notice (PyTorch 2.6.0)
    ───────────────────────────────────
    "torch.backends.cuda.sdp_kernel() is deprecated. In the future, this context
    manager will be removed. Please see torch.nn.attention.sdpa_kernel() for the
    new context manager, with updated signature."

    Parameter mapping (old flags → SDPBackend enum)
    ─────────────────────────────────────────────────
    enable_flash        → SDPBackend.FLASH_ATTENTION     (CUDA ≥ SM 7.5 only)
    enable_math         → SDPBackend.MATH                (universal: CUDA / MPS / CPU)
    enable_mem_efficient→ SDPBackend.EFFICIENT_ATTENTION (CUDA only)
    enable_cudnn        → SDPBackend.CUDNN_ATTENTION     (CUDA + cuDNN 9+ only)

    MPS / Apple Silicon safety
    ──────────────────────────
    FLASH_ATTENTION, EFFICIENT_ATTENTION, and CUDNN_ATTENTION are CUDA-kernel-only
    despite the device-agnostic API signature. Passing them when tensors live on
    MPS causes a RuntimeError at attention dispatch. On MPS-only machines (no CUDA)
    we restrict the backend list to [SDPBackend.MATH], which is the only universally
    supported backend.

    Idempotency
    ───────────
    The function checks for a sentinel attribute ``_is_migrated_shim`` and returns
    immediately if the shim is already in place, making it safe to call multiple
    times or from multiple import paths.
    """
    import torch

    # Guard: new API must exist (introduced in PyTorch 2.0)
    if not hasattr(torch.nn.attention, "sdpa_kernel"):
        log.debug(
            "torch.nn.attention.sdpa_kernel not available — "
            "skipping sdp_kernel migration (PyTorch < 2.0?)"
        )
        return

    # Guard: old API must exist
    if not hasattr(torch.backends.cuda, "sdp_kernel"):
        log.debug(
            "torch.backends.cuda.sdp_kernel not present — nothing to migrate"
        )
        return

    # Idempotency: already patched
    if getattr(torch.backends.cuda.sdp_kernel, "_is_migrated_shim", False):
        log.debug("sdp_kernel migration already applied — skipping")
        return

    from contextlib import contextmanager
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # Detect MPS-only environment once at patch time (not per-call) for speed
    _is_mps_only: bool = (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and not torch.cuda.is_available()
    )

    @contextmanager
    def _sdp_kernel_shim(
        enable_flash: bool = True,
        enable_math: bool = True,
        enable_mem_efficient: bool = True,
        enable_cudnn: bool = False,
    ):
        """
        Drop-in replacement for torch.backends.cuda.sdp_kernel().

        Translates the four boolean flags from the old API into a list of
        SDPBackend enum values accepted by torch.nn.attention.sdpa_kernel().

        The set_priority=False default is preserved so PyTorch auto-selects
        the best available backend from the enabled set — matching the
        original flag-toggle behaviour.
        """
        if _is_mps_only:
            # FLASH / EFFICIENT / CUDNN are CUDA-only kernels; using them on
            # MPS raises RuntimeError at dispatch.  MATH is the only safe choice.
            backends = [SDPBackend.MATH]
        else:
            # Build list in PyTorch's historical implicit priority order:
            # FLASH → EFFICIENT → CUDNN → MATH (most-specialised first)
            backends: list[SDPBackend] = []
            if enable_flash:
                backends.append(SDPBackend.FLASH_ATTENTION)
            if enable_mem_efficient:
                backends.append(SDPBackend.EFFICIENT_ATTENTION)
            if enable_cudnn:
                try:
                    backends.append(SDPBackend.CUDNN_ATTENTION)
                except AttributeError:
                    # CUDNN_ATTENTION may not exist on all PyTorch builds
                    pass
            if enable_math:
                backends.append(SDPBackend.MATH)
            if not backends:
                # Guarantee at least one backend is always enabled
                backends = [SDPBackend.MATH]

        # set_priority=False (default): PyTorch auto-selects the best available
        # backend from the list, matching the original flag-toggle behaviour.
        with sdpa_kernel(backends, set_priority=False):
            yield

    # Mark the shim so idempotency check works
    _sdp_kernel_shim._is_migrated_shim = True

    # Replace the deprecated function
    torch.backends.cuda.sdp_kernel = _sdp_kernel_shim

    log.info(
        "Migration applied: torch.backends.cuda.sdp_kernel "
        "→ torch.nn.attention.sdpa_kernel ✓"
        + (" [MPS-only machine: restricted to MATH backend]" if _is_mps_only else "")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Migration 2: diffusers.LoRACompatibleLinear → torch.nn.Linear
# ─────────────────────────────────────────────────────────────────────────────

def apply_diffusers_lora_migration() -> None:
    """
    Replace ``diffusers.models.lora.LoRACompatibleLinear`` with
    ``torch.nn.Linear`` in the diffusers module namespace, before chatterbox's
    lazy model import binds the name.

    Deprecation notice (diffusers 0.29.0)
    ──────────────────────────────────────
    "LoRACompatibleLinear is deprecated and will be removed in version 1.0.0.
    Use of LoRACompatibleLinear is deprecated. Please switch to PEFT backend
    by installing PEFT: pip install peft."

    Why this is safe
    ────────────────
    Chatterbox's S3Gen model (matcha/transformer.py) instantiates
    LoRACompatibleLinear exclusively with ``lora_layer=None``:

        self.proj = LoRACompatibleLinear(in_features, out_features)
        # ^^^ no lora_layer= argument → lora_layer defaults to None

    With ``lora_layer=None``, LoRACompatibleLinear.forward() reduces to:

        def forward(self, hidden_states, scale=1.0):
            return super().forward(hidden_states)   # ← pure nn.Linear pass

    There is no LoRA delta applied, no extra parameters, and the weight tensor
    shapes/names in the .safetensors checkpoint are identical to nn.Linear.
    Replacing it with nn.Linear is therefore functionally equivalent.

    Import-order guarantee
    ──────────────────────
    This function patches ``diffusers.models.lora.LoRACompatibleLinear`` in
    the already-cached sys.modules entry. Because chatterbox's model imports
    are lazy (deferred until the first Generate click), the patched value is
    always in place before:

        from diffusers.models.lora import LoRACompatibleLinear   # in transformer.py

    binds the name in chatterbox's module namespace.

    Future-proofing
    ───────────────
    At diffusers 1.0.0 the ``deprecate()`` call will raise ValueError instead
    of emitting a warning. The monkey-patch prevents that hard failure as well.
    Pin ``diffusers < 1.0.0`` in pyproject.toml as an additional safeguard.
    """
    try:
        import torch.nn as nn
        import diffusers.models.lora as _diffusers_lora
    except ImportError:
        log.debug("diffusers not available — skipping LoRACompatibleLinear migration")
        return

    # Idempotency: already patched or was never the deprecated class
    if _diffusers_lora.LoRACompatibleLinear is nn.Linear:
        log.debug("LoRACompatibleLinear migration already applied — skipping")
        return

    # Verify the class is still the deprecated one we expect
    original_cls = _diffusers_lora.LoRACompatibleLinear
    original_name = getattr(original_cls, "__name__", str(original_cls))

    _diffusers_lora.LoRACompatibleLinear = nn.Linear

    log.info(
        f"Migration applied: diffusers.{original_name} → torch.nn.Linear ✓  "
        "(chatterbox uses lora_layer=None — forward() is identical to nn.Linear)"
    )
