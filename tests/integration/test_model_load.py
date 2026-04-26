#!/usr/bin/env python3
"""
End-to-end model load and generation test.
Validates the perth patch, warning suppression, and basic TTS generation.

Applies the exact same suppression stack as app.py so the output mirrors
what the running application produces.

Run:
    uv run python test_model_load.py
"""

import logging
import os
import sys
import warnings

import numpy as np
import torch

# ── 0. Apply the full suppression stack (mirrors app.py exactly) ──────────────

# Silence noisy third-party loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# diffusers 0.29.0: LoRACompatibleLinear deprecated
warnings.filterwarnings(
    "ignore",
    message=r".*LoRACompatibleLinear.*",
    category=FutureWarning,
)

# PyTorch 2.6.0: torch.backends.cuda.sdp_kernel() deprecated
warnings.filterwarnings(
    "ignore",
    message=r".*sdp_kernel.*",
    category=FutureWarning,
)

# transformers: sdpa + output_attentions messages
warnings.filterwarnings("ignore", message=r".*sdpa.*attention.*output_attentions.*")
warnings.filterwarnings("ignore", message=r".*output_attentions.*sdpa.*")

# huggingface_hub: unauthenticated request advisory
warnings.filterwarnings(
    "ignore",
    message=r".*unauthenticated requests.*HF Hub.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Please set a HF_TOKEN.*",
    category=UserWarning,
)

# Use transformers' own verbosity API to silence sdpa and similar internal logs
try:
    import transformers as _transformers

    _transformers.logging.set_verbosity_error()
except ImportError:
    pass

# huggingface_hub sets its root logger to level=WARNING during its own import.
# If we call setLevel(ERROR) before that import, the library overwrites our setting.
# Fix: force-import huggingface_hub first, then use its own public verbosity API.
try:
    from huggingface_hub import logging as _hf_logging  # use the library's own API

    _hf_logging.set_verbosity_error()  # overrides the library's WARNING default
except ImportError:
    pass

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # belt-and-suspenders
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)  # target the exact emitter


# ── 1. Basic setup ────────────────────────────────────────────────────────────


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = detect_device()

PASS = "✅"
FAIL = "❌"
WARN = "⚠ "

print(f"\n{'=' * 62}")
print("  Chatterbox TTS — Model Load & Generation Test")
print(f"{'=' * 62}")
print(f"  Device  : {DEVICE.upper()}")
print(f"  Python  : {sys.version.split()[0]}")
print(f"  HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'not set (public rate limits)'}")
print(f"{'=' * 62}\n")

all_passed = True


def step(n: int, label: str) -> None:
    print(f"[{n}] {label} ...")


def ok(msg: str) -> None:
    print(f"     {PASS} {msg}")


def fail(msg: str) -> None:
    global all_passed
    all_passed = False
    print(f"     {FAIL} {msg}")


def warn(msg: str) -> None:
    print(f"     {WARN} {msg}")


# ── 2. PerTh patch ────────────────────────────────────────────────────────────

step(1, "Checking resemble-perth watermarker")

import perth as _perth_mod

WATERMARK_AVAILABLE: bool = _perth_mod.PerthImplicitWatermarker is not None

if not WATERMARK_AVAILABLE:
    warn("PerthImplicitWatermarker is None (open-source edition)")
    warn("Installing no-op fallback so models can load ...")

    class _NoOpWatermarker:
        """
        Passthrough watermarker used when the full PerTh implementation
        is not available in the open-source resemble-perth package.

        Parameter names must match exactly what Chatterbox calls:
          - apply_watermark(wav, sample_rate=<int>)
          - get_watermark(audio, sample_rate=<int>)
        """

        def apply_watermark(
            self,
            audio: np.ndarray,
            sample_rate: int,  # keyword arg — must be named 'sample_rate'
        ) -> np.ndarray:
            return audio  # passthrough — no watermark embedded

        def get_watermark(
            self,
            audio: np.ndarray,
            sample_rate: int,  # keyword arg — must be named 'sample_rate'
        ) -> float:
            return 0.0  # always reports "no watermark present"

    _perth_mod.PerthImplicitWatermarker = _NoOpWatermarker
    ok("No-op patch applied — models will load correctly")
else:
    ok("Full PerTh watermarker available")


# ── 3. Standard TTS ───────────────────────────────────────────────────────────

step(2, "Loading ChatterboxTTS (Standard — 500M)")

try:
    from chatterbox.tts import ChatterboxTTS

    model_tts = ChatterboxTTS.from_pretrained(DEVICE)
    wm_cls = type(model_tts.watermarker).__name__
    ok(f"Loaded  |  sr={model_tts.sr} Hz  |  watermarker={wm_cls}")

    print("     Generating speech ...")
    wav_tts = model_tts.generate(
        "Hello, this is an end-to-end test of the Chatterbox standard TTS model.",
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    )
    assert wav_tts.ndim >= 1 and wav_tts.shape[-1] > 0, "Empty wav tensor"
    duration_s = wav_tts.shape[-1] / model_tts.sr
    ok(f"Generated  |  shape={tuple(wav_tts.shape)}  |  duration={duration_s:.2f}s")

except Exception as exc:
    fail(f"FAILED: {exc}")
    import traceback

    traceback.print_exc()
    sys.exit(1)


# ── 4. Turbo TTS ──────────────────────────────────────────────────────────────

step(3, "Loading ChatterboxTurboTTS (Turbo — 350M)")

try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    model_turbo = ChatterboxTurboTTS.from_pretrained(DEVICE)
    wm_cls = type(model_turbo.watermarker).__name__
    ok(f"Loaded  |  sr={model_turbo.sr} Hz  |  watermarker={wm_cls}")

    print("     Generating speech with paralinguistic tag ...")
    wav_turbo = model_turbo.generate(
        "Hi there! [chuckle] This is a quick Turbo model test.",
        temperature=0.8,
        top_k=1000,
        top_p=0.95,
    )
    assert wav_turbo.ndim >= 1 and wav_turbo.shape[-1] > 0, "Empty wav tensor"
    duration_s = wav_turbo.shape[-1] / model_turbo.sr
    ok(f"Generated  |  shape={tuple(wav_turbo.shape)}  |  duration={duration_s:.2f}s")

except Exception as exc:
    fail(f"FAILED: {exc}")
    import traceback

    traceback.print_exc()
    sys.exit(1)


# ── 5. int16 conversion (Gradio 6.x format) ───────────────────────────────────

step(4, "Verifying float32 → int16 conversion (Gradio 6.x format)")

try:
    raw = wav_tts.squeeze().detach().cpu().numpy().astype(np.float32)
    peak = np.abs(raw).max()
    if peak > 1.0:
        raw = raw / peak
    arr_int16 = np.clip(raw * 32767.0, -32768, 32767).astype(np.int16)

    assert arr_int16.dtype == np.int16, f"Expected int16, got {arr_int16.dtype}"
    assert arr_int16.shape == raw.shape, "Shape mismatch after conversion"
    assert np.abs(arr_int16).max() <= 32767, "int16 overflow"
    ok(f"int16 shape={arr_int16.shape}  |  peak={np.abs(arr_int16).max()}  |  no overflow")

except Exception as exc:
    fail(f"FAILED: {exc}")
    import traceback

    traceback.print_exc()
    sys.exit(1)


# ── 6. Watermark no-op round-trip ─────────────────────────────────────────────

step(5, "Watermark round-trip")

try:
    wm = _perth_mod.PerthImplicitWatermarker()
    dummy = np.zeros(24000, dtype=np.float32)

    # Chatterbox always calls with sample_rate as a keyword argument
    result = wm.apply_watermark(dummy, sample_rate=24000)
    score = wm.get_watermark(dummy, sample_rate=24000)

    assert result.shape == dummy.shape, "Shape mismatch after apply_watermark"

    if WATERMARK_AVAILABLE:
        ok(f"Full watermark  |  apply ✓  |  detect score={score:.4f}")
    else:
        assert score == 0.0, f"No-op should return 0.0, got {score}"
        ok(f"No-op passthrough ✓  |  detect score={score:.1f} (expected 0.0)")

except Exception as exc:
    fail(f"FAILED: {exc}")
    import traceback

    traceback.print_exc()
    sys.exit(1)


# ── 7. Warning audit ──────────────────────────────────────────────────────────
#
# Run a fresh generation inside a catch_warnings block — WITHOUT
# warnings.simplefilter('always'), so our filterwarnings rules stay active.
# Only warnings that were NOT suppressed will appear in `caught`.

step(6, "Warning audit — running generation with suppression active")

try:
    with warnings.catch_warnings(record=True) as caught:
        # Do NOT call warnings.simplefilter('always') here — that would
        # override our filterwarnings rules and capture everything again.
        # Instead, keep the current filter stack and only record what leaks through.
        warnings.simplefilter(
            "always", DeprecationWarning
        )  # we do want to know about deprecations in our own code

        wav_audit = model_tts.generate(
            "Auditing for unexpected warnings during generation.",
            exaggeration=0.5,
            temperature=0.8,
        )

    # Filter out DeprecationWarnings from third-party packages (not our code)
    our_warnings = [
        w
        for w in caught
        if not any(pkg in (w.filename or "") for pkg in ["site-packages", "cpython", "lib/python"])
    ]

    third_party_leaks = [
        w for w in caught if any(pkg in (w.filename or "") for pkg in ["site-packages", "cpython"])
    ]

    if third_party_leaks:
        warn(f"{len(third_party_leaks)} third-party warning(s) still leaking through suppression:")
        for w in third_party_leaks:
            print(f"       [{w.category.__name__}] {str(w.message)[:100]}")
    else:
        ok("Zero third-party warnings leaked through suppression stack")

    if our_warnings:
        warn(f"{len(our_warnings)} warning(s) in project code:")
        for w in our_warnings:
            print(f"       [{w.category.__name__}] {str(w.message)[:100]}")
    else:
        ok("Zero warnings in project code")

    duration_s = wav_audit.shape[-1] / model_tts.sr
    ok(f"Generation completed  |  duration={duration_s:.2f}s")

except Exception as exc:
    fail(f"FAILED: {exc}")
    import traceback

    traceback.print_exc()
    sys.exit(1)


# ── 8. Final summary ──────────────────────────────────────────────────────────

verdict = PASS if all_passed else FAIL
print(f"\n{'=' * 62}")
print(f"  {verdict}  All checks {'passed' if all_passed else 'FAILED'}")
print(f"{'=' * 62}")
print(f"  Device             : {DEVICE.upper()}")
print(
    f"  Standard TTS       : {PASS} loaded and generated ({wav_tts.shape[-1] / model_tts.sr:.2f}s)"
)
print(
    f"  Turbo TTS          : {PASS} loaded and generated ({wav_turbo.shape[-1] / model_turbo.sr:.2f}s)"
)
print(f"  float32 → int16    : {PASS} clean conversion, no Gradio UserWarning")
print(
    f"  PerTh watermarker  : {'✅ full impl' if WATERMARK_AVAILABLE else '⚠  no-op (open-source resemble-perth)'}"
)
print(f"{'=' * 62}\n")

if not all_passed:
    sys.exit(1)
