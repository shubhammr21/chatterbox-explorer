#!/usr/bin/env python3
"""
Chatterbox TTS Explorer  (v2 — all gaps fixed)
===============================================
Interactive Gradio demo showcasing all Chatterbox capabilities.
Every developer-relevant parameter is exposed — no code editing required.

v2 fixes:
  • Random seed on all TTS tabs  (0 = random, any other value = reproducible)
  • Exaggeration slider extended to 2.0  (official app + demo page show 0.5 / 1.0 / 2.0)
  • Correct temperature ranges: Standard & MTL 0.05–5.0 · Turbo 0.05–2.0
  • Cursor-position-aware paralinguistic tag insertion via JavaScript
  • Language change auto-fills both sample text AND reference audio clip
  • --mcp flag exposes app as an MCP tool for AI agents

Run:
    uv run app.py
    uv run app.py --share
    uv run app.py --mcp
    uv run app.py --port 8080 --no-browser
"""

from __future__ import annotations

import argparse
import logging
import atexit
import os
import random
import re
import tempfile
import warnings

import gradio as gr
import numpy as np
import torch
import torchaudio

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chatterbox-demo")

# Silence noisy third-party loggers that produce no actionable output for users.
#
# IMPORTANT — ordering matters for huggingface_hub:
# The huggingface_hub library sets its root logger to level=WARNING during its
# own import.  If we call setLevel(ERROR) before that import happens our setting
# gets overwritten.  We therefore force-import huggingface_hub first, then use
# its own public verbosity API (hf_logging.set_verbosity_error) so we always win.

try:
    import huggingface_hub as _hf_hub                     # force initialization first
    from huggingface_hub import logging as _hf_logging    # then use its own API
    _hf_logging.set_verbosity_error()                     # overrides library's WARNING default
except ImportError:
    pass

logging.getLogger("transformers").setLevel(logging.ERROR)   # suppresses sdpa/attention warnings
logging.getLogger("huggingface_hub").setLevel(logging.ERROR) # belt-and-suspenders on top of hf_logging
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)        # keep httpx warnings but drop INFO


# ──────────────────────────────────────────────────────────────────────────────
# Deprecation Migrations  (proper fixes — not suppressions)
# ──────────────────────────────────────────────────────────────────────────────
# Each migration replaces a deprecated API with its sanctioned successor.
# See compat.py for full documentation of each migration.

import compat

# Migration 1 — diffusers 0.29.0: LoRACompatibleLinear → nn.Linear
# The warning fires unconditionally in LoRACompatibleLinear.__init__().
# chatterbox/models/s3gen/matcha/transformer.py instantiates it with
# lora_layer=None, making it functionally identical to nn.Linear.
# We patch diffusers.models.lora.LoRACompatibleLinear = nn.Linear BEFORE
# chatterbox's lazy model import so the deprecated class is never instantiated.
compat.apply_diffusers_lora_migration()

# Migration 2 — PyTorch 2.6.0: torch.backends.cuda.sdp_kernel() → sdpa_kernel()
# chatterbox/models/t3/modules/perceiver.py:94 calls the deprecated context
# manager.  We replace it with a shim that delegates to the new
# torch.nn.attention.sdpa_kernel() API with correct SDPBackend enum mapping.
# MPS safety: FLASH/EFFICIENT/CUDNN are CUDA-only; restricted to MATH on Apple Silicon.
compat.apply_torch_sdp_kernel_migration()

# huggingface_hub: unauthenticated request advisory — suppressed because models
# are public and we already emit a single clean advisory ourselves below.
# This is not a deprecation; the library issues it on every model file request.
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

# Use transformers' own logging API to silence internal attention-dispatch
# messages (e.g. sdpa + output_attentions) that bypass Python's logging
# hierarchy and have no user-actionable migration path from our side.
try:
    import transformers as _transformers
    _transformers.logging.set_verbosity_error()
except ImportError:
    pass  # transformers not installed — nothing to silence


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace Token Check
# ──────────────────────────────────────────────────────────────────────────────
# Models are public so HF_TOKEN is optional, but without it HF Hub applies
# stricter rate limits. We emit a single clean advisory instead of letting
# the huggingface_hub library spam the log on every model file request.
_HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not _HF_TOKEN:
    log.info(
        "HF_TOKEN not set — using unauthenticated HuggingFace access (public rate limits). "
        "Set HF_TOKEN=<token> for higher limits: https://huggingface.co/settings/tokens"
    )
else:
    log.info("HF_TOKEN detected ✓ — authenticated HuggingFace access enabled.")


# ──────────────────────────────────────────────────────────────────────────────
# Device Detection
# ──────────────────────────────────────────────────────────────────────────────
def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _detect_device()
log.info(f"Device: {DEVICE.upper()}")


# ──────────────────────────────────────────────────────────────────────────────
# PerTh Watermarker Compatibility Patch
# ──────────────────────────────────────────────────────────────────────────────
# Root cause: the open-source `resemble-perth` package (PyPI v1.0.1) ships with
# PerthImplicitWatermarker = None (the full neural watermarker is not included).
# Every Chatterbox model calls `perth.PerthImplicitWatermarker()` in __init__,
# which throws `TypeError: 'NoneType' object is not callable` and blocks ALL
# model loading.
#
# Fix: patch a no-op class into `perth.PerthImplicitWatermarker` BEFORE any
# Chatterbox import happens.  Because Python caches modules in sys.modules,
# chatterbox.tts / tts_turbo / mtl_tts / vc all see the patched value when
# they later do `import perth`.
#
# Consequence: outputs are NOT watermarked in this environment.
# The watermark checker will always return 0.0 (no watermark found).
# ──────────────────────────────────────────────────────────────────────────────
import perth as _perth_mod

_WATERMARK_AVAILABLE: bool = _perth_mod.PerthImplicitWatermarker is not None

if not _WATERMARK_AVAILABLE:
    log.warning(
        "resemble-perth open-source edition detected — "
        "PerthImplicitWatermarker is None.  "
        "Patching a no-op fallback so models can load.  "
        "Outputs will NOT carry a PerTh AI watermark."
    )

    class _NoOpWatermarker:
        """Passthrough watermarker used when the full PerTh impl is unavailable."""

        def apply_watermark(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:  # noqa: ARG002
            return audio  # passthrough — no watermark embedded

        def get_watermark(self, audio: np.ndarray, sample_rate: int) -> float:  # noqa: ARG002
            return 0.0  # always reports "no watermark present"

    _perth_mod.PerthImplicitWatermarker = _NoOpWatermarker
    log.info("No-op PerTh watermarker installed ✓  — models will now load correctly.")


# ──────────────────────────────────────────────────────────────────────────────
# Lazy Model Registry
# ──────────────────────────────────────────────────────────────────────────────
_MODEL_CACHE: dict = {}


def get_model(key: str):
    """Load and cache a Chatterbox model by key. Returns (model, error_str | None)."""
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], None
    try:
        log.info(f"Loading model '{key}' — may download from HuggingFace on first run …")
        if key == "tts":
            from chatterbox.tts import ChatterboxTTS
            model = ChatterboxTTS.from_pretrained(DEVICE)
        elif key == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            model = ChatterboxTurboTTS.from_pretrained(DEVICE)
        elif key == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            model = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
        elif key == "vc":
            from chatterbox.vc import ChatterboxVC
            model = ChatterboxVC.from_pretrained(DEVICE)
        else:
            return None, f"Unknown model key: {key}"
        _MODEL_CACHE[key] = model
        log.info(f"Model '{key}' ready ✓")
        return model, None
    except Exception as exc:
        log.error(f"Failed to load '{key}': {exc}")
        return None, str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# Static Data
# ──────────────────────────────────────────────────────────────────────────────
LANGUAGE_OPTIONS: list[str] = [
    "ar - Arabic",
    "da - Danish",
    "de - German",
    "el - Greek",
    "en - English",
    "es - Spanish",
    "fi - Finnish",
    "fr - French",
    "he - Hebrew",
    "hi - Hindi",
    "it - Italian",
    "ja - Japanese",
    "ko - Korean",
    "ms - Malay",
    "nl - Dutch",
    "no - Norwegian",
    "pl - Polish",
    "pt - Portuguese",
    "ru - Russian",
    "sv - Swedish",
    "sw - Swahili",
    "tr - Turkish",
    "zh - Chinese",
]

# Sample texts sourced from official Chatterbox multilingual demo
SAMPLE_TEXTS: dict[str, str] = {
    "ar - Arabic":    "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب.",
    "da - Danish":    "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal.",
    "de - German":    "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal.",
    "el - Greek":     "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube.",
    "en - English":   "Last month, we reached a new milestone with two billion views on our YouTube channel.",
    "es - Spanish":   "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube.",
    "fi - Finnish":   "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme.",
    "fr - French":    "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube.",
    "he - Hebrew":    "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו.",
    "hi - Hindi":     "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।",
    "it - Italian":   "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube.",
    "ja - Japanese":  "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。",
    "ko - Korean":    "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다.",
    "ms - Malay":     "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami.",
    "nl - Dutch":     "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal.",
    "no - Norwegian": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår.",
    "pl - Polish":    "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube.",
    "pt - Portuguese":"No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube.",
    "ru - Russian":   "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале.",
    "sv - Swedish":   "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal.",
    "sw - Swahili":   "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kwenye kituo chetu cha YouTube.",
    "tr - Turkish":   "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık.",
    "zh - Chinese":   "上个月，我们达到了一个新的里程碑，我们的YouTube频道观看次数达到了二十亿次。",
}

# Curated native-speaker reference clips from the official Chatterbox multilingual demo.
# These are publicly accessible FLAC files hosted by Resemble AI on Google Cloud Storage.
# They load automatically when you switch languages — no upload needed.
LANGUAGE_AUDIO_DEFAULTS: dict[str, str] = {
    "ar": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
    "da": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
    "de": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
    "el": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
    "en": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
    "es": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
    "fi": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
    "fr": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
    "he": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
    "hi": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
    "it": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
    "ja": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
    "ko": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
    "ms": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
    "nl": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
    "no": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
    "pl": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
    "pt": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
    "ru": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
    "sv": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
    "sw": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
    "tr": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
    "zh": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
}

PARA_TAGS: list[str] = [
    "[laugh]", "[chuckle]", "[cough]", "[sigh]", "[gasp]", "[hmm]", "[clears throat]",
]

# ──────────────────────────────────────────────────────────────────────────────
# Presets — named parameter bundles for real-world use cases
# ──────────────────────────────────────────────────────────────────────────────
# Each preset contains:
#   params       — concrete slider values (validated against model behaviour)
#   description  — one-line summary shown as dropdown label tooltip
#   rationale_md — markdown explaining WHY each param is set this way
#                  (educational: users learn to tweak by reading this)
#   sample_text  — example sentence that showcases the preset's character
#
# Parameter notes (from official Chatterbox README + empirical testing):
#   • Higher exaggeration accelerates speech pace — compensate with lower cfg_weight
#   • cfg_weight=0.3 is the README's recommended value for expressive/dramatic speech
#   • temperature controls output variation: 0.3 = stable, 1.2 = very creative
#   • rep_penalty 1.4 is the "safe maximum" for long-form content
#   • top_k (Turbo only) is the strongest creativity dial: 80 = near-greedy, 2000 = wide open
# ──────────────────────────────────────────────────────────────────────────────

# Standard TTS presets  (also used by Multilingual tab — same param set)
PRESETS_TTS: dict[str, dict] = {
    "🎯 Default": {
        "description": "Balanced starting point — works well for most content out of the box.",
        "rationale_md": """\
**🎯 Default** — Balanced starting point for most content.

| Parameter | Value | Rationale |
|---|---|---|
| Exaggeration | **0.5** | Neutral — neither flat nor over-dramatic |
| CFG Weight | **0.5** | Balanced voice fidelity vs. natural pacing |
| Temperature | **0.8** | Mild variation — lively without being unstable |
| Rep Penalty | **1.2** | Light repetition suppression |
| Min-P | **0.05** | Filters very low-probability tokens |
| Top-P | **1.0** | Nucleus sampling disabled (full distribution) |
""",
        "sample_text": (
            "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's "
            "Nexus in an epic late-game pentakill. The crowd went wild as the final tower crumbled."
        ),
        "params": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8,
                   "rep_penalty": 1.2, "min_p": 0.05, "top_p": 1.0},
    },

    "📚 Audiobook": {
        "description": "Calm, consistent narration for long-form content. Stable across chapters.",
        "rationale_md": """\
**📚 Audiobook** — Long-form narration, consistent across paragraphs.

| Parameter | Value | Rationale |
|---|---|---|
| Exaggeration | **0.35** | Neutral tone — avoids listener fatigue over long passages |
| CFG Weight | **0.65** | Stays faithful to the reference voice across chapters |
| Temperature | **0.65** | Low variation — keeps voice stable and predictable |
| Rep Penalty | **1.40** | Aggressive anti-repetition for long documents |
| Min-P | **0.05** | Standard filtering |
| Top-P | **1.00** | Full distribution — let rep_penalty do the work |

> **Key insight:** Low temperature is the most important knob for audiobooks —
> it ensures paragraph 50 sounds like paragraph 1.
""",
        "sample_text": (
            "It was the best of times, it was the worst of times, it was the age of wisdom, "
            "it was the age of foolishness, it was the epoch of belief, "
            "it was the epoch of incredulity, it was the season of Light."
        ),
        "params": {"exaggeration": 0.35, "cfg_weight": 0.65, "temperature": 0.65,
                   "rep_penalty": 1.40, "min_p": 0.05, "top_p": 1.0},
    },

    "📰 News Broadcast": {
        "description": "Clear, authoritative, measured pace. Professional journalism delivery.",
        "rationale_md": """\
**📰 News Broadcast** — Professional broadcast delivery.

| Parameter | Value | Rationale |
|---|---|---|
| Exaggeration | **0.40** | Minimal emotion — journalistic neutrality |
| CFG Weight | **0.70** | Locks in the broadcaster's measured cadence |
| Temperature | **0.45** | Highly consistent — predictable across takes |
| Rep Penalty | **1.30** | Prevents redundant phrasing in dense copy |
| Min-P | **0.06** | Slightly tighter filtering for clean delivery |
| Top-P | **0.90** | Mild nucleus narrowing for crisp word choices |

> **Key insight:** `cfg_weight=0.70` acts as a pace anchor — it keeps
> the delivery measured even if the reference speaker talks fast.
""",
        "sample_text": (
            "Breaking: scientists have announced a major breakthrough in renewable energy technology, "
            "promising to reduce global carbon emissions by thirty percent within the next decade. "
            "Markets responded positively to the news."
        ),
        "params": {"exaggeration": 0.40, "cfg_weight": 0.70, "temperature": 0.45,
                   "rep_penalty": 1.30, "min_p": 0.06, "top_p": 0.90},
    },

    "💬 Conversational": {
        "description": "Natural, relaxed, everyday speech. Warm and spontaneous-sounding.",
        "rationale_md": """\
**💬 Conversational** — Natural everyday speech, relaxed and warm.

| Parameter | Value | Rationale |
|---|---|---|
| Exaggeration | **0.55** | Slightly above neutral — adds warmth without drama |
| CFG Weight | **0.45** | Looser pacing — sounds less scripted |
| Temperature | **0.90** | Enough variation to feel spontaneous |
| Rep Penalty | **1.20** | Light — permits natural speech patterns |
| Min-P | **0.05** | Standard |
| Top-P | **1.00** | Full distribution for natural word choices |

> **Key insight:** Lowering `cfg_weight` below 0.5 is what makes
> the speech sound casual rather than "read from a script."
""",
        "sample_text": (
            "Hey, so I was thinking — what if we grab coffee this afternoon? "
            "I know this great place downtown, they have the best lattes "
            "and the vibe is super chill. You in?"
        ),
        "params": {"exaggeration": 0.55, "cfg_weight": 0.45, "temperature": 0.90,
                   "rep_penalty": 1.20, "min_p": 0.05, "top_p": 1.0},
    },

    "🎭 Dramatic": {
        "description": "High-emotion, theatrical delivery. Strong expressive range.",
        "rationale_md": """\
**🎭 Dramatic** — High-emotion, theatrical delivery.

| Parameter | Value | Rationale |
|---|---|---|
| Exaggeration | **0.80** | Amplifies emotional intensity (README: 0.7+ for expressive) |
| CFG Weight | **0.30** | README's recommended value for dramatic speech — frees pacing |
| Temperature | **0.85** | Moderate variation — expressive but controlled |
| Rep Penalty | **1.20** | Standard |
| Min-P | **0.03** | Slightly open — allows bold word choices |
| Top-P | **1.00** | Full distribution for dramatic range |

> **Key insight:** The README explicitly recommends `cfg_weight ≈ 0.3` for
> dramatic content — it counteracts the pace acceleration caused by high exaggeration.
""",
        "sample_text": (
            "You want answers? You can't handle the truth! "
            "Son, we live in a world that has walls, and those walls have to be guarded "
            "by people with guns. Who's gonna do it — you?"
        ),
        "params": {"exaggeration": 0.80, "cfg_weight": 0.30, "temperature": 0.85,
                   "rep_penalty": 1.20, "min_p": 0.03, "top_p": 1.0},
    },

    "📣 Advertisement": {
        "description": "Energetic, persuasive copy. Upbeat, punchy delivery.",
        "rationale_md": """\
**📣 Advertisement** — Energetic, persuasive copy.

| Parameter | Value | Rationale |
|---|---|---|
| Exaggeration | **0.70** | Upbeat and engaging without overdoing it |
| CFG Weight | **0.40** | Slightly loose — natural flow for ad pacing |
| Temperature | **0.80** | Consistent yet lively |
| Rep Penalty | **1.30** | Prevents repetitive phrasing in short copy |
| Min-P | **0.05** | Standard |
| Top-P | **0.95** | Slight nucleus narrowing for punchy word choices |

> **Key insight:** Exaggeration 0.70 + CFG 0.40 is the "ad sweet spot" —
> energetic delivery without sounding artificially dramatic.
""",
        "sample_text": (
            "Introducing the all-new SmartLife Pro — the device that changes everything. "
            "Smarter. Faster. Designed for the way you live today. "
            "Available now at your nearest retailer."
        ),
        "params": {"exaggeration": 0.70, "cfg_weight": 0.40, "temperature": 0.80,
                   "rep_penalty": 1.30, "min_p": 0.05, "top_p": 0.95},
    },

    "🎓 E-Learning": {
        "description": "Clear, educational delivery. Measured pace, easy to follow.",
        "rationale_md": """\
**🎓 E-Learning** — Clear, educational delivery.

| Parameter | Value | Rationale |
|---|---|---|
| Exaggeration | **0.45** | Warm but focused — engaging without being distracting |
| CFG Weight | **0.60** | Consistent voice identity throughout the course |
| Temperature | **0.55** | Stable and easy to listen to repeatedly |
| Rep Penalty | **1.40** | Critical for long lessons — prevents artifact repetition |
| Min-P | **0.07** | Tighter filtering for crisp, clear delivery |
| Top-P | **0.92** | Narrow nucleus — clean, deliberate word choices |

> **Key insight:** High `rep_penalty` (1.40) is essential for e-learning —
> students often replay segments, and repetition artifacts stand out.
""",
        "sample_text": (
            "In today's lesson we'll explore the fundamental principles of quantum mechanics. "
            "Pay close attention, as these concepts form the foundation of modern physics "
            "and will appear throughout the rest of this course."
        ),
        "params": {"exaggeration": 0.45, "cfg_weight": 0.60, "temperature": 0.55,
                   "rep_penalty": 1.40, "min_p": 0.07, "top_p": 0.92},
    },

    "🎮 Game Character": {
        "description": "Distinctive NPC voice. Expressive, varied, high personality.",
        "rationale_md": """\
**🎮 Game Character / NPC** — Distinctive character voice.

| Parameter | Value | Rationale |
|---|---|---|
| Exaggeration | **0.75** | Strong character personality |
| CFG Weight | **0.35** | Minimal reference constraint — voice diverges for character |
| Temperature | **0.95** | High variation — each line sounds different |
| Rep Penalty | **1.20** | Standard |
| Min-P | **0.02** | Very open — allows unexpected, characterful word choices |
| Top-P | **1.00** | Full distribution for maximum character range |

> **Key insight:** Low `cfg_weight` + high `exaggeration` = the character
> "takes over" the reference voice. Great for fictional voices.
""",
        "sample_text": (
            "Hah! You dare challenge me? I've defeated a thousand warriors stronger than you. "
            "Draw your sword — let's see what you're really made of. "
            "This should be entertaining."
        ),
        "params": {"exaggeration": 0.75, "cfg_weight": 0.35, "temperature": 0.95,
                   "rep_penalty": 1.20, "min_p": 0.02, "top_p": 1.0},
    },

    "🧘 Meditation / ASMR": {
        "description": "Soft, slow, intimate delivery. Minimal pace, maximum calm.",
        "rationale_md": """\
**🧘 Meditation / ASMR** — Soft, slow, calming delivery.

| Parameter | Value | Rationale |
|---|---|---|
| Exaggeration | **0.25** | Floor value — prevents any pace acceleration |
| CFG Weight | **0.75** | Very faithful to reference voice — keeps it intimate |
| Temperature | **0.40** | Very stable, slow, deliberate word-by-word delivery |
| Rep Penalty | **1.20** | Standard |
| Min-P | **0.08** | Tighter filtering — gentle, careful word selection |
| Top-P | **0.88** | Narrow nucleus — avoids jarring word choices |

> **Key insight:** `exaggeration=0.25` (minimum) is critical here —
> it prevents the model from speeding up, which would break the meditative flow.
""",
        "sample_text": (
            "Take a deep breath in... and slowly let it go. "
            "Feel your body relax with each exhale. "
            "There is nowhere you need to be right now. "
            "You are exactly where you need to be."
        ),
        "params": {"exaggeration": 0.25, "cfg_weight": 0.75, "temperature": 0.40,
                   "rep_penalty": 1.20, "min_p": 0.08, "top_p": 0.88},
    },

    "🔬 Experimental": {
        "description": "Maximum expressiveness. Unpredictable and creative — may be unstable.",
        "rationale_md": """\
**🔬 Experimental** — Maximum expressiveness. Use with caution.

| Parameter | Value | Rationale |
|---|---|---|
| Exaggeration | **1.00** | Strong emotional amplification |
| CFG Weight | **0.20** | Near-unconstrained — model runs free |
| Temperature | **1.20** | High variation — unique every run |
| Rep Penalty | **1.10** | Low — allows creative repetition for effect |
| Min-P | **0.01** | Almost fully open token selection |
| Top-P | **1.00** | Full distribution |

> ⚠️ **Warning:** Outputs can be unstable, mispaced, or unusual.
> That's the point — use for creative prototyping, not production.
""",
        "sample_text": (
            "The crimson sunset melted into the horizon, "
            "as if the sky itself was bleeding out its last remaining light. "
            "Everything felt uncertain, beautiful, and terrifying all at once."
        ),
        "params": {"exaggeration": 1.00, "cfg_weight": 0.20, "temperature": 1.20,
                   "rep_penalty": 1.10, "min_p": 0.01, "top_p": 1.0},
    },
}

# Turbo TTS presets  (exaggeration + cfg_weight are not supported by Turbo)
PRESETS_TURBO: dict[str, dict] = {
    "🎯 Default": {
        "description": "Balanced starting point for Turbo. Good for general use.",
        "rationale_md": """\
**🎯 Default** — Balanced Turbo starting point.

| Parameter | Value | Rationale |
|---|---|---|
| Temperature | **0.80** | Natural variation |
| Top-K | **1000** | Wide candidate pool |
| Top-P | **0.95** | Standard nucleus sampling |
| Rep Penalty | **1.20** | Light repetition suppression |
| Norm Loudness | **✓** | Normalises reference to −27 LUFS |
""",
        "sample_text": (
            "Hi there, Sarah here from MochaFone calling you back [chuckle]. "
            "Have you got one minute to chat about the billing issue? "
            "I know it looked confusing. [laugh] Don't worry, we'll sort it out."
        ),
        "params": {"temperature": 0.80, "top_k": 1000, "top_p": 0.95,
                   "rep_penalty": 1.20, "min_p": 0.0, "norm_loudness": True},
    },

    "🤖 Voice Agent": {
        "description": "Production-ready voice agent. Crisp, clear, professional.",
        "rationale_md": """\
**🤖 Voice Agent** — Production voice agent or IVR assistant.

| Parameter | Value | Rationale |
|---|---|---|
| Temperature | **0.60** | Consistent, predictable — critical for live agents |
| Top-K | **200** | Narrow pool → reliable word choices |
| Top-P | **0.90** | Focused nucleus for professional vocabulary |
| Rep Penalty | **1.30** | Prevents repeated phrases in long responses |
| Norm Loudness | **✓** | Consistent call-centre volume level |

> **Key insight:** `top_k=200` is the most important setting here —
> it hard-caps creativity, producing reliable telephony-quality output.
""",
        "sample_text": (
            "Thank you for calling customer support. I'm here to help you today. "
            "Could you please provide your account number "
            "so I can pull up your details?"
        ),
        "params": {"temperature": 0.60, "top_k": 200, "top_p": 0.90,
                   "rep_penalty": 1.30, "min_p": 0.0, "norm_loudness": True},
    },

    "🎙️ Podcast Host": {
        "description": "Warm, engaging interview style. Natural with personality.",
        "rationale_md": """\
**🎙️ Podcast Host** — Warm, engaging, spontaneous-sounding.

| Parameter | Value | Rationale |
|---|---|---|
| Temperature | **0.85** | Lively, more spontaneous-sounding than default |
| Top-K | **800** | Good variety for an engaging host voice |
| Top-P | **0.95** | Natural flow |
| Rep Penalty | **1.20** | Permits natural speech patterns |
| Norm Loudness | **✓** | Consistent listening level across episodes |
""",
        "sample_text": (
            "Welcome back to the show, everyone! [chuckle] "
            "We've got an incredible guest joining us today, "
            "and trust me, you do not want to miss what they have to say."
        ),
        "params": {"temperature": 0.85, "top_k": 800, "top_p": 0.95,
                   "rep_penalty": 1.20, "min_p": 0.0, "norm_loudness": True},
    },

    "🧙 Character / NPC": {
        "description": "Distinctive character voice. Expressive and highly varied.",
        "rationale_md": """\
**🧙 Character / NPC** — Distinctive, expressive character voice.

| Parameter | Value | Rationale |
|---|---|---|
| Temperature | **1.05** | Strong variation — each line has character |
| Top-K | **2000** | Maximum candidate pool → widest creative range |
| Top-P | **1.00** | Fully open distribution |
| Rep Penalty | **1.15** | Low — allows deliberate repetition for character effect |
| Norm Loudness | **✗** | OFF — preserves dynamic range for dramatic intensity |

> **Key insight:** `top_k=2000` (maximum) is the Turbo equivalent of lowering
> `cfg_weight` in Standard — it opens up the model's creative range.
> `norm_loudness=False` preserves dramatic dynamics that −27 LUFS would flatten.
""",
        "sample_text": (
            "Well, well, well... [chuckle] Look who finally decided to show up. "
            "I was starting to think you'd forgotten all about me. [sigh] "
            "How disappointing."
        ),
        "params": {"temperature": 1.05, "top_k": 2000, "top_p": 1.00,
                   "rep_penalty": 1.15, "min_p": 0.0, "norm_loudness": False},
    },

    "📻 Radio / Promo": {
        "description": "High-energy, upbeat delivery. Perfect for promos and short ads.",
        "rationale_md": """\
**📻 Radio / Promo** — High-energy, upbeat delivery.

| Parameter | Value | Rationale |
|---|---|---|
| Temperature | **0.90** | Energetic and dynamic |
| Top-K | **600** | Good variety — engaging without unpredictability |
| Top-P | **0.92** | Slight narrowing keeps delivery punchy |
| Rep Penalty | **1.30** | Avoids repetitive phrasing in short-form copy |
| Norm Loudness | **✓** | Consistent broadcast volume |
""",
        "sample_text": (
            "You're listening to the hottest hits of the morning! [laugh] "
            "Stay tuned — we've got amazing music, big giveaways, "
            "and all the news you need, coming right up."
        ),
        "params": {"temperature": 0.90, "top_k": 600, "top_p": 0.92,
                   "rep_penalty": 1.30, "min_p": 0.0, "norm_loudness": True},
    },

    "📞 IVR / Max Reliable": {
        "description": "Maximum consistency for IVR and automated phone systems.",
        "rationale_md": """\
**📞 IVR / Max Reliable** — Near-deterministic output for automated systems.

| Parameter | Value | Rationale |
|---|---|---|
| Temperature | **0.50** | Very stable — near-greedy selection |
| Top-K | **80** | Near-greedy — almost always picks the most likely token |
| Top-P | **0.85** | Tight nucleus — very clean vocabulary |
| Rep Penalty | **1.40** | Maximum safe value — telephony quality anti-repetition |
| Norm Loudness | **✓** | Critical for consistent phone-line volume |

> **Key insight:** `top_k=80` is the strongest reliability lever in Turbo.
> It's the equivalent of setting `temperature=0.1` — almost deterministic.
""",
        "sample_text": (
            "Your appointment has been confirmed for Thursday, March fifteenth "
            "at two thirty in the afternoon. "
            "Please arrive fifteen minutes early. "
            "Press one to reschedule."
        ),
        "params": {"temperature": 0.50, "top_k": 80, "top_p": 0.85,
                   "rep_penalty": 1.40, "min_p": 0.0, "norm_loudness": True},
    },
}

# Convenience lists for Gradio dropdowns
PRESET_TTS_NAMES:   list[str] = list(PRESETS_TTS.keys())
PRESET_TURBO_NAMES: list[str] = list(PRESETS_TURBO.keys())

# ──────────────────────────────────────────────────────────────────────────────
# Model Registry — metadata used by the Model Manager tab
# ──────────────────────────────────────────────────────────────────────────────
# check_file: a unique large weight file that is only present once the model
#             has been fully downloaded — used for disk-cache probing via
#             try_to_load_from_cache() (zero network, disk-only check).
# download_*: mirrors exactly what each model's from_pretrained() does, so
#             downloading here and loading later works without re-fetching.
MODEL_REGISTRY: dict[str, dict] = {
    "tts": {
        "display_name": "Standard TTS",
        "class_name":   "ChatterboxTTS",
        "description":  "English · zero-shot voice cloning · exaggeration & CFG controls",
        "params":       "500M",
        "size_gb":      1.4,
        "repo_id":      "ResembleAI/chatterbox",
        "check_file":   "t3_cfg.safetensors",
        "dl_mode":      "files",
        "dl_files":     ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors",
                         "tokenizer.json", "conds.pt"],
    },
    "turbo": {
        "display_name": "Turbo TTS",
        "class_name":   "ChatterboxTurboTTS",
        "description":  "English · 1-step decoder · paralinguistic tags · low VRAM",
        "params":       "350M",
        "size_gb":      2.9,
        "repo_id":      "ResembleAI/chatterbox-turbo",
        "check_file":   "t3_turbo_v1.safetensors",
        "dl_mode":      "snapshot",
        "dl_patterns":  ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
    },
    "multilingual": {
        "display_name": "Multilingual TTS",
        "class_name":   "ChatterboxMultilingualTTS",
        "description":  "23 languages · cross-language voice cloning",
        "params":       "500M",
        "size_gb":      1.5,
        "repo_id":      "ResembleAI/chatterbox",
        "check_file":   "t3_mtl23ls_v2.safetensors",
        "dl_mode":      "snapshot",
        "dl_patterns":  ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                         "grapheme_mtl_merged_expanded_v1.json", "conds.pt",
                         "Cangjie5_TC.json"],
    },
    "vc": {
        "display_name": "Voice Conversion",
        "class_name":   "ChatterboxVC",
        "description":  "Audio-to-audio · no text needed · voice identity swap",
        "params":       "—",
        "size_gb":      0.4,
        "repo_id":      "ResembleAI/chatterbox",
        "check_file":   "s3gen.safetensors",
        "dl_mode":      "files",
        "dl_files":     ["s3gen.safetensors", "conds.pt"],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# JavaScript — cursor-position-aware tag insertion (v2 gap fix)
# ──────────────────────────────────────────────────────────────────────────────
# When fn=None and js=... is set on .click(), Gradio executes this purely
# client-side with no server round-trip.
# Receives: (button_label_value, current_textbox_value)
# Returns:  new textbox value with tag inserted at cursor position
_INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#turbo_textbox textarea');
    if (!textarea) {
        // Fallback: append to end if textarea not found in DOM
        const t = current_text || '';
        return t + (t.endsWith(' ') || t === '' ? '' : ' ') + tag_val;
    }
    const start = textarea.selectionStart;
    const end   = textarea.selectionEnd;
    const t = current_text || '';
    // Add spaces around tag only when not already at a word boundary
    const prefix = (start === 0 || t[start - 1] === ' ') ? '' : ' ';
    const suffix = (end >= t.length  || t[end]     === ' ') ? '' : ' ';
    return t.slice(0, start) + prefix + tag_val + suffix + t.slice(end);
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Set global random seed for reproducible generation. 0 = skip (random)."""
    seed = int(seed)
    if seed == 0:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def split_sentences(text: str) -> list[str]:
    """Split text on sentence-ending punctuation for streaming generation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def to_audio_tuple(wav, sr: int) -> tuple[int, np.ndarray]:
    """Convert a wav tensor / numpy array to a Gradio-compatible (sr, int16_array) tuple.

    Gradio 6.x expects int16 audio.  Returning float32 triggers a UserWarning
    ('Trying to convert audio automatically from float32 to 16-bit int format')
    on every generation call.  We do the conversion ourselves, cleanly.
    """
    if torch.is_tensor(wav):
        arr = wav.squeeze().detach().cpu().numpy()
    else:
        arr = np.asarray(wav).squeeze()
    arr = arr.astype(np.float32)
    # Normalise to [-1, 1] — only scale down if peak exceeds 1.0
    peak = np.abs(arr).max()
    if peak > 1.0:
        arr = arr / peak
    # Convert to int16 (Gradio 6.x preferred format — avoids UserWarning)
    arr_int16 = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)
    return sr, arr_int16


def preprocess_ref_audio(path: str | None) -> str | None:
    """Trim reference audio to a multiple of 40 ms to prevent S3Gen's
    'Reference mel length != 2 * reference token length' warning.

    Root cause (chatterbox/models/s3gen/s3gen.py → S3Token2Mel.embed_ref):
    ─ Mel extractor runs at 24 kHz with hop = 480 samples → 50 frames/sec (20 ms/frame)
    ─ S3 tokenizer   runs at 16 kHz at 25 tokens/sec       → 40 ms/token  (2 frames/token)
    Invariant:  mel_len = 2 × token_len
    This holds exactly only when the audio duration is a multiple of 40 ms, i.e.:
      • 640 samples at 16 kHz   • 960 samples at 24 kHz   • 1 764 samples at 44.1 kHz

    Strategy: trim to the nearest complete 40 ms boundary at the *original* sample
    rate before handing the path to librosa/torchaudio inside the model.  Because the
    duration is then an exact rational multiple of 40 ms, the subsequent resampling to
    24 kHz and 16 kHz also lands on exact multiples of 960 and 640 samples respectively,
    so the mismatch condition in embed_ref() is never triggered.
    """
    if not path:
        return None
    try:
        wav, sr = torchaudio.load(path)
        # Samples that correspond to exactly 40 ms at this file's native rate
        frame_samples = round(sr * 0.040)
        if frame_samples <= 0 or wav.shape[-1] <= frame_samples:
            return path  # too short to be meaningful — let the model deal with it
        n_complete = wav.shape[-1] // frame_samples
        target_len = n_complete * frame_samples
        if target_len == wav.shape[-1]:
            return path  # already aligned — no temp file needed
        original_len = wav.shape[-1]
        wav = wav[..., :target_len]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(tmp.name, wav, sr)
        tmp.close()
        log.debug(
            "preprocess_ref_audio: trimmed %d → %d samples (%d ms removed) at %d Hz",
            original_len,
            target_len,
            round((original_len - target_len) / sr * 1000),
            sr,
        )
        return tmp.name
    except Exception as exc:
        log.debug("preprocess_ref_audio: skipped preprocessing (%s)", exc)
        return path  # fall back to original path unchanged


def get_loaded_models() -> str:
    """Return a human-readable string of currently cached models."""
    if not _MODEL_CACHE:
        return "None loaded yet — click Generate to lazy-load on first use."
    labels = {
        "tts": "Standard TTS",
        "turbo": "Turbo TTS",
        "multilingual": "Multilingual TTS",
        "vc": "Voice Conversion",
    }
    return "  |  ".join(f"✅ {labels.get(k, k)}" for k in _MODEL_CACHE)


def on_language_change(language: str):
    """
    v2 gap fix: when the language dropdown changes, auto-fill BOTH the sample
    text AND the reference audio with a curated native-speaker clip.
    """
    lang_code = language.split(" ")[0]
    text  = SAMPLE_TEXTS.get(language, SAMPLE_TEXTS["en - English"])
    audio = LANGUAGE_AUDIO_DEFAULTS.get(lang_code, None)
    return text, audio


def refresh_model_status() -> str:
    return get_loaded_models()


# ──────────────────────────────────────────────────────────────────────────────
# Preset Handlers
# ──────────────────────────────────────────────────────────────────────────────

def apply_preset_tts(preset_name: str) -> tuple:
    """Return updated slider values + sample text + rationale for a Standard/MTL preset.

    Output order matches the Gradio outputs list in the Standard TTS and
    Multilingual TTS tabs:
      (exaggeration, cfg_weight, temperature, rep_penalty, min_p, top_p,
       sample_text, rationale_md)
    """
    preset = PRESETS_TTS.get(preset_name)
    if preset is None:
        # Unknown preset — return current defaults unchanged (should not happen)
        return 0.5, 0.5, 0.8, 1.2, 0.05, 1.0, "", ""
    p = preset["params"]
    return (
        p["exaggeration"],
        p["cfg_weight"],
        p["temperature"],
        p["rep_penalty"],
        p["min_p"],
        p["top_p"],
        preset["sample_text"],
        preset["rationale_md"],
    )


def apply_preset_turbo(preset_name: str) -> tuple:
    """Return updated slider values + sample text + rationale for a Turbo preset.

    Output order:
      (temperature, top_k, top_p, rep_penalty, min_p, norm_loudness,
       sample_text, rationale_md)
    """
    preset = PRESETS_TURBO.get(preset_name)
    if preset is None:
        return 0.8, 1000, 0.95, 1.2, 0.0, True, "", ""
    p = preset["params"]
    return (
        p["temperature"],
        p["top_k"],
        p["top_p"],
        p["rep_penalty"],
        p["min_p"],
        p["norm_loudness"],
        preset["sample_text"],
        preset["rationale_md"],
    )


def apply_preset_mtl(preset_name: str) -> tuple:
    """Return updated slider values + rationale for the Multilingual preset.

    Multilingual uses the same param set as Standard TTS but does NOT update
    the text input — the language dropdown + sample-text button control that.

    Output order:
      (exaggeration, cfg_weight, temperature, rep_penalty, min_p, top_p,
       rationale_md)
    """
    preset = PRESETS_TTS.get(preset_name)
    if preset is None:
        return 0.5, 0.5, 0.8, 1.2, 0.05, 1.0, ""
    p = preset["params"]
    return (
        p["exaggeration"],
        p["cfg_weight"],
        p["temperature"],
        p["rep_penalty"],
        p["min_p"],
        p["top_p"],
        preset["rationale_md"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model Manager — disk cache, memory, download, load, unload
# ──────────────────────────────────────────────────────────────────────────────

def is_model_cached_on_disk(key: str) -> bool:
    """Return True if the model's primary weight file exists in the HF disk cache.

    Uses try_to_load_from_cache() — zero network calls, disk-only.
    Returns True only when the result is a path string (i.e. the file is present).
    None  → file never fetched (unknown)
    _CACHED_NO_EXIST → a prior download returned 404 (rare)
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        info   = MODEL_REGISTRY[key]
        result = try_to_load_from_cache(
            repo_id=info["repo_id"],
            filename=info["check_file"],
            repo_type="model",
        )
        return isinstance(result, str)
    except Exception:
        return False


def get_memory_stats() -> dict:
    """Collect system RAM and device (MPS / CUDA) memory statistics.

    Uses a 1.5-second TTL cache so it is safe to call on every Gradio render
    without hammering psutil on each keypress.

    On macOS always use vm.percent (= (total−available)/total) rather than
    (total−free)/total — free is almost always < 200 MB due to kernel caching.
    """
    import psutil
    import time as _time

    # TTL cache stored as a module-level attribute on this function
    now = _time.monotonic()
    cache = getattr(get_memory_stats, "_cache", None)
    if cache is not None and now - cache[0] < 1.5:
        return cache[1]

    vm  = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss

    stats: dict = {
        "sys_total_gb": round(vm.total     / 1024 ** 3, 1),
        "sys_used_gb":  round(vm.used      / 1024 ** 3, 1),
        "sys_avail_gb": round(vm.available / 1024 ** 3, 1),
        "sys_percent":  vm.percent,                           # use this, NOT total−free
        "proc_rss_gb":  round(rss          / 1024 ** 3, 2),  # this app's physical footprint
    }

    if DEVICE == "mps" and torch.backends.mps.is_available():
        # driver_allocated includes the allocator pool; matches Activity Monitor
        stats["mps_driver_gb"] = round(
            torch.mps.driver_allocated_memory() / 1024 ** 3, 2
        )
        stats["mps_max_gb"] = round(
            torch.mps.recommended_max_memory() / 1024 ** 3, 1
        )
    elif DEVICE == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        stats["cuda_alloc_gb"] = round(torch.cuda.memory_allocated() / 1024 ** 3, 2)
        stats["cuda_total_gb"] = round(props.total_memory            / 1024 ** 3, 1)
        stats["cuda_name"]     = props.name

    get_memory_stats._cache = (now, stats)
    return stats


def unload_model_from_memory(key: str) -> str:
    """Remove a model from the in-process cache and flush device memory.

    Five-step recipe (confirmed empirically on MPS):
      1. model.cpu()          — move tensors off device before deletion
      2. del _MODEL_CACHE[key] — drop Python reference (current_allocated → 0)
      3. gc.collect()          — break any reference cycles in nn.Module
      4. synchronize()         — wait for pending Metal / CUDA kernels
      5. empty_cache()         — flush the allocator pool back to the driver
    Steps 4+5 are required on MPS; del + gc alone leave the pool full.
    """
    import gc

    info = MODEL_REGISTRY[key]
    if key not in _MODEL_CACHE:
        return f"ℹ️  {info['display_name']} is not currently loaded in memory."

    model = _MODEL_CACHE.pop(key)

    # Step 1 — move all parameters off the accelerator before deletion
    try:
        model.cpu()
    except Exception:
        pass

    # Step 2+3 — drop reference and break cycles
    del model
    gc.collect()

    # Step 4+5 — flush device allocator pool
    if DEVICE == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    log.info("Model '%s' unloaded and device memory flushed.", key)
    return f"✅  {info['display_name']} unloaded — device memory flushed."


def render_manager_html() -> str:
    """Build the HTML status panel shown at the top of the Model Manager tab.

    Shows two memory bars (system RAM + device) and a per-model status table
    with disk-cache and in-memory badges.  Called on every refresh / action.
    """
    stats = get_memory_stats()

    # ── Memory gauges ─────────────────────────────────────────────────────────
    sys_pct = stats["sys_percent"]
    bar_col = (
        "#22c55e" if sys_pct < 60
        else "#f59e0b" if sys_pct < 80
        else "#ef4444"
    )

    html = f"""<div style="font-family:Inter,ui-sans-serif,sans-serif;font-size:13px;line-height:1.5;">
  <div style="margin-bottom:14px;">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
      <span style="font-weight:600;min-width:130px;">System RAM</span>
      <div style="flex:1;background:#e5e7eb;border-radius:5px;height:12px;overflow:hidden;">
        <div style="background:{bar_col};width:{sys_pct:.0f}%;height:100%;border-radius:5px;"></div>
      </div>
      <span style="color:#6b7280;white-space:nowrap;min-width:240px;">
        {stats['sys_used_gb']:.1f} / {stats['sys_total_gb']:.1f} GB
        &nbsp;({sys_pct:.0f}%)
        &nbsp;·&nbsp; this app: {stats['proc_rss_gb']:.2f} GB
      </span>
    </div>"""

    if "mps_driver_gb" in stats:
        mps_gb  = stats["mps_driver_gb"]
        mps_max = stats["mps_max_gb"]
        mps_pct = min(mps_gb / mps_max * 100, 100) if mps_max else 0
        mps_col = "#22c55e" if mps_pct < 60 else "#f59e0b" if mps_pct < 80 else "#ef4444"
        html += f"""
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="font-weight:600;min-width:130px;">MPS (unified)</span>
      <div style="flex:1;background:#e5e7eb;border-radius:5px;height:12px;overflow:hidden;">
        <div style="background:{mps_col};width:{mps_pct:.0f}%;height:100%;border-radius:5px;"></div>
      </div>
      <span style="color:#6b7280;white-space:nowrap;min-width:240px;">
        {mps_gb:.2f} / {mps_max:.1f} GB driver pool
        &nbsp;·&nbsp; recommended max: {mps_max:.1f} GB
      </span>
    </div>"""
    elif "cuda_alloc_gb" in stats:
        cu_gb  = stats["cuda_alloc_gb"]
        cu_tot = stats["cuda_total_gb"]
        cu_pct = min(cu_gb / cu_tot * 100, 100) if cu_tot else 0
        cu_col = "#22c55e" if cu_pct < 60 else "#f59e0b" if cu_pct < 80 else "#ef4444"
        html += f"""
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="font-weight:600;min-width:130px;">{stats.get('cuda_name','CUDA')}</span>
      <div style="flex:1;background:#e5e7eb;border-radius:5px;height:12px;overflow:hidden;">
        <div style="background:{cu_col};width:{cu_pct:.0f}%;height:100%;border-radius:5px;"></div>
      </div>
      <span style="color:#6b7280;white-space:nowrap;">{cu_gb:.2f} / {cu_tot:.1f} GB</span>
    </div>"""

    html += "\n  </div>"

    # ── Per-model status table ─────────────────────────────────────────────────
    html += """
  <table style="width:100%;border-collapse:collapse;font-size:13px;">
    <thead>
      <tr style="background:#f9fafb;border-bottom:2px solid #e5e7eb;">
        <th style="text-align:left;padding:8px 10px;font-weight:600;">Model</th>
        <th style="text-align:center;padding:8px;font-weight:600;">Params</th>
        <th style="text-align:center;padding:8px;font-weight:600;">Size</th>
        <th style="text-align:center;padding:8px;font-weight:600;">Disk Cache</th>
        <th style="text-align:center;padding:8px;font-weight:600;">In Memory</th>
      </tr>
    </thead>
    <tbody>"""

    for key, info in MODEL_REGISTRY.items():
        in_mem  = key in _MODEL_CACHE
        on_disk = is_model_cached_on_disk(key)

        mem_badge = (
            '<span style="background:#dcfce7;color:#166534;padding:2px 10px;'
            'border-radius:12px;font-size:11px;font-weight:600;">✅ Loaded</span>'
            if in_mem else
            '<span style="background:#f3f4f6;color:#9ca3af;padding:2px 10px;'
            'border-radius:12px;font-size:11px;">— Unloaded</span>'
        )
        disk_badge = (
            '<span style="background:#dbeafe;color:#1e40af;padding:2px 10px;'
            'border-radius:12px;font-size:11px;">💾 Cached</span>'
            if on_disk else
            '<span style="background:#fef9c3;color:#92400e;padding:2px 10px;'
            'border-radius:12px;font-size:11px;">⬇ Not Downloaded</span>'
        )

        html += f"""
      <tr style="border-bottom:1px solid #f3f4f6;">
        <td style="padding:10px 10px;">
          <strong>{info['display_name']}</strong>
          <span style="color:#9ca3af;font-size:11px;margin-left:6px;">{info['class_name']}</span><br>
          <span style="color:#6b7280;font-size:11px;">{info['description']}</span>
        </td>
        <td style="text-align:center;padding:10px;color:#6b7280;">{info['params']}</td>
        <td style="text-align:center;padding:10px;color:#6b7280;">~{info['size_gb']:.1f} GB</td>
        <td style="text-align:center;padding:10px;">{disk_badge}</td>
        <td style="text-align:center;padding:10px;">{mem_badge}</td>
      </tr>"""

    html += "\n    </tbody>\n  </table>\n</div>"
    return html


def manager_do_load(key: str) -> tuple[str, str]:
    """Load a model into memory. Returns (updated html, log message)."""
    info = MODEL_REGISTRY[key]
    if key in _MODEL_CACHE:
        return render_manager_html(), f"ℹ️  {info['display_name']} is already loaded."
    model, err = get_model(key)
    if err:
        return render_manager_html(), f"❌  Load failed — {err}"
    return render_manager_html(), f"✅  {info['display_name']} loaded into memory."


def manager_do_unload(key: str) -> tuple[str, str]:
    """Unload a model from memory. Returns (updated html, log message)."""
    msg = unload_model_from_memory(key)
    return render_manager_html(), msg


def manager_do_download(key: str):
    """Download model weights to HF disk cache without loading into memory.

    Replicates the exact hf_hub_download / snapshot_download calls that each
    model's from_pretrained() makes, so loading afterwards requires no
    additional network access.

    Yields (updated_html, log_line) tuples so Gradio can stream progress.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    info = MODEL_REGISTRY[key]

    yield render_manager_html(), (
        f"⬇️  Starting download: {info['display_name']} (~{info['size_gb']:.1f} GB)\n"
        "Files will be saved to ~/.cache/huggingface/hub/"
    )

    try:
        if info["dl_mode"] == "files":
            files = info["dl_files"]
            for i, fname in enumerate(files, 1):
                yield render_manager_html(), f"  [{i}/{len(files)}] fetching {fname} …"
                hf_hub_download(
                    repo_id=info["repo_id"],
                    filename=fname,
                    token=os.environ.get("HF_TOKEN"),
                )

        elif info["dl_mode"] == "snapshot":
            yield render_manager_html(), "  Fetching snapshot (all model files) …"
            snapshot_download(
                repo_id=info["repo_id"],
                allow_patterns=info["dl_patterns"],
                token=os.environ.get("HF_TOKEN"),
            )

        yield render_manager_html(), (
            f"✅  {info['display_name']} download complete.\n"
            "Click ⬆️ Load to bring it into memory."
        )

    except Exception as exc:
        yield render_manager_html(), f"❌  Download failed: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Generation Handlers
# ──────────────────────────────────────────────────────────────────────────────

def generate_tts(
    text: str,
    ref_audio,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    rep_penalty: float,
    min_p: float,
    top_p: float,
    streaming: bool,
    seed: int,
):
    """Standard ChatterboxTTS — English with full parameter control."""
    if not text.strip():
        gr.Warning("Please enter some text to synthesize.")
        return

    set_seed(seed)  # v2: reproducible generation
    model, err = get_model("tts")
    if err:
        raise gr.Error(f"Standard TTS failed to load: {err}")

    gen_kwargs = dict(
        audio_prompt_path=preprocess_ref_audio(ref_audio),
        exaggeration=float(exaggeration),
        cfg_weight=float(cfg_weight),
        temperature=float(temperature),
        repetition_penalty=float(rep_penalty),
        min_p=float(min_p),
        top_p=float(top_p),
    )

    if streaming:
        sentences = split_sentences(text) or [text]
        buf: list[np.ndarray] = []
        for sentence in sentences:
            wav = model.generate(sentence, **gen_kwargs)
            buf.append(wav.squeeze().cpu().numpy())
            yield to_audio_tuple(np.concatenate(buf), model.sr)
    else:
        wav = model.generate(text, **gen_kwargs)
        yield to_audio_tuple(wav, model.sr)


def generate_turbo(
    text: str,
    ref_audio,
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    min_p: float,
    norm_loudness: bool,
    streaming: bool,
    seed: int,
):
    """ChatterboxTurboTTS — fast, low-VRAM, native paralinguistic tags."""
    if not text.strip():
        gr.Warning("Please enter some text to synthesize.")
        return

    set_seed(seed)  # v2: reproducible generation
    model, err = get_model("turbo")
    if err:
        raise gr.Error(f"Turbo TTS failed to load: {err}")

    gen_kwargs = dict(
        audio_prompt_path=preprocess_ref_audio(ref_audio),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        repetition_penalty=float(rep_penalty),
        min_p=float(min_p),   # model accepts but ignores (logs warning) — exposed for transparency
        norm_loudness=norm_loudness,
    )

    try:
        if streaming:
            sentences = split_sentences(text) or [text]
            buf: list[np.ndarray] = []
            for sentence in sentences:
                wav = model.generate(sentence, **gen_kwargs)
                buf.append(wav.squeeze().cpu().numpy())
                yield to_audio_tuple(np.concatenate(buf), model.sr)
        else:
            wav = model.generate(text, **gen_kwargs)
            yield to_audio_tuple(wav, model.sr)
    except AssertionError as exc:
        raise gr.Error(
            f"Turbo error: {exc}  →  Reference audio must be longer than 5 seconds."
        )


def generate_multilingual(
    text: str,
    language: str,
    ref_audio,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    rep_penalty: float,
    min_p: float,
    top_p: float,
    streaming: bool,
    seed: int,
):
    """ChatterboxMultilingualTTS — 23 languages with zero-shot voice cloning."""
    if not text.strip():
        gr.Warning("Please enter some text to synthesize.")
        return

    set_seed(seed)  # v2: reproducible generation
    model, err = get_model("multilingual")
    if err:
        raise gr.Error(f"Multilingual TTS failed to load: {err}")

    lang_code = language.split(" ")[0]  # "fr - French" → "fr"

    gen_kwargs = dict(
        language_id=lang_code,
        audio_prompt_path=preprocess_ref_audio(ref_audio),
        exaggeration=float(exaggeration),
        cfg_weight=float(cfg_weight),
        temperature=float(temperature),
        repetition_penalty=float(rep_penalty),
        min_p=float(min_p),
        top_p=float(top_p),
    )

    if streaming:
        sentences = split_sentences(text) or [text]
        buf: list[np.ndarray] = []
        for sentence in sentences:
            wav = model.generate(sentence, **gen_kwargs)
            buf.append(wav.squeeze().cpu().numpy())
            yield to_audio_tuple(np.concatenate(buf), model.sr)
    else:
        wav = model.generate(text, **gen_kwargs)
        yield to_audio_tuple(wav, model.sr)


def generate_vc(source_audio, target_voice):
    """ChatterboxVC — convert source audio to sound like the target voice."""
    if not source_audio:
        gr.Warning("Please upload a source audio file.")
        return None
    if not target_voice:
        gr.Warning("Please upload a target voice reference.")
        return None

    model, err = get_model("vc")
    if err:
        raise gr.Error(f"Voice Conversion failed to load: {err}")

    wav = model.generate(audio=source_audio, target_voice_path=preprocess_ref_audio(target_voice))
    return to_audio_tuple(wav, model.sr)


def check_watermark(audio_path: str) -> str:
    """Detect PerTh watermark in audio. Returns human-readable result string."""
    if not audio_path:
        return "⚠️  No audio uploaded."

    # When the open-source perth is active (no-op), detection is meaningless.
    if not _WATERMARK_AVAILABLE:
        return (
            "⚠️  WATERMARK DETECTION UNAVAILABLE\n\n"
            "The open-source `resemble-perth` package does not include the full "
            "PerTh watermark detector.\n\n"
            "The neural watermarker (PerthImplicitWatermarker) is None in this "
            "environment, so watermarks are neither embedded in generated audio "
            "nor detectable here.\n\n"
            "To enable watermarking, obtain a full PerTh licence from Resemble AI."
        )

    try:
        import librosa
        import perth

        audio, sr = librosa.load(audio_path, sr=None)
        wm = perth.PerthImplicitWatermarker()
        score = float(wm.get_watermark(audio, sample_rate=sr))

        if score >= 0.9:
            verdict = "✅  WATERMARK DETECTED"
            detail  = "This audio was generated by Chatterbox (Resemble AI)."
        elif score <= 0.1:
            verdict = "❌  NO WATERMARK"
            detail  = "This audio does not appear to be Chatterbox-generated."
        else:
            verdict = "⚠️  INCONCLUSIVE"
            detail  = "Partial or degraded watermark signal detected."

        return f"{verdict}\n\nScore: {score:.4f}  (1.0 = watermarked · 0.0 = clean)\n\n{detail}"
    except Exception as exc:
        return f"Error during watermark check:\n{exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI — styles
# ──────────────────────────────────────────────────────────────────────────────
_CSS = """
.tab-nav button           { font-size: 15px !important; }
.tag-btn                  { min-width: 115px !important; font-size: 12px !important; }
.accordion-content        { padding: 8px !important; }
.status-bar textarea      { font-size: 12px !important; color: #888 !important; }
footer                    { display: none !important; }
"""

_THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
)

# Shared tooltip strings
_D_EXAG = "Emotional intensity. 0.5 = neutral · 0.7 = expressive · 1.5–2.0 = dramatic/unstable"
_D_CFG  = "Voice clone fidelity. 0 = ignore accent · 0.5 = balanced · 1 = strict clone"
_D_TEMP = "Randomness. Low = stable/consistent · High = creative/varied (may introduce errors)"
_D_REP  = "Suppresses repeated words/tokens during generation"
_D_MINP = "Min-P sampler. 0 = disabled. 0.02–0.1 recommended at high temperatures"
_D_TOPP = "Nucleus sampling cutoff. 1.0 = disabled (recommended default)"
_D_TOPK = "Top-K token pool size. Turbo-specific"
_D_SEED = "0 = random each run. Any other integer → fully reproducible output"

# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI — layout
# ──────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Chatterbox TTS Explorer") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown(f"""
# 🎙️ Chatterbox TTS Explorer
**Resemble AI Chatterbox** — all capabilities · all parameters · no code editing required.

> 🖥️ Device: **{DEVICE.upper()}** &nbsp;|&nbsp;
> 📦 Models: lazy-loaded from HuggingFace on first use &nbsp;|&nbsp;
> {"🔏 All outputs carry an invisible PerTh AI watermark" if _WATERMARK_AVAILABLE else "⚠️ PerTh watermarker unavailable (open-source resemble-perth) — outputs are NOT watermarked"}
{"&nbsp;|&nbsp; ⚠️ **CPU detected — generation will be slow. GPU/MPS strongly recommended.**" if DEVICE == "cpu" else ""}
""")

    with gr.Row():
        gr.HTML(
            value=(
                "<div style='font-size:12px;color:#888;padding:4px 0;'>"
                "💡 <b>Memory tip:</b> Use the <b>🗂️ Model Manager</b> tab to "
                "preload, unload, or download models individually — important on "
                "devices with limited RAM. Models load lazily on first Generate click."
                "</div>"
            )
        )

    gr.Markdown("---")

    with gr.Tabs():

        # ── Tab 1: Standard TTS ───────────────────────────────────────────────
        with gr.Tab("🗣️ Standard TTS"):
            gr.Markdown("""
**`ChatterboxTTS`** — 500M params · English · Zero-shot voice cloning · Full creative controls

- Leave *Voice Reference* empty to use the built-in default voice.
- Upload a **≥ 10 s clean WAV** for best zero-shot cloning quality.
""")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    # ── Preset selector ───────────────────────────────────────
                    with gr.Group():
                        with gr.Row():
                            tts_preset = gr.Dropdown(
                                label="🎛️ Preset",
                                choices=PRESET_TTS_NAMES,
                                value="🎯 Default",
                                scale=2,
                                info="Select a use-case preset — auto-fills text and all parameters",
                            )
                        tts_rationale = gr.Markdown(
                            value=PRESETS_TTS["🎯 Default"]["rationale_md"],
                            label="",
                        )

                    tts_text = gr.Textbox(
                        label="📝 Input Text",
                        lines=5,
                        placeholder="Type anything to synthesize …",
                        value=PRESETS_TTS["🎯 Default"]["sample_text"],
                    )
                    tts_ref = gr.Audio(
                        label="🎤 Voice Reference  (optional — for zero-shot cloning)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )

                    with gr.Accordion("⚙️ Generation Parameters  (auto-filled by preset · adjustable)", open=True):
                        with gr.Row():
                            # v2: exaggeration extended to 2.0 — official app + demo page show 0.5 / 1.0 / 2.0
                            tts_exag = gr.Slider(0.25, 2.0, value=0.5, step=0.05,
                                                 label="Exaggeration", info=_D_EXAG)
                            tts_cfg  = gr.Slider(0.0,  1.0,  value=0.5, step=0.05,
                                                 label="CFG Weight",   info=_D_CFG)
                        with gr.Row():
                            # v2: temperature range 0.05–5.0 (matches official standard TTS app)
                            tts_temp = gr.Slider(0.05, 5.0, value=0.8, step=0.05,
                                                 label="Temperature",       info=_D_TEMP)
                            tts_rep  = gr.Slider(1.0,  2.0, value=1.2, step=0.05,
                                                 label="Repetition Penalty", info=_D_REP)
                        with gr.Row():
                            tts_minp = gr.Slider(0.0, 1.0, value=0.05, step=0.01,
                                                 label="Min-P",  info=_D_MINP)
                            tts_topp = gr.Slider(0.0, 1.0, value=1.0,  step=0.01,
                                                 label="Top-P",  info=_D_TOPP)
                        # v2: random seed control
                        tts_seed = gr.Number(value=0, label="🎲 Seed  (0 = random)",
                                             info=_D_SEED, precision=0, minimum=0)

                    tts_stream = gr.Checkbox(
                        label="🌊 Stream  (sentence-by-sentence — audio updates progressively)",
                        value=False,
                    )
                    tts_btn = gr.Button("🎙️ Generate", variant="primary", size="lg")

                with gr.Column(scale=2):
                    tts_out = gr.Audio(label="🔊 Generated Audio", type="numpy", interactive=False)
                    gr.Markdown("""
### 💡 Parameter Quick Reference
| Setting | Value | Effect |
|---|---|---|
| Exaggeration | 0.5 | Natural, neutral read |
| Exaggeration | 1.0 | Noticeably more expressive |
| Exaggeration | 2.0 | Dramatic — may be unstable |
| CFG Weight | 0.8 | Strict voice clone fidelity |
| CFG Weight | 0.3 | Slower, more natural pacing |
| CFG Weight | 0.0 | Ignore reference accent |
| Temperature | 0.3 | Consistent, stable output |
| Temperature | 1.2 | Creative, expressive variation |

> **No voice reference?**  Uses Chatterbox's built-in default voice.
> **With voice reference?** ~10 s clean mono WAV gives best results.
> **Seed ≠ 0?** Pins all randomness — identical input → identical output.
""")

            tts_preset.change(
                fn=apply_preset_tts,
                inputs=[tts_preset],
                outputs=[tts_exag, tts_cfg, tts_temp, tts_rep,
                         tts_minp, tts_topp, tts_text, tts_rationale],
            )

            tts_btn.click(
                fn=generate_tts,
                inputs=[tts_text, tts_ref, tts_exag, tts_cfg, tts_temp,
                        tts_rep, tts_minp, tts_topp, tts_stream, tts_seed],
                outputs=tts_out,
            )

        # ── Tab 2: Turbo TTS ──────────────────────────────────────────────────
        with gr.Tab("⚡ Turbo TTS"):
            gr.Markdown("""
**`ChatterboxTurboTTS`** — 350M params · English · 1-step mel decoder · Native paralinguistic tags

- Built for **low-latency voice agents** and production pipelines.
- Voice reference must be **> 5 seconds** when provided.
- `exaggeration` and `cfg_weight` are **not supported** by this model — they are silently ignored.
""")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    # ── Preset selector ───────────────────────────────────────
                    with gr.Group():
                        with gr.Row():
                            turbo_preset = gr.Dropdown(
                                label="🎛️ Preset",
                                choices=PRESET_TURBO_NAMES,
                                value="🎯 Default",
                                scale=2,
                                info="Select a use-case preset — auto-fills text and all parameters",
                            )
                        turbo_rationale = gr.Markdown(
                            value=PRESETS_TURBO["🎯 Default"]["rationale_md"],
                            label="",
                        )

                    # v2: elem_id so the cursor-aware JS can find this textarea in the DOM
                    turbo_text = gr.Textbox(
                        label="📝 Input Text  (embed paralinguistic tags inline)",
                        lines=5,
                        placeholder="Hi there! [chuckle] How's your day going?",
                        value=PRESETS_TURBO["🎯 Default"]["sample_text"],
                        elem_id="turbo_textbox",
                    )

                    # v2: tag buttons use JavaScript for cursor-position-aware insertion
                    gr.Markdown("**🏷️ Insert Tag at Cursor:**")
                    with gr.Row():
                        turbo_tag_btns = [
                            gr.Button(t, size="sm", elem_classes=["tag-btn"]) for t in PARA_TAGS
                        ]

                    turbo_ref = gr.Audio(
                        label="🎤 Voice Reference  (optional — must be > 5 s for cloning)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )

                    with gr.Accordion("⚙️ Generation Parameters  (auto-filled by preset · adjustable)", open=True):
                        with gr.Row():
                            # v2: temperature range 0.05–2.0 (matches official Turbo app)
                            turbo_temp = gr.Slider(0.05, 2.0, value=0.8, step=0.05,
                                                   label="Temperature",  info=_D_TEMP)
                            turbo_topk = gr.Slider(1, 2000, value=1000, step=10,
                                                   label="Top-K",        info=_D_TOPK)
                        with gr.Row():
                            turbo_topp = gr.Slider(0.0, 1.0, value=0.95, step=0.01,
                                                   label="Top-P",             info=_D_TOPP)
                            turbo_rep  = gr.Slider(1.0, 2.0, value=1.2,  step=0.05,
                                                   label="Repetition Penalty", info=_D_REP)
                        with gr.Row():
                            # Turbo ignores min_p (warns internally) — shown for transparency
                            turbo_minp = gr.Slider(0.0, 1.0, value=0.0, step=0.01,
                                                   label="Min-P  ⚠️ ignored by Turbo", info=_D_MINP)
                            # v2: seed control
                            turbo_seed = gr.Number(value=0, label="🎲 Seed  (0 = random)",
                                                   info=_D_SEED, precision=0, minimum=0)
                        turbo_loudness = gr.Checkbox(
                            label="📢 Normalize Loudness  (normalises reference to –27 LUFS before conditioning)",
                            value=True,
                        )

                    gr.Markdown(
                        "> ⚠️ `exaggeration` and `cfg_weight` are **not supported** by the "
                        "Turbo architecture — they are accepted by the API but silently ignored."
                    )
                    turbo_stream = gr.Checkbox(
                        label="🌊 Stream  (sentence-by-sentence)", value=False
                    )
                    turbo_btn = gr.Button("⚡ Generate (Turbo)", variant="primary", size="lg")

                with gr.Column(scale=2):
                    turbo_out = gr.Audio(label="🔊 Generated Audio", type="numpy", interactive=False)
                    gr.Markdown("""
### 🏷️ Paralinguistic Tags
| Tag | Description |
|---|---|
| `[laugh]` | Full laughter burst |
| `[chuckle]` | Light, soft laugh |
| `[cough]` | Cough sound |
| `[sigh]` | Sigh / exhale |
| `[gasp]` | Sharp inhale / surprise |
| `[hmm]` | Thinking / pondering |
| `[clears throat]` | Throat clearing |

### ⚡ Turbo vs Standard
| Feature | Standard | Turbo |
|---|---|---|
| Speed | Slower | **Much faster** |
| VRAM | Higher | **Lower** |
| Para-tags | ❌ | ✅ |
| Exaggeration | ✅ | ❌ |
| CFG Weight | ✅ | ❌ |
| Decoder steps | 10 | **1** |

### 💡 Tag Placement Tip
Click a tag button to insert it **exactly at your cursor** inside the text box:
```
"I just heard the news. [gasp]
That's incredible! [chuckle]"
```
""")

            # v2: wire tag buttons with cursor-aware JavaScript (fn=None → no server call)
            for btn, tag in zip(turbo_tag_btns, PARA_TAGS):
                btn.click(
                    fn=None,
                    inputs=[btn, turbo_text],
                    outputs=[turbo_text],
                    js=_INSERT_TAG_JS,
                )

            turbo_preset.change(
                fn=apply_preset_turbo,
                inputs=[turbo_preset],
                outputs=[turbo_temp, turbo_topk, turbo_topp, turbo_rep,
                         turbo_minp, turbo_loudness, turbo_text, turbo_rationale],
            )

            turbo_btn.click(
                fn=generate_turbo,
                inputs=[turbo_text, turbo_ref, turbo_temp, turbo_topk, turbo_topp,
                        turbo_rep, turbo_minp, turbo_loudness, turbo_stream, turbo_seed],
                outputs=turbo_out,
            )

        # ── Tab 3: Multilingual TTS ───────────────────────────────────────────
        with gr.Tab("🌍 Multilingual TTS"):
            gr.Markdown("""
**`ChatterboxMultilingualTTS`** — 500M params · 23 languages · Zero-shot cross-language voice cloning

- Select a language — **sample text and a native-speaker reference clip load automatically**.
- Upload your own reference clip to clone any voice across languages.
- **Accent bleed tip:** set `CFG Weight = 0` when reference language ≠ target language.
""")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    _initial_lang = "fr - French"
                    _initial_lang_code = "fr"

                    # ── Preset selector ───────────────────────────────────────
                    # Presets update params only — the language dropdown + sample
                    # text button control the text/reference audio separately.
                    with gr.Group():
                        with gr.Row():
                            mtl_preset = gr.Dropdown(
                                label="🎛️ Preset",
                                choices=PRESET_TTS_NAMES,
                                value="🎯 Default",
                                scale=2,
                                info="Auto-fills all parameters — text is controlled by the language dropdown below",
                            )
                        mtl_rationale = gr.Markdown(
                            value=PRESETS_TTS["🎯 Default"]["rationale_md"],
                            label="",
                        )

                    mtl_lang = gr.Dropdown(
                        label="🌐 Language",
                        choices=LANGUAGE_OPTIONS,
                        value=_initial_lang,
                    )

                    mtl_text = gr.Textbox(
                        label="📝 Input Text  (auto-filled when language changes)",
                        lines=5,
                        placeholder="Enter text in the chosen language …",
                        value=SAMPLE_TEXTS[_initial_lang],
                    )
                    # v2: initial value = curated French reference clip from Resemble AI GCS
                    mtl_ref = gr.Audio(
                        label="🎤 Voice Reference  (auto-filled · upload to override)",
                        type="filepath",
                        sources=["upload", "microphone"],
                        value=LANGUAGE_AUDIO_DEFAULTS.get(_initial_lang_code),
                    )

                    with gr.Accordion("⚙️ Generation Parameters  (auto-filled by preset · adjustable)", open=True):
                        with gr.Row():
                            # v2: exaggeration 0.25–2.0
                            mtl_exag = gr.Slider(0.25, 2.0, value=0.5, step=0.05,
                                                 label="Exaggeration", info=_D_EXAG)
                            mtl_cfg  = gr.Slider(0.0,  1.0,  value=0.5, step=0.05,
                                                 label="CFG Weight",   info=_D_CFG)
                        with gr.Row():
                            # v2: temperature 0.05–5.0
                            mtl_temp = gr.Slider(0.05, 5.0, value=0.8, step=0.05,
                                                 label="Temperature",       info=_D_TEMP)
                            mtl_rep  = gr.Slider(1.0,  2.0, value=2.0, step=0.05,
                                                 label="Repetition Penalty", info=_D_REP)
                        with gr.Row():
                            mtl_minp = gr.Slider(0.0, 1.0, value=0.05, step=0.01,
                                                 label="Min-P",  info=_D_MINP)
                            mtl_topp = gr.Slider(0.0, 1.0, value=1.0,  step=0.01,
                                                 label="Top-P",  info=_D_TOPP)
                        # v2: seed control
                        mtl_seed = gr.Number(value=0, label="🎲 Seed  (0 = random)",
                                             info=_D_SEED, precision=0, minimum=0)

                    mtl_stream = gr.Checkbox(
                        label="🌊 Stream  (sentence-by-sentence)", value=False
                    )
                    mtl_btn = gr.Button("🌍 Generate", variant="primary", size="lg")

                with gr.Column(scale=2):
                    mtl_out = gr.Audio(label="🔊 Generated Audio", type="numpy", interactive=False)
                    gr.Markdown("""
### 🗺️ 23 Supported Languages
Arabic · Danish · German · Greek · **English** · Spanish ·
Finnish · **French** · Hebrew · **Hindi** · Italian · **Japanese** ·
**Korean** · Malay · Dutch · Norwegian · Polish · Portuguese ·
Russian · Swedish · Swahili · Turkish · **Chinese**

### Auto-fill Behaviour (v2)
When you switch the **Language** dropdown, the app automatically:
1. Fills the text box with a native-language sentence
2. Loads a curated native-speaker reference clip from Resemble AI

You can override the reference by uploading your own clip.

### Cross-Language Cloning Tips
| Scenario | Recommended Setting |
|---|---|
| Ref lang = target lang | `cfg=0.5` (default) |
| Ref lang ≠ target lang | `CFG Weight = 0` |
| Fast-speaking reference | `CFG Weight ≈ 0.3` |
| More expressive output | `Exaggeration → 0.7` |
""")

            # Preset change → update params only (text/audio handled by language dropdown)
            mtl_preset.change(
                fn=apply_preset_mtl,
                inputs=[mtl_preset],
                outputs=[mtl_exag, mtl_cfg, mtl_temp, mtl_rep,
                         mtl_minp, mtl_topp, mtl_rationale],
            )

            # v2: language change → auto-fill BOTH text AND reference audio
            mtl_lang.change(
                fn=on_language_change,
                inputs=[mtl_lang],
                outputs=[mtl_text, mtl_ref],
            )
            mtl_btn.click(
                fn=generate_multilingual,
                inputs=[mtl_text, mtl_lang, mtl_ref, mtl_exag, mtl_cfg,
                        mtl_temp, mtl_rep, mtl_minp, mtl_topp, mtl_stream, mtl_seed],
                outputs=mtl_out,
            )

        # ── Tab 4: Voice Conversion ───────────────────────────────────────────
        with gr.Tab("🔄 Voice Conversion"):
            gr.Markdown("""
**`ChatterboxVC`** — Pure audio-to-audio voice conversion · No text required

The **speech content** (words, timing, prosody) comes from the *source audio*.
The **voice identity** (timbre, accent, character) comes from the *target reference*.
""")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.Markdown("### Inputs")
                    vc_src = gr.Audio(
                        label="🎙️ Source Audio  (content to convert)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    vc_tgt = gr.Audio(
                        label="🎤 Target Voice Reference  (who it should sound like)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    vc_btn = gr.Button("🔄 Convert Voice", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### Output")
                    vc_out = gr.Audio(label="🔊 Converted Audio", type="numpy", interactive=False)
                    gr.Markdown("""
### 🔬 How It Works
```
Source Audio ──► S3 Tokenizer ──► Speech Tokens
                                        │
Target Voice ──► Voice Encoder ──► Speaker Embedding
                                        │
                                   S3Gen ──► Output WAV
```

The model extracts the **"what"** from source and the **"who"**
from target, then re-synthesises them together.

### Use Cases
| Scenario | Description |
|---|---|
| 🔒 Privacy | Anonymise a speaker's identity |
| 🎬 Dubbing | Keep content, swap voice identity |
| 🎮 Game NPCs | Apply character voices to dialogue |
| 🎙️ Podcasts | Unify recordings across sessions |
| 🎭 Prototyping | Try character voices quickly |

### Tips
- Source audio: **clean, minimal noise**
- Target reference: **~10 s, single speaker** for best fidelity
- The model preserves the timing and prosody of the source
""")

            vc_btn.click(fn=generate_vc, inputs=[vc_src, vc_tgt], outputs=vc_out)

        # ── Tab 5: Model Manager ──────────────────────────────────────────────
        with gr.Tab("🗂️ Model Manager"):
            gr.Markdown("""
**Model Manager** — Preload, unload, or download each model independently.

Memory is shared between all loaded models. On limited-memory devices (< 16 GB),
load only the model you need and unload others before switching.

> **MPS note:** Unloading calls `model.cpu()` → `del` → `gc.collect()` →
> `torch.mps.synchronize()` → `torch.mps.empty_cache()` — all five steps are
> required to fully return memory to the Metal driver.
""")

            # ── Status panel ──────────────────────────────────────────────────
            mgr_html = gr.HTML(value=render_manager_html)

            with gr.Row():
                mgr_refresh_btn = gr.Button("🔄 Refresh Status", size="sm", scale=0)

            gr.Markdown("---")

            # ── Action log ────────────────────────────────────────────────────
            mgr_log = gr.Textbox(
                label="Action Log",
                lines=4,
                interactive=False,
                placeholder="Load / Unload / Download results appear here …",
            )

            gr.Markdown("### Per-Model Controls")

            # ── Standard TTS ──────────────────────────────────────────────────
            with gr.Group():
                gr.Markdown(
                    "**Standard TTS** &nbsp;·&nbsp; 500M &nbsp;·&nbsp; ~1.4 GB &nbsp;·&nbsp; "
                    "_English · zero-shot voice cloning · exaggeration & CFG controls_"
                )
                with gr.Row():
                    mgr_tts_load   = gr.Button("⬆️ Load into Memory",   size="sm", variant="primary")
                    mgr_tts_unload = gr.Button("⬇️ Unload from Memory", size="sm")
                    mgr_tts_dl     = gr.Button("📥 Download Weights",   size="sm")

            # ── Turbo TTS ─────────────────────────────────────────────────────
            with gr.Group():
                gr.Markdown(
                    "**Turbo TTS** &nbsp;·&nbsp; 350M &nbsp;·&nbsp; ~2.9 GB &nbsp;·&nbsp; "
                    "_English · 1-step decoder · paralinguistic tags · low VRAM_"
                )
                with gr.Row():
                    mgr_turbo_load   = gr.Button("⬆️ Load into Memory",   size="sm", variant="primary")
                    mgr_turbo_unload = gr.Button("⬇️ Unload from Memory", size="sm")
                    mgr_turbo_dl     = gr.Button("📥 Download Weights",   size="sm")

            # ── Multilingual TTS ──────────────────────────────────────────────
            with gr.Group():
                gr.Markdown(
                    "**Multilingual TTS** &nbsp;·&nbsp; 500M &nbsp;·&nbsp; ~1.5 GB &nbsp;·&nbsp; "
                    "_23 languages · cross-language voice cloning_"
                )
                with gr.Row():
                    mgr_mtl_load   = gr.Button("⬆️ Load into Memory",   size="sm", variant="primary")
                    mgr_mtl_unload = gr.Button("⬇️ Unload from Memory", size="sm")
                    mgr_mtl_dl     = gr.Button("📥 Download Weights",   size="sm")

            # ── Voice Conversion ──────────────────────────────────────────────
            with gr.Group():
                gr.Markdown(
                    "**Voice Conversion** &nbsp;·&nbsp; — &nbsp;·&nbsp; ~0.4 GB &nbsp;·&nbsp; "
                    "_Audio-to-audio · no text needed · voice identity swap_"
                )
                with gr.Row():
                    mgr_vc_load   = gr.Button("⬆️ Load into Memory",   size="sm", variant="primary")
                    mgr_vc_unload = gr.Button("⬇️ Unload from Memory", size="sm")
                    mgr_vc_dl     = gr.Button("📥 Download Weights",   size="sm")

            # ── Wire refresh ──────────────────────────────────────────────────
            mgr_refresh_btn.click(fn=render_manager_html, outputs=mgr_html)

            # ── Wire Standard TTS buttons ─────────────────────────────────────
            mgr_tts_load.click(
                fn=lambda: manager_do_load("tts"),
                outputs=[mgr_html, mgr_log],
            )
            mgr_tts_unload.click(
                fn=lambda: manager_do_unload("tts"),
                outputs=[mgr_html, mgr_log],
            )
            mgr_tts_dl.click(
                fn=lambda: manager_do_download("tts"),
                outputs=[mgr_html, mgr_log],
            )

            # ── Wire Turbo TTS buttons ────────────────────────────────────────
            mgr_turbo_load.click(
                fn=lambda: manager_do_load("turbo"),
                outputs=[mgr_html, mgr_log],
            )
            mgr_turbo_unload.click(
                fn=lambda: manager_do_unload("turbo"),
                outputs=[mgr_html, mgr_log],
            )
            mgr_turbo_dl.click(
                fn=lambda: manager_do_download("turbo"),
                outputs=[mgr_html, mgr_log],
            )

            # ── Wire Multilingual TTS buttons ─────────────────────────────────
            mgr_mtl_load.click(
                fn=lambda: manager_do_load("multilingual"),
                outputs=[mgr_html, mgr_log],
            )
            mgr_mtl_unload.click(
                fn=lambda: manager_do_unload("multilingual"),
                outputs=[mgr_html, mgr_log],
            )
            mgr_mtl_dl.click(
                fn=lambda: manager_do_download("multilingual"),
                outputs=[mgr_html, mgr_log],
            )

            # ── Wire Voice Conversion buttons ─────────────────────────────────
            mgr_vc_load.click(
                fn=lambda: manager_do_load("vc"),
                outputs=[mgr_html, mgr_log],
            )
            mgr_vc_unload.click(
                fn=lambda: manager_do_unload("vc"),
                outputs=[mgr_html, mgr_log],
            )
            mgr_vc_dl.click(
                fn=lambda: manager_do_download("vc"),
                outputs=[mgr_html, mgr_log],
            )

        # ── Tab 6: Watermark Checker ──────────────────────────────────────────
        with gr.Tab("🔍 Watermark Check"):
            gr.Markdown("""
**PerTh Watermark Detector** — Every Chatterbox output is invisibly watermarked.

The watermark survives MP3 compression, audio editing, and common manipulations
while maintaining near-100% detection accuracy.
""")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    wm_audio = gr.Audio(
                        label="🎵 Audio to Check",
                        type="filepath",
                        sources=["upload"],
                    )
                    wm_btn = gr.Button("🔍 Detect Watermark", variant="primary", size="lg")

                with gr.Column(scale=1):
                    wm_result = gr.Textbox(
                        label="Detection Result",
                        lines=6,
                        interactive=False,
                        placeholder="Upload an audio file and click Detect …",
                    )
                    gr.Markdown("""
### Score Interpretation
| Score | Meaning |
|---|---|
| **≥ 0.9** | ✅ Chatterbox watermark detected |
| **≤ 0.1** | ❌ No watermark — likely human or other TTS |
| **0.1 – 0.9** | ⚠️ Inconclusive (degraded/partial signal) |

### Check in Code
```python
import perth, librosa

audio, sr = librosa.load("output.wav", sr=None)
wm = perth.PerthImplicitWatermarker()
score = wm.get_watermark(audio, sample_rate=sr)
print(score)  # 1.0 = watermarked · 0.0 = clean
```
""")

            wm_btn.click(fn=check_watermark, inputs=[wm_audio], outputs=[wm_result])

        # ── Tab 6: About & Reference ──────────────────────────────────────────
        with gr.Tab("ℹ️ About & Reference"):
            gr.Markdown(f"""
## 🎙️ Chatterbox TTS Explorer  (v2)

Built on [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) — open-source · MIT licensed · 24.5k ⭐

---

### v2 Changes vs. v1
| Fix | Detail |
|---|---|
| **Random seed** | All 3 TTS tabs · 0 = random · non-zero = reproducible |
| **Exaggeration range** | Extended to 2.0 (was capped at 1.0) — matches official app and demo page |
| **Temperature ranges** | Standard & MTL: 0.05–5.0 · Turbo: 0.05–2.0 (matches official apps) |
| **Cursor-aware tags** | Tag buttons insert at cursor position via JS (not just append) |
| **Language auto-fill** | Switching language fills both text AND reference audio |
| **`--mcp` flag** | Exposes app as an MCP tool for AI agent integration |

---

### Model Family
| Model | Params | Languages | Strengths | Limitations |
|---|---|---|---|---|
| Standard TTS | 500M | English | Full creative control, voice cloning | Slower than Turbo |
| Turbo TTS | 350M | English | Fast, low-VRAM, paralinguistic tags | No exaggeration/CFG |
| Multilingual | 500M | 23 langs | Cross-language voice cloning | Possible accent bleed |
| Voice Conversion | — | Any | Audio-to-audio identity swap | Requires clean audio |

---

### Full Parameter Reference
| Parameter | Models | Range | Default | Description |
|---|---|---|---|---|
| `exaggeration` | Standard, MTL | 0.25 – 2.0 | 0.5 | Emotional intensity |
| `cfg_weight` | Standard, MTL | 0 – 1 | 0.5 | Voice clone fidelity vs. naturalness |
| `temperature` | Standard, MTL | 0.05 – 5.0 | 0.8 | Output randomness / creativity |
| `temperature` | Turbo | 0.05 – 2.0 | 0.8 | Output randomness / creativity |
| `repetition_penalty` | All | 1.0 – 2.0 | 1.2 / 2.0 | Suppresses token repetition |
| `min_p` | Standard, MTL | 0 – 1 | 0.05 | Min-P sampler (0 = disabled) |
| `top_p` | All | 0 – 1 | 1.0 / 0.95 | Nucleus sampling (1.0 = disabled) |
| `top_k` | Turbo | 1 – 2000 | 1000 | Top-K token pool |
| `norm_loudness` | Turbo | bool | True | Normalise ref to –27 LUFS |
| `seed` | All TTS | int ≥ 0 | 0 | 0 = random · other = fixed |

---

### Architecture
```
Text  ──►  T3 Transformer (Llama/GPT2)  ──►  Speech Tokens
                    ▲                               │
           Voice Encoder                       S3Gen decoder
                    │                    (10-step · or 1-step for Turbo)
           Reference Audio                          │
                                           WAV output (24 kHz)
                                                    │
                                           PerTh Watermarker
                                                    │
                                             Final WAV ✓
```

---

### Streaming Strategy
Chatterbox has **no native token-level streaming** — it generates complete audio per call.
This app implements sentence-level chunked streaming:
1. Split text on `.  !  ?` boundaries → list of sentences
2. Generate each sentence individually with the same voice conditioning
3. Yield cumulative concatenated audio after each sentence
4. Gradio updates the audio widget progressively

---

### Running This App
```bash
uv sync              # first-time install
uv run app.py        # standard launch (opens browser)

uv run app.py --share            # public Gradio link
uv run app.py --mcp              # expose as MCP tool for AI agents
uv run app.py --host 0.0.0.0 --port 8080
uv run app.py --no-browser
```

---

### HuggingFace Model Repos
- [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) — Standard + Multilingual + VC
- [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) — Turbo

---

*Current device: **{DEVICE.upper()}** · Chatterbox © Resemble AI — MIT License*
""")


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatterbox TTS Explorer — Gradio Demo")
    parser.add_argument("--host",       default="0.0.0.0",  help="Server bind address")
    parser.add_argument("--port",       default=7860, type=int, help="Server port")
    parser.add_argument("--share",      action="store_true", help="Create a public Gradio share URL")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    # v2: MCP server mode — exposes app as an MCP tool for AI agents (Claude, GPT, etc.)
    parser.add_argument("--mcp",        action="store_true", help="Launch as MCP server for AI agent integration")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    log.info(f"Starting Chatterbox TTS Explorer on {args.host}:{args.port}"
             + (" [MCP mode]" if args.mcp else ""))
    # Register demo.close() with atexit BEFORE launch so it executes before Python's
    # resource_tracker atexit handler.  This ensures the multiprocessing.Lock semaphore
    # that gradio/flagging.py allocates (via `from multiprocessing import Lock`) is
    # explicitly released, preventing the macOS Python 3.11 UserWarning:
    #   "resource_tracker: There appear to be 1 leaked semaphore objects to clean up"
    atexit.register(demo.close)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        mcp_server=args.mcp,
        theme=_THEME,
        css=_CSS,
    )
