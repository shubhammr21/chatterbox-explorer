"""
presets.py — Named TTS Parameter Presets for Chatterbox Demo
=============================================================
Provides concrete, research-backed parameter dictionaries for common TTS use-cases.
Two preset banks are defined:

  • STANDARD_PRESETS  — ChatterboxTTS (exaggeration + cfg_weight supported)
  • TURBO_PRESETS     — ChatterboxTurboTTS (top_k + norm_loudness; no exaggeration/cfg)

Design sources
--------------
  • resemble-ai/chatterbox README tips (exaggeration / cfg_weight interaction)
  • chatterbox/tts.py  generate() signature defaults
  • chatterbox/tts_turbo.py  generate() signature defaults
  • Official gradio_tts_app.py  slider defaults
  • Official gradio_tts_turbo_app.py  slider defaults
  • General TTS parameter-tuning literature on temperature / top_p / rep_penalty

Parameter quick-reference
--------------------------
Standard model
  exaggeration  0.25-2.0   Emotional intensity. 0.5 = neutral. Higher → more dramatic AND faster.
  cfg_weight    0.0-1.0    Voice-clone fidelity / pacing anchor. Low → model freedom + slower pace.
  temperature   0.05-5.0   Token sampling randomness. Low → stable. High → creative / unpredictable.
  rep_penalty   1.0-2.0    Penalises repeated tokens. Higher → fewer repetition artifacts.
  min_p         0.0-1.0    Min-probability floor for nucleus. 0 = disabled. Handles high-temp well.
  top_p         0.0-1.0    Nucleus sampling mass. 1.0 = disabled (recommended baseline).

Turbo model  (exaggeration / cfg_weight / min_p are IGNORED — model logs a warning)
  temperature   0.05-2.0   Same semantics as Standard.
  top_k         1-2000     Hard cap on candidate tokens. Lower → more deterministic.
  top_p         0.0-1.0    Same semantics as Standard.
  rep_penalty   1.0-2.0    Same semantics as Standard.
  norm_loudness bool       Normalise output to -27 LUFS via pyloudnorm. Essential for telephony.

README tips baked in
--------------------
  "default settings (exaggeration=0.5, cfg_weight=0.5) work well for most prompts"
  "For expressive/dramatic: lower cfg_weight ~0.3, increase exaggeration to 0.7+"
  "Higher exaggeration tends to speed up speech; reducing cfg_weight helps compensate"
  "If reference speaker has fast speaking style, lowering cfg_weight to ~0.3 improves pacing"

Usage
-----
    from presets import STANDARD_PRESETS, TURBO_PRESETS

    # Iterate for a Gradio dropdown
    choices = [p["label"] for p in STANDARD_PRESETS]

    # Apply a preset to sliders
    preset = next(p for p in STANDARD_PRESETS if p["label"] == "Audiobook / Narration")
    exaggeration_slider.value = preset["params"]["exaggeration"]
    ...

    # Or use the helper
    from presets import get_standard_preset, get_turbo_preset
    params = get_standard_preset("Dramatic / Theatrical")["params"]
"""

from __future__ import annotations

from typing import Any, TypedDict

# ──────────────────────────────────────────────────────────────────────────────
# Type definitions
# ──────────────────────────────────────────────────────────────────────────────


class StandardParams(TypedDict):
    exaggeration: float  # 0.25-2.0
    cfg_weight: float  # 0.0-1.0
    temperature: float  # 0.05-5.0
    rep_penalty: float  # 1.0-2.0
    min_p: float  # 0.0-1.0
    top_p: float  # 0.0-1.0


class TurboParams(TypedDict):
    temperature: float  # 0.05-2.0
    top_k: int  # 1-2000
    top_p: float  # 0.0-1.0
    rep_penalty: float  # 1.0-2.0
    norm_loudness: bool


class StandardPreset(TypedDict):
    label: str
    emoji: str
    description: str
    params: StandardParams
    param_notes: dict[str, str]  # param_name → one-line reason for deviation from default
    sample_text: str


class TurboPreset(TypedDict):
    label: str
    emoji: str
    description: str
    params: TurboParams
    param_notes: dict[str, str]
    sample_text: str


# ──────────────────────────────────────────────────────────────────────────────
# Model defaults  (source-of-truth: tts.py and tts_turbo.py generate() sigs)
# ──────────────────────────────────────────────────────────────────────────────

STANDARD_DEFAULTS: StandardParams = {
    "exaggeration": 0.50,
    "cfg_weight": 0.50,
    "temperature": 0.80,
    "rep_penalty": 1.20,
    "min_p": 0.05,
    "top_p": 1.00,
}

TURBO_DEFAULTS: TurboParams = {
    "temperature": 0.80,
    "top_k": 1000,
    "top_p": 0.95,
    "rep_penalty": 1.20,
    "norm_loudness": True,
}
# NOTE: Turbo accepts min_p in its signature but IGNORES it and logs a warning.
#       All Turbo presets omit min_p intentionally.


# ──────────────────────────────────────────────────────────────────────────────
# Standard TTS Presets  (10 use-cases)
# ──────────────────────────────────────────────────────────────────────────────

STANDARD_PRESETS: list[StandardPreset] = [
    # ── 1. Audiobook / Long-form Narration ────────────────────────────────────
    {
        "label": "Audiobook / Narration",
        "emoji": "📚",
        "description": "Consistent, measured pacing for chapters and long prose. "
        "Prioritises stability and even tone over expressiveness.",
        "params": {
            "exaggeration": 0.35,
            "cfg_weight": 0.65,
            "temperature": 0.65,
            "rep_penalty": 1.40,
            "min_p": 0.05,
            "top_p": 1.00,
        },
        "param_notes": {
            "exaggeration": "↓ 0.5→0.35 — subdued intensity prevents fatigue over long passages "
            "and stops exaggeration from speeding up narration pace.",
            "cfg_weight": "↑ 0.5→0.65 — higher reference adherence keeps narrator voice "
            "consistent across sentences and chapters.",
            "temperature": "↓ 0.8→0.65 — more deterministic output prevents odd tonal departures "
            "mid-chapter; stability matters more than variety here.",
            "rep_penalty": "↑ 1.2→1.40 — long-form content is most vulnerable to token repetition "
            "artifacts across sentence boundaries; higher penalty is critical.",
            "min_p": "= default — standard probability floor is fine for prose.",
            "top_p": "= default (disabled) — let min_p handle filtering; full vocabulary "
            "access needed for natural literary prose.",
        },
        "sample_text": (
            "The afternoon sun cast long shadows across the valley floor, "
            "and somewhere in the distance, a hawk called out once, then fell silent."
        ),
    },
    # ── 2. Voice Agent / Customer Service ─────────────────────────────────────
    {
        "label": "Voice Agent / Customer Service",
        "emoji": "🎧",
        "description": "Warm, professional, and highly reliable. "
        "Optimised for consistent output across repeated calls.",
        "params": {
            "exaggeration": 0.45,
            "cfg_weight": 0.55,
            "temperature": 0.55,
            "rep_penalty": 1.30,
            "min_p": 0.05,
            "top_p": 0.90,
        },
        "param_notes": {
            "exaggeration": "↓ 0.5→0.45 — warm but not performative; professional without robotic flatness.",
            "cfg_weight": "↑ 0.5→0.55 — slightly more reference adherence for consistent pacing "
            "on every call; prevents speed variation.",
            "temperature": "↓ 0.8→0.55 — reliability is paramount for agents; low temperature "
            "produces consistent output across thousands of calls.",
            "rep_penalty": "↑ 1.2→1.30 — slightly elevated to avoid robotic repeated phoneme patterns.",
            "min_p": "= default.",
            "top_p": "↓ 1.0→0.90 — restrict to high-probability tokens; reduces unexpected "
            "phrasing patterns in production deployments.",
        },
        "sample_text": (
            "Thank you for calling Horizon Support. I'd be happy to help you with your account today. "
            "Could you please verify your name and the last four digits of your account number?"
        ),
    },
    # ── 3. News Broadcast / Professional Read ─────────────────────────────────
    {
        "label": "News Broadcast",
        "emoji": "📰",
        "description": "Authoritative, neutral, and confident. "
        "Anchored to the reference voice for maximum credibility.",
        "params": {
            "exaggeration": 0.40,
            "cfg_weight": 0.70,
            "temperature": 0.45,
            "rep_penalty": 1.30,
            "min_p": 0.06,
            "top_p": 0.90,
        },
        "param_notes": {
            "exaggeration": "↓ 0.5→0.40 — near-neutral with slight gravitas; avoids 'flat robot' "
            "but stays strictly professional.",
            "cfg_weight": "↑ 0.5→0.70 — high fidelity to the chosen broadcaster reference voice; "
            "pacing matches the anchor's natural cadence.",
            "temperature": "↓ 0.8→0.45 — highly deterministic for professional, consistent delivery; "
            "news copy must land the same way every read.",
            "rep_penalty": "↑ 1.2→1.30 — clean delivery, no artifacts.",
            "min_p": "↑ 0.05→0.06 — slightly higher threshold filters low-probability tokens "
            "for crisp, broadcast-quality output.",
            "top_p": "↓ 1.0→0.90 — restricted nucleus for consistent professional output.",
        },
        "sample_text": (
            "Global leaders convened in Geneva today for an emergency summit on climate policy, "
            "where delegates from forty-seven nations signed a landmark emissions agreement."
        ),
    },
    # ── 4. Conversational / Casual ────────────────────────────────────────────
    {
        "label": "Conversational / Casual",
        "emoji": "💬",
        "description": "Natural, relaxed, and personable. "
        "Slight variation mimics real human speech patterns.",
        "params": {
            "exaggeration": 0.55,
            "cfg_weight": 0.45,
            "temperature": 0.90,
            "rep_penalty": 1.20,
            "min_p": 0.05,
            "top_p": 1.00,
        },
        "param_notes": {
            "exaggeration": "↑ 0.5→0.55 — slightly above neutral; warmer and more personable, "
            "not flat.",
            "cfg_weight": "↓ 0.5→0.45 — looser adherence to reference allows natural variation "
            "in pacing, as real conversation has.",
            "temperature": "↑ 0.8→0.90 — higher for natural prosodic variation; "
            "real speech is not perfectly uniform.",
            "rep_penalty": "= default.",
            "min_p": "= default.",
            "top_p": "= default (disabled) — full vocabulary; natural variation should flow freely.",
        },
        "sample_text": (
            "Oh, that's actually pretty funny — I was just thinking the same thing. "
            "So what do you want to do tonight, grab some food or just hang back?"
        ),
    },
    # ── 5. Dramatic / Theatrical ──────────────────────────────────────────────
    {
        "label": "Dramatic / Theatrical",
        "emoji": "🎭",
        "description": "Intense, expressive, deliberate pacing. "
        "Applies the README-documented exaggeration + cfg tradeoff directly.",
        "params": {
            "exaggeration": 0.80,
            "cfg_weight": 0.30,
            "temperature": 0.85,
            "rep_penalty": 1.20,
            "min_p": 0.03,
            "top_p": 1.00,
        },
        "param_notes": {
            "exaggeration": "↑ 0.5→0.80 — strong emotional intensity; theatrical presence. "
            "Per README: 'increase exaggeration to 0.7 or higher' for expressive speech.",
            "cfg_weight": "↓ 0.5→0.30 — per README: 'lower cfg_weight ~0.3' for dramatic speech; "
            "counteracts the speed-up caused by high exaggeration, giving deliberate pacing.",
            "temperature": "↑ 0.8→0.85 — slight elevation adds expressive variance between lines.",
            "rep_penalty": "= default.",
            "min_p": "↓ 0.05→0.03 — lower floor allows expressive token choices; "
            "dramatic range benefits from less aggressive filtering.",
            "top_p": "= default (disabled) — full range for dramatic expression.",
        },
        "sample_text": (
            "I gave everything to this city — every breath, every sacrifice, every sleepless night. "
            "And THIS is how it ends?"
        ),
    },
    # ── 6. Advertisement / Promo Copy ─────────────────────────────────────────
    {
        "label": "Advertisement / Promo",
        "emoji": "📣",
        "description": "Energetic, punchy, and enthusiastic. "
        "Confident delivery for radio spots and YouTube pre-rolls.",
        "params": {
            "exaggeration": 0.70,
            "cfg_weight": 0.40,
            "temperature": 0.80,
            "rep_penalty": 1.30,
            "min_p": 0.05,
            "top_p": 0.95,
        },
        "param_notes": {
            "exaggeration": "↑ 0.5→0.70 — noticeably expressive; energy and enthusiasm come through "
            "without going theatrical.",
            "cfg_weight": "↓ 0.5→0.40 — slightly looser to allow expressive freedom and partially "
            "compensate for the pace-acceleration from elevated exaggeration.",
            "temperature": "= default — slight variation is good for promo energy.",
            "rep_penalty": "↑ 1.2→1.30 — avoids repetitive patterns in punchy, dense copy.",
            "min_p": "= default.",
            "top_p": "↓ 1.0→0.95 — minor restriction focuses output on confident, "
            "impactful word choices.",
        },
        "sample_text": (
            "Introducing the all-new Nova Pro — engineered for the driven, the bold, "
            "the ones who refuse to settle. Your future. Starts now."
        ),
    },
    # ── 7. E-learning / Educational ───────────────────────────────────────────
    {
        "label": "E-learning / Educational",
        "emoji": "🎓",
        "description": "Clear, patient, and consistent. "
        "Measured delivery suitable for explainer videos and online courses.",
        "params": {
            "exaggeration": 0.45,
            "cfg_weight": 0.60,
            "temperature": 0.55,
            "rep_penalty": 1.40,
            "min_p": 0.07,
            "top_p": 0.92,
        },
        "param_notes": {
            "exaggeration": "↓ 0.5→0.45 — near-neutral with slight warmth; patient and encouraging "
            "without being over-performed.",
            "cfg_weight": "↑ 0.5→0.60 — higher adherence keeps instructor voice consistent "
            "across multi-lesson modules.",
            "temperature": "↓ 0.8→0.55 — consistent and clear; learners need steady, reliable delivery.",
            "rep_penalty": "↑ 1.2→1.40 — educational content is especially vulnerable to token "
            "repetition artifacts; highest priority after audiobooks.",
            "min_p": "↑ 0.05→0.07 — slightly higher threshold for clean, clear tokens; "
            "no unusual pronunciations in classroom material.",
            "top_p": "↓ 1.0→0.92 — slightly restricted for clear, focused output.",
        },
        "sample_text": (
            "Let's take a closer look at photosynthesis. "
            "Plants use sunlight to convert carbon dioxide and water into glucose — "
            "that's the energy they need to grow."
        ),
    },
    # ── 8. Game Character / NPC ───────────────────────────────────────────────
    {
        "label": "Game Character / NPC",
        "emoji": "🎮",
        "description": "Distinctive, expressive, and varied. "
        "High creativity floor lets character personality emerge fully.",
        "params": {
            "exaggeration": 0.75,
            "cfg_weight": 0.35,
            "temperature": 0.95,
            "rep_penalty": 1.20,
            "min_p": 0.02,
            "top_p": 1.00,
        },
        "param_notes": {
            "exaggeration": "↑ 0.5→0.75 — strong character expressiveness; personality comes through clearly.",
            "cfg_weight": "↓ 0.5→0.35 — lower adherence lets character traits dominate over reference; "
            "also partially compensates for speed-up from elevated exaggeration.",
            "temperature": "↑ 0.8→0.95 — higher for character variation; NPC lines benefit from "
            "sounding slightly different on each playthrough.",
            "rep_penalty": "= default.",
            "min_p": "↓ 0.05→0.02 — low floor allows unusual and character-specific token "
            "patterns; characters have idiosyncratic speech.",
            "top_p": "= default (disabled) — full vocabulary for character-appropriate creativity.",
        },
        "sample_text": (
            "So you've finally arrived, stranger. "
            "I warned the elders you'd come... but they never listen. They never do."
        ),
    },
    # ── 9. Meditation / ASMR ──────────────────────────────────────────────────
    {
        "label": "Meditation / ASMR",
        "emoji": "🧘",
        "description": "Soft, slow, and deeply consistent. "
        "Minimum exaggeration + high cfg anchors gentle pacing.",
        "params": {
            "exaggeration": 0.25,
            "cfg_weight": 0.75,
            "temperature": 0.40,
            "rep_penalty": 1.20,
            "min_p": 0.08,
            "top_p": 0.88,
        },
        "param_notes": {
            "exaggeration": "↓ 0.5→0.25 — minimum intensity; as calm and flat as possible with "
            "no sudden emphasis that would break a meditative state.",
            "cfg_weight": "↑ 0.5→0.75 — high adherence preserves soft reference voice quality "
            "and stabilises a slow, unhurried pace.",
            "temperature": "↓ 0.8→0.40 — very low for extremely stable, gentle output; "
            "no surprises in a guided meditation.",
            "rep_penalty": "= default — gentle, rhythmic speech patterns are acceptable "
            "and even desirable; don't over-penalise.",
            "min_p": "↑ 0.05→0.08 — higher threshold filters unusual tokens for very "
            "clean, soothing output.",
            "top_p": "↓ 1.0→0.88 — restricted nucleus; only high-probability, "
            "calming token choices.",
        },
        "sample_text": (
            "Allow your breath to slow... "
            "gently release any tension you're holding... "
            "and let your body sink into perfect stillness."
        ),
    },
    # ── 10. Experimental / Creative ───────────────────────────────────────────
    {
        "label": "Experimental / Creative",
        "emoji": "🔬",
        "description": "High-risk, high-reward. Intentionally pushes parameters to extremes "
        "for spoken-word art, demos, and unexpected results. May be unstable.",
        "params": {
            "exaggeration": 1.00,
            "cfg_weight": 0.20,
            "temperature": 1.20,
            "rep_penalty": 1.10,
            "min_p": 0.01,
            "top_p": 1.00,
        },
        "param_notes": {
            "exaggeration": "↑ 0.5→1.00 — high intensity; dramatic and potentially unstable effects "
            "are part of the creative palette.",
            "cfg_weight": "↓ 0.5→0.20 — very low; model diverges from reference and makes its "
            "own pacing choices.",
            "temperature": "↑ 0.8→1.20 — high randomness; surprising and highly varied output.",
            "rep_penalty": "↓ 1.2→1.10 — allow patterns; repetition can be intentional and "
            "artistic in creative contexts.",
            "min_p": "↓ 0.05→0.01 — near-disabled; allows unusual, creative token choices "
            "that would be filtered in professional presets.",
            "top_p": "= default (disabled) — full vocabulary access for maximum creativity.",
        },
        "sample_text": (
            "The probability cascade reversed — folding midnight into the algorithm's dream. "
            "A voice that tasted of fractals and forgotten frequencies."
        ),
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Turbo TTS Presets  (6 use-cases)
# ──────────────────────────────────────────────────────────────────────────────
# NOTE: Turbo's top_k is the primary creativity dial (replaces cfg_weight).
#       Lower top_k = more deterministic (like lower temperature + restricted vocab).
#       Higher top_k = more variety (wider candidate pool).
#       min_p is intentionally excluded from all Turbo presets — the model
#       accepts it but ignores it and emits a logger.warning().

TURBO_PRESETS: list[TurboPreset] = [
    # ── 1. Default ────────────────────────────────────────────────────────────
    {
        "label": "Default",
        "emoji": "⚡",
        "description": "The model's calibrated defaults. A solid starting point for most content.",
        "params": {
            "temperature": 0.80,
            "top_k": 1000,
            "top_p": 0.95,
            "rep_penalty": 1.20,
            "norm_loudness": True,
        },
        "param_notes": {
            "temperature": "= model default.",
            "top_k": "= model default — wide candidate pool for natural speech.",
            "top_p": "= model default — slight restriction over full 1.0.",
            "rep_penalty": "= model default.",
            "norm_loudness": "= model default — normalises to -27 LUFS via pyloudnorm.",
        },
        "sample_text": "Hello! I'm here to help with whatever you need today.",
    },
    # ── 2. Voice Agent ────────────────────────────────────────────────────────
    {
        "label": "Voice Agent",
        "emoji": "🤖",
        "description": "Reliable, clear, and professional. "
        "Tuned for production agent pipelines where consistency is critical.",
        "params": {
            "temperature": 0.60,
            "top_k": 200,
            "top_p": 0.90,
            "rep_penalty": 1.30,
            "norm_loudness": True,
        },
        "param_notes": {
            "temperature": "↓ 0.8→0.60 — reliability over creativity; consistent output on every call.",
            "top_k": "↓ 1000→200 — restricted candidate pool for highly predictable speech.",
            "top_p": "↓ 0.95→0.90 — tighter nucleus sampling for consistent professional tone.",
            "rep_penalty": "↑ 1.2→1.30 — slightly elevated for clean agent speech.",
            "norm_loudness": "= True — essential for telephony and agent systems; consistent levels.",
        },
        "sample_text": (
            "Thank you for holding. I can see your account details "
            "and I'm going to get this resolved for you right away."
        ),
    },
    # ── 3. Podcast Host ───────────────────────────────────────────────────────
    {
        "label": "Podcast Host",
        "emoji": "🎙️",
        "description": "Natural, warm, and conversational. "
        "Slight variation adds personality without unpredictability.",
        "params": {
            "temperature": 0.85,
            "top_k": 800,
            "top_p": 0.95,
            "rep_penalty": 1.20,
            "norm_loudness": True,
        },
        "param_notes": {
            "temperature": "↑ 0.8→0.85 — slightly higher for natural prosodic variation and warmth.",
            "top_k": "↓ 1000→800 — good range for natural, varied but focused speech.",
            "top_p": "= default.",
            "rep_penalty": "= default.",
            "norm_loudness": "= True — consistent listening level across a podcast episode.",
        },
        "sample_text": (
            "Honestly? I think we've been looking at this all wrong. [chuckle] "
            "Let me break it down the way I see it — because this is actually really fascinating."
        ),
    },
    # ── 4. Character / NPC ────────────────────────────────────────────────────
    {
        "label": "Character / NPC",
        "emoji": "🧙",
        "description": "Expressive and varied with full character personality. "
        "Ideal with Turbo's native paralinguistic tags ([laugh], [chuckle], etc.).",
        "params": {
            "temperature": 1.05,
            "top_k": 2000,
            "top_p": 1.00,
            "rep_penalty": 1.15,
            "norm_loudness": False,
        },
        "param_notes": {
            "temperature": "↑ 0.8→1.05 — higher for character variation and personality expression.",
            "top_k": "↑ 1000→2000 — max candidate pool; characters can use unusual speech patterns.",
            "top_p": "↑ 0.95→1.00 — disabled; allow full creative range for character voices.",
            "rep_penalty": "↓ 1.2→1.15 — characters can have verbal mannerisms and repeated patterns.",
            "norm_loudness": "↓ True→False — preserve natural character voice dynamics; "
            "loudness normalisation would flatten dramatic intensity.",
        },
        "sample_text": (
            "Well, well... look what the tide dragged in. [chuckle] "
            "Don't just stand there gawking — you'll catch flies with that mouth open."
        ),
    },
    # ── 5. Radio Promo ────────────────────────────────────────────────────────
    {
        "label": "Radio Promo",
        "emoji": "📻",
        "description": "Energetic and punchy. "
        "High-energy ad read with focused, impactful delivery.",
        "params": {
            "temperature": 0.90,
            "top_k": 600,
            "top_p": 0.92,
            "rep_penalty": 1.30,
            "norm_loudness": True,
        },
        "param_notes": {
            "temperature": "↑ 0.8→0.90 — slightly higher for promo energy and enthusiasm.",
            "top_k": "↓ 1000→600 — moderate restriction; focused but not robotic.",
            "top_p": "↓ 0.95→0.92 — slightly tighter nucleus for punchy, confident delivery.",
            "rep_penalty": "↑ 1.2→1.30 — avoid repetitive patterns in dense promotional copy.",
            "norm_loudness": "= True — consistent output level for broadcast.",
        },
        "sample_text": (
            "Only this weekend — buy one, get one FREE on everything in store. "
            "Don't wait — this offer ends Sunday at midnight!"
        ),
    },
    # ── 6. IVR / Reliable ─────────────────────────────────────────────────────
    {
        "label": "IVR / Maximum Reliability",
        "emoji": "📞",
        "description": "Maximum determinism for telephone IVR systems. "
        "Most predictable preset — highest consistency, least variation.",
        "params": {
            "temperature": 0.50,
            "top_k": 80,
            "top_p": 0.85,
            "rep_penalty": 1.40,
            "norm_loudness": True,
        },
        "param_notes": {
            "temperature": "↓ 0.8→0.50 — very low for maximum predictability in automated pipelines.",
            "top_k": "↓ 1000→80 — very restricted; only the most likely tokens selected; "
            "behaves almost like greedy decoding.",
            "top_p": "↓ 0.95→0.85 — further tightened nucleus for absolute consistency.",
            "rep_penalty": "↑ 1.2→1.40 — higher for very clean, professional telephony output.",
            "norm_loudness": "= True — consistent output level; mandatory for telephony systems.",
        },
        "sample_text": (
            "Your appointment is confirmed for Thursday, October ninth, at two thirty PM. "
            "Press one to confirm, or two to reschedule."
        ),
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Convenience lookup helpers
# ──────────────────────────────────────────────────────────────────────────────


def get_standard_preset(label: str) -> StandardPreset | None:
    """Return a Standard preset dict by its label, or None if not found."""
    return next((p for p in STANDARD_PRESETS if p["label"] == label), None)


def get_turbo_preset(label: str) -> TurboPreset | None:
    """Return a Turbo preset dict by its label, or None if not found."""
    return next((p for p in TURBO_PRESETS if p["label"] == label), None)


def standard_preset_choices() -> list[str]:
    """Return a list of (emoji + label) strings suitable for a Gradio Dropdown."""
    return [f"{p['emoji']} {p['label']}" for p in STANDARD_PRESETS]


def turbo_preset_choices() -> list[str]:
    """Return a list of (emoji + label) strings suitable for a Gradio Dropdown."""
    return [f"{p['emoji']} {p['label']}" for p in TURBO_PRESETS]


def apply_standard_preset(label: str) -> tuple[Any, ...] | None:
    """
    Return a tuple of Standard param values in the order expected by the
    Gradio output list: (exaggeration, cfg_weight, temperature, rep_penalty,
    min_p, top_p).

    Returns None if the label is not found.

    Example Gradio wiring:
        preset_dd.change(
            fn=lambda lbl: apply_standard_preset(lbl.split(" ", 1)[1]),
            inputs=[preset_dd],
            outputs=[exaggeration_sl, cfg_weight_sl, temperature_sl,
                     rep_penalty_sl, min_p_sl, top_p_sl],
        )
    """
    preset = get_standard_preset(label)
    if preset is None:
        return None
    p = preset["params"]
    return (
        p["exaggeration"],
        p["cfg_weight"],
        p["temperature"],
        p["rep_penalty"],
        p["min_p"],
        p["top_p"],
    )


def apply_turbo_preset(label: str) -> tuple[Any, ...] | None:
    """
    Return a tuple of Turbo param values in the order expected by the
    Gradio output list: (temperature, top_k, top_p, rep_penalty, norm_loudness).

    Returns None if the label is not found.

    Example Gradio wiring:
        preset_dd.change(
            fn=lambda lbl: apply_turbo_preset(lbl.split(" ", 1)[1]),
            inputs=[preset_dd],
            outputs=[temperature_sl, top_k_sl, top_p_sl,
                     rep_penalty_sl, norm_loudness_cb],
        )
    """
    preset = get_turbo_preset(label)
    if preset is None:
        return None
    p = preset["params"]
    return (
        p["temperature"],
        p["top_k"],
        p["top_p"],
        p["rep_penalty"],
        p["norm_loudness"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Quick-reference table  (printed when module is run directly)
# ──────────────────────────────────────────────────────────────────────────────


def _print_reference_table() -> None:
    """Pretty-print all presets for quick human review."""

    col = 28

    def _hr(char: str = "─", width: int = 100) -> str:
        return char * width

    print(_hr("═"))
    print("  STANDARD TTS PRESETS")
    print(_hr("═"))
    header = (
        f"  {'Label':<{col}} {'exagg':>6} {'cfg':>5} {'temp':>5} "
        f"{'rep_p':>5} {'min_p':>5} {'top_p':>5}"
    )
    print(header)
    print(_hr())
    for p in STANDARD_PRESETS:
        pp = p["params"]
        print(
            f"  {p['emoji']} {p['label']:<{col - 2}} "
            f"{pp['exaggeration']:>6.2f} "
            f"{pp['cfg_weight']:>5.2f} "
            f"{pp['temperature']:>5.2f} "
            f"{pp['rep_penalty']:>5.2f} "
            f"{pp['min_p']:>5.2f} "
            f"{pp['top_p']:>5.2f}"
        )

    print()
    print(_hr("═"))
    print("  TURBO TTS PRESETS")
    print(_hr("═"))
    header2 = f"  {'Label':<{col}} {'temp':>5} {'top_k':>5} {'top_p':>5} {'rep_p':>5} {'norm':>5}"
    print(header2)
    print(_hr())
    for p in TURBO_PRESETS:
        pp = p["params"]
        norm = "✓" if pp["norm_loudness"] else "✗"
        print(
            f"  {p['emoji']} {p['label']:<{col - 2}} "
            f"{pp['temperature']:>5.2f} "
            f"{pp['top_k']:>5} "
            f"{pp['top_p']:>5.2f} "
            f"{pp['rep_penalty']:>5.2f} "
            f"{norm:>5}"
        )

    print()
    print(_hr("═"))
    print("  STANDARD DEFAULTS")
    print(_hr())
    for k, v in STANDARD_DEFAULTS.items():
        print(f"    {k:<16} {v}")
    print()
    print(_hr("═"))
    print("  TURBO DEFAULTS")
    print(_hr())
    for k, v in TURBO_DEFAULTS.items():
        print(f"    {k:<16} {v}")
    print(_hr("═"))


if __name__ == "__main__":
    _print_reference_table()
