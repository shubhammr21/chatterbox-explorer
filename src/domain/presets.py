"""
src/chatterbox_explorer/domain/presets.py
==========================================
Canonical preset bundles for all TTS modes.

Pure Python — zero framework dependencies.
Allowed imports: stdlib only (none needed — pure data + simple dict lookups).
Forbidden: torch, gradio, chatterbox, psutil, huggingface_hub.

Layout
------
PRESETS_TTS    — 10 presets for Standard TTS + Multilingual TTS tabs
PRESETS_TURBO  — 6 presets for Turbo TTS tab
PRESET_TTS_NAMES   — list[str] convenience alias for Gradio dropdowns
PRESET_TURBO_NAMES — list[str] convenience alias for Gradio dropdowns
get_preset_tts(name)   -> dict | None
get_preset_turbo(name) -> dict | None

Each preset dict has exactly these keys:
    params       — concrete slider / widget values
    description  — one-line tooltip shown in the dropdown
    rationale_md — markdown table explaining WHY each param is chosen
    sample_text  — example sentence that showcases the preset's character

Parameter notes (from official Chatterbox README + empirical testing):
    • Higher exaggeration accelerates speech pace — compensate with lower cfg_weight
    • cfg_weight=0.3 is the README's recommended value for expressive/dramatic speech
    • temperature controls output variation: 0.3 = stable, 1.2 = very creative
    • rep_penalty 1.4 is the "safe maximum" for long-form content
    • top_k (Turbo only) is the strongest creativity dial: 80 = near-greedy, 2000 = wide open
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Standard TTS presets  (also used by Multilingual tab — same param set)
# ──────────────────────────────────────────────────────────────────────────────

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
        "params": {
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "temperature": 0.8,
            "rep_penalty": 1.2,
            "min_p": 0.05,
            "top_p": 1.0,
        },
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
        "params": {
            "exaggeration": 0.35,
            "cfg_weight": 0.65,
            "temperature": 0.65,
            "rep_penalty": 1.40,
            "min_p": 0.05,
            "top_p": 1.0,
        },
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
        "params": {
            "exaggeration": 0.40,
            "cfg_weight": 0.70,
            "temperature": 0.45,
            "rep_penalty": 1.30,
            "min_p": 0.06,
            "top_p": 0.90,
        },
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
        "params": {
            "exaggeration": 0.55,
            "cfg_weight": 0.45,
            "temperature": 0.90,
            "rep_penalty": 1.20,
            "min_p": 0.05,
            "top_p": 1.0,
        },
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
        "params": {
            "exaggeration": 0.80,
            "cfg_weight": 0.30,
            "temperature": 0.85,
            "rep_penalty": 1.20,
            "min_p": 0.03,
            "top_p": 1.0,
        },
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
        "params": {
            "exaggeration": 0.70,
            "cfg_weight": 0.40,
            "temperature": 0.80,
            "rep_penalty": 1.30,
            "min_p": 0.05,
            "top_p": 0.95,
        },
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
        "params": {
            "exaggeration": 0.45,
            "cfg_weight": 0.60,
            "temperature": 0.55,
            "rep_penalty": 1.40,
            "min_p": 0.07,
            "top_p": 0.92,
        },
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
        "params": {
            "exaggeration": 0.75,
            "cfg_weight": 0.35,
            "temperature": 0.95,
            "rep_penalty": 1.20,
            "min_p": 0.02,
            "top_p": 1.0,
        },
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
        "params": {
            "exaggeration": 0.25,
            "cfg_weight": 0.75,
            "temperature": 0.40,
            "rep_penalty": 1.20,
            "min_p": 0.08,
            "top_p": 0.88,
        },
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
        "params": {
            "exaggeration": 1.00,
            "cfg_weight": 0.20,
            "temperature": 1.20,
            "rep_penalty": 1.10,
            "min_p": 0.01,
            "top_p": 1.0,
        },
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Turbo TTS presets  (exaggeration + cfg_weight are NOT supported by Turbo)
# ──────────────────────────────────────────────────────────────────────────────

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
| Norm Loudness | **✓** | Normalises reference to -27 LUFS |
""",
        "sample_text": (
            "Hi there, Sarah here from MochaFone calling you back [chuckle]. "
            "Have you got one minute to chat about the billing issue? "
            "I know it looked confusing. [laugh] Don't worry, we'll sort it out."
        ),
        "params": {
            "temperature": 0.80,
            "top_k": 1000,
            "top_p": 0.95,
            "rep_penalty": 1.20,
            "min_p": 0.0,
            "norm_loudness": True,
        },
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
        "params": {
            "temperature": 0.60,
            "top_k": 200,
            "top_p": 0.90,
            "rep_penalty": 1.30,
            "min_p": 0.0,
            "norm_loudness": True,
        },
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
        "params": {
            "temperature": 0.85,
            "top_k": 800,
            "top_p": 0.95,
            "rep_penalty": 1.20,
            "min_p": 0.0,
            "norm_loudness": True,
        },
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
> `norm_loudness=False` preserves dramatic dynamics that -27 LUFS would flatten.
""",
        "sample_text": (
            "Well, well, well... [chuckle] Look who finally decided to show up. "
            "I was starting to think you'd forgotten all about me. [sigh] "
            "How disappointing."
        ),
        "params": {
            "temperature": 1.05,
            "top_k": 2000,
            "top_p": 1.00,
            "rep_penalty": 1.15,
            "min_p": 0.0,
            "norm_loudness": False,
        },
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
        "params": {
            "temperature": 0.90,
            "top_k": 600,
            "top_p": 0.92,
            "rep_penalty": 1.30,
            "min_p": 0.0,
            "norm_loudness": True,
        },
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
        "params": {
            "temperature": 0.50,
            "top_k": 80,
            "top_p": 0.85,
            "rep_penalty": 1.40,
            "min_p": 0.0,
            "norm_loudness": True,
        },
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Convenience name lists — for Gradio dropdowns
# ──────────────────────────────────────────────────────────────────────────────
# Derived from the dict keys so they are always in sync.

PRESET_TTS_NAMES: list[str] = list(PRESETS_TTS.keys())
PRESET_TURBO_NAMES: list[str] = list(PRESETS_TURBO.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Lookup helpers
# ──────────────────────────────────────────────────────────────────────────────


def get_preset_tts(name: str) -> dict | None:
    """Return the PRESETS_TTS entry for *name*, or None if not found.

    Case-sensitive — name must match a key exactly (including emoji prefix).

    Examples
    --------
    >>> get_preset_tts("🎯 Default")["params"]["exaggeration"]
    0.5
    >>> get_preset_tts("unknown") is None
    True
    """
    return PRESETS_TTS.get(name)


def get_preset_turbo(name: str) -> dict | None:
    """Return the PRESETS_TURBO entry for *name*, or None if not found.

    Case-sensitive — name must match a key exactly (including emoji prefix).

    Examples
    --------
    >>> get_preset_turbo("🎯 Default")["params"]["top_k"]
    1000
    >>> get_preset_turbo("unknown") is None
    True
    """
    return PRESETS_TURBO.get(name)
