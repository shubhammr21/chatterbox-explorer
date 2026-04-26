"""
src/chatterbox_explorer/adapters/primary/gradio/ui.py
======================================================
Gradio UI for the Chatterbox TTS Explorer.

Public API
----------
build_demo(tts, turbo, mtl, vc, manager, watermark, config) -> gr.Blocks
    Constructs the complete 7-tab Gradio demo and returns it without launching.
    Call ``demo.launch(...)`` in the entry point (app.py / __main__) to start.

Design contract
---------------
- The ``with gr.Blocks(...) as demo:`` block lives INSIDE ``build_demo()``.
  Module-level Blocks construction triggers Gradio side-effects on import,
  which makes unit-testing impossible and breaks hexagonal architecture.
- All event callbacks delegate to a ``GradioHandlers`` instance that is
  constructed once at the top of ``build_demo()`` with all services injected.
- Module-level constants (CSS, theme, JS, tooltip strings) are pure data —
  zero side-effects, safe to import at any time.
"""
from __future__ import annotations

import gradio as gr

from chatterbox_explorer.adapters.primary.gradio.handlers import GradioHandlers
from chatterbox_explorer.domain.languages import (
    LANGUAGE_AUDIO_DEFAULTS,
    LANGUAGE_OPTIONS,
    PARA_TAGS,
    SAMPLE_TEXTS,
)
from chatterbox_explorer.domain.models import AppConfig
from chatterbox_explorer.domain.presets import (
    PRESET_TTS_NAMES,
    PRESET_TURBO_NAMES,
    PRESETS_TTS,
    PRESETS_TURBO,
)
from chatterbox_explorer.ports.input import (
    IModelManagerService,
    IMultilingualTTSService,
    ITTSService,
    ITurboTTSService,
    IVoiceConversionService,
    IWatermarkService,
)

# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants  (pure data — no side-effects)
# ──────────────────────────────────────────────────────────────────────────────

GRADIO_CSS: str = """
.tab-nav button           { font-size: 15px !important; }
.tag-btn                  { min-width: 115px !important; font-size: 12px !important; }
.accordion-content        { padding: 8px !important; }
.status-bar textarea      { font-size: 12px !important; color: #888 !important; }
footer                    { display: none !important; }
"""

GRADIO_THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
)

# Cursor-position-aware paralinguistic tag insertion.
# Used with fn=None (pure client-side) on each tag button's .click() event.
# Receives: (button_label_value, current_textbox_value)
# Returns:  new textbox value with the tag inserted at the current cursor position.
_INSERT_TAG_JS: str = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#turbo_textbox textarea');
    if (!textarea) {
        // Fallback: append to end when textarea is not yet in the DOM
        const t = current_text || '';
        return t + (t.endsWith(' ') || t === '' ? '' : ' ') + tag_val;
    }
    const start = textarea.selectionStart;
    const end   = textarea.selectionEnd;
    const t = current_text || '';
    // Add spaces around the tag only when not already at a word boundary
    const prefix = (start === 0 || t[start - 1] === ' ') ? '' : ' ';
    const suffix = (end >= t.length  || t[end]     === ' ') ? '' : ' ';
    return t.slice(0, start) + prefix + tag_val + suffix + t.slice(end);
}
"""

# Shared tooltip strings (reused across all three TTS tabs)
_D_EXAG = "Emotional intensity. 0.5 = neutral · 0.7 = expressive · 1.5–2.0 = dramatic/unstable"
_D_CFG  = "Voice clone fidelity. 0 = ignore accent · 0.5 = balanced · 1 = strict clone"
_D_TEMP = "Randomness. Low = stable/consistent · High = creative/varied (may introduce errors)"
_D_REP  = "Suppresses repeated words/tokens during generation"
_D_MINP = "Min-P sampler. 0 = disabled. 0.02–0.1 recommended at high temperatures"
_D_TOPP = "Nucleus sampling cutoff. 1.0 = disabled (recommended default)"
_D_TOPK = "Top-K token pool size. Turbo-specific"
_D_SEED = "0 = random each run. Any other integer → fully reproducible output"


# ──────────────────────────────────────────────────────────────────────────────
# build_demo
# ──────────────────────────────────────────────────────────────────────────────

def build_demo(
    tts: ITTSService,
    turbo: ITurboTTSService,
    mtl: IMultilingualTTSService,
    vc: IVoiceConversionService,
    manager: IModelManagerService,
    watermark: IWatermarkService,
    config: AppConfig,
) -> gr.Blocks:
    """Construct and return the complete Gradio demo without launching it.

    All services are injected into a ``GradioHandlers`` instance, which owns
    every event callback.  The returned ``gr.Blocks`` object can be launched
    by the caller via ``demo.launch(...)`` or used as an ASGI app via
    ``demo.app``.

    Args:
        tts:       Standard TTS service port.
        turbo:     Turbo TTS service port.
        mtl:       Multilingual TTS service port.
        vc:        Voice Conversion service port.
        manager:   Model Manager service port.
        watermark: Watermark detection service port.
        config:    Immutable runtime configuration (device, watermark_available).

    Returns:
        A fully configured ``gr.Blocks`` demo, ready to launch.
    """
    # Single handler instance — constructed once, referenced by all event wires.
    h = GradioHandlers(tts, turbo, mtl, vc, manager, watermark, config)

    # ── Computed header strings (depend on runtime config) ───────────────────
    device_label = config.device.upper()
    watermark_note = (
        "🔏 All outputs carry an invisible PerTh AI watermark"
        if config.watermark_available
        else (
            "⚠️ PerTh watermarker unavailable (open-source resemble-perth) — "
            "outputs are NOT watermarked"
        )
    )
    cpu_warning = (
        "\n> &nbsp;|&nbsp; ⚠️ **CPU detected — generation will be slow. "
        "GPU/MPS strongly recommended.**"
        if config.device == "cpu"
        else ""
    )

    # ── Initial values for the Multilingual tab ──────────────────────────────
    _initial_lang = "fr - French"
    _initial_lang_code = "fr"

    with gr.Blocks(
        title="Chatterbox TTS Explorer",
        theme=GRADIO_THEME,
        css=GRADIO_CSS,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown(
            f"# 🎙️ Chatterbox TTS Explorer\n"
            f"**Resemble AI Chatterbox** — all capabilities · all parameters · "
            f"no code editing required.\n\n"
            f"> 🖥️ Device: **{device_label}** &nbsp;|&nbsp;\n"
            f"> 📦 Models: lazy-loaded from HuggingFace on first use &nbsp;|&nbsp;\n"
            f"> {watermark_note}"
            f"{cpu_warning}"
        )

        with gr.Row():
            gr.HTML(
                "<div style='font-size:12px;color:#888;padding:4px 0;'>"
                "💡 <b>Memory tip:</b> Use the <b>🗂️ Model Manager</b> tab to "
                "preload, unload, or download models individually — important on "
                "devices with limited RAM. Models load lazily on first Generate click."
                "</div>"
            )

        gr.Markdown("---")

        with gr.Tabs():

            # ── Tab 1: Standard TTS ───────────────────────────────────────────
            with gr.Tab("🗣️ Standard TTS"):
                gr.Markdown(
                    "**`ChatterboxTTS`** — 500M params · English · "
                    "Zero-shot voice cloning · Full creative controls\n\n"
                    "- Leave *Voice Reference* empty to use the built-in default voice.\n"
                    "- Upload a **≥ 10 s clean WAV** for best zero-shot cloning quality.\n"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        # Preset selector
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

                        with gr.Accordion(
                            "⚙️ Generation Parameters  (auto-filled by preset · adjustable)",
                            open=True,
                        ):
                            with gr.Row():
                                # v2: exaggeration extended to 2.0
                                tts_exag = gr.Slider(
                                    0.25, 2.0, value=0.5, step=0.05,
                                    label="Exaggeration", info=_D_EXAG,
                                )
                                tts_cfg = gr.Slider(
                                    0.0, 1.0, value=0.5, step=0.05,
                                    label="CFG Weight", info=_D_CFG,
                                )
                            with gr.Row():
                                # v2: temperature range 0.05–5.0
                                tts_temp = gr.Slider(
                                    0.05, 5.0, value=0.8, step=0.05,
                                    label="Temperature", info=_D_TEMP,
                                )
                                tts_rep = gr.Slider(
                                    1.0, 2.0, value=1.2, step=0.05,
                                    label="Repetition Penalty", info=_D_REP,
                                )
                            with gr.Row():
                                tts_minp = gr.Slider(
                                    0.0, 1.0, value=0.05, step=0.01,
                                    label="Min-P", info=_D_MINP,
                                )
                                tts_topp = gr.Slider(
                                    0.0, 1.0, value=1.0, step=0.01,
                                    label="Top-P", info=_D_TOPP,
                                )
                            # v2: random seed control
                            tts_seed = gr.Number(
                                value=0,
                                label="🎲 Seed  (0 = random)",
                                info=_D_SEED,
                                precision=0,
                                minimum=0,
                            )

                        tts_stream = gr.Checkbox(
                            label="🌊 Stream  (sentence-by-sentence — audio updates progressively)",
                            value=False,
                        )
                        tts_btn = gr.Button("🎙️ Generate", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        tts_out = gr.Audio(
                            label="🔊 Generated Audio",
                            type="numpy",
                            interactive=False,
                        )
                        gr.Markdown(
                            "### 💡 Parameter Quick Reference\n"
                            "| Setting | Value | Effect |\n"
                            "|---|---|---|\n"
                            "| Exaggeration | 0.5 | Natural, neutral read |\n"
                            "| Exaggeration | 1.0 | Noticeably more expressive |\n"
                            "| Exaggeration | 2.0 | Dramatic — may be unstable |\n"
                            "| CFG Weight | 0.8 | Strict voice clone fidelity |\n"
                            "| CFG Weight | 0.3 | Slower, more natural pacing |\n"
                            "| CFG Weight | 0.0 | Ignore reference accent |\n"
                            "| Temperature | 0.3 | Consistent, stable output |\n"
                            "| Temperature | 1.2 | Creative, expressive variation |\n\n"
                            "> **No voice reference?**  Uses Chatterbox's built-in default voice.\n"
                            "> **With voice reference?** ~10 s clean mono WAV gives best results.\n"
                            "> **Seed ≠ 0?** Pins all randomness — identical input → identical output.\n"
                        )

                # Event wiring — Standard TTS
                tts_preset.change(
                    fn=h.apply_preset_tts,
                    inputs=[tts_preset],
                    outputs=[tts_exag, tts_cfg, tts_temp, tts_rep,
                             tts_minp, tts_topp, tts_text, tts_rationale],
                )
                tts_btn.click(
                    fn=h.handle_tts,
                    inputs=[tts_text, tts_ref, tts_exag, tts_cfg, tts_temp,
                            tts_rep, tts_minp, tts_topp, tts_stream, tts_seed],
                    outputs=tts_out,
                )

            # ── Tab 2: Turbo TTS ──────────────────────────────────────────────
            with gr.Tab("⚡ Turbo TTS"):
                gr.Markdown(
                    "**`ChatterboxTurboTTS`** — 350M params · English · "
                    "1-step mel decoder · Native paralinguistic tags\n\n"
                    "- Built for **low-latency voice agents** and production pipelines.\n"
                    "- Voice reference must be **> 5 seconds** when provided.\n"
                    "- `exaggeration` and `cfg_weight` are **not supported** by this model "
                    "— they are silently ignored.\n"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        # Preset selector
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

                        # v2: elem_id so the cursor-aware JS can locate this textarea
                        turbo_text = gr.Textbox(
                            label="📝 Input Text  (embed paralinguistic tags inline)",
                            lines=5,
                            placeholder="Hi there! [chuckle] How's your day going?",
                            value=PRESETS_TURBO["🎯 Default"]["sample_text"],
                            elem_id="turbo_textbox",
                        )

                        # v2: tag insertion buttons with cursor-aware JS
                        gr.Markdown("**🏷️ Insert Tag at Cursor:**")
                        with gr.Row():
                            turbo_tag_btns = [
                                gr.Button(t, size="sm", elem_classes=["tag-btn"])
                                for t in PARA_TAGS
                            ]

                        turbo_ref = gr.Audio(
                            label="🎤 Voice Reference  (optional — must be > 5 s for cloning)",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )

                        with gr.Accordion(
                            "⚙️ Generation Parameters  (auto-filled by preset · adjustable)",
                            open=True,
                        ):
                            with gr.Row():
                                # v2: temperature range 0.05–2.0
                                turbo_temp = gr.Slider(
                                    0.05, 2.0, value=0.8, step=0.05,
                                    label="Temperature", info=_D_TEMP,
                                )
                                turbo_topk = gr.Slider(
                                    1, 2000, value=1000, step=10,
                                    label="Top-K", info=_D_TOPK,
                                )
                            with gr.Row():
                                turbo_topp = gr.Slider(
                                    0.0, 1.0, value=0.95, step=0.01,
                                    label="Top-P", info=_D_TOPP,
                                )
                                turbo_rep = gr.Slider(
                                    1.0, 2.0, value=1.2, step=0.05,
                                    label="Repetition Penalty", info=_D_REP,
                                )
                            with gr.Row():
                                # Turbo ignores min_p internally — shown for transparency
                                turbo_minp = gr.Slider(
                                    0.0, 1.0, value=0.0, step=0.01,
                                    label="Min-P  ⚠️ ignored by Turbo", info=_D_MINP,
                                )
                                # v2: seed control
                                turbo_seed = gr.Number(
                                    value=0,
                                    label="🎲 Seed  (0 = random)",
                                    info=_D_SEED,
                                    precision=0,
                                    minimum=0,
                                )
                            turbo_loudness = gr.Checkbox(
                                label="📢 Normalize Loudness  "
                                      "(normalises reference to –27 LUFS before conditioning)",
                                value=True,
                            )

                        gr.Markdown(
                            "> ⚠️ `exaggeration` and `cfg_weight` are **not supported** by the "
                            "Turbo architecture — they are accepted by the API but silently ignored."
                        )
                        turbo_stream = gr.Checkbox(
                            label="🌊 Stream  (sentence-by-sentence)", value=False
                        )
                        turbo_btn = gr.Button(
                            "⚡ Generate (Turbo)", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        turbo_out = gr.Audio(
                            label="🔊 Generated Audio",
                            type="numpy",
                            interactive=False,
                        )
                        gr.Markdown(
                            "### 🏷️ Paralinguistic Tags\n"
                            "| Tag | Description |\n"
                            "|---|---|\n"
                            "| `[laugh]` | Full laughter burst |\n"
                            "| `[chuckle]` | Light, soft laugh |\n"
                            "| `[cough]` | Cough sound |\n"
                            "| `[sigh]` | Sigh / exhale |\n"
                            "| `[gasp]` | Sharp inhale / surprise |\n"
                            "| `[hmm]` | Thinking / pondering |\n"
                            "| `[clears throat]` | Throat clearing |\n\n"
                            "### ⚡ Turbo vs Standard\n"
                            "| Feature | Standard | Turbo |\n"
                            "|---|---|---|\n"
                            "| Speed | Slower | **Much faster** |\n"
                            "| VRAM | Higher | **Lower** |\n"
                            "| Para-tags | ❌ | ✅ |\n"
                            "| Exaggeration | ✅ | ❌ |\n"
                            "| CFG Weight | ✅ | ❌ |\n"
                            "| Decoder steps | 10 | **1** |\n\n"
                            "### 💡 Tag Placement Tip\n"
                            "Click a tag button to insert it **exactly at your cursor** "
                            "inside the text box."
                        )

                # v2: wire tag buttons with cursor-aware JS (fn=None → no server round-trip)
                for _btn, _tag in zip(turbo_tag_btns, PARA_TAGS):
                    _btn.click(
                        fn=None,
                        inputs=[_btn, turbo_text],
                        outputs=[turbo_text],
                        js=_INSERT_TAG_JS,
                    )

                # Event wiring — Turbo TTS
                turbo_preset.change(
                    fn=h.apply_preset_turbo,
                    inputs=[turbo_preset],
                    outputs=[turbo_temp, turbo_topk, turbo_topp, turbo_rep,
                             turbo_minp, turbo_loudness, turbo_text, turbo_rationale],
                )
                turbo_btn.click(
                    fn=h.handle_turbo,
                    inputs=[turbo_text, turbo_ref, turbo_temp, turbo_topk, turbo_topp,
                            turbo_rep, turbo_minp, turbo_loudness, turbo_stream, turbo_seed],
                    outputs=turbo_out,
                )

            # ── Tab 3: Multilingual TTS ───────────────────────────────────────
            with gr.Tab("🌍 Multilingual TTS"):
                gr.Markdown(
                    "**`ChatterboxMultilingualTTS`** — 500M params · 23 languages · "
                    "Zero-shot cross-language voice cloning\n\n"
                    "- Select a language — **sample text and a native-speaker reference "
                    "clip load automatically**.\n"
                    "- Upload your own reference clip to clone any voice across languages.\n"
                    "- **Accent bleed tip:** set `CFG Weight = 0` when reference language "
                    "≠ target language.\n"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        # Preset selector — updates params only; language dropdown controls text
                        with gr.Group():
                            with gr.Row():
                                mtl_preset = gr.Dropdown(
                                    label="🎛️ Preset",
                                    choices=PRESET_TTS_NAMES,
                                    value="🎯 Default",
                                    scale=2,
                                    info="Auto-fills all parameters — text is controlled "
                                         "by the language dropdown below",
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

                        with gr.Accordion(
                            "⚙️ Generation Parameters  (auto-filled by preset · adjustable)",
                            open=True,
                        ):
                            with gr.Row():
                                # v2: exaggeration 0.25–2.0
                                mtl_exag = gr.Slider(
                                    0.25, 2.0, value=0.5, step=0.05,
                                    label="Exaggeration", info=_D_EXAG,
                                )
                                mtl_cfg = gr.Slider(
                                    0.0, 1.0, value=0.5, step=0.05,
                                    label="CFG Weight", info=_D_CFG,
                                )
                            with gr.Row():
                                # v2: temperature 0.05–5.0
                                mtl_temp = gr.Slider(
                                    0.05, 5.0, value=0.8, step=0.05,
                                    label="Temperature", info=_D_TEMP,
                                )
                                mtl_rep = gr.Slider(
                                    1.0, 2.0, value=2.0, step=0.05,
                                    label="Repetition Penalty", info=_D_REP,
                                )
                            with gr.Row():
                                mtl_minp = gr.Slider(
                                    0.0, 1.0, value=0.05, step=0.01,
                                    label="Min-P", info=_D_MINP,
                                )
                                mtl_topp = gr.Slider(
                                    0.0, 1.0, value=1.0, step=0.01,
                                    label="Top-P", info=_D_TOPP,
                                )
                            # v2: seed control
                            mtl_seed = gr.Number(
                                value=0,
                                label="🎲 Seed  (0 = random)",
                                info=_D_SEED,
                                precision=0,
                                minimum=0,
                            )

                        mtl_stream = gr.Checkbox(
                            label="🌊 Stream  (sentence-by-sentence)", value=False
                        )
                        mtl_btn = gr.Button("🌍 Generate", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        mtl_out = gr.Audio(
                            label="🔊 Generated Audio",
                            type="numpy",
                            interactive=False,
                        )
                        gr.Markdown(
                            "### 🗺️ 23 Supported Languages\n"
                            "Arabic · Danish · German · Greek · **English** · Spanish ·\n"
                            "Finnish · **French** · Hebrew · **Hindi** · Italian · **Japanese** ·\n"
                            "**Korean** · Malay · Dutch · Norwegian · Polish · Portuguese ·\n"
                            "Russian · Swedish · Swahili · Turkish · **Chinese**\n\n"
                            "### Auto-fill Behaviour (v2)\n"
                            "When you switch the **Language** dropdown, the app automatically:\n"
                            "1. Fills the text box with a native-language sentence\n"
                            "2. Loads a curated native-speaker reference clip from Resemble AI\n\n"
                            "You can override the reference by uploading your own clip.\n\n"
                            "### Cross-Language Cloning Tips\n"
                            "| Scenario | Recommended Setting |\n"
                            "|---|---|\n"
                            "| Ref lang = target lang | `cfg=0.5` (default) |\n"
                            "| Ref lang ≠ target lang | `CFG Weight = 0` |\n"
                            "| Fast-speaking reference | `CFG Weight ≈ 0.3` |\n"
                            "| More expressive output | `Exaggeration → 0.7` |\n"
                        )

                # Event wiring — Multilingual TTS
                # Preset change → update params only (text/audio handled by language dropdown)
                mtl_preset.change(
                    fn=h.apply_preset_mtl,
                    inputs=[mtl_preset],
                    outputs=[mtl_exag, mtl_cfg, mtl_temp, mtl_rep,
                             mtl_minp, mtl_topp, mtl_rationale],
                )
                # v2: language change → auto-fill BOTH sample text AND reference audio
                mtl_lang.change(
                    fn=h.on_language_change,
                    inputs=[mtl_lang],
                    outputs=[mtl_text, mtl_ref],
                )
                mtl_btn.click(
                    fn=h.handle_multilingual,
                    inputs=[mtl_text, mtl_lang, mtl_ref, mtl_exag, mtl_cfg,
                            mtl_temp, mtl_rep, mtl_minp, mtl_topp, mtl_stream, mtl_seed],
                    outputs=mtl_out,
                )

            # ── Tab 4: Voice Conversion ───────────────────────────────────────
            with gr.Tab("🔄 Voice Conversion"):
                gr.Markdown(
                    "**`ChatterboxVC`** — Pure audio-to-audio voice conversion · No text required\n\n"
                    "The **speech content** (words, timing, prosody) comes from the *source audio*.\n"
                    "The **voice identity** (timbre, accent, character) comes from the "
                    "*target reference*.\n"
                )
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
                        vc_btn = gr.Button(
                            "🔄 Convert Voice", variant="primary", size="lg"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Output")
                        vc_out = gr.Audio(
                            label="🔊 Converted Audio",
                            type="numpy",
                            interactive=False,
                        )
                        gr.Markdown(
                            "### 🔬 How It Works\n"
                            "```\n"
                            "Source Audio ──► S3 Tokenizer ──► Speech Tokens\n"
                            "                                        │\n"
                            "Target Voice ──► Voice Encoder ──► Speaker Embedding\n"
                            "                                        │\n"
                            "                                   S3Gen ──► Output WAV\n"
                            "```\n\n"
                            "The model extracts the **\"what\"** from source and the **\"who\"**\n"
                            "from target, then re-synthesises them together.\n\n"
                            "### Use Cases\n"
                            "| Scenario | Description |\n"
                            "|---|---|\n"
                            "| 🔒 Privacy | Anonymise a speaker's identity |\n"
                            "| 🎬 Dubbing | Keep content, swap voice identity |\n"
                            "| 🎮 Game NPCs | Apply character voices to dialogue |\n"
                            "| 🎙️ Podcasts | Unify recordings across sessions |\n"
                            "| 🎭 Prototyping | Try character voices quickly |\n\n"
                            "### Tips\n"
                            "- Source audio: **clean, minimal noise**\n"
                            "- Target reference: **~10 s, single speaker** for best fidelity\n"
                            "- The model preserves the timing and prosody of the source\n"
                        )

                # Event wiring — Voice Conversion
                vc_btn.click(
                    fn=h.handle_vc,
                    inputs=[vc_src, vc_tgt],
                    outputs=vc_out,
                )

            # ── Tab 5: Model Manager ──────────────────────────────────────────
            with gr.Tab("🗂️ Model Manager"):
                gr.Markdown(
                    "**Model Manager** — Preload, unload, or download each model independently.\n\n"
                    "Memory is shared between all loaded models. On limited-memory devices (< 16 GB),\n"
                    "load only the model you need and unload others before switching.\n\n"
                    "> **MPS note:** Unloading calls `model.cpu()` → `del` → `gc.collect()` →\n"
                    "> `torch.mps.synchronize()` → `torch.mps.empty_cache()` — all five steps are\n"
                    "> required to fully return memory to the Metal driver.\n"
                )

                # Status panel — h.render_manager_html is a callable; Gradio calls it
                # for the initial render and again after every action button click.
                mgr_html = gr.HTML(value=h.render_manager_html)

                with gr.Row():
                    mgr_refresh_btn = gr.Button("🔄 Refresh Status", size="sm", scale=0)

                gr.Markdown("---")

                # Action log
                mgr_log = gr.Textbox(
                    label="Action Log",
                    lines=4,
                    interactive=False,
                    placeholder="Load / Unload / Download results appear here …",
                )

                gr.Markdown("### Per-Model Controls")

                # Standard TTS
                with gr.Group():
                    gr.Markdown(
                        "**Standard TTS** &nbsp;·&nbsp; 500M &nbsp;·&nbsp; ~1.4 GB "
                        "&nbsp;·&nbsp; _English · zero-shot voice cloning · "
                        "exaggeration & CFG controls_"
                    )
                    with gr.Row():
                        mgr_tts_load   = gr.Button("⬆️ Load into Memory",   size="sm", variant="primary")
                        mgr_tts_unload = gr.Button("⬇️ Unload from Memory", size="sm")
                        mgr_tts_dl     = gr.Button("📥 Download Weights",   size="sm")

                # Turbo TTS
                with gr.Group():
                    gr.Markdown(
                        "**Turbo TTS** &nbsp;·&nbsp; 350M &nbsp;·&nbsp; ~2.9 GB "
                        "&nbsp;·&nbsp; _English · 1-step decoder · paralinguistic tags · low VRAM_"
                    )
                    with gr.Row():
                        mgr_turbo_load   = gr.Button("⬆️ Load into Memory",   size="sm", variant="primary")
                        mgr_turbo_unload = gr.Button("⬇️ Unload from Memory", size="sm")
                        mgr_turbo_dl     = gr.Button("📥 Download Weights",   size="sm")

                # Multilingual TTS
                with gr.Group():
                    gr.Markdown(
                        "**Multilingual TTS** &nbsp;·&nbsp; 500M &nbsp;·&nbsp; ~1.5 GB "
                        "&nbsp;·&nbsp; _23 languages · cross-language voice cloning_"
                    )
                    with gr.Row():
                        mgr_mtl_load   = gr.Button("⬆️ Load into Memory",   size="sm", variant="primary")
                        mgr_mtl_unload = gr.Button("⬇️ Unload from Memory", size="sm")
                        mgr_mtl_dl     = gr.Button("📥 Download Weights",   size="sm")

                # Voice Conversion
                with gr.Group():
                    gr.Markdown(
                        "**Voice Conversion** &nbsp;·&nbsp; — &nbsp;·&nbsp; ~0.4 GB "
                        "&nbsp;·&nbsp; _Audio-to-audio · no text needed · voice identity swap_"
                    )
                    with gr.Row():
                        mgr_vc_load   = gr.Button("⬆️ Load into Memory",   size="sm", variant="primary")
                        mgr_vc_unload = gr.Button("⬇️ Unload from Memory", size="sm")
                        mgr_vc_dl     = gr.Button("📥 Download Weights",   size="sm")

                # Event wiring — Model Manager
                mgr_refresh_btn.click(fn=h.render_manager_html, outputs=mgr_html)

                # Standard TTS buttons
                mgr_tts_load.click(
                    fn=lambda: h.handle_load("tts"),
                    outputs=[mgr_html, mgr_log],
                )
                mgr_tts_unload.click(
                    fn=lambda: h.handle_unload("tts"),
                    outputs=[mgr_html, mgr_log],
                )
                mgr_tts_dl.click(
                    fn=lambda: h.handle_download("tts"),
                    outputs=[mgr_html, mgr_log],
                )

                # Turbo TTS buttons
                mgr_turbo_load.click(
                    fn=lambda: h.handle_load("turbo"),
                    outputs=[mgr_html, mgr_log],
                )
                mgr_turbo_unload.click(
                    fn=lambda: h.handle_unload("turbo"),
                    outputs=[mgr_html, mgr_log],
                )
                mgr_turbo_dl.click(
                    fn=lambda: h.handle_download("turbo"),
                    outputs=[mgr_html, mgr_log],
                )

                # Multilingual TTS buttons
                mgr_mtl_load.click(
                    fn=lambda: h.handle_load("multilingual"),
                    outputs=[mgr_html, mgr_log],
                )
                mgr_mtl_unload.click(
                    fn=lambda: h.handle_unload("multilingual"),
                    outputs=[mgr_html, mgr_log],
                )
                mgr_mtl_dl.click(
                    fn=lambda: h.handle_download("multilingual"),
                    outputs=[mgr_html, mgr_log],
                )

                # Voice Conversion buttons
                mgr_vc_load.click(
                    fn=lambda: h.handle_load("vc"),
                    outputs=[mgr_html, mgr_log],
                )
                mgr_vc_unload.click(
                    fn=lambda: h.handle_unload("vc"),
                    outputs=[mgr_html, mgr_log],
                )
                mgr_vc_dl.click(
                    fn=lambda: h.handle_download("vc"),
                    outputs=[mgr_html, mgr_log],
                )

            # ── Tab 6: Watermark Check ────────────────────────────────────────
            with gr.Tab("🔍 Watermark Check"):
                gr.Markdown(
                    "**PerTh Watermark Detector** — Every Chatterbox output is invisibly watermarked.\n\n"
                    "The watermark survives MP3 compression, audio editing, and common manipulations\n"
                    "while maintaining near-100% detection accuracy.\n"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        wm_audio = gr.Audio(
                            label="🎵 Audio to Check",
                            type="filepath",
                            sources=["upload"],
                        )
                        wm_btn = gr.Button(
                            "🔍 Detect Watermark", variant="primary", size="lg"
                        )

                    with gr.Column(scale=1):
                        wm_result = gr.Textbox(
                            label="Detection Result",
                            lines=6,
                            interactive=False,
                            placeholder="Upload an audio file and click Detect …",
                        )
                        gr.Markdown(
                            "### Score Interpretation\n"
                            "| Score | Meaning |\n"
                            "|---|---|\n"
                            "| **≥ 0.9** | ✅ Chatterbox watermark detected |\n"
                            "| **≤ 0.1** | ❌ No watermark — likely human or other TTS |\n"
                            "| **0.1 – 0.9** | ⚠️ Inconclusive (degraded/partial signal) |\n\n"
                            "### Check in Code\n"
                            "```python\n"
                            "import perth, librosa\n\n"
                            "audio, sr = librosa.load(\"output.wav\", sr=None)\n"
                            "wm = perth.PerthImplicitWatermarker()\n"
                            "score = wm.get_watermark(audio, sample_rate=sr)\n"
                            "print(score)  # 1.0 = watermarked · 0.0 = clean\n"
                            "```\n"
                        )

                # Event wiring — Watermark Check
                wm_btn.click(
                    fn=h.handle_watermark,
                    inputs=[wm_audio],
                    outputs=[wm_result],
                )

            # ── Tab 7: About & Reference ──────────────────────────────────────
            with gr.Tab("ℹ️ About & Reference"):
                gr.Markdown(
                    f"## 🎙️ Chatterbox TTS Explorer  (v2)\n\n"
                    "Built on [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) "
                    "— open-source · MIT licensed · 24.5k ⭐\n\n"
                    "---\n\n"
                    "### v2 Changes vs. v1\n"
                    "| Fix | Detail |\n"
                    "|---|---|\n"
                    "| **Random seed** | All 3 TTS tabs · 0 = random · non-zero = reproducible |\n"
                    "| **Exaggeration range** | Extended to 2.0 (was capped at 1.0) — "
                    "matches official app and demo page |\n"
                    "| **Temperature ranges** | Standard & MTL: 0.05–5.0 · Turbo: 0.05–2.0 "
                    "(matches official apps) |\n"
                    "| **Cursor-aware tags** | Tag buttons insert at cursor position via JS "
                    "(not just append) |\n"
                    "| **Language auto-fill** | Switching language fills both text AND "
                    "reference audio |\n"
                    "| **`--mcp` flag** | Exposes app as an MCP tool for AI agent integration |\n\n"
                    "---\n\n"
                    "### Model Family\n"
                    "| Model | Params | Languages | Strengths | Limitations |\n"
                    "|---|---|---|---|---|\n"
                    "| Standard TTS | 500M | English | Full creative control, voice cloning | "
                    "Slower than Turbo |\n"
                    "| Turbo TTS | 350M | English | Fast, low-VRAM, paralinguistic tags | "
                    "No exaggeration/CFG |\n"
                    "| Multilingual | 500M | 23 langs | Cross-language voice cloning | "
                    "Possible accent bleed |\n"
                    "| Voice Conversion | — | Any | Audio-to-audio identity swap | "
                    "Requires clean audio |\n\n"
                    "---\n\n"
                    "### Full Parameter Reference\n"
                    "| Parameter | Models | Range | Default | Description |\n"
                    "|---|---|---|---|---|\n"
                    "| `exaggeration` | Standard, MTL | 0.25 – 2.0 | 0.5 | "
                    "Emotional intensity |\n"
                    "| `cfg_weight` | Standard, MTL | 0 – 1 | 0.5 | "
                    "Voice clone fidelity vs. naturalness |\n"
                    "| `temperature` | Standard, MTL | 0.05 – 5.0 | 0.8 | "
                    "Output randomness / creativity |\n"
                    "| `temperature` | Turbo | 0.05 – 2.0 | 0.8 | "
                    "Output randomness / creativity |\n"
                    "| `repetition_penalty` | All | 1.0 – 2.0 | 1.2 / 2.0 | "
                    "Suppresses token repetition |\n"
                    "| `min_p` | Standard, MTL | 0 – 1 | 0.05 | "
                    "Min-P sampler (0 = disabled) |\n"
                    "| `top_p` | All | 0 – 1 | 1.0 / 0.95 | "
                    "Nucleus sampling (1.0 = disabled) |\n"
                    "| `top_k` | Turbo | 1 – 2000 | 1000 | Top-K token pool |\n"
                    "| `norm_loudness` | Turbo | bool | True | "
                    "Normalise ref to –27 LUFS |\n"
                    "| `seed` | All TTS | int ≥ 0 | 0 | 0 = random · other = fixed |\n\n"
                    "---\n\n"
                    "### Architecture\n"
                    "```\n"
                    "Text  ──►  T3 Transformer (Llama/GPT2)  ──►  Speech Tokens\n"
                    "                    ▲                               │\n"
                    "           Voice Encoder                       S3Gen decoder\n"
                    "                    │                    (10-step · or 1-step for Turbo)\n"
                    "           Reference Audio                          │\n"
                    "                                           WAV output (24 kHz)\n"
                    "                                                    │\n"
                    "                                           PerTh Watermarker\n"
                    "                                                    │\n"
                    "                                             Final WAV ✓\n"
                    "```\n\n"
                    "---\n\n"
                    "### Streaming Strategy\n"
                    "Chatterbox has **no native token-level streaming** — it generates "
                    "complete audio per call.\n"
                    "This app implements sentence-level chunked streaming:\n"
                    "1. Split text on `.  !  ?` boundaries → list of sentences\n"
                    "2. Generate each sentence individually with the same voice conditioning\n"
                    "3. Yield cumulative concatenated audio after each sentence\n"
                    "4. Gradio updates the audio widget progressively\n\n"
                    "---\n\n"
                    "### Running This App\n"
                    "```bash\n"
                    "uv sync              # first-time install\n"
                    "uv run app.py        # standard launch (opens browser)\n\n"
                    "uv run app.py --share            # public Gradio link\n"
                    "uv run app.py --mcp              # expose as MCP tool for AI agents\n"
                    "uv run app.py --host 0.0.0.0 --port 8080\n"
                    "uv run app.py --no-browser\n"
                    "```\n\n"
                    "---\n\n"
                    "### HuggingFace Model Repos\n"
                    "- [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) "
                    "— Standard + Multilingual + VC\n"
                    "- [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) "
                    "— Turbo\n\n"
                    "---\n\n"
                    f"*Current device: **{device_label}** · "
                    "Chatterbox © Resemble AI — MIT License*\n"
                )

    return demo
