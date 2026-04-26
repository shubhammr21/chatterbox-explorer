# PLAN — Chatterbox TTS Explorer (Demo)

## Goal
Build an interactive Gradio demo app that showcases all Chatterbox TTS capabilities
with every developer-relevant parameter exposed in the UI — no code editing required.
This is a prototype for exploration, not a production product.

---

## Requirements

### Functional
| # | Requirement | Chatterbox Class |
|---|---|---|
| 1 | Standard English TTS with voice cloning | `ChatterboxTTS` |
| 2 | Fast Turbo TTS with paralinguistic tags | `ChatterboxTurboTTS` |
| 3 | Multilingual TTS (23 languages) with cross-language cloning | `ChatterboxMultilingualTTS` |
| 4 | Voice Conversion (audio-in → audio-out, no text) | `ChatterboxVC` |
| 5 | Sentence-level streaming (progressive audio output) | All TTS models |
| 6 | All generation params exposed as sliders/toggles | All models |
| 7 | Watermark detection on any audio file | `perth` library |
| 8 | Sample texts per language (multilingual tab) | Static data |

### Non-Functional
- Single-command startup: `uv run app.py`
- Models load lazily on first use (no blocking startup)
- Works on CUDA / MPS (Apple Silicon) / CPU automatically
- Clear UI labels, tooltips, and tips — self-documenting

---

## Constraints
- macOS / Apple Silicon (MPS) primary target; CUDA and CPU also supported
- Python 3.11+ (Chatterbox requirement)
- Demo quality — not production hardened
- No backend server, no auth, no persistence
- Models download from HuggingFace on first run (~1–4 GB total per model)
- Turbo reference audio must be > 5 seconds (model assertion)
- All output audio is PerTh watermarked (mandatory, cannot be disabled)
- Max ~1000 speech tokens per generation call (~1–2 min of speech)

---

## Tech Stack Decision

| Choice | Selected | Reason |
|---|---|---|
| UI Framework | **Gradio** | Native audio widget, generator streaming, Chatterbox already ships Gradio examples |
| Package Manager | **uv** | Fast, lockfile-based, user requirement |
| Python | **3.11** | Chatterbox tested on 3.11 |
| Entry Point | `app.py` | Single-file demo for simplicity |

Streamlit considered but rejected — Gradio's `gr.Audio` component and generator-based streaming are significantly better suited for TTS demos.

---

## Approach

### Architecture
```
app.py
├── Device detection (CUDA → MPS → CPU)
├── Lazy model registry (_MODEL_CACHE dict, load on first generate click)
├── Generation handlers (one per model type)
│   ├── generate_tts()       → generator (streaming support)
│   ├── generate_turbo()     → generator (streaming support)
│   ├── generate_multilingual() → generator (streaming support)
│   └── generate_vc()        → direct return
├── Utility functions
│   ├── split_sentences()    → sentence splitter for streaming
│   ├── to_audio_tuple()     → wav tensor → (sr, np.ndarray)
│   └── load_sample_text()   → language → sample string
└── Gradio UI (gr.Blocks with 6 tabs)
```

### Streaming Strategy
Chatterbox has no native streaming — it generates complete audio per call.
Workaround: split input text on sentence boundaries → generate each sentence →
yield cumulative concatenated audio after each sentence → Gradio updates widget progressively.

```
Text → [sent1, sent2, sent3]
         ↓ generate
       [audio1] → yield (audio1)
       [audio1 + audio2] → yield (cumulative)
       [audio1 + audio2 + audio3] → yield (final)
```

### Tab Layout
| Tab | Model | Key UI Elements |
|---|---|---|
| 🗣️ Standard TTS | ChatterboxTTS | Text, ref audio, 6 param sliders, stream toggle |
| ⚡ Turbo TTS | ChatterboxTurboTTS | Text + tag insert buttons, ref audio, 4 sliders |
| 🌍 Multilingual | ChatterboxMultilingualTTS | Language dropdown, sample loader, text, ref audio |
| 🔄 Voice Conversion | ChatterboxVC | Source audio, target voice audio |
| 🔍 Watermark Check | perth lib | Audio upload, detection score + interpretation |
| ℹ️ About | — | Model comparison, param reference, architecture |

### Parameter Exposure per Model
| Param | Standard | Turbo | Multilingual |
|---|---|---|---|
| exaggeration | ✅ 0–1 | ❌ (unsupported) | ✅ 0–1 |
| cfg_weight | ✅ 0–1 | ❌ (unsupported) | ✅ 0–1 |
| temperature | ✅ 0.1–1.5 | ✅ 0.1–1.5 | ✅ 0.1–1.5 |
| repetition_penalty | ✅ 1.0–2.0 | ✅ 1.0–2.0 | ✅ 1.0–2.0 |
| min_p | ✅ 0–0.2 | ❌ (ignored) | ✅ 0–0.2 |
| top_p | ✅ 0.5–1.0 | ✅ 0.5–1.0 | ✅ 0.5–1.0 |
| top_k | ❌ | ✅ 100–2000 | ❌ |
| norm_loudness | ❌ | ✅ bool | ❌ |

---

## File Structure
```
chatterbox-demo/
├── pyproject.toml        ← uv project, deps: chatterbox-tts + gradio
├── .python-version       ← 3.11
├── app.py                ← full Gradio application (single file)
├── PLAN.md               ← this file
├── DECISIONS.md
├── NOTES.md
└── flow/                 ← dev flow templates
```

---

## Edge Cases

| Case | Handling |
|---|---|
| Empty text input | `gr.Warning` + early return |
| No voice reference | Use built-in default `conds.pt` voice |
| Turbo ref audio < 5s | Model assertion raised → `gr.Error` with clear message |
| Model download fails | `get_model()` catches exception → `gr.Error` with error string |
| Single sentence in streaming mode | Still works — one iteration, one yield |
| Tags in text with sentence splitting | Tags like `[laugh]` won't be cut mid-word (sentence split on `.!?`) |
| MPS not available on macOS | Falls back to CPU (handled by each model's `from_pretrained`) |

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Long first-run model download | High | Document clearly in UI + terminal progress visible |
| Slow generation on CPU | Medium | Warning in UI header when DEVICE=cpu |
| Memory: loading all 4 models simultaneously | Medium | Lazy loading — only load when tab is first used |
| Gradio API changes between 4.x and 5.x | Low | Pin `gradio>=4.40.0`, test basic audio output pattern |
| Turbo `assert` on short reference clip | Medium | Show clear UI note about >5s requirement |

---

## Out of Scope (for this demo)
- Real-time true streaming (byte-level chunked audio)
- User accounts / session management
- Saving outputs to disk
- Custom model fine-tuning
- Production deployment (Docker, auth, rate limiting)
- Batch processing multiple files

---

## Success Criteria
- [ ] All 4 model types generate audio successfully
- [ ] All configurable params are sliders/toggles (no code editing)
- [ ] Streaming mode shows progressive audio update sentence-by-sentence
- [ ] Watermark checker correctly identifies Chatterbox outputs
- [ ] App starts with `uv run app.py` after `uv sync`
- [ ] Works on MPS (Apple Silicon) without modification