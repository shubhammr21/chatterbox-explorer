# DECISIONS

---

## Decision 1: Gradio over Streamlit

### Decision
Use **Gradio** as the UI framework for the Chatterbox demo app.

### Why
- Chatterbox's own example apps (`gradio_tts_app.py`, `gradio_vc_app.py`) are built on Gradio — proven pattern
- `gr.Audio` component has native support for `(sample_rate, np.ndarray)` output — zero boilerplate
- Generator functions (`yield`) stream progressive output to `gr.Audio` out of the box
- Microphone input support built-in — no extra libraries
- Accordion, Slider, Dropdown — all needed controls exist natively
- `gr.themes` give polished UI without custom CSS effort

### Alternatives Rejected
- **Streamlit**: Audio output needs `st.audio()` with BytesIO buffers; streaming requires `st.empty()` hacks; no native mic input; session state is complex for ML model caching
- **FastAPI + React**: Overkill for a feature explorer demo; days of work vs. hours

### Trade-offs
- Gradio apps are less customisable for pixel-perfect UI — acceptable for a demo
- Gradio 5.x has some breaking changes from 4.x — pinned to `>=4.40.0` for safety

---

## Decision 2: uv as Package Manager

### Decision
Use **uv** for dependency management and virtual environment.

### Why
- Significantly faster than pip for dependency resolution and installation
- Single tool: replaces pip + venv + pip-tools
- `uv run` allows running scripts without manually activating the venv
- Lock file (`uv.lock`) ensures reproducible installs across machines
- Native Python version management via `.python-version`

### Alternatives Rejected
- **pip + venv**: Slower, more manual, no lock file by default
- **conda**: Heavy, overkill for this project
- **poetry**: Slower resolver, more config overhead

### Trade-offs
- uv is relatively new — but stable and widely adopted as of 2025
- Requires uv to be installed separately (one-time: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

---

## Decision 3: Lazy Model Loading with In-Memory Cache

### Decision
Load each Chatterbox model on first use and cache it in a module-level dict (`_MODEL_CACHE`).

### Why
- Models are large (350M–500M params); loading all 4 at startup is slow and memory-heavy
- Most demo users will only use 1–2 features per session
- Gradio's process stays alive between requests — in-memory cache persists across button clicks
- HuggingFace `from_pretrained()` already caches weights on disk (`~/.cache/huggingface/`)

### Alternatives Rejected
- **Load all at startup**: Too slow (30–90s startup depending on hardware), wastes VRAM/RAM
- **Reload on every request**: Extremely slow and wasteful
- **Disk-based cache (pickle)**: Unnecessary complexity when in-memory cache works

### Trade-offs
- First generate click for each model will be slow (model load + optional HF download)
- If server restarts, cache is cold again — acceptable for a demo

---

## Decision 4: Sentence-Level Streaming

### Decision
Implement "streaming" by splitting text into sentences, generating each separately, and yielding cumulative audio via Gradio's generator support.

### Why
- Chatterbox has no native token/chunk streaming — it generates full audio per call
- Sentence-level generation gives users progressive feedback ("audio is growing")
- Uses standard Python `yield` + Gradio's built-in generator output handling
- Each sentence is independently valid speech — no artifacts at boundaries when concatenated

### Alternatives Rejected
- **No streaming (full generation only)**: Poor UX for long texts — user waits with no feedback
- **Character-level or word-level splitting**: Too short → poor TTS quality (no context)
- **Async threading with partial audio**: Complex; Gradio's generator model handles this cleanly

### Trade-offs
- Voice consistency may vary slightly between sentences (model re-conditions per sentence)
- Silence gaps can appear at sentence joins — mitigated by the model's own silence handling
- Sentence splitting by `.!?` may not work perfectly for all text styles (e.g., abbreviations)

---

## Decision 5: Single-File App (`app.py`)

### Decision
Put all application code in a single `app.py` file.

### Why
- This is a **demo prototype**, not production code
- Single file is easier to share, inspect, and run
- No import complexity or module resolution issues
- Aligns with "prefer simplicity" rule for this phase

### Alternatives Rejected
- **src/ package structure**: Appropriate for production; overkill for a demo
- **Separate handler modules**: Useful for large apps; unnecessary here

### Trade-offs
- File will be ~400 lines — manageable but not ideal for long-term maintenance
- If this evolves into a real product, restructure into modules (see NOTES.md)
