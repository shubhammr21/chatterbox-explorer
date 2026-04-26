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

---

## Decision 6: Rename `adapters/primary` → `adapters/inbound` and `adapters/secondary` → `adapters/outbound`

### Decision
Rename the adapter sub-directories from `primary/secondary` to `inbound/outbound`.

### Why
- `inbound` / `outbound` communicates **direction of data flow** — inbound adapters are
  driven by external actors (UI, API clients), outbound adapters are driven by the
  application (models, infrastructure).
- `primary/secondary` is an older DDD term that is less intuitive for contributors
  unfamiliar with Evans-era literature.
- Modern hexagonal architecture tooling, blog posts, and reference codebases
  consistently use `inbound/outbound` — lowers the onboarding barrier.
- The rename is pure mechanics (no logic changes), so it carries zero behavioural risk.

### Alternatives Rejected
- **Keep `primary/secondary`**: Working but creates terminology friction for new
  contributors and diverges from current industry convention.
- **Use `driving/driven`**: Also valid hexagonal terminology but even less widely
  recognised than `inbound/outbound`.

### Trade-offs
- All 17 import references had to be updated across source and test files.
- Module docstrings referencing old paths also required manual updates.
- `pyproject.toml` per-file-ignores and coverage omit paths needed updating.

---

## Decision 7: Add `infrastructure/` Layer with `AppSettings` and `AppContainer`

### Decision
Introduce a dedicated `src/infrastructure/` package containing:
- `config.py` — `AppSettings` frozen dataclass (device, watermark_available)
- `container.py` — `AppContainer` declarative DI container

### Why
- The old `bootstrap.py` mixed two concerns: runtime config resolution and manual
  object construction wiring. Neither had a clear conceptual home.
- `AppSettings` belongs in infrastructure (not domain) because it is a runtime-resolved
  value produced by infrastructure probing (device detection, library availability check),
  not a business concept.
- `infrastructure/` is the conventional home for DI wiring in hexagonal architecture
  — it knows about both the port ABCs and the concrete adapter implementations.

### Alternatives Rejected
- **Keep everything in `bootstrap.py`**: Workable but the file conflates config
  resolution, object construction, and adapter selection in one procedural block.
- **Put `AppSettings` in `domain/models.py`**: Domain layer must be free of
  infrastructure concerns; device strings and library flags are not domain concepts.
- **Separate `wiring.py` and `settings.py` at the `src/` root**: Less discoverable
  than a dedicated `infrastructure/` package; breaks convention.

### Trade-offs
- Two new files to maintain; mitigated by comprehensive unit tests for both.
- `container.py` must remain a deferred import (inside `build_app()`) to preserve
  the compat-patch ordering guarantee — this constraint is documented explicitly.

---

## Decision 8: Replace Manual DI Wiring with `python-dependency-injector`

### Decision
Replace the manual object-construction wiring in `bootstrap.py` with a declarative
`AppContainer` using the `dependency-injector` library (v4.49.0).

### Why
- The previous manual wiring was imperative (~120 lines of explicit constructor calls)
  and not testable in isolation — to test that `TTSService` received the right repo,
  you had to call `build_app()` or repeat the construction manually in tests.
- `dependency-injector` provides:
  - `providers.Singleton` — singletons without manual caching
  - `providers.Configuration` — runtime config values injected by name
  - `providers.Object` — inject a callable reference (e.g. `set_seed`)
  - `provider.override()` — replace any provider in tests without touching source
- The override capability is the key gain: tests can now swap out
  `ChatterboxModelLoader` for a `MagicMock(spec=IModelRepository)` without
  loading torch, enabling fast container-level integration tests.
- `bootstrap.py` shrinks from ~120 lines to ~25 lines — pure coordination code.

### Library Selection Rationale
- **4.9k GitHub stars**, actively maintained (v4.49.0 released 2026)
- Written in Cython — provider resolution is fast
- First-class `mypy` stubs included
- Well-documented patterns for Flask, FastAPI, and plain Python apps
- `providers.Singleton` / `providers.Configuration` map directly to the patterns
  already used implicitly in the old manual wiring

### Alternatives Rejected
- **`injector`** (Google's library): Uses decorators (`@inject`, `@singleton`) which
  couple service classes to the DI framework — violates the hexagonal boundary.
- **`pinject`**: Less maintained, requires class-name-based binding which is fragile.
- **Keep manual wiring**: Zero new dependencies but loses testability and explicitness
  of the dependency graph.

### Trade-offs
- New `dependency-injector>=4.41.0` dependency added to `pyproject.toml`.
- The deferred-import constraint (container imported inside `build_app()`) must be
  maintained or compat patches will fire after chatterbox/torch code loads.
- `providers.pyx` (Cython source) cannot be parsed by the coverage tool — produces
  a harmless `CoverageWarning` that is expected and documented.
