# PLAN — Chatterbox TTS Explorer  (v3 — Hexagonal Architecture + TDD)

## Goal
Refactor the current monolithic `app.py` into the Hexagonal (Ports & Adapters)
architecture so that future delivery adapters — REST API, CLI, gRPC — can be
added without touching domain or business logic.  Every change must be verified
by a test written **before** the implementation (TDD).

---

## Why Hexagonal?

```
┌─────────────────────────────────────────────────────────┐
│              Driving Adapters  (Primary)                 │
│         Gradio UI  │  REST API  │  CLI  │  gRPC          │
└──────────────────────────┬──────────────────────────────┘
                           │  calls
             ┌─────────────▼─────────────┐
             │       Input Ports         │  ← ABCs the outside world calls
             └─────────────┬─────────────┘
                           │
             ┌─────────────▼─────────────┐
             │      Domain Services      │  ← pure orchestration logic
             │  TTSService               │
             │  VoiceConversionService   │
             │  ModelManagerService      │
             │  WatermarkService         │
             └─────────────┬─────────────┘
                           │  depends on
             ┌─────────────▼─────────────┐
             │       Output Ports        │  ← ABCs the domain needs from infra
             └─────────────┬─────────────┘
                           │  implemented by
┌──────────────────────────▼──────────────────────────────┐
│              Driven Adapters  (Secondary)                │
│  ChatterboxModelLoader  │  TorchAudioPreprocessor        │
│  PsutilMemoryMonitor    │  PerThWatermarkDetector        │
└─────────────────────────────────────────────────────────┘
```

**Rule:** Dependency arrows point **inward only**.  Domain never imports Gradio,
torch, or chatterbox.  Adapters never import each other directly.

---

## Final Directory Structure

```
chatterbox-demo/
├── app.py                                   # ENTRY POINT (~30 lines)
├── compat.py                                # CROSS-CUTTING (unchanged)
├── pyproject.toml                           # updated — src layout + pytest dev dep
├── src/
│   └── chatterbox_explorer/
│       ├── __init__.py
│       ├── config.py                        # AppConfig dataclass (device, watermark flag)
│       ├── logging_config.py                # logging + warning suppression setup
│       ├── bootstrap.py                     # DI root — wires all adapters → services
│       │
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── models.py                    # Pure dataclasses — zero framework deps
│       │   ├── languages.py                 # LANGUAGE_OPTIONS, SAMPLE_TEXTS, PARA_TAGS
│       │   └── presets.py                   # PRESETS_TTS, PRESETS_TURBO (merged canonical)
│       │
│       ├── ports/
│       │   ├── __init__.py
│       │   ├── input.py                     # ITTSService, ITurboTTSService, … (ABC)
│       │   └── output.py                    # IModelRepository, IAudioPreprocessor, … (ABC)
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   ├── tts.py                       # TTSService, TurboTTSService, MultilingualTTSService
│       │   ├── voice_conversion.py          # VoiceConversionService
│       │   ├── model_manager.py             # ModelManagerService
│       │   └── watermark.py                 # WatermarkService
│       │
│       └── adapters/
│           ├── __init__.py
│           ├── secondary/
│           │   ├── __init__.py
│           │   ├── device.py                # detect_device(), set_seed()
│           │   ├── model_loader.py          # ChatterboxModelLoader  (IModelRepository)
│           │   ├── audio.py                 # TorchAudioPreprocessor (IAudioPreprocessor)
│           │   ├── memory.py                # PsutilMemoryMonitor    (IMemoryMonitor)
│           │   └── watermark.py             # PerThWatermarkDetector (IWatermarkDetector)
│           └── primary/
│               ├── __init__.py
│               └── gradio/
│                   ├── __init__.py
│                   ├── handlers.py          # All Gradio event callbacks
│                   └── ui.py                # build_demo() → gr.Blocks
│
└── tests/
    ├── conftest.py                          # shared fixtures + mock factories
    ├── unit/
    │   ├── domain/
    │   │   ├── test_models.py               # TTSRequest, AudioResult, etc.
    │   │   ├── test_languages.py            # LANGUAGE_OPTIONS, SAMPLE_TEXTS
    │   │   └── test_presets.py              # preset lookup, param ranges
    │   ├── services/
    │   │   ├── test_tts_service.py          # mock IModelRepository + IAudioPreprocessor
    │   │   ├── test_voice_conversion_service.py
    │   │   ├── test_model_manager_service.py
    │   │   └── test_watermark_service.py
    │   └── adapters/
    │       ├── test_audio_preprocessor.py   # 40 ms alignment logic
    │       └── test_device.py               # detect_device, set_seed
    └── integration/
        └── test_model_load.py               # existing test — runs real models
```

---

## Domain Models  (`src/chatterbox_explorer/domain/models.py`)

All dataclasses — **zero framework imports**.

| Class | Fields | Notes |
|---|---|---|
| `TTSRequest` | text, ref_audio_path, exaggeration, cfg_weight, temperature, rep_penalty, min_p, top_p, seed, streaming | Standard model params |
| `TurboTTSRequest` | text, ref_audio_path, temperature, top_k, top_p, rep_penalty, min_p, norm_loudness, seed, streaming | Turbo-specific params |
| `MultilingualTTSRequest` | text, language, ref_audio_path, exaggeration, cfg_weight, temperature, rep_penalty, min_p, top_p, seed, streaming | language = ISO 639-1 code |
| `VoiceConversionRequest` | source_audio_path, target_voice_path | Audio-only, no text |
| `AudioResult` | sample_rate: int, samples: np.ndarray (float32) | `.duration_s` property |
| `ModelStatus` | key, display_name, class_name, description, params, size_gb, in_memory, on_disk | Per-model state |
| `MemoryStats` | sys_total_gb, sys_used_gb, sys_avail_gb, sys_percent, proc_rss_gb, device_name, device_driver_gb, device_max_gb | Nullable device fields |
| `WatermarkResult` | score, verdict, message, available | verdict ∈ {detected, not_detected, inconclusive, unavailable} |
| `AppConfig` | device, watermark_available | Created once in bootstrap |

---

## Port Contracts

### Input Ports (`ports/input.py`)  — what calling code invokes

```python
class ITTSService(ABC):
    def generate(self, request: TTSRequest) -> AudioResult: ...
    def generate_stream(self, request: TTSRequest) -> Iterator[AudioResult]: ...

class ITurboTTSService(ABC):
    def generate(self, request: TurboTTSRequest) -> AudioResult: ...
    def generate_stream(self, request: TurboTTSRequest) -> Iterator[AudioResult]: ...

class IMultilingualTTSService(ABC):
    def generate(self, request: MultilingualTTSRequest) -> AudioResult: ...
    def generate_stream(self, request: MultilingualTTSRequest) -> Iterator[AudioResult]: ...

class IVoiceConversionService(ABC):
    def convert(self, request: VoiceConversionRequest) -> AudioResult: ...

class IModelManagerService(ABC):
    def load(self, key: str) -> str: ...           # returns status message
    def unload(self, key: str) -> str: ...         # returns status message
    def download(self, key: str) -> Iterator[str]: # yields progress lines
    def get_all_status(self) -> list[ModelStatus]: ...
    def get_memory_stats(self) -> MemoryStats: ...

class IWatermarkService(ABC):
    def detect(self, audio_path: str) -> WatermarkResult: ...
```

### Output Ports (`ports/output.py`) — what services need from infrastructure

```python
class IModelRepository(ABC):
    def get_model(self, key: str) -> Any: ...
    def is_loaded(self, key: str) -> bool: ...
    def is_cached_on_disk(self, key: str) -> bool: ...
    def unload(self, key: str) -> None: ...
    def download(self, key: str) -> Iterator[str]: ...   # yields filenames

class IAudioPreprocessor(ABC):
    def preprocess(self, path: str | None) -> str | None: ...

class IMemoryMonitor(ABC):
    def get_stats(self) -> MemoryStats: ...

class IWatermarkDetector(ABC):
    def detect(self, audio_path: str) -> float: ...   # returns raw score
    def is_available(self) -> bool: ...
```

---

## Service Responsibilities

| Service | Depends on | Key behaviours |
|---|---|---|
| `TTSService` | `IModelRepository`, `IAudioPreprocessor` | Raises `ValueError` for empty text; splits sentences for streaming; calls `set_seed`; never imports gradio |
| `TurboTTSService` | same | Same as above; catches `AssertionError` from Turbo 5-second check and re-raises as `ValueError` |
| `MultilingualTTSService` | same | Parses language code; same streaming logic |
| `VoiceConversionService` | `IModelRepository`, `IAudioPreprocessor` | Raises `ValueError` for missing paths |
| `ModelManagerService` | `IModelRepository`, `IMemoryMonitor` | Delegates load/unload/download; aggregates status |
| `WatermarkService` | `IWatermarkDetector` | Wraps raw score into `WatermarkResult` with verdict |

**Services NEVER:** call `gr.Warning`, `gr.Error`, import torch, import gradio.

---

## Secondary Adapter Responsibilities

| Adapter | Implements | Key details |
|---|---|---|
| `ChatterboxModelLoader` | `IModelRepository` | Holds `_cache: dict`; lazy loads from HF; `DEVICE` injected via constructor |
| `TorchAudioPreprocessor` | `IAudioPreprocessor` | 40 ms frame alignment; writes aligned wav to tempfile |
| `PsutilMemoryMonitor` | `IMemoryMonitor` | 1.5 s TTL cache; MPS `driver_allocated_memory()` |
| `PerThWatermarkDetector` | `IWatermarkDetector` | Returns 0.0 + `available=False` when no-op watermarker active |

---

## Primary Adapter — Gradio

`adapters/primary/gradio/handlers.py`
- All Gradio event callbacks (`generate_tts`, `generate_turbo`, etc.)
- Receives domain service instances via constructor injection
- Translates `ValueError` → `gr.Warning`, other exceptions → `gr.Error`
- Converts `AudioResult.samples` (float32) → int16 tuple for `gr.Audio`
- `render_manager_html()` lives here (view rendering is adapter concern)

`adapters/primary/gradio/ui.py`
- Single public function: `build_demo(services, config) -> gr.Blocks`
- `with gr.Blocks(...) as demo:` is **inside this function** (not module-level)
- Receives all service instances + `AppConfig` as parameters
- No direct secondary adapter imports

---

## Bootstrap (`src/chatterbox_explorer/bootstrap.py`)

```
def build_app(device: str) -> tuple[gr.Blocks, AppConfig]:
    # 1. Create secondary adapters (inject device)
    model_repo   = ChatterboxModelLoader(device)
    preprocessor = TorchAudioPreprocessor()
    mem_monitor  = PsutilMemoryMonitor(device)
    wm_detector  = PerThWatermarkDetector()

    # 2. Create domain services (inject secondary adapters via ports)
    tts_svc   = TTSService(model_repo, preprocessor)
    turbo_svc = TurboTTSService(model_repo, preprocessor)
    mtl_svc   = MultilingualTTSService(model_repo, preprocessor)
    vc_svc    = VoiceConversionService(model_repo, preprocessor)
    mgr_svc   = ModelManagerService(model_repo, mem_monitor)
    wm_svc    = WatermarkService(wm_detector)

    config = AppConfig(device=device, watermark_available=wm_detector.is_available())

    # 3. Build primary adapter (inject services)
    from chatterbox_explorer.adapters.primary.gradio.ui import build_demo
    demo = build_demo(
        tts=tts_svc, turbo=turbo_svc, mtl=mtl_svc,
        vc=vc_svc, manager=mgr_svc, watermark=wm_svc,
        config=config,
    )
    return demo, config
```

---

## TDD Strategy

### Principle
**Red → Green → Refactor** for every new unit.  Integration tests (real model
loading) are kept in `tests/integration/` and run separately from unit tests.

### Test Levels

| Level | Location | Framework deps | Run time |
|---|---|---|---|
| Unit — domain | `tests/unit/domain/` | None | < 1 s |
| Unit — services | `tests/unit/services/` | `unittest.mock` | < 1 s |
| Unit — adapters | `tests/unit/adapters/` | `torch`, `torchaudio` (small) | < 5 s |
| Integration | `tests/integration/` | Real chatterbox models | 30–120 s |

### What to test per layer

**Domain models** — field defaults, validation, `AudioResult.duration_s` property.

**Presets** — all 10 standard + 6 turbo presets present; every param in valid
slider range; `get_standard_preset()` returns correct values.

**Languages** — 23 entries in `LANGUAGE_OPTIONS`; all have `SAMPLE_TEXTS`;
all codes in `LANGUAGE_AUDIO_DEFAULTS` are ISO 639-1.

**Services (mocked ports):**
- `ValueError` raised on empty text
- Preprocessor is called with the ref path from the request
- Model `generate()` is called with correct kwargs
- `split_sentences` splits on `.  !  ?` boundaries correctly
- Streaming yields multiple `AudioResult` objects (one per sentence)
- `ModelManagerService.unload()` calls `repo.unload()` and `monitor.get_stats()`

**Audio preprocessor:**
- `None` input → `None` output (no crash)
- Already-aligned audio → same path returned (no tempfile written)
- Unaligned audio → new path with sample count divisible by `frame_samples`

**Device:**
- `set_seed(0)` is a no-op (does not raise)
- `set_seed(42)` sets torch seed without error

### Mock Pattern

```python
# tests/conftest.py
@pytest.fixture
def mock_model():
    m = MagicMock()
    m.sr = 24000
    m.generate.return_value = torch.zeros(1, 24000)
    return m

@pytest.fixture
def mock_model_repo(mock_model):
    repo = MagicMock(spec=IModelRepository)
    repo.get_model.return_value = mock_model
    repo.is_loaded.return_value = False
    return repo

@pytest.fixture
def mock_preprocessor():
    p = MagicMock(spec=IAudioPreprocessor)
    p.preprocess.side_effect = lambda path: path   # passthrough
    return p
```

### TDD Order of Implementation

```
Phase 1  — domain/models.py       + tests/unit/domain/test_models.py
Phase 2  — domain/presets.py      + tests/unit/domain/test_presets.py
Phase 3  — domain/languages.py    + tests/unit/domain/test_languages.py
Phase 4  — ports/input.py  (ABC — no tests needed, verified by service tests)
Phase 5  — ports/output.py (ABC — same)
Phase 6  — services/tts.py        + tests/unit/services/test_tts_service.py
Phase 7  — services/voice_conversion.py + test
Phase 8  — services/model_manager.py    + test
Phase 9  — services/watermark.py        + test
Phase 10 — adapters/secondary/audio.py  + tests/unit/adapters/test_audio_preprocessor.py
Phase 11 — adapters/secondary/device.py + tests/unit/adapters/test_device.py
Phase 12 — adapters/secondary/model_loader.py (integration — uses real HF cache)
Phase 13 — adapters/secondary/memory.py (integration — uses real psutil)
Phase 14 — logging_config.py + bootstrap.py
Phase 15 — adapters/primary/gradio/handlers.py + ui.py
Phase 16 — app.py (thin entry point)
Phase 17 — Run all tests; fix regressions
Phase 18 — git commit
```

---

## Constraints

- Domain layer: **zero imports from torch, gradio, chatterbox, huggingface_hub, psutil**
- Services: zero imports from gradio; may import numpy for `AudioResult`
- Secondary adapters: may import torch, torchaudio, chatterbox, psutil, huggingface_hub
- Primary Gradio adapter: may import gradio; receives services via DI only
- `compat.py`: stays at root, unchanged
- `test_model_load.py` (existing integration test): must still pass after refactor
- `app.py` (new): ≤ 40 lines; only parses args + calls bootstrap + launches

---

## Risks

| Risk | Mitigation |
|---|---|
| Gradio `with gr.Blocks` runs at module import | Wrap in `build_demo()` function |
| `_MODEL_CACHE` global state breaks test isolation | Make it an instance variable on `ChatterboxModelLoader` |
| `render_manager_html` calls secondary adapter functions | Move HTML rendering entirely to primary adapter; secondary returns data objects |
| `presets.py` root file and `PRESETS_TTS` in `app.py` diverge | Single canonical source in `domain/presets.py`; delete both old files |
| 12 hardcoded lambda closures for Model Manager tab | Replace with a loop over `MODEL_REGISTRY.keys()` |
| `_WATERMARK_AVAILABLE` referenced at UI build time | Pass via `AppConfig` to `build_demo()` |

---

## Success Criteria

- [ ] `uv run pytest tests/unit/ -v` — all unit tests GREEN (no real models needed)
- [ ] `uv run pytest tests/integration/ -v` — integration tests GREEN (existing test_model_load.py)
- [ ] `uv run python app.py` — app starts and all 7 tabs function identically to current
- [ ] Domain layer has zero imports from torch/gradio/chatterbox (verified by test)
- [ ] A new delivery adapter (e.g. REST) can be added by only touching `adapters/primary/`
- [ ] `uv run pytest --co -q` shows ≥ 40 collected test items
