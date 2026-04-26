# PLAN — Structural Refactoring: Inbound/Outbound + Infrastructure Layer + Dependency Injector

> **Selected Flow:** Refactor (`flow/refactor.md`)
> **Reason:** Improving structure without changing behavior — renaming adapter directories,
> adding an infrastructure layer, and replacing manual DI wiring with a declarative container.
> **Status:** Planning Phase

---

## 0. Router Output

| Field        | Value                                                                       |
|--------------|-----------------------------------------------------------------------------|
| Flow         | `flow/refactor.md`                                                          |
| Trigger      | Design inconsistency: `primary/secondary` naming, no infrastructure layer,  |
|              | manual DI wiring in `bootstrap.py`                                          |
| Next Step    | Execute planning checklist → approve → implement incrementally              |

---

## 1. Problem Statement

The current codebase has solid hexagonal architecture in place, but three structural issues
reduce clarity and make future growth harder:

### Issue 1 — Adapter Naming (`primary/secondary` vs `inbound/outbound`)

```
current:                         target:
adapters/primary/gradio/     →   adapters/inbound/gradio/
adapters/primary/rest/       →   adapters/inbound/rest/
adapters/secondary/audio.py  →   adapters/outbound/audio.py
adapters/secondary/device.py →   adapters/outbound/device.py
...                          →   ...
```

`inbound` / `outbound` is the dominant convention in modern hexagonal architecture
literature and tooling. It communicates *direction of flow* (inbound = driven by external
actors, outbound = driven by the application). `primary/secondary` is an older DDD term
that is less intuitive for new contributors.

### Issue 2 — No `infrastructure/` Layer

Configuration and dependency-wiring currently live in `bootstrap.py` as a single
procedural function. There is no clear home for:
- Application-level settings (device, watermark flag, server options)
- The DI container that wires everything together

A dedicated `infrastructure/` package makes these concerns explicit and
separate from domain logic.

### Issue 3 — Manual Dependency Injection in `bootstrap.py`

`build_app()` manually constructs and wires every object:

```python
model_repo   = ChatterboxModelLoader(device)
preprocessor = TorchAudioPreprocessor()
tts_svc      = TTSService(model_repo, preprocessor, seed_setter=set_seed)
# ...
```

This is imperative and hard to test or override in isolation.
`python-dependency-injector` provides a declarative container that:
- Makes the dependency graph explicit and readable
- Enables provider overriding in tests (`container.model_loader.override(mock)`)
- Manages singleton lifetimes automatically
- Scales cleanly when new services / adapters are added

---

## 2. What Does NOT Change

To be explicit about scope — the following are **frozen** during this refactoring:

| Layer                    | Status  | Reason                                              |
|--------------------------|---------|-----------------------------------------------------|
| `domain/`                | FROZEN  | Pure models, zero deps, already correct             |
| `ports/input.py`         | FROZEN  | Port ABCs are well-defined, no issues               |
| `ports/output.py`        | FROZEN  | Port ABCs are well-defined, no issues               |
| `services/tts.py`        | FROZEN  | Domain logic is correct and well-tested             |
| `services/turbo_tts.py`  | FROZEN  | Same as above                                       |
| `services/multilingual`  | FROZEN  | Same as above                                       |
| `services/vc.py`         | FROZEN  | Same as above                                       |
| `services/watermark.py`  | FROZEN  | Same as above                                       |
| `services/model_manager` | FROZEN  | Same as above                                       |
| All adapter internals    | FROZEN  | Logic unchanged — only location + imports change    |
| All tests                | FROZEN* | Only import paths updated, assertions unchanged     |
| `compat.py`              | FROZEN  | Migration shims, no structural concern              |
| `logging_config.py`      | FROZEN  | Cross-cutting concern, correct as-is                |

> (*) Tests get import-path updates only — no assertion or fixture logic changes.

**No `application/` use-case layer is added.** For a TTS inference service,
the domain services ARE the use cases. Wrapping them in a separate application
layer would be forced abstraction with no benefit at this scale.

---

## 3. Target Directory Structure

```
src/
│
├── domain/                         # UNCHANGED — pure domain layer
│   ├── models.py
│   ├── types.py
│   ├── presets.py
│   └── languages.py
│
├── ports/                          # UNCHANGED — port ABCs
│   ├── input.py                    # ITTSService, ITurboTTSService, ...
│   └── output.py                   # IModelRepository, IAudioPreprocessor, ...
│
├── services/                       # UNCHANGED — domain service implementations
│   ├── tts.py
│   ├── voice_conversion.py
│   ├── model_manager.py
│   └── watermark.py
│
├── adapters/
│   ├── inbound/                    # RENAMED from primary/
│   │   ├── __init__.py
│   │   ├── gradio/
│   │   │   ├── __init__.py
│   │   │   ├── ui.py               # build_demo() — unchanged logic
│   │   │   └── handlers.py         # GradioHandlers — unchanged logic
│   │   └── rest/
│   │       └── __init__.py         # Future FastAPI stub — unchanged
│   │
│   └── outbound/                   # RENAMED from secondary/
│       ├── __init__.py
│       ├── audio.py                # TorchAudioPreprocessor — unchanged
│       ├── device.py               # detect_device, set_seed — unchanged
│       ├── memory.py               # PsutilMemoryMonitor — unchanged
│       ├── model_loader.py         # ChatterboxModelLoader — unchanged
│       └── watermark.py            # PerThWatermarkDetector — unchanged
│
├── infrastructure/                 # NEW — config + DI wiring
│   ├── __init__.py
│   ├── config.py                   # NEW — AppSettings dataclass
│   └── container.py                # NEW — AppContainer (dependency-injector)
│
├── bootstrap.py                    # SIMPLIFIED — thin shell → delegates to container
├── cli.py                          # UPDATED — import paths only
├── compat.py                       # UNCHANGED
├── logging_config.py               # UNCHANGED
├── __init__.py                     # UNCHANGED
└── __main__.py                     # UNCHANGED
```

---

## 4. New File Contracts

### 4.1 `infrastructure/config.py`

Owns application-level settings that are resolved once at startup and passed
through the system. Currently these live as bare variables inside `build_app()`.

```python
# infrastructure/config.py
@dataclass(frozen=True)
class AppSettings:
    watermark_available: bool
    device: DeviceType          # "cuda" | "mps" | "cpu"
```

**Why a separate file?**
- Single source of truth for runtime config
- Enables the DI container to reference settings by name
- Makes it obvious what "application config" consists of

### 4.2 `infrastructure/container.py`

Declarative DI container using `dependency-injector`.

```python
# infrastructure/container.py
from dependency_injector import containers, providers

class AppContainer(containers.DeclarativeContainer):

    # ── Configuration ──────────────────────────────────────────────────
    config = providers.Configuration()

    # ── Outbound adapters (infrastructure) ─────────────────────────────
    device = providers.Callable(detect_device)

    model_loader = providers.Singleton(
        ChatterboxModelLoader,
        device=device,
    )
    audio_preprocessor = providers.Singleton(TorchAudioPreprocessor)
    memory_monitor     = providers.Singleton(PsutilMemoryMonitor, device=device)
    watermark_detector = providers.Singleton(
        PerThWatermarkDetector,
        available=config.watermark_available,
    )

    # ── Domain services ────────────────────────────────────────────────
    tts_service = providers.Singleton(
        TTSService,
        model_repo=model_loader,
        preprocessor=audio_preprocessor,
        seed_setter=providers.Object(set_seed),
    )
    turbo_service = providers.Singleton(
        TurboTTSService,
        model_repo=model_loader,
        preprocessor=audio_preprocessor,
        seed_setter=providers.Object(set_seed),
    )
    multilingual_service = providers.Singleton(
        MultilingualTTSService,
        model_repo=model_loader,
        preprocessor=audio_preprocessor,
        seed_setter=providers.Object(set_seed),
    )
    vc_service = providers.Singleton(
        VoiceConversionService,
        model_repo=model_loader,
        preprocessor=audio_preprocessor,
    )
    model_manager_service = providers.Singleton(
        ModelManagerService,
        model_repo=model_loader,
        mem_monitor=memory_monitor,
    )
    watermark_service = providers.Singleton(
        WatermarkService,
        detector=watermark_detector,
    )

    # ── App config value-object ────────────────────────────────────────
    app_config = providers.Factory(
        AppConfig,
        device=device,
        watermark_available=config.watermark_available,
    )
```

**Critical design constraint preserved:**
All imports inside `container.py` fire only when that module is imported.
`bootstrap.py` imports `container.py` lazily (inside the function body),
so compat patches in `cli.py` still fire before any chatterbox/torch code loads.
The import order guarantee is unchanged.

### 4.3 `bootstrap.py` (simplified)

```python
def build_app(watermark_available: bool) -> tuple[gr.Blocks, AppConfig]:
    # Deferred import ensures compat patches fire first (unchanged contract)
    from infrastructure.container import AppContainer
    from adapters.inbound.gradio.ui import build_demo

    container = AppContainer()
    container.config.watermark_available.from_value(watermark_available)

    demo = build_demo(
        tts=container.tts_service(),
        turbo=container.turbo_service(),
        mtl=container.multilingual_service(),
        vc=container.vc_service(),
        manager=container.model_manager_service(),
        watermark=container.watermark_service(),
        config=container.app_config(),
    )
    return demo, container.app_config()
```

`bootstrap.py` shrinks from ~120 lines to ~20 lines. All wiring knowledge
moves into the container where it is declarative, inspectable, and overridable.

---

## 5. Dependency Changes

### Add to `pyproject.toml`

```toml
[project]
dependencies = [
    "chatterbox-tts",
    "dependency-injector>=4.41.0",    # NEW — declarative DI container
    "diffusers>=0.29.0,<1.0.0",
    "gradio>=4.40.0",
    "peft>=0.6.0",
]
```

**Why `dependency-injector`?**
- 4.9k stars, actively maintained, production-grade
- Provides `Singleton`, `Factory`, `Callable`, `Object`, `Configuration` providers
  covering every pattern used in this codebase
- First-class test support via `container.provider.override(mock)`
- Avoids the need for global module state or manual singleton guards
- Written in Cython — fast provider resolution
- Preferred over alternatives (pinject, injector) due to explicit provider graph
  and excellent documentation

### Update `pyproject.toml` — ruff isort

```toml
[tool.ruff.lint.isort]
known-first-party = [
    "domain", "ports", "services", "adapters",
    "bootstrap", "logging_config", "cli",
    "infrastructure",                            # NEW
]
```

### Update `pyproject.toml` — coverage omit

```toml
[tool.coverage.run]
omit = [
    "src/adapters/inbound/*",      # renamed from primary
    "src/cli.py",
    "src/bootstrap.py",
    "src/__init__.py",
    "src/__main__.py",
]
```

---

## 6. Complete Import Change Map

Every reference to the old paths must be updated. Full inventory:

### Source files

| File                                      | Old import                            | New import                           |
|-------------------------------------------|---------------------------------------|--------------------------------------|
| `src/bootstrap.py`                        | `adapters.secondary.audio`            | `adapters.outbound.audio`            |
| `src/bootstrap.py`                        | `adapters.secondary.device`           | `adapters.outbound.device`           |
| `src/bootstrap.py`                        | `adapters.secondary.memory`           | `adapters.outbound.memory`           |
| `src/bootstrap.py`                        | `adapters.secondary.model_loader`     | `adapters.outbound.model_loader`     |
| `src/bootstrap.py`                        | `adapters.secondary.watermark`        | `adapters.outbound.watermark`        |
| `src/bootstrap.py`                        | `adapters.primary.gradio.ui`          | `adapters.inbound.gradio.ui`         |
| `src/cli.py`                              | `adapters.primary.gradio.ui`          | `adapters.inbound.gradio.ui`         |
| `src/adapters/primary/gradio/handlers.py` | `adapters.secondary.audio`            | `adapters.outbound.audio`            |
| `src/adapters/primary/gradio/ui.py`       | `adapters.primary.gradio.handlers`    | `adapters.inbound.gradio.handlers`   |
| `src/infrastructure/container.py`         | (new — uses `adapters.outbound.*`)    | —                                    |

### Test files

| File                                          | Old import / patch                              | New import / patch                             |
|-----------------------------------------------|-------------------------------------------------|------------------------------------------------|
| `tests/unit/adapters/test_audio_preprocessor` | `adapters.secondary.audio`                      | `adapters.outbound.audio`                      |
| `tests/unit/adapters/test_device`             | `adapters.secondary.device`                     | `adapters.outbound.device`                     |
| `tests/unit/adapters/test_memory_monitor`     | `adapters.secondary.memory`                     | `adapters.outbound.memory`                     |
| `tests/unit/adapters/test_model_loader`       | `adapters.secondary.model_loader`               | `adapters.outbound.model_loader`               |
| `tests/unit/adapters/test_model_loader`       | `patch("adapters.secondary.model_loader.gc")`   | `patch("adapters.outbound.model_loader.gc")`   |
| `tests/unit/adapters/test_model_loader`       | `patch("adapters.secondary.model_loader.MODEL…")` | `patch("adapters.outbound.model_loader.MODEL…")` |
| `tests/unit/adapters/test_watermark_detector` | `adapters.secondary.watermark`                  | `adapters.outbound.watermark`                  |
| `tests/unit/adapters/test_watermark_detector` | `logger="adapters.secondary.watermark"`         | `logger="adapters.outbound.watermark"`         |
| `tests/unit/adapters/test_device`             | docstring mentions `.secondary.`                | update docstring                               |

### Docstrings / comments (non-functional but should be consistent)

| File                                   | Content to update                                    |
|----------------------------------------|------------------------------------------------------|
| `src/adapters/primary/rest/__init__.py`| "Entry point would be: adapters.primary.rest.app:app" |
| `src/bootstrap.py`                     | "Secondary adapters (infrastructure layer)" comment   |
| `app.py`                               | Directory listing in module docstring                 |

---

## 7. Implementation Phases

Phases are ordered for minimal breakage. Each phase must leave tests passing
before the next phase begins.

### Phase 1 — Directory Rename (no logic change)

**Goal:** Move files, keep all imports temporarily broken, fix imports in same commit.

**Steps:**
1. Copy `src/adapters/primary/` → `src/adapters/inbound/`
2. Copy `src/adapters/secondary/` → `src/adapters/outbound/`
3. Update all import strings across all files (see table above)
4. Delete old `src/adapters/primary/` and `src/adapters/secondary/`
5. Verify: `uv run pytest tests/unit/ -x` — all tests pass

**Files touched:** All files in the import change map above.

**Risk:** Low. Pure rename + import update. No logic changes.

### Phase 2 — Add `infrastructure/` Package

**Goal:** Create the package with `config.py` and stub `container.py`.

**Steps:**
1. Create `src/infrastructure/__init__.py`
2. Create `src/infrastructure/config.py` with `AppSettings` dataclass
3. Create `src/infrastructure/container.py` stub (empty container, no providers yet)
4. Update `pyproject.toml` isort known-first-party
5. Verify: `uv run pytest tests/unit/ -x` — all tests pass (nothing wired yet)

**Files touched:** 3 new files + `pyproject.toml`.

**Risk:** Minimal. No existing code changes.

### Phase 3 — Add `dependency-injector` + Wire `AppContainer`

**Goal:** Implement the full declarative container, simplify `bootstrap.py`.

**Steps:**
1. Add `dependency-injector>=4.41.0` to `pyproject.toml` dependencies
2. Run `uv lock` to update the lockfile
3. Implement `infrastructure/container.py` (all providers wired)
4. Refactor `bootstrap.py` to use `AppContainer` (replaces manual wiring)
5. Update `cli.py` if needed (minimal changes expected)
6. Verify: `uv run pytest tests/unit/ -x` — all tests pass

**Files touched:** `pyproject.toml`, `uv.lock`, `infrastructure/container.py`, `bootstrap.py`.

**Risk:** Medium. Core wiring changes. Mitigated by: unit tests don't call `build_app()`,
so failures would be isolated to the integration test scope.

### Phase 4 — Update Coverage Config + Docstrings

**Goal:** Keep config accurate, clean up all documentation references.

**Steps:**
1. Update `pyproject.toml` coverage `omit` paths (`primary→inbound`)
2. Update docstrings in `rest/__init__.py`, `bootstrap.py`, `app.py`
3. Update `tests/unit/adapters/test_device.py` docstring
4. Verify: `uv run pytest --cov=src tests/unit/ -x` — coverage threshold maintained

**Files touched:** `pyproject.toml` + 4 docstrings.

**Risk:** None. Documentation only.

### Phase 5 — Final Validation

```bash
uv run pytest tests/ -x --tb=short          # all tests pass
uv run pytest --cov=src tests/unit/          # coverage ≥ 95%
uv run ruff check src/ tests/               # zero lint errors
uv run ruff format --check src/ tests/      # formatting clean
```

---

## 8. Risk Register

| Risk                              | Likelihood | Impact | Mitigation                                      |
|-----------------------------------|------------|--------|-------------------------------------------------|
| Missed import reference           | Medium     | Low    | `grep -r "adapters.primary\|adapters.secondary"` before closing |
| `dependency-injector` version conflict | Low   | Medium | Pin to `>=4.41.0` (tested with Singleton + Callable) |
| Singleton lifecycle mismatch      | Low        | Medium | Container is constructed once in `build_app()` — same as current |
| `device` provider called twice    | Low        | Low    | `providers.Callable` calls fn each time; wrap in `providers.Singleton` if needed |
| Lazy import order broken          | Low        | High   | `container.py` imported inside `build_app()` body — patches guaranteed to fire first |
| Coverage drops below 95%          | Low        | Low    | Infrastructure files (container, config) excluded from measurement or tested via unit tests |
| Patch strings in tests broken     | Medium     | Medium | Full table in Section 6 covers every patch string; grep validation in Phase 5 |

---

## 9. Testing Strategy for New Code

### `infrastructure/config.py`
- Unit test `AppSettings` creation with known values
- Verify frozen dataclass rejects mutation

### `infrastructure/container.py`
- Unit test: container wires without raising (all providers resolve)
- Unit test: `container.model_loader.override(mock)` correctly replaces the singleton
- Do NOT test with real chatterbox models — use mock providers

```python
# Example container override test (no real models needed)
def test_container_override():
    from infrastructure.container import AppContainer
    container = AppContainer()
    container.config.watermark_available.from_value(False)

    mock_loader = Mock(spec=IModelRepository)
    with container.model_loader.override(mock_loader):
        tts = container.tts_service()
        assert tts._repo is mock_loader
```

---

## 10. Success Criteria

- [ ] `src/adapters/primary/` and `src/adapters/secondary/` no longer exist
- [ ] `src/adapters/inbound/` and `src/adapters/outbound/` exist with identical content
- [ ] `src/infrastructure/config.py` contains `AppSettings`
- [ ] `src/infrastructure/container.py` contains `AppContainer` with all providers declared
- [ ] `bootstrap.py` is ≤ 30 lines — delegates entirely to `AppContainer`
- [ ] `dependency-injector>=4.41.0` is in `pyproject.toml` and `uv.lock`
- [ ] Zero `grep` hits for `adapters.primary` or `adapters.secondary` in any `.py` file
- [ ] `uv run pytest tests/unit/ -x` passes with no failures
- [ ] `uv run pytest --cov=src tests/unit/` reports ≥ 95% coverage
- [ ] `uv run ruff check src/ tests/` reports zero errors
- [ ] App launches correctly: `uv run chatterbox-explorer`

---

## 11. Constraints

1. **Behavior unchanged** — no feature additions, no bug fixes mixed in
2. **Deferred import order preserved** — `container.py` must be imported
   inside `build_app()`, never at module level
3. **No new abstraction layers** — no `application/` use-case layer added
4. **No domain changes** — `domain/`, `ports/`, `services/` are frozen
5. **Incremental** — each phase leaves the test suite green

---

## 12. Decisions Made in This Plan

| # | Decision                                | Alternatives Rejected                        |
|---|-----------------------------------------|----------------------------------------------|
| 1 | Rename to `inbound/outbound`            | Keep `primary/secondary` (less intuitive)    |
| 2 | Use `dependency-injector` library       | `injector`, `pinject`, custom DI (less battle-tested) |
| 3 | Keep `ports/` directory name            | Move to `domain/interfaces/` (unnecessary churn) |
| 4 | No `application/` use-case layer        | Add thin wrappers (over-engineering for TTS app) |
| 5 | Container instantiated in `build_app()` | Module-level container (breaks lazy import order) |
| 6 | `AppSettings` in `infrastructure/`      | Keep in `domain/models.py` (infra concern, not domain) |
