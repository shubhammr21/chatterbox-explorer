# PLAN — Infrastructure Enhancements (apnaai-api Reference + 2025 Research)

> **Selected Flow:** System Design (`flow/system-design.md`)
> **Trigger:** Cross-project reference study of `apnaai-api/src/config/` + online
> research into 2025 best practices for domain exceptions, pydantic-settings, and
> non-blocking logging.
> **Status:** Planning Phase — awaiting approval before implementation begins.

---

## 0. Router Output

| Field     | Value                                                                             |
|-----------|-----------------------------------------------------------------------------------|
| Flow      | `flow/system-design.md`                                                           |
| Trigger   | apnaai-api reference study + research rounds (domain exceptions, settings, logging)|
| Research  | Python exception hierarchies · pydantic-settings 2.x · QueueHandler Python 3.11  |
| Next Step | Plan approved → implement phases in order, TDD + ty + ruff before each commit     |

---

## 1. Research Findings (Before Any Decision)

### 1.1 Domain Exceptions

**Source: Python docs + DDD / Clean Architecture literature (2024–2025)**

Using `ValueError` and `RuntimeError` in domain services is an anti-pattern:

| Problem | Explanation |
|---------|-------------|
| Ambiguity | `except ValueError` catches both domain failures AND Python bugs (`int("foo")`) |
| No domain meaning | `ValueError` says nothing about *which* business rule was violated |
| Not catchable by type | You cannot do `except EmptyTextError` vs `except ReferenceTooShortError` |
| Hides programming errors | `AttributeError` from a typo becomes indistinguishable from an intentional failure |
| `except Exception` in services | Swallows bugs silently — the service lies, returning garbage instead of failing fast |

**Correct pattern:**
- Domain services define their own exception hierarchy rooted at a single base class
- Services only catch exceptions they are specifically prepared to handle
- Everything else propagates — bugs surface immediately, not silently
- HTTP status codes belong ONLY in the REST adapter layer, never in the domain

### 1.2 pydantic-settings 2.x

**Source: https://pypi.org/project/pydantic-settings/ (latest: 2.14.0, April 2026)**

Key facts confirmed by research:
- `pydantic-settings` v2 **requires pydantic v2** — they are tightly coupled
- `model_config = SettingsConfigDict(...)` is the current pattern (not `class Config:`)
- `SecretStr` masks tokens in logs/repr; unwrap only at the boundary with `.get_secret_value()`
- Priority order: init kwargs > env vars > `.env` file > field defaults
- `env_nested_delimiter="__"` allows `SERVER__PORT=8001` → `settings.server.port`
- **Architecture rule confirmed:** `BaseSettings` must NOT leak into the domain layer
  — use a factory function to convert `BaseSettings` → plain `dataclass` for domain use

**User constraint confirmed:** pydantic (and therefore pydantic-settings) is limited
to the REST adapter layer. It must live in the `rest` optional extra, not core deps.

### 1.3 Non-blocking Logging (Python 3.11)

**Source: https://docs.python.org/3/library/logging.handlers.html**

`QueueHandler` + `QueueListener` is still the stdlib-recommended approach in 2025.
The Python docs explicitly recommend it for asyncio applications.

**Python 3.11 caveats (project uses Python 3.11):**
- `QueueHandler.prepare()` clears `exc_text` by setting it to `None` (for pickling)
  — the listener-side handler cannot re-format exceptions from the original `exc_info`
- The `.listener` attribute on `QueueHandler` was only added in **Python 3.12**
- Context manager support for `QueueListener` only added in **Python 3.14**

**Workaround for Python 3.11:** subclass `QueueHandler` and override `prepare()` to
preserve `exc_text` before the base class clears it.

**`structlog` 25.x** is the structured logging leader in 2025, but replacing the entire
logging stack (`python-json-logger` + `CorrelationIdFilter`) is too disruptive for this
phase. Deferred to a separate future task.

---

## 2. What Applies to chatterbox-demo

### Adopted (this plan)

| # | Enhancement | Source | Why |
|---|-------------|--------|-----|
| 1 | **Domain exception hierarchy** | apnaai + 2025 research | Services raise `ValueError`/`RuntimeError` — anti-pattern confirmed |
| 2 | **pydantic-settings `BaseSettings`** | apnaai `settings.py` | REST mode needs env var config; limited to `rest` extra |
| 3 | **`Environment` enum** | apnaai `constants.py` | No environment concept; dev and prod behave identically |
| 4 | **Non-blocking `QueueHandler`** | apnaai `logging.py` + research | Sync `StreamHandler` can stall event loop under load |
| 5 | **`sys.excepthook` override** | apnaai `logging.py` | Process crashes go to stderr unstructured |
| 6 | **Consistent `ErrorResponse` shape** | apnaai `entities/response.py` | Mixed error shapes: `{"detail":"str"}` vs `{"detail":[...]}` |

### Rejected (documented)

| Pattern | Reason |
|---------|--------|
| `pydantic` in domain/services | User constraint: pydantic limited to REST adapter |
| `BaseError` with `status_code` | HTTP is adapter concern — confirmed by research |
| `PublicEntity`/`InternalEntity` Pydantic base classes | Domain uses stdlib dataclasses — hexagonal boundary |
| `except Exception` anywhere in services | Anti-pattern confirmed by research |
| `structlog` full adoption | Too disruptive for this phase — deferred |
| Rate limiting, OpenTelemetry, Firebase | Out of scope |

---

## 3. Architecture Constraints (Non-Negotiable)

1. **Domain layer is frozen for features** — `domain/models.py`, `ports/`, `services/`
   receive ONLY the exception hierarchy addition. No Pydantic, no HTTP, no framework.

2. **Exception rule** — Services catch ONLY what they are prepared to handle.
   `except Exception` is forbidden everywhere. Programming errors must propagate.

3. **HTTP status codes are adapter-only** — Domain exceptions carry no `status_code`.
   The REST `exception_handlers.py` owns the domain-exception → HTTP mapping.

4. **pydantic is REST-only** — `pydantic-settings` goes into the `rest` optional extra.
   The domain, Gradio adapter, and `cli.py` for `--mode ui` never import pydantic.

5. **Deferred import order preserved** — compat patches fire before torch/chatterbox.
   Any new module in `infrastructure/` must be safe to defer to function bodies.

6. **TDD** — failing test before every new module.

7. **ty + ruff + zero suppressions before every commit** — no `# type: ignore`, no `# noqa`.

---

## 4. Enhancement 1 — Domain Exception Hierarchy

### Problem

Every service raises Python builtins:

```python
# services/tts.py  ← current, incorrect
if not request.text.strip():
    raise ValueError("Text input is empty.")
...
except AssertionError as exc:
    raise ValueError(f"Reference audio must be longer than 5 seconds: {exc}") from exc
```

`ValueError` is ambiguous — callers cannot distinguish "empty text" from a Python
programming error. The REST exception handler has to infer HTTP intent from the type
alone, which is fragile.

### Solution: `domain/exceptions.py`

A single new file. Zero stdlib-beyond-`Exception` imports. No HTTP knowledge.

```
ChatterboxError                      ← root; catch-all for all domain failures
├── TTSInputError                    ← business rule violated by the caller
│   ├── EmptyTextError               ← text is empty or whitespace-only
│   └── ReferenceTooShortError       ← reference audio below minimum duration
├── VoiceConversionInputError        ← VC-specific input failures
│   ├── MissingSourceAudioError
│   └── MissingTargetVoiceError
├── ModelError                       ← model / infrastructure failure
│   ├── ModelNotLoadedError          ← model has not been initialised
│   ├── ModelLoadError               ← failure while loading / downloading
│   └── InferenceError               ← failure during model.generate() call
└── WatermarkError                   ← watermark detection failure
```

No `status_code` on any exception. The REST adapter decides the HTTP mapping.

### Updated service contracts

| Before | After |
|--------|-------|
| `raise ValueError("Text input is empty.")` | `raise EmptyTextError(text=request.text)` |
| `raise ValueError(f"Reference audio must be longer than 5 seconds: {exc}")` | `raise ReferenceTooShortError(duration_sec=actual, minimum_sec=5.0)` |
| `raise ValueError(...)` for missing VC paths | `raise MissingSourceAudioError()` / `MissingTargetVoiceError()` |
| Catch `AssertionError` in service body | Let it propagate — it is a bug, not a domain failure |

### REST adapter handler mapping

```python
# exception_handlers.py — new handlers for domain exceptions
# (ValueError and RuntimeError handlers are REMOVED — no longer raised by services)

async def tts_input_error_handler(request, exc):       # → 422
async def model_error_handler(request, exc):            # → 503
async def vc_input_error_handler(request, exc):         # → 422
async def watermark_error_handler(request, exc):        # → 500
async def chatterbox_error_handler(request, exc):       # → 500 catch-all for domain
```

The generic `ValueError`/`RuntimeError` handlers that currently exist are removed —
they are no longer needed because services no longer raise those types.

### Gradio adapter

`GradioHandlers` also catches `ValueError` and `AssertionError` today. It must be
updated to catch the typed domain exceptions instead.

---

## 5. Enhancement 2 — `pydantic-settings` `BaseSettings` (REST-only)

### Constraint

`pydantic-settings` installs pydantic v2 as a hard dependency. Per user instruction,
pydantic is limited to the REST adapter layer. Therefore:
- `pydantic-settings` goes into `[project.optional-dependencies].rest`
- `infrastructure/settings.py` is only imported inside `build_rest_app()` (deferred)
- `cli.py` for `--mode ui` never touches it

### `infrastructure/settings.py` (new)

```python
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from .constants import Environment   # infrastructure-layer enum

class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7860

class LoggingSettings(BaseModel):
    level: str = "INFO"
    json_logs: bool = False

class HuggingFaceSettings(BaseModel):
    token: SecretStr | None = None   # HF_TOKEN env var — masked in logs

class RestSettings(BaseSettings):
    """Settings for the FastAPI REST adapter. Loaded from env + .env file.

    Never imported at module level — always inside build_rest_app() to
    preserve the deferred-import guarantee.
    """
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    environment: Environment = Environment.LOCAL
    server: ServerSettings = ServerSettings()
    logging: LoggingSettings = LoggingSettings()
    huggingface: HuggingFaceSettings = HuggingFaceSettings()

    @field_validator("logging")
    @classmethod
    def normalise_log_level(cls, v: LoggingSettings) -> LoggingSettings:
        v.level = v.level.upper()
        return v
```

### `infrastructure/constants.py` (new)

Pure Python — no pydantic. Safe to import anywhere.

```python
from enum import Enum

class Environment(str, Enum):
    LOCAL      = "LOCAL"
    STAGING    = "STAGING"
    TESTING    = "TESTING"
    PRODUCTION = "PRODUCTION"

    @property
    def is_debug(self) -> bool:
        return self in (Environment.LOCAL, Environment.TESTING)

    @property
    def is_deployed(self) -> bool:
        return self in (Environment.STAGING, Environment.PRODUCTION)

    @property
    def use_json_logs(self) -> bool:
        return self.is_deployed
```

### How settings flow into the app

```python
# bootstrap.py — build_rest_app() (deferred — patches already fired)
def build_rest_app(watermark_available: bool) -> FastAPI:
    from infrastructure.settings import RestSettings
    settings = RestSettings()   # reads .env + env vars

    # Pass only what each layer needs — settings never leak into services
    app = ChatterboxAPI(
        title="Chatterbox TTS API",
        version=_read_version_from_pyproject(),
        debug=settings.environment.is_debug,
        lifespan=lifespan,
    )
    # configure logging level from settings
    configure_json(log_level=settings.logging.level)
    ...
```

`device` and `watermark_available` remain runtime-detected (not settings fields) —
they depend on hardware and library availability, not operator configuration.

---

## 6. Enhancement 3 — Non-blocking `QueueHandler` + `sys.excepthook`

### Problem

`configure_json()` uses `logging.StreamHandler` — synchronous stdout write.
Under inference load, each log write holds the GIL and adds latency to the event loop.

### Solution

Wrap the existing `StreamHandler` (with `CorrelationIdFilter` + `JsonFormatter`)
inside a `QueueHandler` → `QueueListener`. The event loop only does a fast
`queue.put_nowait()`; a dedicated background thread handles the actual I/O.

### Python 3.11 caveats and workarounds

The project targets Python 3.11. Two stdlib limitations apply:

**1. `exc_text` is cleared by `QueueHandler.prepare()`**

`QueueHandler.prepare()` sets `record.exc_text = None` before enqueuing (for
pickling safety). The listener's handler receives a record without formatted
exception text. Workaround: subclass `QueueHandler` and preserve `exc_text`:

```python
class _PreservingQueueHandler(logging.handlers.QueueHandler):
    """QueueHandler that preserves exc_text for downstream handlers.

    Python's QueueHandler.prepare() clears exc_text (sets it to None)
    before enqueuing the record, for pickling safety. When the listener
    picks up the record on the background thread, the handler cannot
    re-format the exception. This subclass formats exc_text before the
    base class clears it, so the listener receives an already-formatted
    string it can emit directly.

    Required for Python 3.11 compatibility. Python 3.12+ is unaffected
    because the base class was updated to preserve exception information.
    """
    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        if record.exc_info and not record.exc_text:
            # Format the exception text NOW, before the base class clears exc_info
            record.exc_text = self.formatter.formatException(record.exc_info) \
                if self.formatter else logging.Formatter().formatException(record.exc_info)
        return super().prepare(record)
```

**2. No `.listener` attribute on `QueueHandler` in Python 3.11**

The `.listener` attribute was added in Python 3.12. In 3.11, the `QueueListener`
must be stored separately and stopped via `atexit`. The `configure_json()` function
returns the listener so the caller can register it with `atexit`.

### Updated `configure_json()` signature

```python
def configure_json(log_level: str = "INFO") -> None:
    """Configure non-blocking structured JSON logging.

    Uses QueueHandler → QueueListener so log writes never block the event loop.
    Registers the listener's stop() with atexit for clean shutdown.
    Overrides sys.excepthook to route process-level crashes to the structured
    logger instead of unstructured stderr.
    """
    # ... validate deps ...
    # ... build JsonFormatter + CorrelationIdFilter (unchanged) ...
    # ... wrap in _PreservingQueueHandler ...
    # ... start QueueListener ...
    # ... atexit.register(listener.stop) ...
    # ... sys.excepthook = _handle_uncaught_exception ...
```

### `sys.excepthook` override

```python
def _handle_uncaught_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: types.TracebackType | None,
) -> None:
    """Route uncaught process-level exceptions to the structured logger.

    KeyboardInterrupt is re-dispatched to the default handler so Ctrl-C
    still works correctly. Everything else is logged at ERROR level and
    will flow through the QueueHandler → JSON formatter.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.getLogger("chatterbox-explorer").error(
        "Uncaught exception — process will terminate",
        exc_info=(exc_type, exc_value, exc_traceback),
    )
```

---

## 7. Enhancement 4 — Consistent `ErrorResponse` Shape

### Problem

The REST API currently returns two different error shapes:

```json
// From domain exception handlers (ValueError/RuntimeError):
{"detail": "text must not be empty"}

// From FastAPI's default RequestValidationError handler:
{"detail": [{"loc": ["body", "text"], "msg": "...", "type": "..."}]}
```

Clients must parse two completely different error structures.

### Solution

Add `ErrorResponse` and `ErrorResponseMulti` to `adapters/inbound/rest/schemas.py`
and use them in all exception handlers. These are Pydantic models that live in
the REST adapter — not in the domain.

```python
class ErrorDetail(BaseModel):
    """Single error item."""
    message: str
    path: list[str | int] = Field(default_factory=list)

class ErrorResponse(BaseModel):
    """Consistent error envelope for all REST error responses."""
    errors: list[ErrorDetail]
```

Updated handler responses:

```python
# All domain exception handlers return the same shape:
return JSONResponse(
    status_code=422,
    content=ErrorResponse(
        errors=[ErrorDetail(message=str(exc))]
    ).model_dump(),
)

# Validation errors include field paths:
return JSONResponse(
    status_code=422,
    content=ErrorResponse(
        errors=[
            ErrorDetail(message=err["msg"], path=list(err["loc"]))
            for err in exc.errors()
        ]
    ).model_dump(),
)
```

---

## 8. New File Structure

Only `domain/`, `infrastructure/`, `adapters/inbound/rest/`, and `logging_config.py`
change. Everything else is frozen.

```
src/
│
├── domain/
│   ├── exceptions.py            NEW  — ChatterboxError hierarchy (pure Python)
│   ├── models.py                FROZEN
│   ├── types.py                 FROZEN
│   ├── presets.py               FROZEN
│   └── languages.py             FROZEN
│
├── ports/                       FROZEN
├── services/                    UPDATED — raise typed domain exceptions
│   ├── tts.py                   replace ValueError → EmptyTextError / ReferenceTooShortError
│   ├── turbo_tts.py             same
│   ├── multilingual_tts.py      same
│   ├── voice_conversion.py      replace ValueError → MissingSourceAudioError etc.
│   ├── model_manager.py         raise ModelLoadError / ModelNotLoadedError
│   └── watermark.py             raise WatermarkError
│
├── infrastructure/
│   ├── __init__.py              UNCHANGED
│   ├── config.py                REMOVED (superseded by constants.py)
│   ├── constants.py             NEW  — Environment enum (pure Python, no pydantic)
│   ├── settings.py              NEW  — RestSettings BaseSettings (rest extra only)
│   ├── errors.py                NOT CREATED (errors live in domain/exceptions.py)
│   └── container.py             UPDATED — minor wiring changes for new exceptions
│
├── adapters/
│   ├── inbound/
│   │   ├── gradio/
│   │   │   └── handlers.py      UPDATED — catch typed domain exceptions
│   │   └── rest/
│   │       ├── exception_handlers.py  UPDATED — domain exc → HTTP, consistent shape
│   │       └── schemas.py             UPDATED — add ErrorDetail, ErrorResponse
│   └── outbound/                FROZEN
│
└── logging_config.py            UPDATED — QueueHandler, sys.excepthook, log_level param
```

---

## 9. Dependencies

### Core (`[project.dependencies]`) — unchanged

No new core dependencies. `pydantic-settings` is NOT added here.

### REST extra (`[project.optional-dependencies].rest`) — one addition

```toml
rest = [
    "asgi-correlation-id>=4.0.0",
    "fastapi>=0.100.0",
    "python-json-logger>=3.2.0",
    "pydantic-settings>=2.0.0",   # NEW — BaseSettings for REST config
    "uvicorn[standard]>=0.20.0",
]
```

`pydantic-settings>=2.0.0` requires pydantic v2. FastAPI already pulls pydantic v2
as a dependency, so this adds no net new runtime dependency when the `rest` extra
is installed. It is absent when only `--extra ui` is installed.

---

## 10. Implementation Phases

Each phase ends with: `ty check` = 0 errors, `ruff check src/ tests/` = 0 errors,
`ruff format --check src/ tests/` = clean, `pytest tests/unit/` = all green,
coverage ≥ 95%. **Tests written before implementation (TDD).**

---

### Phase 1 — Domain Exception Hierarchy

**New file:** `domain/exceptions.py`

**TDD steps:**
1. Write `tests/unit/domain/test_exceptions.py` (RED):
   - `ChatterboxError` is a subclass of `Exception`
   - `EmptyTextError` is a subclass of `TTSInputError` which is a subclass of `ChatterboxError`
   - `ReferenceTooShortError(duration_sec=3.0, minimum_sec=5.0)` stores both attributes
   - `MissingSourceAudioError`, `MissingTargetVoiceError` are subclasses of `VoiceConversionInputError`
   - `ModelNotLoadedError("tts")` stores `model_key`
   - `ModelLoadError` and `InferenceError` are subclasses of `ModelError`
   - `WatermarkError` is a subclass of `ChatterboxError`
   - No exception class imports from `starlette`, `fastapi`, or any third-party package
   - No exception class carries `status_code`

2. Implement `domain/exceptions.py` (GREEN)

3. Update `services/tts.py`, `services/voice_conversion.py`, `services/model_manager.py`,
   `services/watermark.py` — replace `ValueError`/`RuntimeError` raises with typed exceptions.
   Remove all `except AssertionError` patterns (they were masking programming errors).

4. Update `tests/unit/services/test_tts_service.py` and siblings — assertions that
   previously checked `pytest.raises(ValueError)` now check the specific domain type.

5. Update `adapters/inbound/gradio/handlers.py` — replace `except ValueError` with
   `except TTSInputError`, `except ModelError`, etc.

6. Update `adapters/inbound/rest/exception_handlers.py`:
   - Add `tts_input_error_handler` → 422
   - Add `model_error_handler` → 503
   - Add `vc_input_error_handler` → 422
   - Add `chatterbox_error_handler` → 500 (root catch-all)
   - Remove `value_error_handler` and `runtime_error_handler` (no longer raised by services)

7. Update `bootstrap.py` registration — register new handlers, deregister old ones.

8. `ty + ruff + pytest`

---

### Phase 2 — `Environment` Enum + `infrastructure/constants.py`

**New file:** `infrastructure/constants.py` (pure Python — no pydantic)

**TDD steps:**
1. Write `tests/unit/infrastructure/test_constants.py` (RED):
   - `Environment.LOCAL.is_debug is True`
   - `Environment.TESTING.is_debug is True`
   - `Environment.STAGING.is_debug is False`
   - `Environment.PRODUCTION.is_debug is False`
   - `Environment.STAGING.is_deployed is True`
   - `Environment.PRODUCTION.is_deployed is True`
   - `Environment.LOCAL.use_json_logs is False`
   - `Environment.PRODUCTION.use_json_logs is True`
   - `Environment("LOCAL")` works (str enum — readable from env var)
   - Importing `infrastructure.constants` does not import pydantic

2. Implement `infrastructure/constants.py` (GREEN)

3. Remove `infrastructure/config.py` — `AppSettings` was the only thing there.
   Update `tests/unit/infrastructure/test_config.py` → `test_constants.py`.

4. `ty + ruff + pytest`

---

### Phase 3 — Non-blocking Logging (`QueueHandler` + `sys.excepthook`)

**Modified file:** `logging_config.py`

**TDD steps:**
1. Write additions to `tests/unit/test_logging_config.py` (RED):
   - After `configure_json()`, root logger has a `QueueHandler` (not `StreamHandler` directly)
   - After `configure_json()`, `sys.excepthook` is overridden (not `sys.__excepthook__`)
   - `KeyboardInterrupt` dispatches to `sys.__excepthook__` (not to the logger)
   - After `configure_json()`, at least one handler with `JsonFormatter` exists
     on the `QueueListener`'s handlers (not directly on root logger)
   - `configure_json(log_level="DEBUG")` sets root logger level to `DEBUG`
   - After `configure_json()`, `uvicorn.access` is still silenced at `CRITICAL`
   - `CorrelationIdFilter` is still on the listener-side handler

2. Rewrite `configure_json()` — `QueueHandler` wrapping, `_PreservingQueueHandler`
   subclass, `atexit.register(listener.stop)`, `sys.excepthook` override.

3. Add `log_level: str = "INFO"` parameter to both `configure()` and `configure_json()`.

4. Update `cli.py` `_launch_rest()` — pass `settings.logging.level` to `configure_json()`.

5. `ty + ruff + pytest`

---

### Phase 4 — `pydantic-settings` `RestSettings` (REST-only)

**New file:** `infrastructure/settings.py`

**Dependency update:**
```bash
uv add pydantic-settings --optional rest
```

**TDD steps:**
1. Write `tests/unit/infrastructure/test_settings.py` (RED):
   - `RestSettings()` constructs with all defaults when no `.env` exists
   - `RestSettings().server.host == "0.0.0.0"`
   - `RestSettings().server.port == 7860`
   - `RestSettings().logging.level == "INFO"`
   - `RestSettings().environment == Environment.LOCAL`
   - `RestSettings(environment="PRODUCTION").environment == Environment.PRODUCTION`
   - `RestSettings(huggingface={"token": "secret"}).huggingface.token` is a `SecretStr`
   - `str(settings.huggingface.token)` does NOT reveal the secret value
   - Importing `infrastructure.settings` without the `rest` extra installed raises
     `ModuleNotFoundError` with a clear message

2. Implement `infrastructure/settings.py` (GREEN)

3. Update `build_rest_app()` — defer-import `RestSettings` inside the function body,
   pass `log_level` to `configure_json()`, pass `environment.is_debug` to `FastAPI(debug=...)`.

4. `ty + ruff + pytest`

---

### Phase 5 — Consistent `ErrorResponse` Shape

**Modified file:** `adapters/inbound/rest/schemas.py`,
`adapters/inbound/rest/exception_handlers.py`

**TDD steps:**
1. Write tests (RED):
   - `ErrorDetail(message="bad input")` serializes to `{"message": "bad input", "path": []}`
   - `ErrorDetail(message="required", path=["body", "text"])` serializes with path
   - `ErrorResponse(errors=[ErrorDetail(message="x")])` serializes to
     `{"errors": [{"message": "x", "path": []}]}`
   - Domain exception handler returns body with `errors` key (not `detail`)
   - Validation error handler returns body with `errors` list where each item has
     `message` and `path` from the Pydantic error
   - Update existing route tests that assert `resp.json()["detail"]` → `resp.json()["errors"]`

2. Add `ErrorDetail` and `ErrorResponse` to `schemas.py` (GREEN)

3. Update all handlers in `exception_handlers.py` to use `ErrorResponse`

4. Update `test_rest_routes.py` — error body assertions updated to new shape

5. `ty + ruff + pytest`

---

### Phase 6 — Final Validation

```bash
# Type checking
uv run ty check
# → 0 errors

# Linting + formatting
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
# → All checks passed

# Tests + coverage
uv run pytest tests/unit/ --cov=src -q
# → ≥ 95% coverage, all green

# Smoke tests
uv run chatterbox-explorer --help
uv run chatterbox-explorer --mode rest --port 8001
curl http://localhost:8001/api/v1/health
curl -X POST http://localhost:8001/api/v1/tts/generate \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}' --output test.wav
# → {"errors": [...]} shape on failures
# → RIFF on success
```

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Services currently have tests asserting `pytest.raises(ValueError)` — all must be updated | **HIGH** | Medium | Phase 1 step 4 explicitly updates all service tests before moving on |
| Gradio handlers catch `ValueError`/`AssertionError` — missed update | Medium | High | Phase 1 step 5 covers Gradio; integration smoke test catches runtime errors |
| `ty` flags `ChatterboxError` handlers with `exc: Exception` protocol issue | Medium | Low | Same `isinstance` narrowing pattern already proven in current `exception_handlers.py` |
| `QueueHandler.prepare()` clears exc_info in Python 3.11 | **HIGH** | Medium | `_PreservingQueueHandler` subclass explicitly addresses this — tested in Phase 3 |
| `pydantic-settings` imported at module level (breaks deferred import) | Medium | High | `RestSettings` is only imported inside `build_rest_app()` function body — explicit constraint |
| `ErrorResponse` shape breaks existing clients / tests | Medium | Medium | Phase 5 updates all test assertions; documented as breaking change in API version comment |
| `Environment.use_json_logs` bypasses explicit `--mode rest` flag | Low | Low | Settings only active in REST mode; UI mode never loads `RestSettings` |

---

## 12. Decisions Made in This Plan

| # | Decision | Alternatives Rejected | Reason |
|---|----------|-----------------------|--------|
| 1 | Domain exceptions in `domain/exceptions.py` | `infrastructure/errors.py` with `status_code` | HTTP is adapter concern — confirmed by 2025 research |
| 2 | Domain services keep raising typed domain exceptions, not stdlib builtins | Keep `ValueError`/`RuntimeError` | Anti-pattern confirmed — ambiguous, hides bugs |
| 3 | `except Exception` is forbidden in all services | Status quo | Research confirms it swallows bugs silently |
| 4 | `pydantic-settings` in `rest` optional extra only | Core dependency | User constraint; pydantic limited to REST layer |
| 5 | `Environment` in `infrastructure/constants.py` (no pydantic) | `domain/types.py` | Deployment environment is infrastructure, not domain |
| 6 | `QueueHandler` with `_PreservingQueueHandler` workaround | structlog full adoption | structlog adoption too disruptive for this phase; deferred |
| 7 | `ErrorResponse(errors=[...])` envelope | Keep FastAPI `{"detail": ...}` | Inconsistent shapes confirmed problematic; `errors` key more explicit than `detail` |
| 8 | Remove `value_error_handler` and `runtime_error_handler` | Keep alongside domain handlers | Services no longer raise these types — dead handlers |

---

## 13. Success Criteria

- [ ] `domain/exceptions.py` exists; every exception class imports ONLY from stdlib
- [ ] No service file contains `raise ValueError(...)` or `raise RuntimeError(...)` as domain errors
- [ ] No service file contains `except Exception` or `except AssertionError`
- [ ] `pytest.raises(EmptyTextError)` passes where `pytest.raises(ValueError)` used to pass
- [ ] `uv run ty check` reports 0 errors
- [ ] `uv run ruff check src/ tests/` reports 0 errors
- [ ] `uv run pytest tests/unit/ --cov=src -q` reports ≥ 95% coverage
- [ ] `ENVIRONMENT=PRODUCTION uv run chatterbox-explorer --mode rest` auto-sets JSON logs
- [ ] Log writes use `QueueHandler`; `StreamHandler` is on the listener thread only
- [ ] Process crash → JSON-formatted error record in structured log
- [ ] Every REST error response has `{"errors": [{"message": "...", "path": [...]}]}`
- [ ] Validation errors include `"path": ["body", "field_name"]` for field-level context
- [ ] `uv run chatterbox-explorer` (no flags) → Gradio UI unchanged
