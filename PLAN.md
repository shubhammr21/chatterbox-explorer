# PLAN — FastAPI REST Adapter (v1)

> **Research status:** ✅ Round 1 + Round 2 complete.
> Sources: official dependency-injector FastAPI examples · uv docs · FastAPI async/concurrency docs ·
> anyio thread docs · Starlette concurrency source · FastAPI events/lifespan docs ·
> uvicorn settings docs · python-json-logger PyPI · pytest-benchmark PyPI.
> No assumptions — every pattern below is docs-backed.

> **Selected Flow:** System Design (`flow/system-design.md`)  
> **Research rounds:**
> - Round 1: `dependency-injector` wiring · uv optional extras · `TestClient` vs `AsyncClient`
> - Round 2: sync-in-async P0 bug · `run_in_threadpool` · `asyncio.Semaphore` · lifespan · structured logging · `pytest-benchmark`
>
> **Reason:** New delivery mechanism — a fully functional inbound REST adapter that
> exposes every domain service through a versioned HTTP API. No domain changes required.
> **Status:** Research complete — plan updated — ready for implementation.

---

## 0. Router Output

| Field     | Value                                                                         |
|-----------|-------------------------------------------------------------------------------|
| Flow      | `flow/system-design.md`                                                       |
| Trigger   | Refactoring complete; `adapters/inbound/rest/routes.py` stub ready to activate |
| Next Step | Plan approved → implement phases 1–6 incrementally                            |

---

## 1. Problem Definition

### What we are building
A production-ready FastAPI REST adapter that exposes every Chatterbox capability
(Standard TTS, Turbo TTS, Multilingual TTS, Voice Conversion, Model Manager,
Watermark Detection) as versioned HTTP endpoints.

### Why now
- The hexagonal structure is in place — adding an inbound adapter requires zero
  domain or service changes.
- The `adapters/inbound/rest/routes.py` stub already contains the full route design.
- The `AppContainer` DI container makes it trivial to wire FastAPI with the same
  singleton service instances used by the Gradio adapter.

### Scope boundaries

**In scope (v1):**
- All one-shot generation endpoints (Standard, Turbo, Multilingual TTS; VC)
- Model management endpoints (status, memory, load, unload)
- Watermark detection endpoint
- Health check endpoint
- `--mode api` CLI flag that launches FastAPI + uvicorn instead of Gradio
- Pydantic request/response schemas
- Full test suite using `TestClient` with container overrides

**Out of scope (deferred to v2):**
- Sentence-level streaming over HTTP (requires SSE or chunked transfer — separate design)
- Authentication / API keys
- Rate limiting
- Running Gradio and FastAPI simultaneously on two ports
- OpenAPI SDK generation

---

## 2. Architecture

### Flow diagram

```mermaid
graph LR
    subgraph inbound["adapters/inbound/rest/"]
        SC[schemas.py<br/>Pydantic models]
        RT[routes.py<br/>@inject handlers]
        AP[app.py<br/>build_rest_app()]
    end

    subgraph infra["infrastructure/"]
        CT[container.py<br/>AppContainer]
    end

    subgraph services["services/"]
        TS[TTSService]
        TU[TurboService]
        ML[MultilingualService]
        VC[VCService]
        MM[ModelManagerService]
        WM[WatermarkService]
    end

    HTTP[HTTP Client] -->|JSON body| SC
    SC -->|domain request| RT
    RT -->|Depends Provide| CT
    CT -->|singleton| TS & TU & ML & VC & MM & WM
    TS & TU & ML & VC -->|AudioResult| RT
    RT -->|WAV bytes| HTTP
```

### Hexagonal position

```
[ HTTP Client ]
      │  HTTP POST/GET
      ▼
[ FastAPI routes ]    ← NEW inbound adapter (same boundary as Gradio)
      │  port ABC calls
      ▼
[ Domain Services ]   ← UNCHANGED
      │  port ABC calls
      ▼
[ Outbound Adapters ] ← UNCHANGED
      │
      ▼
[ Chatterbox Models / psutil / PerTh ]
```

The REST adapter sits at exactly the same hexagonal boundary as the Gradio adapter.
Both call the same port ABCs. Neither knows about the other.

---

## 3. New File Structure

```
src/adapters/inbound/rest/
├── __init__.py          UPDATED  — docstring only
├── app.py               NEW      — FastAPI app factory: build_rest_app()
├── schemas.py           NEW      — Pydantic request + response models
└── routes.py            UPDATED  — uncomment + adapt stub; wire @inject
```

No changes to `domain/`, `ports/`, `services/`, or `adapters/outbound/`.

---

## 4. API Contract

Base prefix: `/api/v1`

### 4.1 Health

| Method | Path              | Auth | Response         |
|--------|-------------------|------|------------------|
| GET    | `/api/v1/health`  | —    | `200 {"status":"ok","device":"mps"}` |

### 4.2 Standard TTS

| Method | Path                   | Body              | Response          |
|--------|------------------------|-------------------|-------------------|
| POST   | `/api/v1/tts/generate` | `TTSRequestSchema`| `200 audio/wav`   |

### 4.3 Turbo TTS

| Method | Path                     | Body                 | Response        |
|--------|--------------------------|----------------------|-----------------|
| POST   | `/api/v1/turbo/generate` | `TurboRequestSchema` | `200 audio/wav` |

### 4.4 Multilingual TTS

| Method | Path                            | Body                        | Response        |
|--------|---------------------------------|-----------------------------|-----------------|
| POST   | `/api/v1/multilingual/generate` | `MultilingualRequestSchema` | `200 audio/wav` |

### 4.5 Voice Conversion

| Method | Path                  | Body                      | Response        |
|--------|-----------------------|---------------------------|-----------------|
| POST   | `/api/v1/vc/convert`  | `multipart/form-data`     | `200 audio/wav` |

Two file fields: `source_audio: UploadFile`, `target_voice: UploadFile`.

### 4.6 Model Management

| Method | Path                          | Body | Response           |
|--------|-------------------------------|------|--------------------|
| GET    | `/api/v1/models/status`       | —    | `200 [ModelStatusResponse]` |
| GET    | `/api/v1/models/memory`       | —    | `200 MemoryStatsResponse`   |
| POST   | `/api/v1/models/{key}/load`   | —    | `200 {"message": "…"}`      |
| POST   | `/api/v1/models/{key}/unload` | —    | `200 {"message": "…"}`      |

### 4.7 Watermark Detection

| Method | Path                      | Body               | Response                |
|--------|---------------------------|--------------------|-------------------------|
| POST   | `/api/v1/watermark/detect`| `file: UploadFile` | `200 WatermarkResponse` |

### 4.8 Error responses

| HTTP Status | Trigger                                    |
|-------------|--------------------------------------------|
| 422         | `ValueError` from service (empty text, audio too short, missing paths) |
| 503         | `RuntimeError` from service (model load failure, OOM, etc.)            |
| 404         | Unknown `key` in model management endpoints                             |
| 500         | Any unhandled exception (catch-all)                                     |

---

## 5. Pydantic Schemas (`schemas.py`)

Schemas live in the inbound adapter — they are **not** domain models. Each schema
maps to a domain request dataclass via a `.to_domain()` helper method, keeping
the adapter layer responsible for translation.

```python
# adapters/inbound/rest/schemas.py  (outline)

class TTSRequestSchema(BaseModel):
    text: str
    ref_audio_path: str | None = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    rep_penalty: float = 1.2
    min_p: float = 0.05
    top_p: float = 1.0
    seed: int = 0

    def to_domain(self) -> TTSRequest: ...

class TurboRequestSchema(BaseModel):
    text: str
    ref_audio_path: str | None = None
    temperature: float = 0.8
    top_k: int = 1000
    top_p: float = 0.95
    rep_penalty: float = 1.2
    min_p: float = 0.0
    norm_loudness: bool = True
    seed: int = 0

    def to_domain(self) -> TurboTTSRequest: ...

class MultilingualRequestSchema(BaseModel):
    text: str
    language: str = "en"
    ref_audio_path: str | None = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    rep_penalty: float = 2.0
    min_p: float = 0.05
    top_p: float = 1.0
    seed: int = 0

    def to_domain(self) -> MultilingualTTSRequest: ...

class ModelStatusResponse(BaseModel):
    key: str
    display_name: str
    class_name: str
    description: str
    params: str
    size_gb: float
    in_memory: bool
    on_disk: bool

class MemoryStatsResponse(BaseModel):
    sys_total_gb: float
    sys_used_gb: float
    sys_avail_gb: float
    sys_percent: float
    proc_rss_gb: float
    device_name: str
    device_driver_gb: float | None
    device_max_gb: float | None

class WatermarkResponse(BaseModel):
    score: float
    verdict: str
    message: str
    available: bool
```

---

## 6. P0 Performance Bug — Sync Blocking Inside `async def`

### Research finding (FastAPI async docs + anyio thread docs)

**Every current route handler is broken for concurrency.**

The asyncio event loop runs on a single OS thread. When an `async def` handler
calls a blocking sync function directly (`tts.generate(...)`, `model.generate(...)`),
the event loop freezes completely — every other in-flight request (health checks,
status polls, other TTS calls) hangs until the inference finishes.

> *FastAPI docs: "When you declare a path operation function with normal `def` instead
> of `async def`, it is run in an external thread pool that is then awaited, instead of
> being called directly (as it would block the server)."*

The corollary: `async def` + direct blocking call = **worst of both worlds** — you
lose the thread-pool offload but also can't suspend the event loop.

### Root cause in written code

```python
# ❌ CURRENT — freezes event loop for 5-30 s per TTS request
async def generate_tts(body, tts) -> Response:
    result = tts.generate(body.to_domain())   # ← blocking; event loop stuck
    ...
```

### Fix: `run_in_threadpool` + `asyncio.Semaphore`

`starlette.concurrency.run_in_threadpool` is the FastAPI/Starlette-blessed wrapper
over `anyio.to_thread.run_sync`. FastAPI uses it internally for every `def` route:

```python
# ✅ FIXED — event loop free during inference; only the worker thread blocks
from starlette.concurrency import run_in_threadpool

async def generate_tts(body, tts) -> Response:
    async with inference_semaphore:          # GPU serialization gate
        result = await run_in_threadpool(tts.generate, body.to_domain())
    ...
```

### Why `run_in_threadpool` over alternatives

| API | Verdict |
|-----|---------|
| `await run_in_threadpool(fn, *args)` | ✅ Official Starlette/FastAPI pattern |
| `await anyio.to_thread.run_sync(fn, *args)` | ✅ Same — lower level |
| `loop.run_in_executor(pool, fn, *args)` | ⚠️ asyncio-only, manual pool lifecycle |
| Switch route to `def` (sync) | ⚠️ Valid for pure-sync routes — but VC/watermark routes have `await upload.read()` so must stay `async def` |

### `asyncio.Semaphore(1)` — GPU inference gate

A single GPU (or MPS device) serializes kernel execution anyway. The semaphore
prevents multiple requests from piling into the thread pool simultaneously:

```python
# concurrency.py — created once at module import, safe in Python 3.11+
import asyncio
inference_semaphore = asyncio.Semaphore(1)
```

While a request waits on the semaphore, the event loop is **free** — it serves
health checks, model-status polls, and other non-inference requests normally.

### Blocking calls requiring `run_in_threadpool` (complete list)

| Route | Blocking call |
|-------|---------------|
| `POST /tts/generate` | `tts.generate(...)` |
| `POST /turbo/generate` | `turbo.generate(...)` |
| `POST /multilingual/generate` | `mtl.generate(...)` |
| `POST /vc/convert` | `vc.convert(...)` |
| `POST /models/{key}/load` | `manager.load(key)` — downloads weights, seconds |
| `POST /models/{key}/unload` | `manager.unload(key)` — flushes GPU cache |
| `GET /models/status` | `manager.get_all_status()` — disk probes |
| `GET /models/memory` | `manager.get_memory_stats()` — psutil calls |
| `POST /watermark/detect` | `watermark.detect(path)` — neural inference |

---

## 7. Lifespan — Model Pre-warming and Resource Setup

### Research finding (FastAPI events docs)

`@app.on_event("startup")` is **soft-deprecated** since FastAPI 0.93 (Feb 2023).
The canonical 2025 pattern is `@asynccontextmanager` lifespan passed to `FastAPI(lifespan=...)`.

```python
# bootstrap.py — inside build_rest_app()
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP — runs before uvicorn accepts the first connection ─────────
    container = app.container
    # Configure AnyIO thread pool: 1 inference thread + headroom for admin ops
    import anyio.to_thread
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 10   # conservative for single-GPU server

    # Optional: pre-warm TTS model so first request isn't cold.
    # Comment out to keep startup fast and load lazily on first request.
    # _ = container.tts_service()

    yield   # ← server accepts connections from here

    # ── SHUTDOWN — runs after last request drains ──────────────────────────
    container.unwire()

app = FastAPI(title="Chatterbox TTS API", version="1.0.0", lifespan=lifespan)
```

### `TestClient` and lifespan

`TestClient` with a `with` block **does** trigger lifespan — startup on `__enter__`,
shutdown on `__exit__`. Session-scoped fixtures must use `with TestClient(app) as client:`.

```python
# conftest.py — session-scoped client that fires lifespan once per test session
@pytest.fixture(scope="session")
def rest_client(rest_app):
    with TestClient(rest_app) as client:
        yield client
```

---

## 8. Structured Logging and Access Logs

### Research finding (uvicorn settings docs + python-json-logger PyPI)

#### 8.1 `BaseHTTPMiddleware` for per-request timing

```python
# adapters/inbound/rest/middleware.py
import logging, time, uuid
from starlette.middleware.base import BaseHTTPMiddleware

access_log = logging.getLogger("chatterbox.access")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            access_log.error("Request failed", extra={
                "request_id": request_id, "method": request.method,
                "path": request.url.path, "status_code": 500,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            })
            raise
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        access_log.info("Request completed", extra={
            "request_id": request_id, "method": request.method,
            "path": request.url.path, "status_code": response.status_code,
            "duration_ms": duration_ms,
        })
        response.headers["X-Request-Id"] = request_id
        return response
```

#### 8.2 uvicorn launch — prevent double logging

Research confirmed: `log_config=None` does **not** suppress uvicorn access logs —
it only prevents uvicorn from overwriting the app's log config. To suppress uvicorn's
native access log (so `RequestLoggingMiddleware` is the sole source), use `access_log=False`:

```python
uvicorn.run(
    app,
    host=args.host,
    port=args.port,
    log_config=None,     # don't overwrite app's log configuration
    access_log=False,    # suppress uvicorn's own access logger
)
```

#### 8.3 JSON structured logging (`python-json-logger`)

Latest stable: **4.1.0** (March 2025). Added to `rest` optional extra.

```python
# logging_config.py — new configure_json() function for REST mode
from pythonjsonlogger.json import JsonFormatter

def configure_json() -> None:
    """JSON log format for production REST deployments."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)
    for lib in ("transformers", "huggingface_hub", "diffusers"):
        logging.getLogger(lib).setLevel(logging.ERROR)
```

---

## 6. Dependency Injection Wiring

### Research findings (official dependency-injector FastAPI example)

Source: `ets-labs/python-dependency-injector` → `examples/miniapps/fastapi`

#### 6.1 `container.wire()` vs `wiring_config`

The modern preferred pattern is to declare `wiring_config` on the container class
so wiring fires automatically on instantiation:

```python
class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=[".routes"])
```

**We cannot use this** because `routes.py` is a deferred import (loaded only inside
`build_rest_app()`, after compat patches fire). Module-level `wiring_config` would
try to import `routes` at class-definition time — breaking the patch ordering guarantee.

**Resolution:** Use explicit `container.wire(modules=[routes_module])` inside
`build_rest_app()`, after the deferred import of `routes_module`. This is equally
valid and is the correct approach for deferred-import architectures.

#### 6.2 Decorator order — CRITICAL

`@inject` **must be the inner decorator**, directly above `def`, below the route
decorator. Reversed order silently breaks injection:

```python
# ✅ CORRECT
@router.post("/tts/generate")
@inject
async def generate_tts(...):

# ❌ WRONG — injection silently broken
@inject
@router.post("/tts/generate")
async def generate_tts(...):
```

#### 6.3 `app.container` (not `app.state.container`)

The official dependency-injector pattern attaches the container directly to the
app object as `app.container`, not `app.state.container`. Tests access it via
`app.container.provider.override(mock)`:

```python
# build_rest_app() — correct attachment
app.container = container   # NOT app.state.container
```

#### 6.4 Multiple `container.wire()` calls — gotcha

`container.wire()` modifies the routes module globally (Python module cache).
Calling it multiple times on the same module across tests causes `AlreadyWiredError`
or double-wrapping.

**Rule enforced in test design:** Tests share **one module-level `app` instance**.
Individual test customisation uses `app.container.provider.override()` context
managers — never calls `build_rest_app()` per test.

#### 6.5 Full wiring pattern (research-validated)

```python
# bootstrap.py — build_rest_app()
from fastapi import FastAPI
from adapters.inbound.rest import routes as routes_module
from infrastructure.container import AppContainer

container = AppContainer()
container.config.watermark_available.from_value(watermark_available)
container.wire(modules=[routes_module])   # once per process

app = FastAPI(title="Chatterbox TTS API", version="1.0.0")
app.container = container                 # attached for test override access
app.include_router(routes_module.router)
return app
```

```python
# routes.py — correct handler pattern
from dependency_injector.wiring import Provide, inject  # dependency_injector.wiring
from infrastructure.container import AppContainer

router = APIRouter(prefix="/api/v1")

@router.post("/tts/generate")   # route decorator FIRST
@inject                         # inject decorator SECOND, directly above def
async def generate_tts(
    body: TTSRequestSchema,
    tts: ITTSService = Depends(Provide[AppContainer.tts_service]),
):
    ...
```

### Why `container.wire()` over alternatives

| Approach | Pro | Con |
|---|---|---|
| `container.wire()` + `@inject` + `Provide[]` | Official pattern, full test override support, explicit | Routes import `AppContainer` at module level (safe here — deferred module import) |
| `app.state` + `Depends(lambda: app.state.tts)` | No container import in routes | Couples route to `Request`, defeats DI library test override API |
| Constructor injection (`RestHandlers` class) | Mirrors Gradio pattern | Does not compose with FastAPI's `Depends()` system cleanly |

---

## 9. Audio Encoding Helper

WAV encoding lives in the REST adapter layer — never in domain or services.

```python
# adapters/inbound/rest/schemas.py  (or a private _audio.py)
def audio_result_to_wav_bytes(result: AudioResult) -> bytes:
    """Encode float32 AudioResult as in-memory 16-bit WAV bytes.

    Uses scipy.io.wavfile — lightweight, no torchaudio dep in the REST layer.
    Normalises to [-1, 1] if peak > 1.0 to prevent clipping on encode.
    """
    import io
    import numpy as np
    from scipy.io import wavfile

    arr = result.samples.astype(np.float32)
    peak = float(np.abs(arr).max())
    if peak > 1.0:
        arr = arr / peak
    int16 = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, result.sample_rate, int16)
    return buf.getvalue()
```

`scipy` is already a transitive dependency of `chatterbox-tts`. No new install required.

---

## 10. CLI Changes (`cli.py`)

Add `--mode` argument:

```
uv run chatterbox-explorer --mode gradio        # default — unchanged behaviour
uv run chatterbox-explorer --mode api            # launch FastAPI + uvicorn
uv run chatterbox-explorer --mode api --port 8000 --host 0.0.0.0
```

The `--share`, `--no-browser`, `--mcp` flags remain valid for `--mode gradio`.
For `--mode api`, only `--host` and `--port` are relevant.

```python
# cli.py addition
parser.add_argument(
    "--mode",
    choices=["gradio", "api"],
    default="gradio",
    help="Delivery mode: 'gradio' launches the Gradio UI (default); "
         "'api' launches the FastAPI REST server via uvicorn.",
)
```

`build_app()` continues to build the Gradio demo.
`build_rest_app()` builds the FastAPI app.
Both call `AppContainer()` independently — same wiring pattern.

---

## 11. Dependency Changes

### Research findings (uv docs)

Source: `docs.astral.sh/uv/concepts/projects/dependencies/`

Key facts confirmed:
- `uv add <pkg> --optional <extra-name>` is the correct command to add to an optional extra.
- `uv sync --all-extras` installs all optional extras into the venv.
- `uv.lock` **always resolves ALL extras** (including optional ones) — the lockfile
  is universal. Only the installation step is selective.
- `httpx` is NOT a transitive dep of plain `fastapi` — only of `fastapi[standard]`.
  Must be added explicitly to the test group.
- `chatterbox-explorer[all]` self-reference inside `[dependency-groups].dev` requires
  a `[tool.uv.sources]` entry. Simpler alternative: use `uv sync --all-extras` for
  local dev without a self-reference in the dev group.

### `pyproject.toml` structure (adopted)

```toml
[project]
dependencies = [
    "chatterbox-tts",
    "dependency-injector>=4.41.0",
    "diffusers>=0.29.0,<1.0.0",
    "peft>=0.6.0",
    # gradio and fastapi/uvicorn are optional — see below
]

[project.optional-dependencies]
ui = [
    "gradio>=4.40.0",
]
rest = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.20.0",
    "python-json-logger>=3.2.0",   # structured JSON logs for production
]
all = [
    "chatterbox-explorer[ui,rest]",
]

[dependency-groups]
test = [
    "pytest>=9.0.3",
    "pytest-cov>=7.1.0",
    "httpx>=0.27.0",          # required by TestClient; NOT transitive from plain fastapi
    "pytest-benchmark>=5.1.0", # route micro-benchmarks; compatible with pytest>=9
]
```

### Add commands (reference — how these were added)

```bash
# These were added via uv (correct syntax confirmed from docs):
uv add gradio --optional ui
uv add fastapi --optional rest
uv add "uvicorn[standard]" --optional rest
uv add python-json-logger --optional rest
uv add httpx --dev            # added to test group
uv add pytest-benchmark --dev # added to test group
```

### Install commands (reference)

```bash
# Add to an optional extra (correct uv syntax):
uv add gradio --optional ui
uv add fastapi --optional rest
uv add "uvicorn[standard]" --optional rest

# Local dev — install everything:
uv sync --all-extras

# CI — only what the job needs:
uv sync --extra rest --no-dev   # REST API tests
uv sync --extra ui --no-dev     # UI tests
uv sync                          # core only (no extras)
```

`scipy` is a transitive dep of `chatterbox-tts` — relied on by `audio_result_to_wav_bytes()`
but not pinned explicitly. Document in `NOTES.md`.

---

## 12. Testing Strategy

### Research findings (FastAPI testing docs + dependency-injector test examples + pytest-benchmark PyPI)

#### 12.1 `TestClient` vs `AsyncClient`

**`TestClient` (sync) is the correct choice** for this codebase. From FastAPI docs:
> *"The TestClient does some magic inside to call the asynchronous FastAPI application
> in your normal `def` test functions."*

`AsyncClient` is only needed when the test function itself must `await` something
beyond HTTP calls. Our services are all synchronous — `TestClient` is sufficient.

Dependencies required: `httpx` (not a transitive dep of plain `fastapi` — only of
`fastapi[standard]`). Added explicitly to the `test` dependency group.

#### 12.2 Provider override pattern (research-validated)

**One shared `app` instance per test module.** Override per test using the context
manager — it auto-resets on exit, is thread-safe, and works with `TestClient`:

```python
# ✅ CORRECT — shared app, per-test override
from fastapi.testclient import TestClient
from bootstrap import build_rest_app

# Module-level: built once, wired once — never per-test
app = build_rest_app(watermark_available=False)
client = TestClient(app)

def test_tts_generate_returns_wav():
    mock_tts = MagicMock(spec=ITTSService)
    mock_tts.generate.return_value = AudioResult(
        sample_rate=24000, samples=np.zeros(24000, dtype=np.float32)
    )
    with app.container.tts_service.override(mock_tts):
        resp = client.post("/api/v1/tts/generate", json={"text": "Hello"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
```

```python
# ❌ WRONG — build_rest_app() per test causes container.wire() to be
# called multiple times on the same module → AlreadyWiredError
def test_something():
    app = build_rest_app(watermark_available=False)  # DO NOT DO THIS
    ...
```

#### 12.3 `httpx` dependency note

`httpx` is used by `TestClient` internally. It is **not** automatically installed
with plain `fastapi` — only with `fastapi[standard]`. Added to `[dependency-groups].test`.

#### 12.4 Session-scoped app + lifespan

`TestClient` triggers lifespan only inside a `with TestClient(app) as client:` block.
Session-scoped fixtures must use the context manager form to fire startup/shutdown
exactly once per test session:

```python
@pytest.fixture(scope="session")
def rest_app():
    from bootstrap import build_rest_app
    return build_rest_app(watermark_available=False)

@pytest.fixture(scope="session")
def rest_client(rest_app):
    with TestClient(rest_app) as client:   # triggers lifespan on enter
        yield client
    # lifespan shutdown fires automatically on exit
```

#### 12.5 Benchmark tests (`pytest-benchmark>=5.1.0`)

Source: pytest-benchmark PyPI — latest **5.2.3** (Nov 2025), compatible with `pytest>=9.0.3`.

```python
# test_rest_benchmark.py — mocked service, measures routing overhead only
@pytest.mark.benchmark(group="rest-routes")
def test_benchmark_tts_generate(benchmark, rest_client, mock_tts):
    with rest_app.container.tts_service.override(mock_tts):
        benchmark(
            rest_client.post,
            "/api/v1/tts/generate",
            json={"text": "Hello world"},
        )
    stats = benchmark.stats
    assert stats["median"] < 0.050, f"Median {stats['median']:.3f}s exceeds 50 ms"
```

Metrics to assert (mocked service — measures routing/serialization overhead only):

| Metric | Threshold |
|--------|-----------|
| Median (p50) | < 50 ms |
| p99 (max) | < 200 ms |
| Stddev | < 2× median |

#### 12.6 Middleware logging test

```python
def test_request_logging_middleware_adds_request_id(rest_client):
    resp = rest_client.get("/api/v1/health")
    assert "X-Request-Id" in resp.headers
    # UUID4 format: 8-4-4-4-12 hex chars
    import re
    assert re.match(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        resp.headers["X-Request-Id"],
    )
```

### Test categories

| Category | Location | What is tested | Models needed |
|---|---|---|---|
| Schemas | `tests/unit/adapters/rest/test_rest_schemas.py` | Pydantic validation, `to_domain()` round-trip, WAV encoding | No |
| Routes — happy path | `tests/unit/adapters/rest/test_rest_routes.py` | Each endpoint: status, content-type, service called with correct args | No (mocked via `app.container.*.override()`) |
| Routes — error paths | `tests/unit/adapters/rest/test_rest_routes.py` | `ValueError`→422, `RuntimeError`→503, unknown key→404 | No |
| Middleware | `tests/unit/adapters/rest/test_rest_routes.py` | `X-Request-Id` header present, timing logged | No |
| Benchmarks | `tests/unit/adapters/rest/test_rest_benchmark.py` | Routing overhead: median < 50 ms (mocked) | No |
| Integration | `tests/integration/test_rest_integration.py` | Real model + full HTTP round-trip | Yes (`--slow` marker) |

### Coverage target
`schemas.py`, `routes.py`, `middleware.py`, `concurrency.py` must reach ≥ 90% individually.
Overall project threshold remains ≥ 95%.
`bootstrap.py` remains excluded from measurement (integration-level concern).

---

## 13. Implementation Phases

Each phase leaves tests green before proceeding.
**All research is complete (2 rounds). Patterns below are fully docs-validated.**

### Phase 1 — Dependencies ✅ DONE
- `fastapi>=0.100.0`, `uvicorn[standard]>=0.20.0`, `python-json-logger>=3.2.0` → `[project.optional-dependencies].rest`
- `gradio>=4.40.0` → `[project.optional-dependencies].ui`
- `httpx>=0.27.0`, `pytest-benchmark>=5.1.0` → `[dependency-groups].test`
- `uv lock && uv sync --all-extras` run
- `--mode ui|rest` flag and extra guards in `cli.py`

### Phase 2 — Concurrency + Middleware (new files)
- `src/adapters/inbound/rest/concurrency.py` — `inference_semaphore = asyncio.Semaphore(1)`
- `src/adapters/inbound/rest/middleware.py` — `RequestLoggingMiddleware` with `X-Request-Id`, timing

### Phase 3 — Schemas (`schemas.py`) ✅ IN PROGRESS (fix WAV encoding)
- Pydantic request schemas with `.to_domain()` methods
- Response schemas: `ModelStatusResponse`, `MemoryStatsResponse`, `WatermarkResponse`, `MessageResponse`, `HealthResponse`
- `audio_result_to_wav_bytes()` helper using `scipy.io.wavfile`
- `scipy` is transitive from `chatterbox-tts` — no explicit pin

### Phase 4 — Routes (`routes.py`) — FIX P0 BUG
- All blocking calls wrapped: `await run_in_threadpool(service.method, args)`
- `async with inference_semaphore:` wrapping every GPU/CPU-bound call
- `@router.post/get(...)` → `@inject` → `async def` order confirmed correct
- `Depends(Provide[AppContainer.*])` for all services
- Error translation: `ValueError`→422, `RuntimeError`→503, unknown key→404

### Phase 5 — Bootstrap wiring (`build_rest_app`)
- `@asynccontextmanager` lifespan (not deprecated `on_event`)
- AnyIO thread pool tuning: `limiter.total_tokens = 10`
- `app.container = container` (confirmed official pattern)
- `app.add_middleware(RequestLoggingMiddleware)`

### Phase 6 — Logging (`logging_config.py`)
- New `configure_json()` function for REST/production mode
- `python-json-logger>=4.1.0` formatter
- Suppress `uvicorn.access` logger (middleware is sole access-log source)
- Called from `_launch_rest()` in `cli.py`

### Phase 7 — CLI (`cli.py`) — update `_launch_rest`
- `access_log=False` added to `uvicorn.run()` (research: `log_config=None` alone does NOT suppress access logs)
- Call `configure_json()` before uvicorn starts

### Phase 8 — Tests
- `tests/unit/adapters/rest/__init__.py`
- `test_rest_schemas.py` — Pydantic validation, `to_domain()`, WAV RIFF header
- `test_rest_routes.py` — session-scoped `app`+`client`; per-test provider overrides; happy path + error path for all endpoints; middleware header check
- `test_rest_benchmark.py` — `pytest-benchmark` micro-benchmarks; median < 50 ms (mocked)

### Phase 9 — Validation
```bash
uv run pytest tests/ -x --tb=short -q
uv run pytest --cov=src tests/unit/
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
# Benchmark:
uv run pytest tests/unit/adapters/rest/test_rest_benchmark.py --benchmark-only -v
# Smoke test:
uv run chatterbox-explorer --mode rest --port 8001
curl http://localhost:8001/api/v1/health
```

---

## 14. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **P0: sync blocking in `async def` routes** | **Fixed** | **Critical** | All inference calls wrapped with `await run_in_threadpool(fn, args)` |
| `asyncio.Semaphore` not GPU-proof alone | Low | Medium | `run_in_threadpool` offloads to thread; semaphore is the application-level gate |
| `container.wire()` called multiple times across tests | **HIGH** | High | Session-scoped `rest_app` fixture — `build_rest_app()` called once per test session |
| `@inject` placed above route decorator (silent breakage) | Low | High | Confirmed order: `@router.*` → `@inject` → `async def` ✓ |
| `app.container` vs `app.state.container` | Fixed | High | `app.container = container` confirmed from official DI library examples |
| `log_config=None` alone doesn't suppress uvicorn access logs | Fixed | Medium | `access_log=False` added to `uvicorn.run()` separately |
| `on_event("startup")` deprecated | Fixed | Low | Using `@asynccontextmanager` lifespan |
| `scipy` not explicitly pinned | Low | Medium | Transitive from `chatterbox-tts`; add import guard in `test_rest_schemas.py` |
| VC / watermark temp files leak on exception | Fixed | Medium | `try/finally` cleans up all temp files |
| `httpx` not transitive from plain `fastapi` | Fixed | Medium | Added to `[dependency-groups].test` |
| `pytest-benchmark` incompatible with `pytest>=9` | Verified | Low | `pytest-benchmark>=5.1.0` (5.2.3 latest) is compatible with `pytest>=9.0.3` |
| lifespan not triggered in tests | Fixed | Medium | Session-scoped `with TestClient(app) as client:` triggers lifespan correctly |

---

## 15. Resolved Design Questions

All open questions resolved before implementation:

1. **Sync blocking in `async def`** — P0 bug confirmed by research. Fix: `await run_in_threadpool(fn, args)` for every service call. `asyncio.Semaphore(1)` for GPU serialization.

2. **Lifespan** — `@asynccontextmanager` + `FastAPI(lifespan=lifespan)`. `on_event` deprecated. AnyIO thread pool tuned in startup.

3. **Access logging** — `BaseHTTPMiddleware` with per-request timing and `X-Request-Id`. `log_config=None` + `access_log=False` in `uvicorn.run()`.

4. **JSON structured logs** — `python-json-logger>=4.1.0`. New `configure_json()` in `logging_config.py` called from `_launch_rest()`.

5. **VC file upload** — `UploadFile` → temp file → path → `VoiceConversionRequest`. `try/finally` cleans up.

6. **Model manager unknown key** — check `manager.get_all_status()` first → 404 if missing. Load/unload failures → 503.

7. **`ref_audio_path` in TTS v1** — Always `None`. Voice cloning via file upload deferred to v2.

8. **`app.container`** — confirmed from official DI library examples. Not `app.state.container`.

9. **`TestClient` vs `AsyncClient`** — `TestClient` correct. Session-scoped with `with TestClient(app) as client:` triggers lifespan.

10. **`container.wire()` once per process** — session-scoped `rest_app` fixture ensures single call.

11. **Benchmark tooling** — `pytest-benchmark>=5.1.0` (5.2.3 latest, compatible with `pytest>=9.0.3`). Locust/k6 for load testing (out of scope for this PR).

12. **`asyncio.Semaphore` at module level** — Safe in Python 3.11 (deprecation was about `get_event_loop()`, not primitive construction). `concurrency.py` creates it at import time.

---

## 16. Success Criteria

- [ ] `uv run chatterbox-explorer --mode rest` starts uvicorn on port 7860 without error
- [ ] `curl -X GET http://localhost:7860/api/v1/health` returns `{"status":"ok","device":"..."}`
- [ ] Every inference request logged with `request_id`, `method`, `path`, `status_code`, `duration_ms`
- [ ] `X-Request-Id` header present in every response
- [ ] `curl -X POST http://localhost:7860/api/v1/tts/generate -H 'Content-Type: application/json' -d '{"text":"Hello"}' --output out.wav` produces a valid WAV file (starts with `RIFF`)
- [ ] No event loop blocking — `/api/v1/health` responds immediately even during active inference
- [ ] `uv run pytest tests/ -x --tb=short` — all existing 538 + all new REST tests pass
- [ ] `uv run pytest --cov=src tests/unit/` — coverage ≥ 95%
- [ ] `uv run ruff check src/ tests/` — zero errors
- [ ] `uv run pytest tests/unit/adapters/rest/test_rest_benchmark.py --benchmark-only` — median < 50 ms (mocked)
- [ ] `GET /api/v1/docs` serves OpenAPI interactive documentation
- [ ] Every endpoint: `ValueError` → 422, `RuntimeError` → 503, unknown model key → 404
- [ ] `uv run chatterbox-explorer` (no flags / `--mode ui`) continues to launch Gradio unchanged

---

## 17. Constraints

1. **Zero domain changes** — `domain/`, `ports/`, `services/` are frozen
2. **Zero Gradio changes** — existing inbound Gradio adapter is untouched
3. **Deferred import order preserved** — `build_rest_app()` imports inside function body
4. **No new domain concepts** — Pydantic schemas live in the inbound adapter, not domain
5. **No streaming in v1** — one-shot endpoints only; streaming deferred to v2
6. **No auth in v1** — add in a future `adapters/inbound/rest/middleware/` layer
7. **`run_in_threadpool` on every blocking call** — no exceptions; any sync service method called from `async def` must be wrapped
8. **`inference_semaphore` on all GPU/CPU-bound operations** — TTS, VC, watermark, model load/unload
9. **`@asynccontextmanager` lifespan only** — `on_event` is deprecated and must not be used
10. **`access_log=False` in `uvicorn.run()`** — middleware is the sole access-log source; no double-logging
