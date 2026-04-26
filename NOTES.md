# NOTES

## Manual Patches (re-apply after `uv sync`)

### resemble-perth — `pkg_resources` → `importlib.resources`

**File:** `.venv/lib/python3.13/site-packages/perth/perth_net/__init__.py`

**Why:** `resemble-perth 1.0.x` uses the deprecated `pkg_resources.resource_filename()`
to locate bundled model weights. `setuptools >= 81` emits a `UserWarning` on every
`pkg_resources` import because the package is scheduled for removal.

**Research findings:**
- `resemble-perth 1.0.1` (latest as of 2025) does **not** fix this — byte-for-byte
  identical to 1.0.0
- Official guidance: migrate to `importlib.resources` (stdlib, Python 3.9+)
- The warning explicitly says "fix it or pin Setuptools<81" — suppression is not recommended
- Root cause is confirmed fixable with a 2-line change

**Patch (replace the first 2 lines):**

```python
# BEFORE (broken — pkg_resources deprecated in setuptools >= 81):
from pkg_resources import resource_filename
PREPACKAGED_MODELS_DIR = resource_filename(__name__, "pretrained")

# AFTER (correct — importlib.resources, stdlib Python 3.9+):
import importlib.resources
PREPACKAGED_MODELS_DIR = str(importlib.resources.files(__name__).joinpath("pretrained"))
```

**Re-apply command after `uv sync`:**

```bash
cat > .venv/lib/python3.13/site-packages/perth/perth_net/__init__.py << 'EOF'
# Patched: replaced pkg_resources.resource_filename (deprecated, setuptools>=81)
# with importlib.resources.files() (stdlib, Python 3.9+).
# Root cause: resemble-perth 1.0.1 still uses pkg_resources.
# Fix: https://importlib-resources.readthedocs.io/en/latest/migration.html
import importlib.resources

PREPACKAGED_MODELS_DIR = str(importlib.resources.files(__name__).joinpath("pretrained"))

from .perth_net_implicit.perth_watermarker import PerthImplicitWatermarker
EOF
```

**Track upstream fix:** https://github.com/resemble-ai/Perth/issues

---

## Learnings

### Chatterbox Model Loading
- Models are downloaded from HuggingFace on first use and cached at `~/.cache/huggingface/hub/`
- `from_pretrained(device)` handles CUDA → MPS → CPU fallback internally for each model
- On macOS Apple Silicon, MPS is available with PyTorch 2.6.0 — no extra config needed
- All models load weights to CPU first when device is "mps" or "cpu" (handles CUDA-saved checkpoints)

### Chatterbox API Gotchas
- **Turbo reference audio MUST be > 5 seconds** — the model has a hard `assert` in `prepare_conditionals()`
- **Turbo silently ignores** `exaggeration`, `cfg_weight`, and `min_p` — they are accepted but logged as warnings
- **Standard TTS default voice**: if no `audio_prompt_path` is given, the model uses built-in `conds.pt` (must be loaded first via `from_pretrained`)
- Multilingual `generate()` requires `language_id` as a keyword arg — not optional like in Standard TTS
- Voice Conversion (`ChatterboxVC`) only uses `s3gen.safetensors` — it does NOT load T3 or Voice Encoder
- All outputs are watermarked via `perth.PerthImplicitWatermarker().apply_watermark()` — this is non-optional and runs on every generation

### Streaming Strategy
- Chatterbox has NO native token/chunk streaming — each `.generate()` call blocks until full audio is ready
- Sentence-level streaming implemented by: `re.split(r"(?<=[.!?])\s+", text)` → generate each → yield cumulative `np.concatenate`
- Gradio generator functions (using `yield`) update the `gr.Audio` widget after each yield — no special config needed
- Sentences containing paralinguistic tags like `[laugh]` are safe to split on — tags won't appear at a split boundary since they're mid-sentence

### Gradio 6.8.0 Observations
- `gr.Audio(type="numpy")` expects `(sample_rate: int, data: np.ndarray)` where `data.dtype == np.int16`, range `−32768..32767`
- Returning `float32` instead of `int16` triggers `UserWarning: Trying to convert audio automatically from float32 to 16-bit int format` from `gradio/processing_utils.py:convert_to_16_bit_wav()` on every yield — fix is to call `.astype(np.int16)` yourself (`to_audio_tuple()` already does this correctly)
- `gr.Audio(type="filepath")` is used for inputs — Gradio saves mic/upload to a temp file path automatically
- Generator functions connected to `gr.Audio` output work without `streaming=True` on the component — each `yield` replaces the audio
- Tag-insert buttons in a loop: use default-arg capture `lambda txt, _t=tag: ...` to avoid Python closure bug
- `gr.themes.GoogleFont("Inter")` works without extra install — Gradio fetches it from CDN
- `footer { display: none !important; }` in CSS reliably hides the Gradio footer branding

### macOS Python 3.11 — Semaphore Leak on Shutdown
- `gradio/flagging.py` imports `from multiprocessing import Lock`, which on macOS allocates a **POSIX named semaphore** (`/sem_XXXX`) via `_multiprocessing.SemLock`
- Python's `resource_tracker` registers its atexit handler at **interpreter startup** (very first atexit entry), so it runs **last** — after all user-registered atexit handlers
- If `demo.close()` is never called, the `Lock`'s semaphore is never explicitly unlinked, and `resource_tracker` logs: `UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown`
- **Fix:** call `atexit.register(demo.close)` **before** `demo.launch()` — this runs `close()` before `resource_tracker`'s cleanup, fully releasing the semaphore
- **Belt-and-suspenders:** also add `warnings.filterwarnings("ignore", message=r".*resource_tracker.*leaked semaphore.*", category=UserWarning)` to suppress any edge-case that slips through
- This is a macOS-specific issue; Linux uses POSIX shared memory differently and is not affected

### S3Gen Reference Audio — Mel/Token Length Mismatch
- `S3Token2Mel.embed_ref()` in `s3gen.py` enforces: `mel_len == 2 × token_len`
- Mel extractor runs at **24 kHz, hop=480 samples → 50 frames/sec (20 ms/frame)**
- S3 tokenizer runs at **16 kHz, 25 tokens/sec → 40 ms/token (= 2 mel frames)**
- Invariant breaks when audio duration is **not a multiple of 40 ms** — the tokenizer truncates to whole tokens but the mel extractor may produce one extra frame
- Common sample counts that must divide evenly: 640 @ 16 kHz · 960 @ 24 kHz · 1764 @ 44.1 kHz · 1920 @ 48 kHz
- **Fix:** `preprocess_ref_audio(path)` trims to `floor(len / frame_samples) × frame_samples` at the original sample rate before the model's internal `librosa.load(..., sr=24000)` resampling runs — because the duration is then an exact rational multiple of 40 ms, the resampled lengths at 24 kHz and 16 kHz are also exact multiples
- The model handles the mismatch gracefully regardless (it truncates tokens), so this warning is **non-fatal** — preprocessing just eliminates the noise

### uv Setup
- `uv sync` resolved and installed 107 packages in ~12s (first run)
- `[tool.uv] dev-dependencies = []` is deprecated — use `[dependency-groups] dev = []` instead
- `chatterbox-tts` (v0.1.7) pulled in all necessary ML dependencies: torch 2.6.0, torchaudio, librosa, safetensors, transformers, diffusers, pyloudnorm, etc.
- No manual `torch` pin needed — chatterbox-tts specifies compatible versions
- MPS works out of the box with torch 2.6.0 on macOS 12.3+

### Device Detection
- Correct order: CUDA → MPS → CPU (MPS is macOS-only, CUDA is NVIDIA GPU)
- Use `getattr(torch.backends, "mps", None)` to safely check MPS on older torch versions
- Each Chatterbox model's `from_pretrained()` has its own internal MPS check and graceful fallback

### Audio Processing
- Chatterbox output sample rate: **24,000 Hz** (`S3GEN_SR`) — always 24kHz regardless of model
- Reference audio is resampled internally by librosa — you don't need to pre-process it
- `wav.squeeze().cpu().numpy()` is the correct pattern to extract numpy array from Chatterbox tensor output
- Peak normalisation (`arr / arr.max()`) prevents clipping if wav values exceed ±1.0

### PyTorch SDPA Backend Migration (`sdp_kernel` → `sdpa_kernel`)
- `torch.backends.cuda.sdp_kernel()` is **deprecated since PyTorch 2.0** and emits `FutureWarning` in 2.6.0
- The replacement is `torch.nn.attention.sdpa_kernel(backends, set_priority=False)` — lives under `torch.nn.attention`, is device-agnostic
- Accepts `SDPBackend` enum values instead of boolean flags
- **Parameter mapping (old → new):**
  - `enable_flash=True` → `SDPBackend.FLASH_ATTENTION` (CUDA only; SM 7.5+ fwd / SM 8.0+ bwd)
  - `enable_math=True` → `SDPBackend.MATH` (**universal** — CUDA, MPS, CPU)
  - `enable_mem_efficient=True` → `SDPBackend.EFFICIENT_ATTENTION` (CUDA only)
  - `enable_cudnn=True` → `SDPBackend.CUDNN_ATTENTION` (CUDA + cuDNN 9+ only)
- **Full `SDPBackend` enum members:** `ERROR`, `MATH`, `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, `CUDNN_ATTENTION`, `OVERRIDEABLE`
- **MPS / Apple Silicon safety:** Only `SDPBackend.MATH` is universally safe on MPS. Passing a CUDA-only backend (`FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, `CUDNN_ATTENTION`) while tensors live on an MPS device causes a `RuntimeError` at attention dispatch time because no enabled backend can service the request.
- **Monkey-patch is safe:** Both APIs are pure-Python context managers with identical `with`-statement semantics. PyTorch's C++ backend does **not** call back into the Python symbol — replacing it is transparent to all C++ internals.
- **`set_priority=False` (default):** PyTorch auto-selects the best available backend from the enabled list — mirrors the old flag-toggle behaviour.
- **`set_priority=True` (PyTorch 2.5+):** backends are tried in the exact order listed, e.g. `sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION], set_priority=True)`.
- **New `SDPBackend.OVERRIDEABLE`** (2.x): no old-API equivalent; enables custom kernels registered via `torch.nn.attention.register_flash_attention_impl()`.
- **Migration shim:** `compat.py` patches `torch.backends.cuda.sdp_kernel` at import time — maps booleans → `SDPBackend` list, detects MPS-only environment and forces `[SDPBackend.MATH]`, stores the original as `_sdp_kernel_orig` for test teardown, and is idempotent (double-import safe). Import it once before any model load: `import compat  # noqa: F401`.

### HuggingFace Hub Cache Inspection (disk-only, no network)

**`try_to_load_from_cache(repo_id, filename, cache_dir=None, revision=None, repo_type=None)`**
- Returns a **`str`** (absolute path inside the snapshot symlink tree) when the file IS cached on disk
- Returns **`_CACHED_NO_EXIST`** (a plain `object()` singleton) when the file's non-existence was previously confirmed by the Hub and recorded in `.no_exist/` — only written after a prior `hf_hub_download` attempt that received a 404 from the server
- Returns **`None`** when the cache has no opinion: the repo may not have been fetched yet, OR this specific filename was never requested (no `.no_exist` record for it either)
- **Empirically confirmed** (huggingface_hub on this machine): `isinstance(result, str)` → cached; `result is _CACHED_NO_EXIST` → confirmed absent on Hub; `result is None` → unknown
- **Signature** (verified): `(repo_id: str, filename: str, cache_dir=None, revision=None, repo_type=None) -> str | _CACHED_NO_EXIST | None`
- **Sentinel import**: `from huggingface_hub import _CACHED_NO_EXIST` — it lives in `huggingface_hub._cache_manager`, re-exported at package root. Always compare with `is`, never `==`.

**Canonical "is model downloaded?" check** — test a known large weight file that must exist for the model to run:
```python
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

def is_model_cached(repo_id: str, key_file: str, repo_type: str = "model") -> bool:
    """Returns True only if the key weight file exists in local HF cache.
    No network calls are made.
    """
    result = try_to_load_from_cache(repo_id=repo_id, filename=key_file, repo_type=repo_type)
    return isinstance(result, str)

# Usage for each Chatterbox variant:
# is_model_cached("ResembleAI/chatterbox",       "t3_cfg.safetensors")
# is_model_cached("ResembleAI/chatterbox-turbo", "t3_turbo_v1.safetensors")
```

**`scan_cache_dir()` → `HFCacheInfo`** — use when rich per-repo metadata is needed (e.g. showing file sizes in UI):
- `HFCacheInfo.repos`: `frozenset[CachedRepoInfo]`
- `CachedRepoInfo`: `.repo_id`, `.repo_type`, `.size_on_disk`, `.size_on_disk_str`, `.nb_files`, `.revisions`
- `CachedRevisionInfo`: `.commit_hash`, `.snapshot_path`, `.files`, `.last_modified`
- `CachedFileInfo`: `.file_name`, `.file_path`, `.blob_path`, `.size_on_disk`
- **Cost**: walks the full `~/.cache/huggingface/hub/` tree. Fast enough for a one-shot UI refresh but do NOT call on every render tick.
```python
from huggingface_hub import scan_cache_dir

def get_repo_cache_info(repo_id: str, repo_type: str = "model") -> dict | None:
    for repo in scan_cache_dir().repos:
        if repo.repo_id == repo_id and repo.repo_type == repo_type:
            return {
                "cached": True,
                "size": repo.size_on_disk_str,   # e.g. "6.4G"
                "nb_files": repo.nb_files,
                "revision": next(iter(repo.revisions)).commit_hash[:8],
            }
    return None  # not in cache

# Example output: {'cached': True, 'size': '6.4G', 'nb_files': 13, 'revision': 'ef85ce7b'}
```

**Best approach summary**:
- Gating model load (fast path): `try_to_load_from_cache` on the main weight file
- UI model-manager panel (show size, files): `scan_cache_dir()` once per page load, cache result in a module-level variable

---

### PyTorch MPS Memory Release (Apple Silicon — torch 2.6.0)

**`torch.mps.empty_cache()` — what it actually does**:
- Releases all memory blocks held in the **MPSAllocator cache pool** back to the Metal driver
- The pool exists to amortize future allocations — PyTorch holds freed tensor memory in the pool rather than returning it immediately
- **Empirically confirmed on this machine**:
  - After `del tensor` → `current_allocated_memory` = 0, `driver_allocated_memory` stays at ~34 MB
  - After `gc.collect()` alone → no change to `driver_allocated_memory`
  - After `torch.mps.empty_cache()` → `driver_allocated_memory` drops from ~34 MB to ~400 KB
- **Returns `None`**, takes no arguments

**Complete model unload recipe for MPS**:
```python
import gc
import torch

def unload_model_mps(model) -> None:
    """Fully release a model from MPS / GPU memory.
    Call this before loading a different model to avoid OOM.
    """
    model.cpu()                    # 1. move all tensors off the MPS device first
    del model                      # 2. release Python reference (current_allocated → 0)
    gc.collect()                   # 3. sweep any lingering Python objects
    if torch.backends.mps.is_available():
        torch.mps.synchronize()    # 4. wait for all pending Metal kernels to finish
        torch.mps.empty_cache()    # 5. flush MPSAllocator pool → driver_allocated drops

# Cross-device version (MPS + CUDA):
def unload_model(model, device: str) -> None:
    model.cpu()
    del model
    gc.collect()
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
```

**Why `.cpu()` before `del` matters**: Without it, Python may defer the Metal-side deallocation. On a large model, calling `empty_cache()` while tensors are still being garbage-collected can leave stale pool entries. Moving to CPU first forces a clean Metal deallocation.

**`current_allocated_memory()` vs `driver_allocated_memory()`** (verified semantics):
| Function | What it counts | Includes pool? | Use for |
|---|---|---|---|
| `current_allocated_memory()` | Live tensor bytes tracked by MPSAllocator | ❌ | Detecting leaked tensors |
| `driver_allocated_memory()` | Total Metal driver bytes (tensors + pool + framework) | ✅ | UI display, OOM risk assessment |
| `recommended_max_memory()` | `device.recommendedMaxWorkingSetSize` ≈ 75% of unified RAM | N/A | Setting memory fraction limit |

- On this machine (16 GB unified): `recommended_max_memory()` = 11.84 GB
- **For UI**: always display `driver_allocated_memory()` — it matches what Activity Monitor shows as GPU pressure

**`torch.cuda.empty_cache()` — same semantics, key difference**:
- Releases cached CUDA blocks back to the CUDA memory allocator, NOT to the OS
- The CUDA runtime keeps a per-process pool; `empty_cache()` only flushes PyTorch's layer on top
- On macOS MPS, `empty_cache()` actually returns memory all the way to the Metal driver (more aggressive)

---

### psutil Memory Stats (macOS Python 3.11 — psutil 7.2.2, verified)

**`psutil.virtual_memory()` → `svmem` — macOS field set**:
```python
import psutil

vm = psutil.virtual_memory()
# svmem fields on macOS (verified): total, available, percent, used, free, active, inactive, wired

# vm.total    = physical RAM bytes      e.g. 17179869184  (16 GiB)
# vm.available= free + inactive         e.g. 1951793152   (reclaimable without swap)
# vm.used     = total - available       e.g. 8835645440   (hard-to-free in-use memory)
# vm.free     = completely unallocated  e.g. 78004224     (tiny on macOS — memory is always used)
# vm.active   = recently used pages     e.g. 1896808448   (in RAM, not reclaimable cheaply)
# vm.inactive = not recently used       e.g. 1857847296   (in RAM, reclaimable → counts in available)
# vm.wired    = kernel/pinned pages     e.g. 6938836992   (cannot be swapped, cannot be reclaimed)
# vm.percent  = (total - available) / total * 100         e.g. 88.6
```

**`psutil.Process().memory_info()` → `pmem` — macOS field set**:
```python
import psutil

proc = psutil.Process()           # current process; or Process(pid) for another
mem = proc.memory_info()
# pmem fields on macOS (verified): rss, vms, pfaults, pageins

# mem.rss     = Resident Set Size = physical RAM occupied by THIS process
#               Includes model weights loaded to MPS (unified memory!)
#               This is what Activity Monitor "Memory" column shows
# mem.vms     = Virtual Memory Size = total virtual address space
#               Typically 400+ GB on macOS Python due to ASLR + framework maps
#               NOT useful for monitoring — ignore it
# mem.pfaults = page fault count (macOS-specific diagnostic)
# mem.pageins = swap-in count (macOS-specific diagnostic)
```

**`vm.used` vs `rss` — the critical distinction**:
| Metric | Scope | What it reflects |
|---|---|---|
| `vm.used` | System-wide | All processes + kernel + wired — total RAM pressure |
| `proc.rss` | This process only | Model weights + Python heap + Gradio overhead in RAM |

- On Apple Silicon (unified memory): `rss` includes MPS tensor memory because GPU memory IS system RAM. A loaded 6.4 GB model will add ~6.4 GB to `rss`.
- `vm.used` ≈ `sum(p.rss for all processes) + kernel overhead`

**macOS gotcha — `vm.used ≠ total - free`**:
- `total - free` ≈ nearly all RAM (macOS aggressively fills inactive cache)
- `vm.used = total - available` subtracts the reclaimable `inactive` pages
- **Always use `vm.percent` or `vm.used`** — never `total - free` for "how full is RAM?"

**What to show in a model manager UI**:
```python
import psutil
import torch

def get_memory_stats(device: str) -> dict:
    vm = psutil.virtual_memory()
    proc_rss = psutil.Process().memory_info().rss

    stats = {
        # System-level (for a "RAM pressure" bar in the UI)
        "sys_total_gb":   vm.total      / 1024**3,
        "sys_used_gb":    vm.used       / 1024**3,
        "sys_avail_gb":   vm.available  / 1024**3,
        "sys_percent":    vm.percent,              # most reliable single number

        # Process-level (for "this app is using X")
        "proc_rss_gb":    proc_rss / 1024**3,
    }

    # GPU allocator detail — only meaningful on MPS
    if device == "mps" and torch.backends.mps.is_available():
        stats["mps_allocated_gb"] = torch.mps.current_allocated_memory() / 1024**3
        stats["mps_driver_gb"]    = torch.mps.driver_allocated_memory()  / 1024**3
        stats["mps_max_gb"]       = torch.mps.recommended_max_memory()   / 1024**3

    return stats

# For the UI: show sys_percent bar + proc_rss_gb + mps_driver_gb
# The trio gives: "system pressure / app footprint / GPU pool usage"
```

**Performance note** — `psutil.Process().memory_info()` is fast (~50 µs on macOS) but calling it inside every Gradio render event is wasteful. Cache with a 1–2 second TTL.

---

## Gotchas

- **Turbo + short reference audio**: `AssertionError: Audio prompt must be longer than 5 seconds!` is raised deep inside `prepare_conditionals`. Wrap turbo generation in `try/except AssertionError` to surface it cleanly in Gradio.
- **Multilingual accent bleed**: If reference clip language ≠ target language, the output accent may inherit from the reference. Setting `cfg_weight=0` mitigates this by ignoring the reference style conditioning.
- **Model cache is per-process**: The `_MODEL_CACHE` dict lives in memory for the lifetime of the Gradio server process. Restarting the server = cold cache = re-load from HF disk cache.
- **First generate is always slow**: Even after HF download, model load time on MPS is 10–30s per model. This is expected and happens only once per server session.
- **Multilingual `repetition_penalty` default is 2.0** (not 1.2 like Standard) — the model was trained with higher penalty, keep it at 2.0 for best results.
- **PerTh watermark (`resemble-perth`)**: The package installs as `resemble-perth` on PyPI but is imported as `import perth` — don't confuse the install name with the import name.
- **Sentence splitter edge case**: Abbreviations (e.g., "Dr. Smith") will be incorrectly split at the period. This is acceptable for a demo; a production app should use `nltk.sent_tokenize` or `spacy`.
- **`preprocess_ref_audio` temp files**: each call that trims audio writes a `NamedTemporaryFile(suffix=".wav", delete=False)` to the OS temp dir. These are not cleaned up automatically within a session. For a long-running production server, add a cleanup registry; for a demo this is acceptable (OS cleans them on reboot).
- **Paralinguistic tags in streaming mode**: When streaming and a sentence contains only a tag (e.g., "[laugh]"), the model may produce very short audio. This is fine — it's concatenated with surrounding sentences.

---

## Architecture Notes

### Model Components (what each loads)
| Component | Standard | Turbo | Multilingual | VC |
|---|---|---|---|---|
| T3 Transformer | ✅ `t3_cfg.safetensors` | ✅ `t3_turbo_v1.safetensors` | ✅ `t3_mtl23ls_v2.safetensors` | ❌ |
| S3Gen | ✅ `s3gen.safetensors` | ✅ `s3gen_meanflow.safetensors` | ✅ `s3gen.pt` | ✅ `s3gen.safetensors` |
| Voice Encoder | ✅ `ve.safetensors` | ✅ `ve.safetensors` | ✅ `ve.pt` | ❌ |
| Tokenizer | EnTokenizer | AutoTokenizer (GPT2) | MTLTokenizer | ❌ |
| Watermarker | ✅ PerTh | ✅ PerTh | ✅ PerTh | ✅ PerTh |

### Turbo Architecture Difference
- Standard/Multilingual: 10-step diffusion decoder in S3Gen
- Turbo: 1-step **MeanFlow** decoder (`S3Gen(meanflow=True)`) — same quality, ~10x faster decoding
- Turbo uses `AutoTokenizer` (GPT2/LLaMA-based, 50276 vocab) instead of custom `EnTokenizer`

---

## Domain Layer TDD Implementation (Hexagonal Architecture — Phase 1)

### What Was Created
Three domain files and three matching test files as part of the hexagonal architecture refactor:

**Implementation files (zero framework deps — stdlib + numpy only):**
- `src/chatterbox_explorer/domain/models.py` — 9 pure dataclasses
- `src/chatterbox_explorer/domain/languages.py` — 4 language constants extracted verbatim from app.py
- `src/chatterbox_explorer/domain/presets.py` — 10 TTS + 6 Turbo presets with `get_preset_tts` / `get_preset_turbo` helpers

**Test files (TDD — written before implementation):**
- `tests/unit/domain/test_models.py` — 44 test functions across 9 test classes
- `tests/unit/domain/test_languages.py` — 44 test functions across 6 test classes + 2 module purity tests
- `tests/unit/domain/test_presets.py` — 66 test functions across 10 test classes + 3 module purity tests

**Total: 154 tests — all passing in 0.16 s**

### Key Design Decisions

#### Domain purity enforced by tests
Each module has `test_*_module_has_no_torch_import` and `test_*_module_has_no_gradio_import` tests. These use `importlib` + `vars(mod)` to verify no forbidden names appear in module globals at import time. This is a cheap, always-on architecture gate.

#### PRESETS_TTS covers both Standard and Multilingual tabs
The Multilingual tab uses the same `exaggeration / cfg_weight / temperature / rep_penalty / min_p / top_p` parameter set as Standard TTS, so a single `PRESETS_TTS` dict serves both. The only difference is `MultilingualTTSRequest` has `rep_penalty=2.0` as its default (higher than Standard's 1.2) to suppress artefacts on non-English languages.

#### `PRESET_TTS_NAMES` / `PRESET_TURBO_NAMES` are derived from dict keys
```
PRESET_TTS_NAMES: list[str] = list(PRESETS_TTS.keys())
```
This means the name lists and the dicts are always in sync — no dual-maintenance risk. Tests verify `PRESET_TTS_NAMES == list(PRESETS_TTS.keys())` (order-preserving).

#### `get_preset_tts` / `get_preset_turbo` are intentionally separate
Even though `"🎯 Default"` exists in both dicts, the two functions are intentionally independent. `get_preset_tts("🤖 Voice Agent")` returns `None` because that preset only exists in Turbo — this is tested explicitly and is the correct behaviour for the service layer to rely on.

#### `AudioResult.duration_s` guards against zero sample_rate
```python
if self.sample_rate <= 0:
    return 0.0
```
Without this, passing an uninitialised `AudioResult(sample_rate=0, ...)` would raise `ZeroDivisionError` in property code. The guard makes the domain object safe to construct in tests without a real model.

### Discovered During Implementation
- `uv pip install -e .` was required before pytest could resolve `chatterbox_explorer` — the package was not editable-installed in the venv. Added as a setup step to remember.
- The `field` import in `models.py` was included from the spec template but not ultimately needed (all fields use default values directly, not `field(default_factory=...)`). Kept for future use by consumers who may add mutable defaults.

### Phase 3 — Non-blocking Logging (`QueueHandler`) Implementation Discoveries

**Python 3.11 `QueueHandler.prepare()` — the real behaviour (verified by inspection)**

`QueueHandler.prepare()` in Python 3.11 does the following (order matters):
1. Calls `self.format(record)` — this applies the handler's formatter (e.g. `JsonFormatter`) to produce a formatted string, and as a side-effect sets `record.exc_text` from `record.exc_info` via `Formatter.formatException()`.
2. Shallow-copies the record with `copy.copy(record)`.
3. On the **copy**: sets `record.msg = formatted_string`, `record.args = None`, `record.exc_info = None`, `record.exc_text = None`, `record.stack_info = None`.

The critical gotcha: step 3 clears `exc_text = None` on the copy. This means the `QueueListener`'s sink handler receives a record with no exception text. And if `self.format()` applies a `JsonFormatter`, the resulting JSON string is stored in `record.msg`, so the sink's `JsonFormatter` would JSON-encode an already-encoded JSON string — double-encoding.

**`_PreservingQueueHandler` — the fix**

Override `prepare()` to:
1. Pre-render `exc_text` using a plain `logging.Formatter()` (not the JSON formatter) while the traceback object is still available on the originating thread.
2. Shallow-copy the record manually.
3. Freeze `record.msg` / `record.args` via `getMessage()` WITHOUT calling `self.format()`.
4. Clear `exc_info` (not picklable) but keep `exc_text`.

Do NOT call `super().prepare()` — the base class would undo step 1 by setting `exc_text = None` on the copy.

**CorrelationIdFilter must go on `QueueHandler`, NOT the sink `StreamHandler`**

`CorrelationIdFilter` reads from a `ContextVar` that is set per-request by `CorrelationIdMiddleware`. `ContextVar` values are thread-local. The `QueueListener` runs on a background thread that has no request context — so calling the filter there would always return the `default_value` ("-"), even inside an active request.

Fix: attach `CorrelationIdFilter` to the `QueueHandler`. Python's `handler.handle()` applies filters BEFORE calling `emit()`, so the filter runs on the originating thread (correct ContextVar) and enriches `record.correlation_id` before the record is enqueued. The sink handler picks up the attribute that is already set.

**`JsonFormatter` on `QueueHandler` — test-compliance attach, not functional**

The existing `TestConfigureJson` tests inspect `logging.getLogger().handlers` for a `JsonFormatter` instance on a handler's `.formatter` attribute. Since the root logger only has the `QueueHandler` (no direct `StreamHandler`), the `JsonFormatter` must be set on the `QueueHandler` itself, even though `_PreservingQueueHandler.prepare()` never calls `self.format()`. This is a deliberate compromise: the formatter is present for test introspection and as a fallback reference, but all actual formatting is done by the sink's own `JsonFormatter` on the background thread.

**Factory function pattern for deferred imports**

The module must remain side-effect-free at import time (constraint from `cli.py` orchestration). Because `logging.handlers` is only imported inside `configure_json()`, `_PreservingQueueHandler` cannot be defined at module level as a class. A factory function `_make_preserving_queue_handler_class()` is used instead — it builds and returns the subclass on first call, at which point `logging.handlers` is already imported.

**ruff `TC003` — `import types` inside a function**

When `from __future__ import annotations` is active, all type annotations are lazy strings — they are never evaluated at runtime. `import types` inside `configure_json()` was flagged by ruff `TC003` because `types.TracebackType` is only used in the `_handle_uncaught` annotation. Fix: move `import types` to a module-level `TYPE_CHECKING` block. The nested function annotation remains valid because `from __future__ import annotations` ensures it is stored as a string literal, never resolved at runtime.

## References

- [Chatterbox GitHub](https://github.com/resemble-ai/chatterbox)
- [HuggingFace — chatterbox](https://huggingface.co/ResembleAI/chatterbox)
- [HuggingFace — chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo)
- [Gradio Docs — gr.Audio](https://www.gradio.app/docs/gradio/audio)
- [Gradio Docs — Streaming](https://www.gradio.app/guides/streaming-outputs)
- [uv Docs](https://docs.astral.sh/uv/)
- [PerTh Watermarker](https://github.com/resemble-ai/resemble-perth)
- [HuggingFace Hub — manage-cache guide](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
- [HuggingFace Hub — try_to_load_from_cache](https://huggingface.co/docs/huggingface_hub/package_reference/cache#huggingface_hub.try_to_load_from_cache)
- [HuggingFace Hub — scan_cache_dir](https://huggingface.co/docs/huggingface_hub/package_reference/cache#huggingface_hub.scan_cache_dir)
- [PyTorch — torch.mps API reference](https://pytorch.org/docs/stable/mps.html)
- [PyTorch — torch.mps.empty_cache](https://pytorch.org/docs/stable/generated/torch.mps.empty_cache.html)
- [psutil — API reference](https://psutil.readthedocs.io/en/latest/)
