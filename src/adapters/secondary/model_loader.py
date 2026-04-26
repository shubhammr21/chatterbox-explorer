"""
src/adapters/secondary/model_loader.py  (updated: ModelKey + ModelMetadata types)
============================================================
Secondary adapter: wraps lazy model loading, HuggingFace downloads,
disk-cache probing, and memory unloading behind the IModelRepository port.

Architecture contract:
    - NO Gradio, NO primary-adapter concerns (HTML rendering is NOT here).
    - download() yields plain strings — HTML markup is the primary adapter's job.
    - unload() follows the strict 5-step MPS recipe to fully flush device memory.
    - get_model() always raises RuntimeError on failure (never returns None).
"""

from __future__ import annotations

import gc
import logging
import os
from typing import TYPE_CHECKING, Any

from ports.output import IModelRepository

if TYPE_CHECKING:
    from collections.abc import Iterator

    from domain.types import ModelKey, ModelMetadata

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Model registry — single source of truth for all model metadata
# ──────────────────────────────────────────────────────────────────────────────
# Each entry describes:
#   display_name  — human-readable label shown in the UI
#   class_name    — Python class instantiated by from_pretrained()
#   description   — one-line capability summary
#   params        — parameter count label (informational)
#   size_gb       — approximate download size (informational)
#   repo_id       — HuggingFace Hub repository identifier
#   check_file    — primary weight file used to probe disk-cache presence
#   dl_mode       — "files" (individual hf_hub_download) or
#                   "snapshot" (snapshot_download with allow_patterns)
#   dl_files      — list of filenames (dl_mode == "files" only)
#   dl_patterns   — glob patterns   (dl_mode == "snapshot" only)

MODEL_REGISTRY: dict[ModelKey, ModelMetadata] = {
    "tts": {
        "display_name": "Standard TTS",
        "class_name": "ChatterboxTTS",
        "description": "English · zero-shot voice cloning · exaggeration & CFG controls",
        "params": "500M",
        "size_gb": 1.4,
        "repo_id": "ResembleAI/chatterbox",
        "check_file": "t3_cfg.safetensors",
        "dl_mode": "files",
        "dl_files": [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ],
    },
    "turbo": {
        "display_name": "Turbo TTS",
        "class_name": "ChatterboxTurboTTS",
        "description": "English · 1-step decoder · paralinguistic tags · low VRAM",
        "params": "350M",
        "size_gb": 2.9,
        "repo_id": "ResembleAI/chatterbox-turbo",
        "check_file": "t3_turbo_v1.safetensors",
        "dl_mode": "snapshot",
        "dl_patterns": ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
    },
    "multilingual": {
        "display_name": "Multilingual TTS",
        "class_name": "ChatterboxMultilingualTTS",
        "description": "23 languages · cross-language voice cloning",
        "params": "500M",
        "size_gb": 1.5,
        "repo_id": "ResembleAI/chatterbox",
        "check_file": "t3_mtl23ls_v2.safetensors",
        "dl_mode": "snapshot",
        "dl_patterns": [
            "ve.pt",
            "t3_mtl23ls_v2.safetensors",
            "s3gen.pt",
            "grapheme_mtl_merged_expanded_v1.json",
            "conds.pt",
            "Cangjie5_TC.json",
        ],
    },
    "vc": {
        "display_name": "Voice Conversion",
        "class_name": "ChatterboxVC",
        "description": "Audio-to-audio · no text needed · voice identity swap",
        "params": "—",
        "size_gb": 0.4,
        "repo_id": "ResembleAI/chatterbox",
        "check_file": "s3gen.safetensors",
        "dl_mode": "files",
        "dl_files": ["s3gen.safetensors", "conds.pt"],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Adapter implementation
# ──────────────────────────────────────────────────────────────────────────────


class ChatterboxModelLoader(IModelRepository):
    """Concrete IModelRepository backed by the Chatterbox model family.

    Responsibilities:
        - Lazy-loads models on first access and keeps them in an instance-level
          cache (``self._cache``) so subsequent calls are O(1).
        - Probes the HuggingFace Hub disk cache without network calls to report
          whether weights are already downloaded.
        - Unloads models from memory using the confirmed 5-step MPS recipe that
          fully flushes the device allocator pool (not just the Python reference).
        - Streams download progress as plain text lines — no HTML markup; that
          concern belongs to the primary (Gradio) adapter.

    Args:
        device: Compute device string — one of ``"cpu"``, ``"cuda"``, ``"mps"``.
                Passed to every ``from_pretrained()`` call.
    """

    def __init__(self, device: str) -> None:
        self._device = device
        self._cache: dict[ModelKey, Any] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # IModelRepository — load / unload
    # ──────────────────────────────────────────────────────────────────────────

    def get_model(self, key: ModelKey) -> Any:
        """Return the live model object for *key*, loading it if necessary.

        Models are loaded lazily on first access.  Subsequent calls return the
        cached instance immediately without any I/O or network activity.

        Chatterbox import mapping:
            "tts"          → chatterbox.tts.ChatterboxTTS
            "turbo"        → chatterbox.tts_turbo.ChatterboxTurboTTS
            "multilingual" → chatterbox.mtl_tts.ChatterboxMultilingualTTS
            "vc"           → chatterbox.vc.ChatterboxVC

        Args:
            key: One of the keys in MODEL_REGISTRY.

        Returns:
            The loaded model object (framework-specific; callers cast as needed).

        Raises:
            RuntimeError: if *key* is unknown, or if model loading fails for
                          any reason (OOM, missing weights, device error, …).
        """
        if key in self._cache:
            return self._cache[key]

        if key not in MODEL_REGISTRY:
            raise RuntimeError(
                f"Unknown model key {key!r}. Valid keys: {list(MODEL_REGISTRY.keys())}"
            )

        log.info(
            "Loading model '%s' (%s) on %s — may download from HuggingFace on first run …",
            key,
            MODEL_REGISTRY[key]["display_name"],
            self._device.upper(),
        )

        try:
            model = self._load(key)
        except RuntimeError:
            # Propagate RuntimeError (OOM, device errors) unchanged so callers
            # such as ModelManagerService can re-raise them with context.
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{key}' ({MODEL_REGISTRY[key]['display_name']}): {exc}"
            ) from exc

        self._cache[key] = model
        log.info("Model '%s' ready ✓", key)
        return model

    def is_loaded(self, key: ModelKey) -> bool:
        """Return True if the model for *key* is currently held in memory."""
        return key in self._cache

    def is_cached_on_disk(self, key: ModelKey) -> bool:
        """Return True if the primary weight file for *key* exists in the HF cache.

        Uses ``huggingface_hub.try_to_load_from_cache()`` which is a pure
        disk-lookup — zero network calls, zero model loading.

        Return value semantics of try_to_load_from_cache:
            str                  → file is present (return True)
            None                 → file was never fetched (unknown)
            _CACHED_NO_EXIST     → a prior download returned HTTP 404 (rare)
        """
        if key not in MODEL_REGISTRY:
            return False
        try:
            from huggingface_hub import try_to_load_from_cache

            info = MODEL_REGISTRY[key]
            result = try_to_load_from_cache(
                repo_id=info["repo_id"],
                filename=info["check_file"],
                repo_type="model",
            )
            # Only a string path means the file is genuinely on disk.
            return isinstance(result, str)
        except Exception:
            return False

    def unload(self, key: ModelKey) -> None:
        """Remove the model for *key* from memory and flush device memory.

        Five-step MPS-confirmed recipe:
            1. model.cpu()          — move all tensors off the accelerator
                                      BEFORE dropping the Python reference;
                                      skipping this leaves the MPS allocator
                                      pool full even after gc.collect().
            2. del self._cache[key] — drop the last Python reference so the
                                      nn.Module becomes eligible for GC.
            3. gc.collect()         — break reference cycles inside nn.Module
                                      (PyTorch modules have circular refs via
                                      _parameters / _modules dicts).
            4. synchronize()        — wait for all pending Metal / CUDA kernels
                                      to finish; required before empty_cache()
                                      or the driver may still report the memory
                                      as in-use.
            5. empty_cache()        — flush the allocator pool back to the
                                      driver.  Without this step on MPS, the
                                      pool stays reserved even though Python
                                      no longer holds any tensor references.

        No-op if the model is not currently loaded.
        """
        if key not in self._cache:
            log.debug("unload('%s'): not in cache — no-op", key)
            return

        import torch

        model = self._cache.pop(key)

        # Step 1 — evict tensors from device memory before dropping the ref.
        try:
            model.cpu()
        except Exception as exc:
            # cpu() may fail on partially-initialised models; proceed anyway.
            log.debug("unload('%s'): model.cpu() raised %s — continuing", key, exc)

        # Steps 2 + 3 — drop reference and break nn.Module cycles.
        del model
        gc.collect()

        # Steps 4 + 5 — flush the device allocator pool.
        if self._device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        log.info("Model '%s' unloaded and device memory flushed.", key)

    # ──────────────────────────────────────────────────────────────────────────
    # IModelRepository — download
    # ──────────────────────────────────────────────────────────────────────────

    def download(self, key: ModelKey) -> Iterator[str]:
        """Download weights for *key* to the HF disk cache.

        Replicates the exact ``hf_hub_download`` / ``snapshot_download`` calls
        that each model's ``from_pretrained()`` makes internally, so that a
        subsequent ``get_model()`` requires no additional network access.

        Yields plain text progress lines — no HTML markup.  The primary
        adapter (Gradio UI) is responsible for rendering these strings.

        Args:
            key: One of the keys in MODEL_REGISTRY.

        Yields:
            Human-readable progress / status strings.
        """
        if key not in MODEL_REGISTRY:
            yield f"Unknown model key: {key!r}"
            return

        from huggingface_hub import hf_hub_download, snapshot_download

        info = MODEL_REGISTRY[key]

        yield (
            f"Starting download: {info['display_name']} "
            f"(~{info['size_gb']:.1f} GB) — "
            "saving to ~/.cache/huggingface/hub/"
        )

        try:
            if info["dl_mode"] == "files":
                files: list[str] = info["dl_files"]
                for i, fname in enumerate(files, 1):
                    yield f"  [{i}/{len(files)}] fetching {fname} …"
                    hf_hub_download(
                        repo_id=info["repo_id"],
                        filename=fname,
                        token=os.environ.get("HF_TOKEN"),
                    )

            elif info["dl_mode"] == "snapshot":
                yield "  Fetching snapshot (all model files) …"
                snapshot_download(
                    repo_id=info["repo_id"],
                    allow_patterns=info["dl_patterns"],
                    token=os.environ.get("HF_TOKEN"),
                )

            else:
                yield f"  Unknown dl_mode {info['dl_mode']!r} — skipping"
                return

            yield (f"{info['display_name']} download complete. Click Load to bring it into memory.")

        except Exception as exc:
            yield f"Download failed: {exc}"

    # ──────────────────────────────────────────────────────────────────────────
    # IModelRepository — metadata
    # ──────────────────────────────────────────────────────────────────────────

    def get_all_keys(self) -> list[ModelKey]:
        """Return every model key known to this repository, in registry order."""
        return list(MODEL_REGISTRY.keys())

    def get_display_name(self, key: ModelKey) -> str:
        """Return the human-readable label for *key* (e.g. ``'Standard TTS'``).

        Falls back to the key itself if not found in the registry.
        """
        return MODEL_REGISTRY.get(key, {}).get("display_name", key)

    def get_model_metadata(self, key: ModelKey) -> ModelMetadata:
        """Return a metadata dict for *key* with the following keys:

        ``size_gb``     (float)  — weight file size on disk
        ``params``      (str)    — parameter count label, e.g. ``'500M'``
        ``description`` (str)    — one-line capability description
        ``class_name``  (str)    — Python class name, e.g. ``'ChatterboxTTS'``

        Returns an empty dict if *key* is not in the registry.
        """
        info = MODEL_REGISTRY.get(key, {})
        return {
            "size_gb": info.get("size_gb", 0.0),
            "params": info.get("params", "—"),
            "description": info.get("description", ""),
            "class_name": info.get("class_name", ""),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load(self, key: ModelKey) -> Any:
        """Dispatch to the correct Chatterbox from_pretrained() factory.

        All chatterbox imports are deferred to this method so that importing
        this module does NOT trigger any chatterbox code (and therefore does
        NOT require the compat patches to have fired yet).  The patches are
        guaranteed to run before cli.main() calls build_app() → get_model().

        Args:
            key: A validated key present in MODEL_REGISTRY.

        Returns:
            The loaded model object.

        Raises:
            RuntimeError: for unknown keys (should not happen after the guard
                          in get_model(), but included for safety).
        """
        if key == "tts":
            from chatterbox.tts import ChatterboxTTS

            return ChatterboxTTS.from_pretrained(self._device)

        if key == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            return ChatterboxTurboTTS.from_pretrained(self._device)

        if key == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            return ChatterboxMultilingualTTS.from_pretrained(self._device)

        if key == "vc":
            from chatterbox.vc import ChatterboxVC

            return ChatterboxVC.from_pretrained(self._device)

        raise RuntimeError(f"No loader defined for model key {key!r}")
