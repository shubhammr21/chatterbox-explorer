"""
tests/unit/adapters/test_model_loader.py
==========================================
TDD unit tests for ChatterboxModelLoader and the MODEL_REGISTRY constant.

All external I/O is mocked — no real model downloads, GPU memory operations,
or network calls occur.  The chatterbox model classes, huggingface_hub helpers,
torch device APIs, and gc are all replaced with MagicMocks.

Mock strategy
─────────────
* chatterbox.tts / .tts_turbo / .mtl_tts / .vc   → patch.dict(sys.modules)
  These sub-modules are deferred-imported inside _load(); injecting fakes into
  sys.modules intercepts them before any real package code runs.

* huggingface_hub.try_to_load_from_cache           → patch() on the hf module
  huggingface_hub.hf_hub_download / snapshot_download  → same

* torch                                            → patch.dict(sys.modules)
  torch is imported lazily inside unload(); replacing sys.modules['torch']
  gives us a disposable mock for the MPS/CUDA flush calls.

* gc                                               → patch the name in the
  adapter's module namespace (gc is imported at module level there).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from adapters.secondary.model_loader import MODEL_REGISTRY, ChatterboxModelLoader

# ──────────────────────────────────────────────────────────────────────────────
# MODEL_REGISTRY constant
# ──────────────────────────────────────────────────────────────────────────────


class TestModelRegistry:
    """The MODEL_REGISTRY dict is the single source of truth for all models."""

    def test_has_exactly_four_keys(self):
        assert len(MODEL_REGISTRY) == 4

    def test_contains_tts_key(self):
        assert "tts" in MODEL_REGISTRY

    def test_contains_turbo_key(self):
        assert "turbo" in MODEL_REGISTRY

    def test_contains_multilingual_key(self):
        assert "multilingual" in MODEL_REGISTRY

    def test_contains_vc_key(self):
        assert "vc" in MODEL_REGISTRY

    def test_every_entry_has_display_name(self):
        for key, info in MODEL_REGISTRY.items():
            assert "display_name" in info, f"'display_name' missing for key {key!r}"

    def test_every_entry_has_repo_id(self):
        for key, info in MODEL_REGISTRY.items():
            assert "repo_id" in info, f"'repo_id' missing for key {key!r}"

    def test_every_entry_has_check_file(self):
        for key, info in MODEL_REGISTRY.items():
            assert "check_file" in info, f"'check_file' missing for key {key!r}"


# ──────────────────────────────────────────────────────────────────────────────
# get_all_keys
# ──────────────────────────────────────────────────────────────────────────────


class TestGetAllKeys:
    def test_returns_exactly_four_keys(self):
        loader = ChatterboxModelLoader(device="cpu")
        assert len(loader.get_all_keys()) == 4

    def test_contains_all_four_model_keys(self):
        loader = ChatterboxModelLoader(device="cpu")
        assert set(loader.get_all_keys()) == {"tts", "turbo", "multilingual", "vc"}

    def test_return_type_is_list(self):
        loader = ChatterboxModelLoader(device="cpu")
        assert isinstance(loader.get_all_keys(), list)


# ──────────────────────────────────────────────────────────────────────────────
# get_display_name
# ──────────────────────────────────────────────────────────────────────────────


class TestGetDisplayName:
    def test_tts_display_name_matches_registry(self):
        loader = ChatterboxModelLoader(device="cpu")
        assert loader.get_display_name("tts") == MODEL_REGISTRY["tts"]["display_name"]

    def test_turbo_display_name_matches_registry(self):
        loader = ChatterboxModelLoader(device="cpu")
        assert loader.get_display_name("turbo") == MODEL_REGISTRY["turbo"]["display_name"]

    def test_multilingual_display_name_matches_registry(self):
        loader = ChatterboxModelLoader(device="cpu")
        assert (
            loader.get_display_name("multilingual")
            == MODEL_REGISTRY["multilingual"]["display_name"]
        )

    def test_vc_display_name_matches_registry(self):
        loader = ChatterboxModelLoader(device="cpu")
        assert loader.get_display_name("vc") == MODEL_REGISTRY["vc"]["display_name"]

    def test_unknown_key_falls_back_to_key_string(self):
        loader = ChatterboxModelLoader(device="cpu")
        assert loader.get_display_name("does_not_exist") == "does_not_exist"


# ──────────────────────────────────────────────────────────────────────────────
# get_model_metadata
# ──────────────────────────────────────────────────────────────────────────────


class TestGetModelMetadata:
    _REQUIRED_FIELDS = ("size_gb", "params", "description", "class_name")

    def test_tts_metadata_contains_all_required_fields(self):
        loader = ChatterboxModelLoader(device="cpu")
        meta = loader.get_model_metadata("tts")
        for field in self._REQUIRED_FIELDS:
            assert field in meta, f"Field {field!r} missing from tts metadata"

    def test_tts_class_name(self):
        loader = ChatterboxModelLoader(device="cpu")
        assert loader.get_model_metadata("tts")["class_name"] == "ChatterboxTTS"

    def test_turbo_class_name(self):
        loader = ChatterboxModelLoader(device="cpu")
        assert loader.get_model_metadata("turbo")["class_name"] == "ChatterboxTurboTTS"

    def test_multilingual_class_name(self):
        loader = ChatterboxModelLoader(device="cpu")
        assert (
            loader.get_model_metadata("multilingual")["class_name"] == "ChatterboxMultilingualTTS"
        )

    def test_vc_class_name(self):
        loader = ChatterboxModelLoader(device="cpu")
        assert loader.get_model_metadata("vc")["class_name"] == "ChatterboxVC"

    def test_unknown_key_returns_safe_defaults(self):
        loader = ChatterboxModelLoader(device="cpu")
        meta = loader.get_model_metadata("nonexistent")
        assert meta["size_gb"] == 0.0
        assert meta["params"] == "—"
        assert meta["description"] == ""
        assert meta["class_name"] == ""


# ──────────────────────────────────────────────────────────────────────────────
# is_loaded
# ──────────────────────────────────────────────────────────────────────────────


class TestIsLoaded:
    def test_all_keys_false_on_fresh_loader(self):
        loader = ChatterboxModelLoader(device="cpu")
        for key in ("tts", "turbo", "multilingual", "vc"):
            assert loader.is_loaded(key) is False, f"Expected is_loaded({key!r}) to be False"

    def test_true_after_direct_cache_insertion(self):
        loader = ChatterboxModelLoader(device="cpu")
        loader._cache["tts"] = MagicMock()
        assert loader.is_loaded("tts") is True

    def test_only_the_cached_key_reports_loaded(self):
        loader = ChatterboxModelLoader(device="cpu")
        loader._cache["tts"] = MagicMock()
        assert loader.is_loaded("turbo") is False
        assert loader.is_loaded("multilingual") is False
        assert loader.is_loaded("vc") is False


# ──────────────────────────────────────────────────────────────────────────────
# is_cached_on_disk
# ──────────────────────────────────────────────────────────────────────────────


class TestIsCachedOnDisk:
    def test_returns_true_when_hf_returns_string_path(self):
        loader = ChatterboxModelLoader(device="cpu")
        with patch(
            "huggingface_hub.try_to_load_from_cache", return_value="/hf/cache/t3_cfg.safetensors"
        ):
            assert loader.is_cached_on_disk("tts") is True

    def test_returns_false_when_hf_returns_none(self):
        loader = ChatterboxModelLoader(device="cpu")
        with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
            assert loader.is_cached_on_disk("tts") is False

    def test_returns_false_when_hf_raises_exception(self):
        loader = ChatterboxModelLoader(device="cpu")
        with patch("huggingface_hub.try_to_load_from_cache", side_effect=OSError("disk error")):
            assert loader.is_cached_on_disk("tts") is False

    def test_returns_false_for_unknown_key_without_calling_hf(self):
        """Unknown keys must short-circuit before any HuggingFace call."""
        loader = ChatterboxModelLoader(device="cpu")
        with patch("huggingface_hub.try_to_load_from_cache") as mock_hf:
            result = loader.is_cached_on_disk("nonexistent")
        assert result is False
        mock_hf.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# get_model
# ──────────────────────────────────────────────────────────────────────────────


class TestGetModel:
    def test_returns_model_returned_by_load(self):
        loader = ChatterboxModelLoader(device="cpu")
        fake_model = MagicMock()
        with patch.object(loader, "_load", return_value=fake_model):
            result = loader.get_model("tts")
        assert result is fake_model

    def test_caches_model_so_load_called_only_once(self):
        """Second call must return from cache without calling _load again."""
        loader = ChatterboxModelLoader(device="cpu")
        with patch.object(loader, "_load", return_value=MagicMock()) as mock_load:
            loader.get_model("tts")
            loader.get_model("tts")
        mock_load.assert_called_once()

    def test_returns_cached_instance_without_reloading(self):
        loader = ChatterboxModelLoader(device="cpu")
        fake_model = MagicMock()
        loader._cache["tts"] = fake_model
        with patch.object(loader, "_load") as mock_load:
            result = loader.get_model("tts")
        mock_load.assert_not_called()
        assert result is fake_model

    def test_wraps_generic_exception_in_runtime_error(self):
        loader = ChatterboxModelLoader(device="cpu")
        with (
            patch.object(loader, "_load", side_effect=ValueError("weights missing")),
            pytest.raises(RuntimeError, match="Failed to load model 'tts'"),
        ):
            loader.get_model("tts")

    def test_propagates_runtime_error_from_load_unchanged(self):
        """RuntimeError (e.g. OOM) from _load must NOT be wrapped."""
        loader = ChatterboxModelLoader(device="cpu")
        with (
            patch.object(loader, "_load", side_effect=RuntimeError("CUDA out of memory")),
            pytest.raises(RuntimeError, match="CUDA out of memory"),
        ):
            loader.get_model("tts")

    def test_raises_runtime_error_for_unknown_key(self):
        loader = ChatterboxModelLoader(device="cpu")
        with pytest.raises(RuntimeError, match="Unknown model key"):
            loader.get_model("definitely_not_a_real_key")


# ──────────────────────────────────────────────────────────────────────────────
# unload
# ──────────────────────────────────────────────────────────────────────────────


class TestUnload:
    """The 5-step MPS-confirmed unload recipe must be exercised correctly."""

    @pytest.fixture
    def cpu_torch_mock(self):
        """A torch mock with MPS and CUDA reported as unavailable (CPU test)."""
        t = MagicMock()
        t.backends.mps.is_available.return_value = False
        t.cuda.is_available.return_value = False
        return t

    def test_noop_when_model_not_in_cache(self, cpu_torch_mock):
        """Unloading a key that was never loaded must not raise."""
        loader = ChatterboxModelLoader(device="cpu")
        with patch.dict(sys.modules, {"torch": cpu_torch_mock}):
            loader.unload("tts")  # must not raise
        # Device flush methods must not be called on a no-op unload
        cpu_torch_mock.mps.empty_cache.assert_not_called()
        cpu_torch_mock.cuda.empty_cache.assert_not_called()

    def test_removes_key_from_cache(self, cpu_torch_mock):
        loader = ChatterboxModelLoader(device="cpu")
        loader._cache["tts"] = MagicMock()
        with patch.dict(sys.modules, {"torch": cpu_torch_mock}):
            loader.unload("tts")
        assert "tts" not in loader._cache

    def test_calls_model_cpu_before_eviction(self, cpu_torch_mock):
        """Step 1 of the recipe: evict tensors from the accelerator first."""
        loader = ChatterboxModelLoader(device="cpu")
        fake_model = MagicMock()
        loader._cache["tts"] = fake_model
        with patch.dict(sys.modules, {"torch": cpu_torch_mock}):
            loader.unload("tts")
        fake_model.cpu.assert_called_once()

    def test_calls_gc_collect(self, cpu_torch_mock):
        """Step 3: gc.collect() breaks nn.Module reference cycles."""
        loader = ChatterboxModelLoader(device="cpu")
        loader._cache["tts"] = MagicMock()
        with (
            patch.dict(sys.modules, {"torch": cpu_torch_mock}),
            patch("adapters.secondary.model_loader.gc") as mock_gc,
        ):
            loader.unload("tts")
        mock_gc.collect.assert_called_once()

    def test_cpu_device_does_not_touch_mps_or_cuda(self, cpu_torch_mock):
        """On CPU the device-flush steps (4 & 5) must be skipped entirely."""
        loader = ChatterboxModelLoader(device="cpu")
        loader._cache["tts"] = MagicMock()
        with patch.dict(sys.modules, {"torch": cpu_torch_mock}):
            loader.unload("tts")
        cpu_torch_mock.mps.synchronize.assert_not_called()
        cpu_torch_mock.mps.empty_cache.assert_not_called()
        cpu_torch_mock.cuda.synchronize.assert_not_called()
        cpu_torch_mock.cuda.empty_cache.assert_not_called()

    def test_mps_device_synchronizes_and_empties_cache(self):
        """Steps 4 & 5 for MPS: synchronize then empty_cache."""
        loader = ChatterboxModelLoader(device="mps")
        loader._cache["tts"] = MagicMock()
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            loader.unload("tts")
        mock_torch.mps.synchronize.assert_called_once()
        mock_torch.mps.empty_cache.assert_called_once()

    def test_cpu_continues_if_model_cpu_raises(self, cpu_torch_mock):
        """model.cpu() failures must not abort the rest of the unload recipe."""
        loader = ChatterboxModelLoader(device="cpu")
        bad_model = MagicMock()
        bad_model.cpu.side_effect = RuntimeError("partially initialised model")
        loader._cache["tts"] = bad_model
        # Must not raise even though model.cpu() fails.
        with patch.dict(sys.modules, {"torch": cpu_torch_mock}):
            loader.unload("tts")
        assert "tts" not in loader._cache

    def test_unload_cuda_flushes_cuda_cache(self):
        """Steps 4 & 5 for CUDA: synchronize then empty_cache must be called
        when device='cuda' and torch.cuda.is_available() is True."""
        loader = ChatterboxModelLoader(device="cuda")
        loader._cache["tts"] = MagicMock()

        cuda_torch_mock = MagicMock()
        cuda_torch_mock.cuda.is_available.return_value = True
        cuda_torch_mock.backends.mps.is_available.return_value = False

        with (
            patch.dict(sys.modules, {"torch": cuda_torch_mock}),
            patch("adapters.secondary.model_loader.gc"),
        ):
            loader.unload("tts")

        cuda_torch_mock.cuda.synchronize.assert_called_once()
        cuda_torch_mock.cuda.empty_cache.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# download
# ──────────────────────────────────────────────────────────────────────────────


class TestDownload:
    def test_unknown_key_yields_single_error_line(self):
        loader = ChatterboxModelLoader(device="cpu")
        results = list(loader.download("not_a_real_key"))
        assert len(results) == 1
        assert "Unknown model key" in results[0]

    def test_files_mode_starts_with_starting_download_message(self):
        """tts uses dl_mode='files' — first line describes the download."""
        loader = ChatterboxModelLoader(device="cpu")
        with patch("huggingface_hub.hf_hub_download"):
            results = list(loader.download("tts"))
        assert any("Starting download" in r for r in results)

    def test_files_mode_yields_one_progress_line_per_file(self):
        """Each file in dl_files must produce exactly one 'fetching …' line."""
        loader = ChatterboxModelLoader(device="cpu")
        expected = len(MODEL_REGISTRY["tts"]["dl_files"])
        with patch("huggingface_hub.hf_hub_download"):
            results = list(loader.download("tts"))
        fetching_lines = [r for r in results if "fetching" in r]
        assert len(fetching_lines) == expected

    def test_files_mode_calls_hf_hub_download_for_each_file(self):
        loader = ChatterboxModelLoader(device="cpu")
        expected = len(MODEL_REGISTRY["tts"]["dl_files"])
        with patch("huggingface_hub.hf_hub_download") as mock_dl:
            list(loader.download("tts"))
        assert mock_dl.call_count == expected

    def test_files_mode_ends_with_completion_message(self):
        loader = ChatterboxModelLoader(device="cpu")
        with patch("huggingface_hub.hf_hub_download"):
            results = list(loader.download("tts"))
        assert any("download complete" in r.lower() for r in results)

    def test_snapshot_mode_starts_with_starting_download_message(self):
        """turbo uses dl_mode='snapshot'."""
        loader = ChatterboxModelLoader(device="cpu")
        with patch("huggingface_hub.snapshot_download"):
            results = list(loader.download("turbo"))
        assert any("Starting download" in r for r in results)

    def test_snapshot_mode_calls_snapshot_download_once(self):
        loader = ChatterboxModelLoader(device="cpu")
        with patch("huggingface_hub.snapshot_download") as mock_snap:
            list(loader.download("turbo"))
        mock_snap.assert_called_once()

    def test_snapshot_mode_does_not_call_hf_hub_download(self):
        loader = ChatterboxModelLoader(device="cpu")
        with (
            patch("huggingface_hub.hf_hub_download") as mock_file_dl,
            patch("huggingface_hub.snapshot_download"),
        ):
            list(loader.download("turbo"))
        mock_file_dl.assert_not_called()

    def test_yields_download_failed_message_on_exception(self):
        loader = ChatterboxModelLoader(device="cpu")
        with patch("huggingface_hub.hf_hub_download", side_effect=OSError("network down")):
            results = list(loader.download("tts"))
        assert any("Download failed" in r for r in results)

    def test_exception_message_contains_original_error(self):
        loader = ChatterboxModelLoader(device="cpu")
        with patch("huggingface_hub.hf_hub_download", side_effect=OSError("timeout after 30s")):
            results = list(loader.download("tts"))
        error_lines = [r for r in results if "Download failed" in r]
        assert len(error_lines) == 1
        assert "timeout after 30s" in error_lines[0]

    def test_download_unknown_dl_mode_yields_error(self):
        """When a registry entry carries an unrecognised dl_mode, download()
        must yield a message containing 'Unknown dl_mode' and then stop —
        covering the else branch at the bottom of the dl_mode dispatch."""
        loader = ChatterboxModelLoader(device="cpu")

        # Build a patched registry with dl_mode set to an unknown value.
        patched_registry = {**MODEL_REGISTRY}
        patched_registry["tts"] = {**MODEL_REGISTRY["tts"], "dl_mode": "unknown"}

        with patch("adapters.secondary.model_loader.MODEL_REGISTRY", patched_registry):
            messages = list(loader.download("tts"))

        assert any("Unknown dl_mode" in m for m in messages), (
            f"Expected a message containing 'Unknown dl_mode'; got: {messages}"
        )
        # The generator must return after the error line — no completion message.
        assert not any("download complete" in m.lower() for m in messages)


# ──────────────────────────────────────────────────────────────────────────────
# _load — dispatch to correct Chatterbox from_pretrained factory
# ──────────────────────────────────────────────────────────────────────────────


class TestLoad:
    """_load() must dispatch each key to the correct Chatterbox class.

    We inject fake chatterbox sub-modules into sys.modules so that the
    deferred ``from chatterbox.tts import ChatterboxTTS`` (etc.) calls inside
    _load() receive our controlled mocks rather than the real library.
    """

    @pytest.fixture
    def chatterbox_modules(self):
        """Fake chatterbox sub-modules wired up so from_pretrained returns a sentinel."""
        fake_model = MagicMock(name="fake_model")
        fake_model.sr = 24_000

        tts_mod = MagicMock()
        tts_mod.ChatterboxTTS.from_pretrained.return_value = fake_model

        turbo_mod = MagicMock()
        turbo_mod.ChatterboxTurboTTS.from_pretrained.return_value = fake_model

        mtl_mod = MagicMock()
        mtl_mod.ChatterboxMultilingualTTS.from_pretrained.return_value = fake_model

        vc_mod = MagicMock()
        vc_mod.ChatterboxVC.from_pretrained.return_value = fake_model

        patches = {
            "chatterbox.tts": tts_mod,
            "chatterbox.tts_turbo": turbo_mod,
            "chatterbox.mtl_tts": mtl_mod,
            "chatterbox.vc": vc_mod,
        }
        with patch.dict(sys.modules, patches):
            yield {
                "model": fake_model,
                "tts_mod": tts_mod,
                "turbo_mod": turbo_mod,
                "mtl_mod": mtl_mod,
                "vc_mod": vc_mod,
            }

    def test_tts_dispatches_to_chatterbox_tts(self, chatterbox_modules):
        loader = ChatterboxModelLoader(device="cpu")
        result = loader._load("tts")
        assert result is chatterbox_modules["model"]
        chatterbox_modules["tts_mod"].ChatterboxTTS.from_pretrained.assert_called_once_with("cpu")

    def test_turbo_dispatches_to_chatterbox_turbo_tts(self, chatterbox_modules):
        loader = ChatterboxModelLoader(device="cpu")
        result = loader._load("turbo")
        assert result is chatterbox_modules["model"]
        chatterbox_modules["turbo_mod"].ChatterboxTurboTTS.from_pretrained.assert_called_once_with(
            "cpu"
        )

    def test_multilingual_dispatches_to_chatterbox_multilingual_tts(self, chatterbox_modules):
        loader = ChatterboxModelLoader(device="cpu")
        result = loader._load("multilingual")
        assert result is chatterbox_modules["model"]
        chatterbox_modules[
            "mtl_mod"
        ].ChatterboxMultilingualTTS.from_pretrained.assert_called_once_with("cpu")

    def test_vc_dispatches_to_chatterbox_vc(self, chatterbox_modules):
        loader = ChatterboxModelLoader(device="cpu")
        result = loader._load("vc")
        assert result is chatterbox_modules["model"]
        chatterbox_modules["vc_mod"].ChatterboxVC.from_pretrained.assert_called_once_with("cpu")

    def test_device_is_forwarded_to_from_pretrained(self, chatterbox_modules):
        """The device string passed to ChatterboxModelLoader must reach from_pretrained."""
        loader = ChatterboxModelLoader(device="mps")
        loader._load("tts")
        chatterbox_modules["tts_mod"].ChatterboxTTS.from_pretrained.assert_called_once_with("mps")

    def test_unknown_key_raises_runtime_error(self):
        loader = ChatterboxModelLoader(device="cpu")
        with pytest.raises(RuntimeError, match="No loader defined for model key"):
            loader._load("unknown_model_key")
