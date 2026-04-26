"""
tests/unit/services/test_model_manager_service.py
===================================================
Unit tests for ModelManagerService.

Covers:
    - load()   : success path and already-loaded guard
    - unload() : success path and not-loaded guard
    - download(): delegation to repo.download()
    - get_all_status(): aggregates repo metadata into ModelStatus list
    - get_memory_stats(): delegates to IMemoryMonitor

No real models are loaded; all infra is faked via conftest.py fixtures.
"""
from __future__ import annotations

import pytest

from chatterbox_explorer.domain.models import MemoryStats, ModelStatus
from chatterbox_explorer.services.model_manager import ModelManagerService


# ──────────────────────────────────────────────────────────────────────────────
# load()
# ──────────────────────────────────────────────────────────────────────────────

class TestModelManagerLoad:
    def test_load_returns_loaded_message(self, mock_model_repo, mock_memory_monitor):
        """When model is not in memory, load() calls get_model and returns a
        success message containing the word 'loaded'."""
        mock_model_repo.is_loaded.return_value = False

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        result = svc.load("tts")

        mock_model_repo.get_model.assert_called_once_with("tts")
        assert "loaded" in result.lower()

    def test_load_already_loaded_returns_info_message(
        self, mock_model_repo, mock_memory_monitor
    ):
        """When the model is already in memory, load() must NOT call get_model
        and must return a message containing the word 'already'."""
        mock_model_repo.is_loaded.return_value = True

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        result = svc.load("tts")

        mock_model_repo.get_model.assert_not_called()
        assert "already" in result.lower()

    def test_load_includes_display_name_in_message(
        self, mock_model_repo, mock_memory_monitor
    ):
        """The returned message should mention the model's display name."""
        mock_model_repo.is_loaded.return_value = False

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        result = svc.load("tts")

        # mock get_display_name returns k.upper() → "TTS"
        assert "TTS" in result

    def test_load_already_loaded_includes_display_name(
        self, mock_model_repo, mock_memory_monitor
    ):
        mock_model_repo.is_loaded.return_value = True

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        result = svc.load("turbo")

        assert "TURBO" in result

    def test_load_raises_runtime_error_on_repo_failure(
        self, mock_model_repo, mock_memory_monitor
    ):
        """If get_model raises, load() must re-raise as RuntimeError."""
        mock_model_repo.is_loaded.return_value = False
        mock_model_repo.get_model.side_effect = RuntimeError("weights missing")

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        with pytest.raises(RuntimeError):
            svc.load("tts")


# ──────────────────────────────────────────────────────────────────────────────
# unload()
# ──────────────────────────────────────────────────────────────────────────────

class TestModelManagerUnload:
    def test_unload_returns_unloaded_message(
        self, mock_model_repo, mock_memory_monitor
    ):
        """When the model is in memory, unload() calls repo.unload and returns
        a message containing the word 'unloaded'."""
        mock_model_repo.is_loaded.return_value = True

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        result = svc.unload("tts")

        mock_model_repo.unload.assert_called_once_with("tts")
        assert "unloaded" in result.lower()

    def test_unload_not_loaded_returns_info_message(
        self, mock_model_repo, mock_memory_monitor
    ):
        """When the model is not in memory, unload() must NOT call repo.unload
        and must return a message indicating it was not loaded."""
        mock_model_repo.is_loaded.return_value = False

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        result = svc.unload("tts")

        mock_model_repo.unload.assert_not_called()
        assert "not" in result.lower()

    def test_unload_includes_display_name_in_message(
        self, mock_model_repo, mock_memory_monitor
    ):
        mock_model_repo.is_loaded.return_value = True

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        result = svc.unload("tts")

        assert "TTS" in result

    def test_unload_not_loaded_includes_display_name(
        self, mock_model_repo, mock_memory_monitor
    ):
        mock_model_repo.is_loaded.return_value = False

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        result = svc.unload("multilingual")

        assert "MULTILINGUAL" in result


# ──────────────────────────────────────────────────────────────────────────────
# download()
# ──────────────────────────────────────────────────────────────────────────────

class TestModelManagerDownload:
    def test_download_yields_progress_lines(
        self, mock_model_repo, mock_memory_monitor
    ):
        """download() must delegate to repo.download() and pass every yielded
        line through unchanged."""
        expected_lines = ["Downloading... 50%", "Downloading... 100%", "Done."]
        mock_model_repo.download.return_value = iter(expected_lines)

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        result = list(svc.download("tts"))

        mock_model_repo.download.assert_called_once_with("tts")
        assert result == expected_lines

    def test_download_empty_stream(self, mock_model_repo, mock_memory_monitor):
        """If repo yields nothing, download() should yield nothing."""
        mock_model_repo.download.return_value = iter([])

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        result = list(svc.download("tts"))

        assert result == []

    def test_download_forwards_key_to_repo(
        self, mock_model_repo, mock_memory_monitor
    ):
        mock_model_repo.download.return_value = iter(["line"])

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        list(svc.download("turbo"))

        mock_model_repo.download.assert_called_once_with("turbo")


# ──────────────────────────────────────────────────────────────────────────────
# get_all_status()
# ──────────────────────────────────────────────────────────────────────────────

class TestModelManagerGetAllStatus:
    def test_get_all_status_returns_model_status_list(
        self, mock_model_repo, mock_memory_monitor
    ):
        """get_all_status() must return one ModelStatus per key."""
        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        statuses = svc.get_all_status()

        # conftest mock returns ["tts", "turbo", "multilingual", "vc"]
        assert len(statuses) == 4
        assert all(isinstance(s, ModelStatus) for s in statuses)

    def test_get_all_status_correct_keys(
        self, mock_model_repo, mock_memory_monitor
    ):
        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        statuses = svc.get_all_status()
        keys = [s.key for s in statuses]
        assert keys == ["tts", "turbo", "multilingual", "vc"]

    def test_get_all_status_display_names(
        self, mock_model_repo, mock_memory_monitor
    ):
        """display_name comes from repo.get_display_name (k.upper() in mock)."""
        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        statuses = svc.get_all_status()

        assert statuses[0].display_name == "TTS"
        assert statuses[1].display_name == "TURBO"
        assert statuses[2].display_name == "MULTILINGUAL"
        assert statuses[3].display_name == "VC"

    def test_get_all_status_metadata_fields(
        self, mock_model_repo, mock_memory_monitor
    ):
        """Metadata dict fields must be mapped to ModelStatus fields."""
        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        status = svc.get_all_status()[0]

        assert status.size_gb == 1.0
        assert status.params == "500M"
        assert status.description == "test model"
        assert status.class_name == "MockModel"

    def test_get_all_status_in_memory_flag(
        self, mock_model_repo, mock_memory_monitor
    ):
        """in_memory must reflect repo.is_loaded()."""
        mock_model_repo.is_loaded.return_value = True

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        statuses = svc.get_all_status()

        assert all(s.in_memory is True for s in statuses)

    def test_get_all_status_on_disk_flag(
        self, mock_model_repo, mock_memory_monitor
    ):
        """on_disk must reflect repo.is_cached_on_disk()."""
        mock_model_repo.is_cached_on_disk.return_value = True

        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        statuses = svc.get_all_status()

        assert all(s.on_disk is True for s in statuses)

    def test_get_all_status_not_loaded_by_default(
        self, mock_model_repo, mock_memory_monitor
    ):
        """Default conftest mock has is_loaded=False and is_cached_on_disk=False."""
        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        statuses = svc.get_all_status()

        assert all(s.in_memory is False for s in statuses)
        assert all(s.on_disk is False for s in statuses)


# ──────────────────────────────────────────────────────────────────────────────
# get_memory_stats()
# ──────────────────────────────────────────────────────────────────────────────

class TestModelManagerGetMemoryStats:
    def test_get_memory_stats_delegates_to_monitor(
        self, mock_model_repo, mock_memory_monitor
    ):
        """get_memory_stats() must call monitor.get_stats() exactly once."""
        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        svc.get_memory_stats()

        mock_memory_monitor.get_stats.assert_called_once()

    def test_get_memory_stats_returns_memory_stats_instance(
        self, mock_model_repo, mock_memory_monitor
    ):
        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        stats = svc.get_memory_stats()

        assert isinstance(stats, MemoryStats)

    def test_get_memory_stats_correct_values(
        self, mock_model_repo, mock_memory_monitor
    ):
        """Values must match what the mock monitor returns."""
        svc = ModelManagerService(mock_model_repo, mock_memory_monitor)
        stats = svc.get_memory_stats()

        assert stats.sys_total_gb == 16.0
        assert stats.sys_used_gb == 8.0
        assert stats.sys_avail_gb == 8.0
        assert stats.sys_percent == 50.0
        assert stats.proc_rss_gb == 2.0
