"""
tests/conftest.py
==================
Shared pytest fixtures for all unit and integration tests.

Fixture dependency graph:
    mock_wav_tensor  (requires torch — skips if absent)
        └─ mock_model
            └─ mock_model_repo
    mock_preprocessor          (no torch dependency)
    mock_memory_monitor        (no torch dependency)
    mock_watermark_detector    (no torch dependency)
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from domain.types import ALL_MODEL_KEYS

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ports.output import (
    IAudioPreprocessor,
    IMemoryMonitor,
    IModelRepository,
    IWatermarkDetector,
)
from domain.models import MemoryStats


# ──────────────────────────────────────────────────────────────────────────────
# Low-level tensor / model mocks (require torch)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_wav_tensor():
    """Returns a fake wav tensor shaped (1, 24000) — 1 second at 24 kHz.

    Skips the test automatically when torch is not installed.
    """
    if not HAS_TORCH:
        pytest.skip("torch not available")
    import torch
    return torch.zeros(1, 24000)


@pytest.fixture
def mock_model(mock_wav_tensor):
    """A MagicMock that mimics a Chatterbox model instance.

    Attributes:
        sr:       sample rate (24 000)
        generate: returns mock_wav_tensor regardless of arguments
    """
    m = MagicMock()
    m.sr = 24_000
    m.generate.return_value = mock_wav_tensor
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Repository / infrastructure mocks
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model_repo(mock_model):
    """A spec'd MagicMock for IModelRepository.

    Defaults:
        get_model(any)           → mock_model
        is_loaded(any)           → False
        is_cached_on_disk(any)   → False
        get_all_keys()           → ["tts", "turbo", "multilingual", "vc"]
        get_display_name(k)      → k.upper()   e.g. "tts" → "TTS"
        get_model_metadata(any)  → {size_gb, params, description, class_name}
    """
    repo = MagicMock(spec=IModelRepository)
    repo.get_model.return_value = mock_model
    repo.is_loaded.return_value = False
    repo.is_cached_on_disk.return_value = False
    repo.get_all_keys.return_value = list(ALL_MODEL_KEYS)
    repo.get_display_name.side_effect = lambda k: k.upper()
    repo.get_model_metadata.return_value = {
        "size_gb": 1.0,
        "params": "500M",
        "description": "test model",
        "class_name": "MockModel",
    }
    return repo


@pytest.fixture
def mock_preprocessor():
    """A spec'd MagicMock for IAudioPreprocessor.

    preprocess(path) is a no-op passthrough: returns the path unchanged.
    """
    p = MagicMock(spec=IAudioPreprocessor)
    p.preprocess.side_effect = lambda path: path
    return p


@pytest.fixture
def mock_memory_monitor():
    """A spec'd MagicMock for IMemoryMonitor.

    get_stats() returns a fixed MemoryStats snapshot:
        16 GB total, 8 GB used, 8 GB available, 50 % usage, 2 GB RSS.
    """
    m = MagicMock(spec=IMemoryMonitor)
    m.get_stats.return_value = MemoryStats(
        sys_total_gb=16.0,
        sys_used_gb=8.0,
        sys_avail_gb=8.0,
        sys_percent=50.0,
        proc_rss_gb=2.0,
    )
    return m


@pytest.fixture
def mock_watermark_detector():
    """A spec'd MagicMock for IWatermarkDetector.

    Defaults:
        is_available() → True
        detect(path)   → 1.0  (fully watermarked)
    """
    d = MagicMock(spec=IWatermarkDetector)
    d.is_available.return_value = True
    d.detect.return_value = 1.0
    return d
