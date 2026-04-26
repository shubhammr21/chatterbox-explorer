"""
tests/unit/adapters/test_audio_preprocessor.py
================================================
TDD unit tests for TorchAudioPreprocessor and the to_gradio_audio helper.

All tests that require real audio I/O use torchaudio to create minimal WAV
fixtures in a temporary directory — no fixtures from conftest.py are needed.

Frame-alignment arithmetic (24 kHz reference):
    frame_samples = round(24_000 * 0.040) = 960
    960  samples → short-circuit (≤ frame_samples) → original path returned
    970  samples → 970 // 960 = 1 → target_len = 960 ≠ 970 → trim + new file
    1920 samples → 1920 // 960 = 2 → target_len = 1920 == 1920 → already aligned
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── optional-dependency guards ────────────────────────────────────────────────
# Both torch and torchaudio are required by chatterbox-tts, so they should
# always be present in the project venv.  We skip gracefully if they are not
# (e.g. a stripped CI environment running only pure-Python tests).
torch = pytest.importorskip("torch", reason="torch required for audio adapter tests")
torchaudio = pytest.importorskip("torchaudio", reason="torchaudio required for audio adapter tests")

from chatterbox_explorer.adapters.secondary.audio import TorchAudioPreprocessor, to_gradio_audio
from chatterbox_explorer.domain.models import AudioResult


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_wav(tmp_path: Path, n_samples: int, sr: int = 24_000) -> str:
    """Write a mono silent WAV file with *n_samples* at *sr* Hz.

    Returns the path as a str (matching the adapter's str | None signature).
    """
    wav = torch.zeros(1, n_samples)
    p = tmp_path / f"test_{n_samples}samples_{sr}hz.wav"
    torchaudio.save(str(p), wav, sr)
    return str(p)


def _load_sample_count(path: str) -> int:
    """Return the number of samples in the (first channel of a) WAV file."""
    wav, _ = torchaudio.load(path)
    return wav.shape[-1]


# ──────────────────────────────────────────────────────────────────────────────
# TorchAudioPreprocessor
# ──────────────────────────────────────────────────────────────────────────────

class TestTorchAudioPreprocessor:
    """Tests for the IAudioPreprocessor adapter."""

    @pytest.fixture(autouse=True)
    def preprocessor(self) -> TorchAudioPreprocessor:
        return TorchAudioPreprocessor()

    # ── None path ─────────────────────────────────────────────────────────────

    def test_preprocess_none_returns_none(self, preprocessor):
        """Contract: None input → None output (no reference audio case)."""
        assert preprocessor.preprocess(None) is None

    def test_preprocess_empty_string_returns_none(self, preprocessor):
        """Empty string is also treated as 'no audio' and returns None."""
        assert preprocessor.preprocess("") is None

    # ── Already aligned ───────────────────────────────────────────────────────

    def test_preprocess_already_aligned_returns_same_path(
        self, preprocessor, tmp_path
    ):
        """A WAV of exactly 960 samples (one 40 ms frame at 24 kHz) is a
        short-circuit case: wav.shape[-1] <= frame_samples, so the original
        path is returned without any I/O.
        """
        path = _make_wav(tmp_path, n_samples=960, sr=24_000)
        result = preprocessor.preprocess(path)
        assert result == path, (
            "Already-aligned (or single-frame) audio should return the original path"
        )

    def test_preprocess_two_complete_frames_returns_same_path(
        self, preprocessor, tmp_path
    ):
        """1920 samples = exactly 2 × 40 ms frames — target_len == wav.shape[-1],
        so the fast path triggers and no temp file is written.
        """
        path = _make_wav(tmp_path, n_samples=1920, sr=24_000)
        result = preprocessor.preprocess(path)
        assert result == path

    # ── Unaligned → new path ──────────────────────────────────────────────────

    def test_preprocess_unaligned_returns_new_path(
        self, preprocessor, tmp_path
    ):
        """970 samples at 24 kHz (960 + 10 remainder) must produce a new
        temp file, so the returned path differs from the input path.
        """
        path = _make_wav(tmp_path, n_samples=970, sr=24_000)
        result = preprocessor.preprocess(path)
        assert result is not None
        assert result != path, (
            "Unaligned audio should be written to a new temp file"
        )

    def test_preprocess_unaligned_output_aligned(
        self, preprocessor, tmp_path
    ):
        """The trimmed output must be a whole multiple of 40 ms (960 samples)
        when loaded back from disk.
        """
        path = _make_wav(tmp_path, n_samples=970, sr=24_000)
        result = preprocessor.preprocess(path)

        assert result is not None
        out_samples = _load_sample_count(result)

        # At 24 kHz, frame_samples = round(24_000 * 0.040) = 960
        frame_samples = round(24_000 * 0.040)
        assert out_samples % frame_samples == 0, (
            f"Output sample count {out_samples} is not a multiple of "
            f"{frame_samples} (40 ms at 24 kHz)"
        )

    def test_preprocess_unaligned_output_shorter_than_input(
        self, preprocessor, tmp_path
    ):
        """Trimming must reduce the sample count, never increase it."""
        n_input = 970
        path = _make_wav(tmp_path, n_samples=n_input, sr=24_000)
        result = preprocessor.preprocess(path)

        assert result is not None
        out_samples = _load_sample_count(result)
        assert out_samples < n_input

    def test_preprocess_unaligned_output_is_valid_wav(
        self, preprocessor, tmp_path
    ):
        """The new temp file must be a loadable WAV (not a partial write)."""
        path = _make_wav(tmp_path, n_samples=2500, sr=24_000)
        result = preprocessor.preprocess(path)

        assert result is not None
        # torchaudio.load raises if the file is malformed
        wav, sr = torchaudio.load(result)
        assert sr == 24_000
        assert wav.shape[0] == 1  # mono

    def test_preprocess_large_unaligned_audio(self, preprocessor, tmp_path):
        """5 seconds + 10 spare samples → trim to nearest 40 ms boundary."""
        sr = 24_000
        five_seconds = sr * 5          # 120_000 samples
        n_input = five_seconds + 10    # 120_010
        path = _make_wav(tmp_path, n_samples=n_input, sr=sr)
        result = preprocessor.preprocess(path)

        assert result is not None
        out_samples = _load_sample_count(result)
        frame_samples = round(sr * 0.040)
        assert out_samples % frame_samples == 0
        assert out_samples == five_seconds  # 120_000

    def test_preprocess_nonexistent_path_returns_original(self, preprocessor):
        """When torchaudio.load fails (file not found), the original path is
        returned unchanged rather than raising an exception.
        """
        bad_path = "/tmp/this_file_does_not_exist_xyz123.wav"
        result = preprocessor.preprocess(bad_path)
        assert result == bad_path, (
            "On I/O failure the adapter should return the original path as a "
            "graceful degradation (the model will raise a more descriptive error)"
        )

    def test_preprocess_preserves_sample_rate(self, preprocessor, tmp_path):
        """The output WAV should have the same sample rate as the input."""
        sr = 22_050
        # 22050 * 0.040 = 882.0 — need an unaligned count
        path = _make_wav(tmp_path, n_samples=1000, sr=sr)  # 1000 > 882
        result = preprocessor.preprocess(path)

        assert result is not None
        _, out_sr = torchaudio.load(result)
        assert out_sr == sr


# ──────────────────────────────────────────────────────────────────────────────
# to_gradio_audio
# ──────────────────────────────────────────────────────────────────────────────

class TestToGradioAudio:
    """Tests for the float32 → int16 conversion helper."""

    def _make_result(self, samples: np.ndarray, sr: int = 24_000) -> AudioResult:
        return AudioResult(sample_rate=sr, samples=samples.astype(np.float32))

    # ── dtype ─────────────────────────────────────────────────────────────────

    def test_to_gradio_audio_returns_int16(self):
        """Output array must be dtype int16 — required by Gradio 6.x."""
        result = self._make_result(np.array([0.0, 0.5, -0.5]))
        _, arr = to_gradio_audio(result)
        assert arr.dtype == np.int16, (
            "Gradio 6.x expects int16; returning float32 triggers a UserWarning"
        )

    def test_to_gradio_audio_returns_sample_rate(self):
        """The first element of the tuple must be the original sample rate."""
        result = self._make_result(np.array([0.0]), sr=22_050)
        sr, _ = to_gradio_audio(result)
        assert sr == 22_050

    # ── range / overflow ──────────────────────────────────────────────────────

    def test_to_gradio_audio_no_overflow(self):
        """All output samples must stay within the int16 range [-32768, 32767]."""
        # Use values that are already in [-1, 1] — no normalisation needed.
        samples = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
        result = self._make_result(samples)
        _, arr = to_gradio_audio(result)

        assert arr.min() >= -32768, f"Under-range: min = {arr.min()}"
        assert arr.max() <= 32767, f"Over-range:  max = {arr.max()}"

    def test_to_gradio_audio_no_overflow_extreme_values(self):
        """Edge case: samples at exactly ±1.0 must not exceed int16 bounds."""
        samples = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        result = self._make_result(samples)
        _, arr = to_gradio_audio(result)

        assert int(arr.max()) <= 32767
        assert int(arr.min()) >= -32768

    # ── normalisation ─────────────────────────────────────────────────────────

    def test_to_gradio_audio_normalizes_if_peak_exceeds_one(self):
        """When peak > 1.0 the signal is normalised before int16 scaling so
        the loudest sample maps to exactly ±32767 and there is no clipping.
        """
        # Peak = 2.0 — without normalisation this would overflow int16.
        samples = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        result = self._make_result(samples)
        _, arr = to_gradio_audio(result)

        # After normalising by peak=2.0: [1.0, -0.5, 0.25]
        # After ×32767:                  [32767, -16383, 8191]
        assert arr.max() <= 32767
        assert arr.min() >= -32768
        # The positive peak (2.0) should map to the maximum int16 positive value
        assert arr[0] == 32767

    def test_to_gradio_audio_does_not_normalize_if_peak_at_most_one(self):
        """A signal with peak = 0.5 should NOT be amplified — the normalisation
        guard only kicks in when peak > 1.0.
        """
        samples = np.array([0.5, -0.5], dtype=np.float32)
        result = self._make_result(samples)
        _, arr = to_gradio_audio(result)

        # 0.5 × 32767 = 16383.5 → 16383 (truncated to int16)
        expected_max = int(0.5 * 32767)
        assert arr.max() == expected_max or arr.max() == expected_max + 1  # rounding tolerance

    def test_to_gradio_audio_all_zeros(self):
        """Silent audio (all zeros) must remain silent after conversion."""
        samples = np.zeros(100, dtype=np.float32)
        result = self._make_result(samples)
        _, arr = to_gradio_audio(result)

        assert arr.dtype == np.int16
        assert (arr == 0).all()

    def test_to_gradio_audio_preserves_sign(self):
        """Positive samples must stay positive, negative must stay negative."""
        samples = np.array([0.8, -0.8], dtype=np.float32)
        result = self._make_result(samples)
        _, arr = to_gradio_audio(result)

        assert arr[0] > 0
        assert arr[1] < 0

    def test_to_gradio_audio_shape_preserved(self):
        """Output array shape must match input shape."""
        n = 48_000  # 2 seconds at 24 kHz
        samples = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
        result = self._make_result(samples)
        _, arr = to_gradio_audio(result)

        assert arr.shape == (n,)
