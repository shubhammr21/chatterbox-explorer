"""Audio infrastructure: frame alignment and format conversion."""
import logging
import tempfile
from pathlib import Path

import numpy as np

from chatterbox_explorer.domain.models import AudioResult
from chatterbox_explorer.ports.output import IAudioPreprocessor

log = logging.getLogger(__name__)


class TorchAudioPreprocessor(IAudioPreprocessor):
    """Trims reference audio to a multiple of 40 ms before model conditioning.

    Root cause (chatterbox/models/s3gen/s3gen.py → embed_ref):
      mel extractor  — 24 kHz, hop=480 → 50 frames/sec (20 ms/frame)
      S3 tokenizer   — 16 kHz at 25 tokens/sec → 40 ms/token (2 frames/token)
      Invariant: mel_len = 2 × token_len

    This invariant holds exactly when audio duration is a multiple of 40 ms.
    If the tail of the reference clip does not fill a complete 40 ms frame the
    model's mel and token lengths go out of sync, causing an index error deep
    inside embed_ref.  We trim to the nearest complete frame boundary to
    prevent this without resampling or quality loss.

    Implementation notes:
      - frame_samples is computed from the file's *native* sample rate, not a
        fixed 24 kHz, so the trim is correct for any input sample rate.
      - If the audio is already aligned (or shorter than one frame) the
        original path is returned with no I/O cost.
      - On any failure the original path is returned unchanged so generation
        can still proceed (the model may raise later, but that error will be
        more descriptive than a silent wrong-type failure here).
    """

    def preprocess(self, path: str | None) -> str | None:
        """Trim *path* to a 40 ms frame boundary and return the result path.

        Args:
            path: Absolute or relative path to a WAV/FLAC/MP3 file, or
                  ``None`` if no reference audio was provided.

        Returns:
            Path to the trimmed file (may be a new temp file), or ``None``
            if *path* was ``None``.  The original path is returned unchanged
            when the audio is already aligned or trimming is not needed.
        """
        if not path:
            return None

        try:
            import torchaudio

            wav, sr = torchaudio.load(path)

            # 40 ms expressed as an integer number of samples at the file's
            # native sample rate.  round() handles floating-point imprecision
            # (e.g. 22050 Hz → 882.0 samples exactly).
            frame_samples = round(sr * 0.040)

            # Guard: degenerate sample rate or clip shorter than one frame —
            # nothing useful to trim, return as-is.
            if frame_samples <= 0 or wav.shape[-1] <= frame_samples:
                return path

            n_complete = wav.shape[-1] // frame_samples
            target_len = n_complete * frame_samples

            # Already aligned — skip the temp-file write entirely.
            if target_len == wav.shape[-1]:
                return path

            original_len = wav.shape[-1]
            wav = wav[..., :target_len]

            # Write trimmed audio to a named temp file so torchaudio can read
            # it back via a file path (required by Chatterbox's audio_prompt_path
            # argument which does not accept raw tensors).
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            torchaudio.save(tmp.name, wav, sr)
            tmp.close()

            log.debug(
                "preprocess_ref_audio: trimmed %d→%d samples (%d ms removed) at %d Hz",
                original_len,
                target_len,
                round((original_len - target_len) / sr * 1000),
                sr,
            )
            return tmp.name

        except Exception as exc:
            # Broad catch: any I/O error, unsupported codec, or torchaudio
            # issue falls through here.  We log at DEBUG (not WARNING) because
            # the failure is non-fatal — the model will attempt generation with
            # the original unprocessed path.
            log.debug(
                "preprocess_ref_audio: skipped (%s) — using original path", exc
            )
            return path


def to_gradio_audio(result: AudioResult) -> tuple[int, np.ndarray]:
    """Convert an :class:`AudioResult` (float32) to the format Gradio expects.

    Gradio 6.x ``gr.Audio(type='numpy')`` requires ``int16`` arrays.
    Returning float32 triggers:

        UserWarning: Trying to convert audio automatically from float32 to
        16-bit int format.

    We perform the conversion ourselves — cleanly and with peak normalisation
    so that values never exceed the int16 range, eliminating the warning.

    Normalisation policy:
        - If ``peak > 1.0`` the signal is divided by its peak before scaling
          so that the loudest sample maps to exactly ±32767.
        - If ``peak ≤ 1.0`` the signal is scaled directly; no gain is applied,
          preserving the original relative loudness.
        - ``np.clip`` is applied as a final guard against floating-point
          rounding that could push a value just outside [-32768, 32767].

    Args:
        result: A domain :class:`AudioResult` holding float32 samples and a
                sample rate in Hz.

    Returns:
        ``(sample_rate, int16_array)`` tuple ready for assignment to a Gradio
        ``gr.Audio`` output component.
    """
    arr = result.samples.astype(np.float32)

    peak = np.abs(arr).max()
    if peak > 1.0:
        # Normalise to [-1.0, 1.0] before converting to int16.
        arr = arr / peak

    int16_arr = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)
    return result.sample_rate, int16_arr
