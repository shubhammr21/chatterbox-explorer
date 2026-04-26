"""PerTh watermark detection adapter."""
import logging

from chatterbox_explorer.ports.output import IWatermarkDetector

log = logging.getLogger(__name__)


class PerThWatermarkDetector(IWatermarkDetector):
    """Wraps the resemble-perth PerTh neural watermark detector.

    When the open-source resemble-perth package is active (PerthImplicitWatermarker=None),
    this detector reports available=False and always returns score=0.0.
    The no-op patch in cli.py ensures models still load; this adapter reports
    the unavailability honestly via is_available().
    """

    def __init__(self, available: bool = True) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def detect(self, audio_path: str) -> float:
        if not self._available:
            return 0.0
        try:
            import librosa
            import perth

            audio, sr = librosa.load(audio_path, sr=None)
            wm = perth.PerthImplicitWatermarker()
            return float(wm.get_watermark(audio, sample_rate=sr))
        except Exception as exc:
            log.warning("Watermark detection failed: %s", exc)
            return 0.0
