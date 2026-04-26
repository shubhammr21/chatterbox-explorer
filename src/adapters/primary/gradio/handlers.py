"""
src/adapters/primary/gradio/handlers.py
===========================================================
GradioHandlers — owns all Gradio event callbacks for the Chatterbox TTS Explorer UI.

Architecture contract:
  - All domain services are injected via __init__; this class never instantiates
    secondary adapters directly (except importing to_gradio_audio(), a pure
    format-conversion utility with no infrastructure concerns).
  - Exception translation policy:
        ValueError     → gr.Warning(str(e)) + return  (user-facing input error)
        AssertionError → gr.Error("Reference audio must be > 5 seconds")
        Exception      → gr.Error(str(e))              (unexpected runtime failure)
  - render_manager_html() reads memory and model state exclusively through
    self._manager.get_all_status() and self._manager.get_memory_stats() — never
    via secondary adapters, psutil, or torch directly.
  - handle_download() is a generator that yields (html_str, log_str) tuples so
    Gradio can stream download progress to the UI.
  - Streaming TTS handlers are generator functions that yield
    to_gradio_audio(result) after every AudioResult, giving Gradio progressive
    audio updates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gradio as gr

from adapters.secondary.audio import to_gradio_audio
from domain.languages import (
    LANGUAGE_AUDIO_DEFAULTS,
    SAMPLE_TEXTS,
)
from domain.models import (
    AppConfig,
    MultilingualTTSRequest,
    TTSRequest,
    TurboTTSRequest,
    VoiceConversionRequest,
)
from domain.presets import PRESETS_TTS, PRESETS_TURBO

if TYPE_CHECKING:
    # Port ABCs are only referenced in type annotations (constructor parameters).
    # Moving them here avoids importing the ABC hierarchy at runtime while
    # keeping full type-checker visibility.  from __future__ import annotations
    # (above) ensures all annotations are treated as strings, so these imports
    # are never needed at runtime.
    from ports.input import (
        IModelManagerService,
        IMultilingualTTSService,
        ITTSService,
        ITurboTTSService,
        IVoiceConversionService,
        IWatermarkService,
    )

log = logging.getLogger(__name__)


class GradioHandlers:
    """Owns all Gradio event callbacks for the Chatterbox TTS Explorer.

    Constructed once inside ``build_demo()`` with all domain services injected.
    Each public method maps to one or more Gradio ``.click()`` / ``.change()``
    events and is responsible for:

    1. Building domain request value-objects from raw Gradio inputs.
    2. Delegating to the appropriate service port.
    3. Converting ``AudioResult`` domain objects to Gradio-compatible tuples
       via ``to_gradio_audio()``.
    4. Translating domain exceptions into Gradio user-facing notifications.
    """

    def __init__(
        self,
        tts: ITTSService,
        turbo: ITurboTTSService,
        mtl: IMultilingualTTSService,
        vc: IVoiceConversionService,
        manager: IModelManagerService,
        watermark: IWatermarkService,
        config: AppConfig,
    ) -> None:
        self._tts = tts
        self._turbo = turbo
        self._mtl = mtl
        self._vc = vc
        self._manager = manager
        self._watermark = watermark
        self._config = config

    # ──────────────────────────────────────────────────────────────────────────
    # TTS generators  (yield for streaming support; always yield at least once)
    # ──────────────────────────────────────────────────────────────────────────

    def handle_tts(
        self,
        text: str,
        ref_audio,
        exag: float,
        cfg: float,
        temp: float,
        rep: float,
        minp: float,
        topp: float,
        stream: bool,
        seed: int,
    ):
        """Standard TTS generator.

        Yields ``(sample_rate, int16_array)`` tuples compatible with
        ``gr.Audio(type='numpy')``.  When *stream* is ``False`` a single
        tuple is yielded after the full clip is generated; when ``True``
        the cumulative audio is yielded after each sentence.
        """
        try:
            request = TTSRequest(
                text=text,
                ref_audio_path=ref_audio or None,
                exaggeration=float(exag),
                cfg_weight=float(cfg),
                temperature=float(temp),
                rep_penalty=float(rep),
                min_p=float(minp),
                top_p=float(topp),
                seed=int(seed),
                streaming=bool(stream),
            )
            if stream:
                for result in self._tts.generate_stream(request):
                    yield to_gradio_audio(result)
            else:
                result = self._tts.generate(request)
                yield to_gradio_audio(result)
        except ValueError as e:
            gr.Warning(str(e))
            return
        except AssertionError:
            raise gr.Error("Reference audio must be > 5 seconds")
        except Exception as e:
            raise gr.Error(str(e))

    def handle_turbo(
        self,
        text: str,
        ref_audio,
        temp: float,
        topk: int,
        topp: float,
        rep: float,
        minp: float,
        loudness: bool,
        stream: bool,
        seed: int,
    ):
        """Turbo TTS generator.

        Yields ``(sample_rate, int16_array)`` tuples.  Reference audio shorter
        than 5 s causes an ``AssertionError`` inside the model which is
        translated to a visible ``gr.Error``.
        """
        try:
            request = TurboTTSRequest(
                text=text,
                ref_audio_path=ref_audio or None,
                temperature=float(temp),
                top_k=int(topk),
                top_p=float(topp),
                rep_penalty=float(rep),
                min_p=float(minp),
                norm_loudness=bool(loudness),
                seed=int(seed),
                streaming=bool(stream),
            )
            if stream:
                for result in self._turbo.generate_stream(request):
                    yield to_gradio_audio(result)
            else:
                result = self._turbo.generate(request)
                yield to_gradio_audio(result)
        except ValueError as e:
            gr.Warning(str(e))
            return
        except AssertionError:
            raise gr.Error("Reference audio must be > 5 seconds")
        except Exception as e:
            raise gr.Error(str(e))

    def handle_multilingual(
        self,
        text: str,
        lang: str,
        ref_audio,
        exag: float,
        cfg: float,
        temp: float,
        rep: float,
        minp: float,
        topp: float,
        stream: bool,
        seed: int,
    ):
        """Multilingual TTS generator.

        Extracts the ISO-639-1 code from the ``"<code> - <Name>"`` dropdown
        value (e.g. ``"fr - French"`` → ``"fr"``), then yields
        ``(sample_rate, int16_array)`` tuples.
        """
        try:
            lang_code = lang.split(" ")[0]  # "fr - French" → "fr"
            request = MultilingualTTSRequest(
                text=text,
                language=lang_code,
                ref_audio_path=ref_audio or None,
                exaggeration=float(exag),
                cfg_weight=float(cfg),
                temperature=float(temp),
                rep_penalty=float(rep),
                min_p=float(minp),
                top_p=float(topp),
                seed=int(seed),
                streaming=bool(stream),
            )
            if stream:
                for result in self._mtl.generate_stream(request):
                    yield to_gradio_audio(result)
            else:
                result = self._mtl.generate(request)
                yield to_gradio_audio(result)
        except ValueError as e:
            gr.Warning(str(e))
            return
        except AssertionError:
            raise gr.Error("Reference audio must be > 5 seconds")
        except Exception as e:
            raise gr.Error(str(e))

    def handle_vc(self, source, target):
        """Voice Conversion handler.

        Returns a single ``(sample_rate, int16_array)`` tuple, or ``None``
        if a ``ValueError`` was raised (e.g. missing source/target paths).
        """
        try:
            request = VoiceConversionRequest(
                source_audio_path=source or "",
                target_voice_path=target or "",
            )
            result = self._vc.convert(request)
            return to_gradio_audio(result)
        except ValueError as e:
            gr.Warning(str(e))
            return None
        except Exception as e:
            raise gr.Error(str(e))

    def handle_watermark(self, audio_path: str) -> str:
        """Detect PerTh watermark in *audio_path*.

        Calls ``self._watermark.detect()`` and formats the ``WatermarkResult``
        into a human-readable string suitable for a ``gr.Textbox``.
        Formatting is an adapter concern — the domain ``WatermarkResult``
        carries only the raw verdict and score.
        """
        if not audio_path:
            return "⚠️  No audio uploaded."

        try:
            result = self._watermark.detect(audio_path)

            if not result.available or result.verdict == "unavailable":
                return "⚠️  WATERMARK DETECTION UNAVAILABLE\n\n" + result.message

            if result.verdict == "detected":
                verdict_str = "✅  WATERMARK DETECTED"
                detail = "This audio was generated by Chatterbox (Resemble AI)."
            elif result.verdict == "not_detected":
                verdict_str = "❌  NO WATERMARK"
                detail = "This audio does not appear to be Chatterbox-generated."
            else:  # "inconclusive"
                verdict_str = "⚠️  INCONCLUSIVE"
                detail = "Partial or degraded watermark signal detected."

            return (
                f"{verdict_str}\n\n"
                f"Score: {result.score:.4f}  (1.0 = watermarked · 0.0 = clean)\n\n"
                f"{detail}"
            )
        except Exception as e:
            return f"Error during watermark check:\n{e}"

    # ──────────────────────────────────────────────────────────────────────────
    # Model Manager
    # ──────────────────────────────────────────────────────────────────────────

    def handle_load(self, key: str) -> tuple[str, str]:
        """Load a model into memory.

        Returns ``(html_panel, log_message)`` where *html_panel* is a fresh
        render of the status table after the load attempt.
        """
        try:
            log_msg = self._manager.load(key)
        except Exception as e:
            log_msg = f"❌  Load failed — {e}"
        return self.render_manager_html(), log_msg

    def handle_unload(self, key: str) -> tuple[str, str]:
        """Unload a model from memory.

        Returns ``(html_panel, log_message)``.  The service handles the
        5-step MPS/CUDA memory-flush recipe; this handler only wraps it.
        """
        log_msg = self._manager.unload(key)
        return self.render_manager_html(), log_msg

    def handle_download(self, key: str):
        """Download model weights, yielding ``(html_panel, log_str)`` tuples.

        The ``IModelManagerService.download()`` port is itself a generator
        that yields progress strings; we wrap each string with a fresh HTML
        status panel so the UI reflects current disk/memory state in real time.
        """
        try:
            for log_line in self._manager.download(key):
                yield self.render_manager_html(), log_line
        except Exception as e:
            yield self.render_manager_html(), f"❌  Download failed: {e}"

    def render_manager_html(self) -> str:
        """Build the HTML status panel shown at the top of the Model Manager tab.

        Reads from:
            - ``self._manager.get_memory_stats()`` → ``MemoryStats`` domain object
            - ``self._manager.get_all_status()``   → ``list[ModelStatus]``

        Never calls psutil, torch, or any secondary adapter directly — that
        would break the hexagonal architecture boundary.

        Returns an HTML string containing:
            1. System RAM progress bar (colour-coded by usage %).
            2. Device memory bar (MPS / CUDA) when a GPU/MPS device is present.
            3. Per-model status table with Disk Cache and In Memory badges.
        """
        stats = self._manager.get_memory_stats()
        model_statuses = self._manager.get_all_status()

        # ── System RAM gauge ──────────────────────────────────────────────────
        sys_pct = stats.sys_percent
        bar_col = "#22c55e" if sys_pct < 60 else "#f59e0b" if sys_pct < 80 else "#ef4444"

        html = (
            '<div style="font-family:Inter,ui-sans-serif,sans-serif;'
            'font-size:13px;line-height:1.5;">\n'
            '  <div style="margin-bottom:14px;">\n'
            '    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">\n'
            '      <span style="font-weight:600;min-width:130px;">System RAM</span>\n'
            '      <div style="flex:1;background:#e5e7eb;border-radius:5px;'
            'height:12px;overflow:hidden;">\n'
            f'        <div style="background:{bar_col};width:{sys_pct:.0f}%;'
            'height:100%;border-radius:5px;"></div>\n'
            "      </div>\n"
            '      <span style="color:#6b7280;white-space:nowrap;min-width:240px;">\n'
            f"        {stats.sys_used_gb:.1f} / {stats.sys_total_gb:.1f} GB\n"
            f"        &nbsp;({sys_pct:.0f}%)\n"
            f"        &nbsp;·&nbsp; this app: {stats.proc_rss_gb:.2f} GB\n"
            "      </span>\n"
            "    </div>"
        )

        # ── Device memory gauge (MPS / CUDA only) ─────────────────────────────
        if stats.device_driver_gb is not None and stats.device_max_gb:
            dev_gb = stats.device_driver_gb
            dev_max = stats.device_max_gb
            dev_pct = min(dev_gb / dev_max * 100.0, 100.0) if dev_max > 0 else 0.0
            dev_col = "#22c55e" if dev_pct < 60 else "#f59e0b" if dev_pct < 80 else "#ef4444"

            is_mps = stats.device_name == "Apple Silicon MPS"
            label = "MPS (unified)" if is_mps else stats.device_name

            if is_mps:
                detail = (
                    f"{dev_gb:.2f} / {dev_max:.1f} GB driver pool"
                    f" &nbsp;·&nbsp; recommended max: {dev_max:.1f} GB"
                )
            else:
                detail = f"{dev_gb:.2f} / {dev_max:.1f} GB"

            html += (
                '\n    <div style="display:flex;align-items:center;gap:10px;">\n'
                f'      <span style="font-weight:600;min-width:130px;">{label}</span>\n'
                '      <div style="flex:1;background:#e5e7eb;border-radius:5px;'
                'height:12px;overflow:hidden;">\n'
                f'        <div style="background:{dev_col};width:{dev_pct:.0f}%;'
                'height:100%;border-radius:5px;"></div>\n'
                "      </div>\n"
                f'      <span style="color:#6b7280;white-space:nowrap;'
                f'min-width:240px;">{detail}</span>\n'
                "    </div>"
            )

        html += "\n  </div>\n"

        # ── Per-model status table ─────────────────────────────────────────────
        html += (
            '  <table style="width:100%;border-collapse:collapse;font-size:13px;">\n'
            "    <thead>\n"
            '      <tr style="background:#f9fafb;border-bottom:2px solid #e5e7eb;">\n'
            '        <th style="text-align:left;padding:8px 10px;'
            'font-weight:600;">Model</th>\n'
            '        <th style="text-align:center;padding:8px;'
            'font-weight:600;">Params</th>\n'
            '        <th style="text-align:center;padding:8px;'
            'font-weight:600;">Size</th>\n'
            '        <th style="text-align:center;padding:8px;'
            'font-weight:600;">Disk Cache</th>\n'
            '        <th style="text-align:center;padding:8px;'
            'font-weight:600;">In Memory</th>\n'
            "      </tr>\n"
            "    </thead>\n"
            "    <tbody>"
        )

        for status in model_statuses:
            mem_badge = (
                '<span style="background:#dcfce7;color:#166534;padding:2px 10px;'
                'border-radius:12px;font-size:11px;font-weight:600;">✅ Loaded</span>'
                if status.in_memory
                else '<span style="background:#f3f4f6;color:#9ca3af;padding:2px 10px;'
                'border-radius:12px;font-size:11px;">— Unloaded</span>'
            )
            disk_badge = (
                '<span style="background:#dbeafe;color:#1e40af;padding:2px 10px;'
                'border-radius:12px;font-size:11px;">💾 Cached</span>'
                if status.on_disk
                else '<span style="background:#fef9c3;color:#92400e;padding:2px 10px;'
                'border-radius:12px;font-size:11px;">⬇ Not Downloaded</span>'
            )

            html += (
                '\n      <tr style="border-bottom:1px solid #f3f4f6;">\n'
                '        <td style="padding:10px 10px;">\n'
                f"          <strong>{status.display_name}</strong>"
                f'          <span style="color:#9ca3af;font-size:11px;'
                f'margin-left:6px;">{status.class_name}</span><br>\n'
                f'          <span style="color:#6b7280;font-size:11px;">'
                f"{status.description}</span>\n"
                "        </td>\n"
                f'        <td style="text-align:center;padding:10px;'
                f'color:#6b7280;">{status.params}</td>\n'
                f'        <td style="text-align:center;padding:10px;'
                f'color:#6b7280;">~{status.size_gb:.1f} GB</td>\n'
                f'        <td style="text-align:center;padding:10px;">'
                f"{disk_badge}</td>\n"
                f'        <td style="text-align:center;padding:10px;">'
                f"{mem_badge}</td>\n"
                "      </tr>"
            )

        html += "\n    </tbody>\n  </table>\n</div>"
        return html

    # ──────────────────────────────────────────────────────────────────────────
    # Presets
    # ──────────────────────────────────────────────────────────────────────────

    def apply_preset_tts(self, preset_name: str) -> tuple:
        """Return an 8-tuple for Standard TTS Gradio outputs.

        Output order (matches the ``.change()`` outputs list in ``build_demo``):
            (exaggeration, cfg_weight, temperature, rep_penalty,
             min_p, top_p, sample_text, rationale_md)
        """
        preset = PRESETS_TTS.get(preset_name)
        if preset is None:
            # Unknown preset — return safe defaults; should not happen in practice.
            return 0.5, 0.5, 0.8, 1.2, 0.05, 1.0, "", ""
        p = preset["params"]
        return (
            p["exaggeration"],
            p["cfg_weight"],
            p["temperature"],
            p["rep_penalty"],
            p["min_p"],
            p["top_p"],
            preset["sample_text"],
            preset["rationale_md"],
        )

    def apply_preset_turbo(self, preset_name: str) -> tuple:
        """Return an 8-tuple for Turbo TTS Gradio outputs.

        Output order:
            (temperature, top_k, top_p, rep_penalty,
             min_p, norm_loudness, sample_text, rationale_md)
        """
        preset = PRESETS_TURBO.get(preset_name)
        if preset is None:
            return 0.8, 1000, 0.95, 1.2, 0.0, True, "", ""
        p = preset["params"]
        return (
            p["temperature"],
            p["top_k"],
            p["top_p"],
            p["rep_penalty"],
            p["min_p"],
            p["norm_loudness"],
            preset["sample_text"],
            preset["rationale_md"],
        )

    def apply_preset_mtl(self, preset_name: str) -> tuple:
        """Return a 7-tuple for Multilingual TTS Gradio outputs.

        The text input is intentionally *excluded* — it is controlled
        separately by the language dropdown and sample-text autofill.

        Output order:
            (exaggeration, cfg_weight, temperature, rep_penalty,
             min_p, top_p, rationale_md)
        """
        preset = PRESETS_TTS.get(preset_name)
        if preset is None:
            return 0.5, 0.5, 0.8, 1.2, 0.05, 1.0, ""
        p = preset["params"]
        return (
            p["exaggeration"],
            p["cfg_weight"],
            p["temperature"],
            p["rep_penalty"],
            p["min_p"],
            p["top_p"],
            preset["rationale_md"],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Language
    # ──────────────────────────────────────────────────────────────────────────

    def on_language_change(self, language: str) -> tuple:
        """Auto-fill sample text and reference audio when the language changes.

        Parses the ISO-639-1 code from the ``"<code> - <Name>"`` dropdown
        value, then looks up the curated sample text and native-speaker
        reference-audio URL.

        Returns:
            ``(sample_text, audio_url)`` — audio_url may be ``None`` if no
            default clip is registered for that language code.
        """
        lang_code = language.split(" ")[0]  # "fr - French" → "fr"
        text = SAMPLE_TEXTS.get(language, SAMPLE_TEXTS.get("en - English", ""))
        audio = LANGUAGE_AUDIO_DEFAULTS.get(lang_code)
        return text, audio
