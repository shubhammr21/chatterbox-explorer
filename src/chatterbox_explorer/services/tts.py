"""
src/chatterbox_explorer/services/tts.py
=========================================
Domain services for text-to-speech generation.

Architecture rules enforced here:
  - ZERO imports from torch, gradio, or chatterbox packages
  - ValueError   → caller-visible input errors (empty text, bad ref audio)
  - RuntimeError → infrastructure failures surfaced from the model layer
  - seed_setter is injected as a plain callable — no torch.manual_seed here

Services:
  TTSService             — Standard ChatterboxTTS (full parameter control)
  TurboTTSService        — ChatterboxTurboTTS (fast, low-VRAM)
  MultilingualTTSService — ChatterboxMultilingualTTS (23 languages)
"""
from __future__ import annotations

import re
from typing import Callable, Iterator

import numpy as np

from chatterbox_explorer.domain.models import (
    AudioResult,
    MultilingualTTSRequest,
    TTSRequest,
    TurboTTSRequest,
)
from chatterbox_explorer.ports.output import IAudioPreprocessor, IModelRepository


# ──────────────────────────────────────────────────────────────────────────────
# Pure helper — no dependencies, fully testable in isolation
# ──────────────────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split *text* on sentence-ending punctuation (``.``, ``!``, ``?``).

    Rules:
        - Splits on whitespace that immediately follows ``.``, ``!``, or ``?``.
        - Strips each part and discards empty parts.
        - Pure stdlib — zero third-party dependencies.

    Examples::

        >>> split_sentences("Hello. World!")
        ['Hello.', 'World!']
        >>> split_sentences("")
        []
        >>> split_sentences("No punctuation")
        ['No punctuation']
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Private conversion helper
# ──────────────────────────────────────────────────────────────────────────────

def _tensor_to_numpy(wav) -> np.ndarray:
    """Convert a model output tensor to a 1-D float32 NumPy array.

    Accepts any tensor-like with ``.squeeze()``, ``.detach()``, ``.cpu()``,
    and ``.numpy()``.  Result is always contiguous float32 of shape ``(N,)``.
    """
    return wav.squeeze().detach().cpu().numpy().astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Standard TTS
# ──────────────────────────────────────────────────────────────────────────────

class TTSService:
    """Domain service for Standard ChatterboxTTS generation.

    Depends on:
        model_repo   — :class:`~chatterbox_explorer.ports.output.IModelRepository`
                       (loads / caches the ``"tts"`` model)
        preprocessor — :class:`~chatterbox_explorer.ports.output.IAudioPreprocessor`
                       (resamples reference audio before conditioning)
        seed_setter  — optional ``Callable[[int], None]`` injected at
                       construction time; defaults to a no-op so that unit
                       tests can omit it without side-effects.
    """

    def __init__(
        self,
        model_repo: IModelRepository,
        preprocessor: IAudioPreprocessor,
        seed_setter: Callable[[int], None] | None = None,
    ) -> None:
        self._repo = model_repo
        self._prep = preprocessor
        self._set_seed: Callable[[int], None] = seed_setter or (lambda _: None)

    # ── public API ────────────────────────────────────────────────────────────

    def generate(self, request: TTSRequest) -> AudioResult:
        """Generate a complete audio clip from *request* in one shot.

        Raises:
            ValueError: if ``request.text`` is empty or whitespace-only.
            RuntimeError: if the model layer raises for any infrastructure reason.
        """
        if not request.text.strip():
            raise ValueError("Text input is empty.")

        self._set_seed(request.seed)
        model = self._repo.get_model("tts")
        ref_path = self._prep.preprocess(request.ref_audio_path)

        wav = model.generate(
            request.text,
            audio_prompt_path=ref_path,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
            repetition_penalty=request.rep_penalty,
            min_p=request.min_p,
            top_p=request.top_p,
        )
        return AudioResult(
            sample_rate=model.sr,
            samples=_tensor_to_numpy(wav),
        )

    def generate_stream(self, request: TTSRequest) -> Iterator[AudioResult]:
        """Generate audio sentence-by-sentence, yielding cumulative results.

        Each yielded :class:`~chatterbox_explorer.domain.models.AudioResult`
        contains *all* audio produced so far (results are cumulative, not
        differential).

        Raises:
            ValueError: if ``request.text`` is empty or whitespace-only.
        """
        if not request.text.strip():
            raise ValueError("Text input is empty.")

        self._set_seed(request.seed)
        model = self._repo.get_model("tts")
        ref_path = self._prep.preprocess(request.ref_audio_path)

        gen_kwargs = dict(
            audio_prompt_path=ref_path,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
            repetition_penalty=request.rep_penalty,
            min_p=request.min_p,
            top_p=request.top_p,
        )

        sentences = split_sentences(request.text) or [request.text]
        buf: list[np.ndarray] = []

        for sentence in sentences:
            wav = model.generate(sentence, **gen_kwargs)
            buf.append(_tensor_to_numpy(wav))
            yield AudioResult(
                sample_rate=model.sr,
                samples=np.concatenate(buf),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Turbo TTS
# ──────────────────────────────────────────────────────────────────────────────

class TurboTTSService:
    """Domain service for ChatterboxTurboTTS generation.

    Key differences from :class:`TTSService`:

    * Uses the ``"turbo"`` model key.
    * Accepts ``top_k`` and ``norm_loudness`` instead of ``exaggeration`` /
      ``cfg_weight``.
    * Catches ``AssertionError`` raised by the model when the reference audio
      clip is shorter than 5 seconds and re-raises it as ``ValueError`` so
      callers receive a clean, framework-agnostic error.
    """

    def __init__(
        self,
        model_repo: IModelRepository,
        preprocessor: IAudioPreprocessor,
        seed_setter: Callable[[int], None] | None = None,
    ) -> None:
        self._repo = model_repo
        self._prep = preprocessor
        self._set_seed: Callable[[int], None] = seed_setter or (lambda _: None)

    # ── public API ────────────────────────────────────────────────────────────

    def generate(self, request: TurboTTSRequest) -> AudioResult:
        """Generate a complete audio clip from *request* in one shot.

        Raises:
            ValueError: if ``request.text`` is empty, or if the reference
                        audio is shorter than 5 seconds (model assertion).
            RuntimeError: if the model layer raises for any infrastructure reason.
        """
        if not request.text.strip():
            raise ValueError("Text input is empty.")

        self._set_seed(request.seed)
        model = self._repo.get_model("turbo")
        ref_path = self._prep.preprocess(request.ref_audio_path)

        try:
            wav = model.generate(
                request.text,
                audio_prompt_path=ref_path,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.rep_penalty,
                min_p=request.min_p,
                norm_loudness=request.norm_loudness,
            )
        except AssertionError as exc:
            raise ValueError(
                f"Reference audio must be longer than 5 seconds: {exc}"
            ) from exc

        return AudioResult(
            sample_rate=model.sr,
            samples=_tensor_to_numpy(wav),
        )

    def generate_stream(self, request: TurboTTSRequest) -> Iterator[AudioResult]:
        """Generate audio sentence-by-sentence, yielding cumulative results.

        Raises:
            ValueError: if ``request.text`` is empty, or if the reference
                        audio is shorter than 5 seconds.
        """
        if not request.text.strip():
            raise ValueError("Text input is empty.")

        self._set_seed(request.seed)
        model = self._repo.get_model("turbo")
        ref_path = self._prep.preprocess(request.ref_audio_path)

        gen_kwargs = dict(
            audio_prompt_path=ref_path,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.rep_penalty,
            min_p=request.min_p,
            norm_loudness=request.norm_loudness,
        )

        sentences = split_sentences(request.text) or [request.text]
        buf: list[np.ndarray] = []

        for sentence in sentences:
            try:
                wav = model.generate(sentence, **gen_kwargs)
            except AssertionError as exc:
                raise ValueError(
                    f"Reference audio must be longer than 5 seconds: {exc}"
                ) from exc
            buf.append(_tensor_to_numpy(wav))
            yield AudioResult(
                sample_rate=model.sr,
                samples=np.concatenate(buf),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Multilingual TTS
# ──────────────────────────────────────────────────────────────────────────────

class MultilingualTTSService:
    """Domain service for ChatterboxMultilingualTTS generation (23 languages).

    Language handling:
        ``request.language`` may be a bare code (``"fr"``) or the full label
        from the UI dropdown (``"fr - French"``).  In either case only the
        token before the first space is forwarded to the model as
        ``language_id``.
    """

    def __init__(
        self,
        model_repo: IModelRepository,
        preprocessor: IAudioPreprocessor,
        seed_setter: Callable[[int], None] | None = None,
    ) -> None:
        self._repo = model_repo
        self._prep = preprocessor
        self._set_seed: Callable[[int], None] = seed_setter or (lambda _: None)

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_lang_code(language: str) -> str:
        """Return the bare two-letter code from a language string.

        Examples::

            "fr - French" → "fr"
            "en"          → "en"
        """
        return language.split(" ")[0]

    # ── public API ────────────────────────────────────────────────────────────

    def generate(self, request: MultilingualTTSRequest) -> AudioResult:
        """Generate a complete audio clip from *request* in one shot.

        Raises:
            ValueError: if ``request.text`` is empty or whitespace-only.
            RuntimeError: if the model layer raises for any infrastructure reason.
        """
        if not request.text.strip():
            raise ValueError("Text input is empty.")

        self._set_seed(request.seed)
        model = self._repo.get_model("multilingual")
        ref_path = self._prep.preprocess(request.ref_audio_path)
        lang_code = self._extract_lang_code(request.language)

        wav = model.generate(
            request.text,
            language_id=lang_code,
            audio_prompt_path=ref_path,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
            repetition_penalty=request.rep_penalty,
            min_p=request.min_p,
            top_p=request.top_p,
        )
        return AudioResult(
            sample_rate=model.sr,
            samples=_tensor_to_numpy(wav),
        )

    def generate_stream(self, request: MultilingualTTSRequest) -> Iterator[AudioResult]:
        """Generate audio sentence-by-sentence, yielding cumulative results.

        Raises:
            ValueError: if ``request.text`` is empty or whitespace-only.
        """
        if not request.text.strip():
            raise ValueError("Text input is empty.")

        self._set_seed(request.seed)
        model = self._repo.get_model("multilingual")
        ref_path = self._prep.preprocess(request.ref_audio_path)
        lang_code = self._extract_lang_code(request.language)

        gen_kwargs = dict(
            language_id=lang_code,
            audio_prompt_path=ref_path,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
            repetition_penalty=request.rep_penalty,
            min_p=request.min_p,
            top_p=request.top_p,
        )

        sentences = split_sentences(request.text) or [request.text]
        buf: list[np.ndarray] = []

        for sentence in sentences:
            wav = model.generate(sentence, **gen_kwargs)
            buf.append(_tensor_to_numpy(wav))
            yield AudioResult(
                sample_rate=model.sr,
                samples=np.concatenate(buf),
            )
