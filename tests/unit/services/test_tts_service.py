"""
tests/unit/services/test_tts_service.py
=========================================
Unit tests for TTSService, TurboTTSService, MultilingualTTSService,
and the split_sentences helper.

All model I/O is faked via the fixtures in tests/conftest.py — no real
Chatterbox weights are loaded.  torch IS required at test-time (via the
mock_wav_tensor fixture) but is NOT imported by the service modules under test.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from chatterbox_explorer.domain.models import (
    AudioResult,
    MultilingualTTSRequest,
    TTSRequest,
    TurboTTSRequest,
)
from chatterbox_explorer.services.tts import (
    MultilingualTTSService,
    TTSService,
    TurboTTSService,
    split_sentences,
)


# ──────────────────────────────────────────────────────────────────────────────
# split_sentences — pure function, no fixtures needed
# ──────────────────────────────────────────────────────────────────────────────

class TestSplitSentences:
    def test_split_sentences_basic(self):
        result = split_sentences("A. B. C.")
        assert result == ["A.", "B.", "C."]

    def test_split_sentences_single(self):
        """Text with no sentence-ending punctuation returns it as one element."""
        result = split_sentences("text")
        assert result == ["text"]

    def test_split_sentences_empty(self):
        result = split_sentences("")
        assert result == []

    def test_split_sentences_question_mark(self):
        result = split_sentences("Hi? Yes!")
        assert result == ["Hi?", "Yes!"]

    def test_split_sentences_whitespace_only(self):
        result = split_sentences("   ")
        assert result == []

    def test_split_sentences_mixed_punctuation(self):
        result = split_sentences("Hello! How are you? I am fine.")
        assert result == ["Hello!", "How are you?", "I am fine."]

    def test_split_sentences_preserves_trailing_punctuation(self):
        result = split_sentences("First. Second.")
        assert result[0] == "First."
        assert result[1] == "Second."


# ──────────────────────────────────────────────────────────────────────────────
# TTSService
# ──────────────────────────────────────────────────────────────────────────────

class TestTTSService:
    # ── generate (non-streaming) ──────────────────────────────────────────────

    def test_generate_raises_on_empty_text(self, mock_model_repo, mock_preprocessor):
        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(text="")
        with pytest.raises(ValueError, match="empty"):
            svc.generate(req)

    def test_generate_raises_on_whitespace_text(self, mock_model_repo, mock_preprocessor):
        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(text="   ")
        with pytest.raises(ValueError):
            svc.generate(req)

    def test_generate_calls_preprocessor_with_ref_path(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(text="Hello.", ref_audio_path="/tmp/ref.wav")
        svc.generate(req)
        mock_preprocessor.preprocess.assert_called_once_with("/tmp/ref.wav")

    def test_generate_calls_preprocessor_with_none_when_no_ref(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(text="Hello.")
        svc.generate(req)
        mock_preprocessor.preprocess.assert_called_once_with(None)

    def test_generate_calls_model_with_correct_kwargs(
        self, mock_model_repo, mock_preprocessor, mock_model
    ):
        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(
            text="Hello world.",
            exaggeration=0.7,
            cfg_weight=0.3,
            temperature=0.9,
            rep_penalty=1.5,
            min_p=0.1,
            top_p=0.95,
        )
        svc.generate(req)

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["exaggeration"] == 0.7
        assert call_kwargs["cfg_weight"] == 0.3
        assert call_kwargs["temperature"] == 0.9
        assert call_kwargs["repetition_penalty"] == 1.5
        assert call_kwargs["min_p"] == 0.1
        assert call_kwargs["top_p"] == 0.95

    def test_generate_passes_text_as_positional(
        self, mock_model_repo, mock_preprocessor, mock_model
    ):
        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(text="Hello world.")
        svc.generate(req)
        positional = mock_model.generate.call_args[0]
        assert positional[0] == "Hello world."

    def test_generate_returns_audio_result(self, mock_model_repo, mock_preprocessor):
        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(text="Hello.")
        result = svc.generate(req)

        assert isinstance(result, AudioResult)
        assert result.sample_rate == 24000
        assert result.samples.shape == (24000,)

    def test_generate_returns_float32_samples(self, mock_model_repo, mock_preprocessor):
        import numpy as np

        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(text="Hello.")
        result = svc.generate(req)
        assert result.samples.dtype == np.float32

    def test_generate_calls_seed_setter(self, mock_model_repo, mock_preprocessor):
        seed_mock = MagicMock()
        svc = TTSService(mock_model_repo, mock_preprocessor, seed_setter=seed_mock)
        req = TTSRequest(text="Hello.", seed=42)
        svc.generate(req)
        seed_mock.assert_called_once_with(42)

    def test_generate_uses_noop_seed_setter_by_default(
        self, mock_model_repo, mock_preprocessor
    ):
        """No exception when seed_setter is omitted."""
        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(text="Hello.", seed=7)
        # Should not raise
        svc.generate(req)

    # ── generate_stream ───────────────────────────────────────────────────────

    def test_generate_stream_raises_on_empty_text(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(text="")
        with pytest.raises(ValueError, match="empty"):
            # consume the generator to trigger the error
            list(svc.generate_stream(req))

    def test_generate_stream_yields_for_each_sentence(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(text="Hello. World.")
        results = list(svc.generate_stream(req))
        assert len(results) == 2

    def test_generate_stream_yields_cumulative_audio(
        self, mock_model_repo, mock_preprocessor
    ):
        """Second yield must contain more samples than the first."""
        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(text="Hello. World.")
        results = list(svc.generate_stream(req))
        assert len(results[1].samples) > len(results[0].samples)

    def test_generate_stream_all_yields_are_audio_results(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(text="One. Two. Three.")
        results = list(svc.generate_stream(req))
        assert all(isinstance(r, AudioResult) for r in results)
        assert all(r.sample_rate == 24000 for r in results)

    def test_generate_stream_single_sentence_yields_once(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = TTSService(mock_model_repo, mock_preprocessor)
        req = TTSRequest(text="No punctuation here")
        results = list(svc.generate_stream(req))
        assert len(results) == 1


# ──────────────────────────────────────────────────────────────────────────────
# TurboTTSService
# ──────────────────────────────────────────────────────────────────────────────

class TestTurboTTSService:
    def test_turbo_generate_raises_on_empty_text(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = TurboTTSService(mock_model_repo, mock_preprocessor)
        req = TurboTTSRequest(text="")
        with pytest.raises(ValueError):
            svc.generate(req)

    def test_turbo_generate_calls_model_with_turbo_kwargs(
        self, mock_model_repo, mock_preprocessor, mock_model
    ):
        svc = TurboTTSService(mock_model_repo, mock_preprocessor)
        req = TurboTTSRequest(
            text="Hello.",
            top_k=500,
            top_p=0.9,
            norm_loudness=False,
            temperature=0.7,
            rep_penalty=1.1,
            min_p=0.01,
        )
        svc.generate(req)

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["top_k"] == 500
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["norm_loudness"] is False
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["repetition_penalty"] == 1.1
        assert call_kwargs["min_p"] == 0.01

    def test_turbo_generate_uses_turbo_model_key(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = TurboTTSService(mock_model_repo, mock_preprocessor)
        req = TurboTTSRequest(text="Hello.")
        svc.generate(req)
        mock_model_repo.get_model.assert_called_with("turbo")

    def test_turbo_assertion_error_becomes_value_error(
        self, mock_model_repo, mock_preprocessor, mock_model
    ):
        """AssertionError from the 5-second ref-audio check must become ValueError."""
        mock_model.generate.side_effect = AssertionError("audio shorter than 5 s")
        svc = TurboTTSService(mock_model_repo, mock_preprocessor)
        req = TurboTTSRequest(text="Hello.")
        with pytest.raises(ValueError, match="5 seconds"):
            svc.generate(req)

    def test_turbo_generate_returns_audio_result(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = TurboTTSService(mock_model_repo, mock_preprocessor)
        req = TurboTTSRequest(text="Hello.")
        result = svc.generate(req)
        assert isinstance(result, AudioResult)
        assert result.sample_rate == 24000

    def test_turbo_generate_calls_seed_setter(self, mock_model_repo, mock_preprocessor):
        seed_mock = MagicMock()
        svc = TurboTTSService(mock_model_repo, mock_preprocessor, seed_setter=seed_mock)
        req = TurboTTSRequest(text="Hello.", seed=99)
        svc.generate(req)
        seed_mock.assert_called_once_with(99)

    def test_turbo_stream_raises_on_empty_text(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = TurboTTSService(mock_model_repo, mock_preprocessor)
        req = TurboTTSRequest(text="")
        with pytest.raises(ValueError):
            list(svc.generate_stream(req))

    def test_turbo_stream_assertion_error_becomes_value_error(
        self, mock_model_repo, mock_preprocessor, mock_model
    ):
        mock_model.generate.side_effect = AssertionError("ref too short")
        svc = TurboTTSService(mock_model_repo, mock_preprocessor)
        req = TurboTTSRequest(text="Hello.")
        with pytest.raises(ValueError, match="5 seconds"):
            list(svc.generate_stream(req))


# ──────────────────────────────────────────────────────────────────────────────
# MultilingualTTSService
# ──────────────────────────────────────────────────────────────────────────────

class TestMultilingualTTSService:
    def test_multilingual_generate_raises_on_empty_text(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = MultilingualTTSService(mock_model_repo, mock_preprocessor)
        req = MultilingualTTSRequest(text="")
        with pytest.raises(ValueError):
            svc.generate(req)

    def test_multilingual_generate_passes_language_id(
        self, mock_model_repo, mock_preprocessor, mock_model
    ):
        """Language string 'fr - French' must be trimmed to 'fr' before passing."""
        svc = MultilingualTTSService(mock_model_repo, mock_preprocessor)
        req = MultilingualTTSRequest(text="Bonjour.", language="fr - French")
        svc.generate(req)

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["language_id"] == "fr"

    def test_multilingual_generate_passes_bare_language_code(
        self, mock_model_repo, mock_preprocessor, mock_model
    ):
        """A bare two-letter code such as 'en' must be passed through unchanged."""
        svc = MultilingualTTSService(mock_model_repo, mock_preprocessor)
        req = MultilingualTTSRequest(text="Hello.", language="en")
        svc.generate(req)

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["language_id"] == "en"

    def test_multilingual_generate_uses_multilingual_model_key(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = MultilingualTTSService(mock_model_repo, mock_preprocessor)
        req = MultilingualTTSRequest(text="Hello.")
        svc.generate(req)
        mock_model_repo.get_model.assert_called_with("multilingual")

    def test_multilingual_generate_calls_model_with_correct_kwargs(
        self, mock_model_repo, mock_preprocessor, mock_model
    ):
        svc = MultilingualTTSService(mock_model_repo, mock_preprocessor)
        req = MultilingualTTSRequest(
            text="Hola.",
            language="es",
            exaggeration=0.6,
            cfg_weight=0.4,
            temperature=0.75,
            rep_penalty=2.5,
            min_p=0.05,
            top_p=0.98,
        )
        svc.generate(req)

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["exaggeration"] == 0.6
        assert call_kwargs["cfg_weight"] == 0.4
        assert call_kwargs["temperature"] == 0.75
        assert call_kwargs["repetition_penalty"] == 2.5
        assert call_kwargs["min_p"] == 0.05
        assert call_kwargs["top_p"] == 0.98

    def test_multilingual_generate_returns_audio_result(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = MultilingualTTSService(mock_model_repo, mock_preprocessor)
        req = MultilingualTTSRequest(text="Hello.")
        result = svc.generate(req)
        assert isinstance(result, AudioResult)
        assert result.sample_rate == 24000

    def test_multilingual_generate_stream_yields_cumulative(
        self, mock_model_repo, mock_preprocessor
    ):
        """Each stream yield accumulates audio; second result must be longer."""
        svc = MultilingualTTSService(mock_model_repo, mock_preprocessor)
        req = MultilingualTTSRequest(text="Bonjour. Au revoir.", language="fr")
        results = list(svc.generate_stream(req))

        assert len(results) == 2
        assert len(results[1].samples) > len(results[0].samples)

    def test_multilingual_generate_stream_raises_on_empty_text(
        self, mock_model_repo, mock_preprocessor
    ):
        svc = MultilingualTTSService(mock_model_repo, mock_preprocessor)
        req = MultilingualTTSRequest(text="")
        with pytest.raises(ValueError):
            list(svc.generate_stream(req))

    def test_multilingual_generate_calls_seed_setter(
        self, mock_model_repo, mock_preprocessor
    ):
        seed_mock = MagicMock()
        svc = MultilingualTTSService(
            mock_model_repo, mock_preprocessor, seed_setter=seed_mock
        )
        req = MultilingualTTSRequest(text="Hello.", seed=123)
        svc.generate(req)
        seed_mock.assert_called_once_with(123)
