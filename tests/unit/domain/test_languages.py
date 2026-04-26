"""
tests/unit/domain/test_languages.py
=====================================
TDD — RED phase: tests for domain.languages

Run before implementation exists → all tests must FAIL initially.
Run after implementation → all tests must PASS.

Rules:
- No torch, gradio, chatterbox, psutil, or huggingface_hub imports.
- Pure constant validation only.
"""

from __future__ import annotations

import re

from domain.languages import (
    LANGUAGE_AUDIO_DEFAULTS,
    LANGUAGE_OPTIONS,
    PARA_TAGS,
    SAMPLE_TEXTS,
)

# ──────────────────────────────────────────────────────────────────────────────
# LANGUAGE_OPTIONS
# ──────────────────────────────────────────────────────────────────────────────


class TestLanguageOptions:
    def test_language_options_has_23_entries(self):
        assert len(LANGUAGE_OPTIONS) == 23

    def test_language_options_is_list(self):
        assert isinstance(LANGUAGE_OPTIONS, list)

    def test_all_entries_are_strings(self):
        for entry in LANGUAGE_OPTIONS:
            assert isinstance(entry, str), f"Expected str, got {type(entry)}: {entry!r}"

    def test_all_entries_have_separator(self):
        """Every entry must follow the '<code> - <Name>' pattern."""
        for entry in LANGUAGE_OPTIONS:
            assert " - " in entry, f"Entry {entry!r} does not contain ' - ' separator"

    def test_all_entries_have_two_letter_code(self):
        """The prefix before ' - ' must be exactly 2 lowercase letters."""
        pattern = re.compile(r"^[a-z]{2} - ")
        for entry in LANGUAGE_OPTIONS:
            assert pattern.match(entry), (
                f"Entry {entry!r} does not start with a two-letter ISO code"
            )

    def test_all_entries_have_non_empty_name(self):
        """The part after ' - ' must not be blank."""
        for entry in LANGUAGE_OPTIONS:
            _code, _, name = entry.partition(" - ")
            assert name.strip(), f"Entry {entry!r} has an empty language name"

    def test_language_options_contains_english(self):
        assert "en - English" in LANGUAGE_OPTIONS

    def test_language_options_contains_french(self):
        assert "fr - French" in LANGUAGE_OPTIONS

    def test_language_options_contains_chinese(self):
        assert "zh - Chinese" in LANGUAGE_OPTIONS

    def test_language_options_no_duplicates(self):
        assert len(LANGUAGE_OPTIONS) == len(set(LANGUAGE_OPTIONS)), (
            "LANGUAGE_OPTIONS contains duplicate entries"
        )

    def test_known_languages_present(self):
        """Spot-check that the expected 23 languages are all present."""
        expected = {
            "ar - Arabic",
            "da - Danish",
            "de - German",
            "el - Greek",
            "en - English",
            "es - Spanish",
            "fi - Finnish",
            "fr - French",
            "he - Hebrew",
            "hi - Hindi",
            "it - Italian",
            "ja - Japanese",
            "ko - Korean",
            "ms - Malay",
            "nl - Dutch",
            "no - Norwegian",
            "pl - Polish",
            "pt - Portuguese",
            "ru - Russian",
            "sv - Swedish",
            "sw - Swahili",
            "tr - Turkish",
            "zh - Chinese",
        }
        assert set(LANGUAGE_OPTIONS) == expected


# ──────────────────────────────────────────────────────────────────────────────
# SAMPLE_TEXTS
# ──────────────────────────────────────────────────────────────────────────────


class TestSampleTexts:
    def test_sample_texts_is_dict(self):
        assert isinstance(SAMPLE_TEXTS, dict)

    def test_all_language_options_have_sample_texts(self):
        """Every entry in LANGUAGE_OPTIONS must have a corresponding sample text."""
        for lang in LANGUAGE_OPTIONS:
            assert lang in SAMPLE_TEXTS, f"SAMPLE_TEXTS missing entry for {lang!r}"

    def test_sample_texts_count_matches_language_options(self):
        assert len(SAMPLE_TEXTS) == len(LANGUAGE_OPTIONS)

    def test_all_sample_texts_are_non_empty_strings(self):
        for lang, text in SAMPLE_TEXTS.items():
            assert isinstance(text, str), f"SAMPLE_TEXTS[{lang!r}] is not a str: {type(text)}"
            assert text.strip(), f"SAMPLE_TEXTS[{lang!r}] is empty"

    def test_sample_texts_keys_match_language_options(self):
        """Keys must be identical to LANGUAGE_OPTIONS entries — no extra or missing."""
        assert set(SAMPLE_TEXTS.keys()) == set(LANGUAGE_OPTIONS)

    def test_english_sample_text_is_in_english(self):
        """Basic sanity: English sample should use ASCII characters."""
        english_text = SAMPLE_TEXTS.get("en - English", "")
        assert english_text, "English sample text must not be empty"
        # ASCII-only check — real English should pass this
        assert (
            english_text.encode("ascii", errors="replace").decode("ascii").replace("?", "") != ""
        ), "English sample text appears to be empty after ASCII normalisation"

    def test_arabic_sample_text_contains_arabic_characters(self):
        """Arabic text must contain Arabic Unicode characters."""
        arabic_text = SAMPLE_TEXTS.get("ar - Arabic", "")
        assert arabic_text, "Arabic sample text must not be empty"
        has_arabic = any("\u0600" <= ch <= "\u06ff" for ch in arabic_text)
        assert has_arabic, "Arabic sample text does not contain Arabic Unicode characters"

    def test_chinese_sample_text_contains_cjk_characters(self):
        """Chinese text must contain CJK Unicode characters."""
        chinese_text = SAMPLE_TEXTS.get("zh - Chinese", "")
        assert chinese_text, "Chinese sample text must not be empty"
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in chinese_text)
        assert has_cjk, "Chinese sample text does not contain CJK characters"

    def test_japanese_sample_text_contains_japanese_characters(self):
        """Japanese text must contain Hiragana, Katakana, or CJK characters."""
        japanese_text = SAMPLE_TEXTS.get("ja - Japanese", "")
        assert japanese_text, "Japanese sample text must not be empty"
        has_japanese = any(
            "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in japanese_text
        )
        assert has_japanese, "Japanese sample text does not contain Japanese characters"


# ──────────────────────────────────────────────────────────────────────────────
# LANGUAGE_AUDIO_DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────


class TestLanguageAudioDefaults:
    def test_language_audio_defaults_is_dict(self):
        assert isinstance(LANGUAGE_AUDIO_DEFAULTS, dict)

    def test_all_language_options_have_audio_defaults(self):
        """
        Every language in LANGUAGE_OPTIONS must have a matching entry in
        LANGUAGE_AUDIO_DEFAULTS, keyed by its 2-letter ISO code.
        """
        for lang_option in LANGUAGE_OPTIONS:
            code = lang_option.split(" - ")[0]  # e.g. "fr"
            assert code in LANGUAGE_AUDIO_DEFAULTS, (
                f"LANGUAGE_AUDIO_DEFAULTS missing entry for code {code!r} "
                f"(from LANGUAGE_OPTIONS entry {lang_option!r})"
            )

    def test_audio_defaults_count_matches_language_options(self):
        assert len(LANGUAGE_AUDIO_DEFAULTS) == len(LANGUAGE_OPTIONS)

    def test_all_keys_are_two_letter_codes(self):
        code_pattern = re.compile(r"^[a-z]{2}$")
        for key in LANGUAGE_AUDIO_DEFAULTS:
            assert code_pattern.match(key), (
                f"LANGUAGE_AUDIO_DEFAULTS key {key!r} is not a 2-letter ISO code"
            )

    def test_language_audio_defaults_are_urls(self):
        """All audio default values must be valid HTTP/HTTPS URLs."""
        url_pattern = re.compile(r"^https?://")
        for code, url in LANGUAGE_AUDIO_DEFAULTS.items():
            assert isinstance(url, str), f"LANGUAGE_AUDIO_DEFAULTS[{code!r}] is not a string"
            assert url_pattern.match(url), (
                f"LANGUAGE_AUDIO_DEFAULTS[{code!r}] = {url!r} is not a valid URL"
            )

    def test_audio_defaults_are_gcs_urls(self):
        """All URLs should point to the Resemble AI Google Cloud Storage bucket."""
        gcs_prefix = "https://storage.googleapis.com/chatterbox-demo-samples/"
        for code, url in LANGUAGE_AUDIO_DEFAULTS.items():
            assert url.startswith(gcs_prefix), (
                f"LANGUAGE_AUDIO_DEFAULTS[{code!r}] = {url!r} "
                f"does not start with expected GCS prefix"
            )

    def test_audio_defaults_are_flac_files(self):
        """All audio defaults should reference FLAC audio files."""
        for code, url in LANGUAGE_AUDIO_DEFAULTS.items():
            assert url.endswith(".flac"), (
                f"LANGUAGE_AUDIO_DEFAULTS[{code!r}] = {url!r} is not a .flac file"
            )

    def test_english_audio_default_exists(self):
        assert "en" in LANGUAGE_AUDIO_DEFAULTS
        assert LANGUAGE_AUDIO_DEFAULTS["en"]

    def test_audio_defaults_no_duplicates(self):
        """Each language should have a unique audio file URL."""
        urls = list(LANGUAGE_AUDIO_DEFAULTS.values())
        assert len(urls) == len(set(urls)), "LANGUAGE_AUDIO_DEFAULTS contains duplicate URL values"


# ──────────────────────────────────────────────────────────────────────────────
# PARA_TAGS
# ──────────────────────────────────────────────────────────────────────────────


class TestParaTags:
    def test_para_tags_is_list(self):
        assert isinstance(PARA_TAGS, list)

    def test_para_tags_not_empty(self):
        assert len(PARA_TAGS) > 0, "PARA_TAGS must not be empty"

    def test_para_tags_have_brackets(self):
        """Every paralinguistic tag must be wrapped in square brackets."""
        for tag in PARA_TAGS:
            assert tag.startswith("[") and tag.endswith("]"), (
                f"Tag {tag!r} is not wrapped in square brackets"
            )

    def test_all_tags_are_strings(self):
        for tag in PARA_TAGS:
            assert isinstance(tag, str), f"Tag {tag!r} is not a string"

    def test_para_tags_no_duplicates(self):
        assert len(PARA_TAGS) == len(set(PARA_TAGS)), "PARA_TAGS contains duplicate entries"

    def test_para_tags_contain_laugh(self):
        assert "[laugh]" in PARA_TAGS

    def test_para_tags_contain_chuckle(self):
        assert "[chuckle]" in PARA_TAGS

    def test_para_tags_contain_sigh(self):
        assert "[sigh]" in PARA_TAGS

    def test_para_tags_contain_expected_set(self):
        expected = {
            "[laugh]",
            "[chuckle]",
            "[cough]",
            "[sigh]",
            "[gasp]",
            "[hmm]",
            "[clears throat]",
        }
        assert expected.issubset(set(PARA_TAGS)), (
            f"PARA_TAGS is missing expected tags: {expected - set(PARA_TAGS)}"
        )

    def test_para_tags_content_not_empty_inside_brackets(self):
        """Each tag must have non-empty content between its brackets."""
        for tag in PARA_TAGS:
            inner = tag[1:-1]  # strip leading '[' and trailing ']'
            assert inner.strip(), f"Tag {tag!r} has empty content inside brackets"


# ──────────────────────────────────────────────────────────────────────────────
# Cross-constant consistency
# ──────────────────────────────────────────────────────────────────────────────


class TestCrossConstantConsistency:
    def test_language_options_codes_match_audio_default_keys(self):
        """
        The set of 2-letter codes extracted from LANGUAGE_OPTIONS must exactly
        match the keys of LANGUAGE_AUDIO_DEFAULTS.
        """
        option_codes = {opt.split(" - ")[0] for opt in LANGUAGE_OPTIONS}
        default_keys = set(LANGUAGE_AUDIO_DEFAULTS.keys())
        assert option_codes == default_keys, (
            f"Mismatch between LANGUAGE_OPTIONS codes and LANGUAGE_AUDIO_DEFAULTS keys.\n"
            f"  Only in OPTIONS: {option_codes - default_keys}\n"
            f"  Only in DEFAULTS: {default_keys - option_codes}"
        )

    def test_language_options_codes_match_sample_text_keys(self):
        """
        The full LANGUAGE_OPTIONS entries must exactly match SAMPLE_TEXTS keys.
        """
        assert set(LANGUAGE_OPTIONS) == set(SAMPLE_TEXTS.keys()), (
            f"Mismatch between LANGUAGE_OPTIONS and SAMPLE_TEXTS keys.\n"
            f"  Only in OPTIONS: {set(LANGUAGE_OPTIONS) - set(SAMPLE_TEXTS.keys())}\n"
            f"  Only in SAMPLE_TEXTS: {set(SAMPLE_TEXTS.keys()) - set(LANGUAGE_OPTIONS)}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level purity check
# ──────────────────────────────────────────────────────────────────────────────


def test_languages_module_has_no_torch_import():
    """languages.py must not import torch — domain layer must be framework-free."""
    import importlib
    import sys

    mod_name = "domain.languages"
    mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
    assert "torch" not in vars(mod), "languages.py must not import torch"


def test_languages_module_has_no_gradio_import():
    """languages.py must not import gradio — domain layer must be framework-free."""
    import importlib
    import sys

    mod_name = "domain.languages"
    mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
    assert "gradio" not in vars(mod), "languages.py must not import gradio"
