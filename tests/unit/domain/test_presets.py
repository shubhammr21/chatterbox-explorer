"""
tests/unit/domain/test_presets.py
===================================
TDD — RED phase: tests for chatterbox_explorer.domain.presets

Run before implementation exists → all tests must FAIL initially.
Run after implementation → all tests must PASS.

Rules:
- No torch, gradio, chatterbox, psutil, or huggingface_hub imports.
- Pure constant and function validation only.
"""
from __future__ import annotations

import pytest

from chatterbox_explorer.domain.presets import (
    PRESET_TTS_NAMES,
    PRESET_TURBO_NAMES,
    PRESETS_TTS,
    PRESETS_TURBO,
    get_preset_tts,
    get_preset_turbo,
)


# ──────────────────────────────────────────────────────────────────────────────
# Counts
# ──────────────────────────────────────────────────────────────────────────────

class TestPresetCounts:
    def test_standard_presets_count(self):
        """PRESETS_TTS must contain exactly 10 presets."""
        assert len(PRESETS_TTS) == 10

    def test_turbo_presets_count(self):
        """PRESETS_TURBO must contain exactly 6 presets."""
        assert len(PRESETS_TURBO) == 6

    def test_preset_tts_names_count_matches_dict(self):
        assert len(PRESET_TTS_NAMES) == len(PRESETS_TTS)

    def test_preset_turbo_names_count_matches_dict(self):
        assert len(PRESET_TURBO_NAMES) == len(PRESETS_TURBO)


# ──────────────────────────────────────────────────────────────────────────────
# Name lists match dict keys exactly
# ──────────────────────────────────────────────────────────────────────────────

class TestPresetNameLists:
    def test_preset_tts_names_list_matches_dict_keys(self):
        """PRESET_TTS_NAMES must be exactly the keys of PRESETS_TTS (same order)."""
        assert PRESET_TTS_NAMES == list(PRESETS_TTS.keys())

    def test_preset_turbo_names_list_matches_dict_keys(self):
        """PRESET_TURBO_NAMES must be exactly the keys of PRESETS_TURBO (same order)."""
        assert PRESET_TURBO_NAMES == list(PRESETS_TURBO.keys())

    def test_preset_tts_names_is_list(self):
        assert isinstance(PRESET_TTS_NAMES, list)

    def test_preset_turbo_names_is_list(self):
        assert isinstance(PRESET_TURBO_NAMES, list)

    def test_preset_tts_names_are_strings(self):
        for name in PRESET_TTS_NAMES:
            assert isinstance(name, str), f"Expected str, got {type(name)}: {name!r}"

    def test_preset_turbo_names_are_strings(self):
        for name in PRESET_TURBO_NAMES:
            assert isinstance(name, str), f"Expected str, got {type(name)}: {name!r}"

    def test_preset_tts_names_no_duplicates(self):
        assert len(PRESET_TTS_NAMES) == len(set(PRESET_TTS_NAMES))

    def test_preset_turbo_names_no_duplicates(self):
        assert len(PRESET_TURBO_NAMES) == len(set(PRESET_TURBO_NAMES))


# ──────────────────────────────────────────────────────────────────────────────
# Default preset present in both dicts
# ──────────────────────────────────────────────────────────────────────────────

class TestDefaultPreset:
    def test_default_preset_exists_in_tts(self):
        assert "🎯 Default" in PRESETS_TTS, (
            "PRESETS_TTS must contain a '🎯 Default' preset"
        )

    def test_default_preset_exists_in_turbo(self):
        assert "🎯 Default" in PRESETS_TURBO, (
            "PRESETS_TURBO must contain a '🎯 Default' preset"
        )

    def test_default_is_first_tts_preset(self):
        """'🎯 Default' should be the first preset — it's the dropdown's initial value."""
        assert PRESET_TTS_NAMES[0] == "🎯 Default"

    def test_default_is_first_turbo_preset(self):
        assert PRESET_TURBO_NAMES[0] == "🎯 Default"


# ──────────────────────────────────────────────────────────────────────────────
# Required keys in every preset dict
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_PRESET_KEYS = {"params", "description", "rationale_md", "sample_text"}


class TestPresetRequiredKeys:
    def test_all_tts_presets_have_required_keys(self):
        for name, preset in PRESETS_TTS.items():
            missing = _REQUIRED_PRESET_KEYS - set(preset.keys())
            assert not missing, (
                f"TTS preset {name!r} is missing keys: {missing}"
            )

    def test_all_turbo_presets_have_required_keys(self):
        for name, preset in PRESETS_TURBO.items():
            missing = _REQUIRED_PRESET_KEYS - set(preset.keys())
            assert not missing, (
                f"Turbo preset {name!r} is missing keys: {missing}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# sample_text — non-empty strings
# ──────────────────────────────────────────────────────────────────────────────

class TestPresetSampleTexts:
    def test_all_tts_presets_have_sample_text(self):
        for name, preset in PRESETS_TTS.items():
            text = preset.get("sample_text", "")
            assert isinstance(text, str), (
                f"TTS preset {name!r}: sample_text is not a str"
            )
            assert text.strip(), (
                f"TTS preset {name!r}: sample_text is empty"
            )

    def test_all_turbo_presets_have_sample_text(self):
        for name, preset in PRESETS_TURBO.items():
            text = preset.get("sample_text", "")
            assert isinstance(text, str), (
                f"Turbo preset {name!r}: sample_text is not a str"
            )
            assert text.strip(), (
                f"Turbo preset {name!r}: sample_text is empty"
            )

    def test_tts_sample_texts_are_unique(self):
        """Each TTS preset should showcase a distinct use case."""
        texts = [p["sample_text"] for p in PRESETS_TTS.values()]
        assert len(texts) == len(set(texts)), (
            "PRESETS_TTS contains duplicate sample_text values"
        )

    def test_turbo_sample_texts_are_unique(self):
        texts = [p["sample_text"] for p in PRESETS_TURBO.values()]
        assert len(texts) == len(set(texts)), (
            "PRESETS_TURBO contains duplicate sample_text values"
        )


# ──────────────────────────────────────────────────────────────────────────────
# rationale_md — non-empty markdown strings
# ──────────────────────────────────────────────────────────────────────────────

class TestPresetRationaleMd:
    def test_all_tts_presets_have_rationale_md(self):
        for name, preset in PRESETS_TTS.items():
            md = preset.get("rationale_md", "")
            assert isinstance(md, str), (
                f"TTS preset {name!r}: rationale_md is not a str"
            )
            assert md.strip(), (
                f"TTS preset {name!r}: rationale_md is empty"
            )

    def test_all_turbo_presets_have_rationale_md(self):
        for name, preset in PRESETS_TURBO.items():
            md = preset.get("rationale_md", "")
            assert isinstance(md, str), (
                f"Turbo preset {name!r}: rationale_md is not a str"
            )
            assert md.strip(), (
                f"Turbo preset {name!r}: rationale_md is empty"
            )

    def test_tts_rationale_md_contains_markdown_table(self):
        """Each rationale should have a markdown param table for educational value."""
        for name, preset in PRESETS_TTS.items():
            md = preset.get("rationale_md", "")
            assert "|" in md, (
                f"TTS preset {name!r}: rationale_md appears to be missing a markdown table"
            )

    def test_turbo_rationale_md_contains_markdown_table(self):
        for name, preset in PRESETS_TURBO.items():
            md = preset.get("rationale_md", "")
            assert "|" in md, (
                f"Turbo preset {name!r}: rationale_md appears to be missing a markdown table"
            )


# ──────────────────────────────────────────────────────────────────────────────
# description — non-empty strings
# ──────────────────────────────────────────────────────────────────────────────

class TestPresetDescriptions:
    def test_all_tts_presets_have_description(self):
        for name, preset in PRESETS_TTS.items():
            desc = preset.get("description", "")
            assert isinstance(desc, str), (
                f"TTS preset {name!r}: description is not a str"
            )
            assert desc.strip(), (
                f"TTS preset {name!r}: description is empty"
            )

    def test_all_turbo_presets_have_description(self):
        for name, preset in PRESETS_TURBO.items():
            desc = preset.get("description", "")
            assert isinstance(desc, str), (
                f"Turbo preset {name!r}: description is not a str"
            )
            assert desc.strip(), (
                f"Turbo preset {name!r}: description is empty"
            )


# ──────────────────────────────────────────────────────────────────────────────
# TTS preset params validation
# ──────────────────────────────────────────────────────────────────────────────

_TTS_REQUIRED_PARAM_KEYS = {
    "exaggeration", "cfg_weight", "temperature", "rep_penalty", "min_p", "top_p"
}

# Validated ranges: permissive enough for all 10 presets, strict enough to
# catch obviously wrong values (e.g. someone swapping exaggeration with top_k).
_TTS_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "exaggeration": (0.25, 2.0),
    "cfg_weight":   (0.0,  1.0),
    "temperature":  (0.05, 5.0),
    "rep_penalty":  (1.0,  3.0),
    "min_p":        (0.0,  1.0),
    "top_p":        (0.0,  1.0),
}


class TestTTSPresetParams:
    def test_all_standard_presets_have_required_param_keys(self):
        for name, preset in PRESETS_TTS.items():
            params = preset.get("params", {})
            missing = _TTS_REQUIRED_PARAM_KEYS - set(params.keys())
            assert not missing, (
                f"TTS preset {name!r} params is missing keys: {missing}"
            )

    def test_all_standard_preset_params_in_range(self):
        """All numeric TTS params must fall within their validated ranges."""
        for name, preset in PRESETS_TTS.items():
            params = preset["params"]
            for key, (lo, hi) in _TTS_PARAM_RANGES.items():
                value = params[key]
                assert lo <= value <= hi, (
                    f"TTS preset {name!r}: {key}={value} is outside [{lo}, {hi}]"
                )

    def test_default_tts_preset_params_are_correct(self):
        """Spot-check the '🎯 Default' preset params against the documented defaults."""
        params = PRESETS_TTS["🎯 Default"]["params"]
        assert params["exaggeration"] == pytest.approx(0.5)
        assert params["cfg_weight"] == pytest.approx(0.5)
        assert params["temperature"] == pytest.approx(0.8)
        assert params["rep_penalty"] == pytest.approx(1.2)
        assert params["min_p"] == pytest.approx(0.05)
        assert params["top_p"] == pytest.approx(1.0)

    def test_meditation_preset_has_minimum_exaggeration(self):
        """Meditation preset must use exaggeration=0.25 (documented floor value)."""
        params = PRESETS_TTS["🧘 Meditation / ASMR"]["params"]
        assert params["exaggeration"] == pytest.approx(0.25)

    def test_experimental_preset_has_high_exaggeration(self):
        """Experimental preset must use exaggeration=1.0."""
        params = PRESETS_TTS["🔬 Experimental"]["params"]
        assert params["exaggeration"] == pytest.approx(1.0)

    def test_dramatic_preset_cfg_weight(self):
        """Dramatic preset must use cfg_weight=0.30 (documented README recommendation)."""
        params = PRESETS_TTS["🎭 Dramatic"]["params"]
        assert params["cfg_weight"] == pytest.approx(0.30)

    def test_all_tts_params_are_floats(self):
        for name, preset in PRESETS_TTS.items():
            for key, value in preset["params"].items():
                assert isinstance(value, (int, float)), (
                    f"TTS preset {name!r}: param {key!r} = {value!r} is not numeric"
                )

    def test_known_tts_preset_names_present(self):
        expected_names = {
            "🎯 Default", "📚 Audiobook", "📰 News Broadcast", "💬 Conversational",
            "🎭 Dramatic", "📣 Advertisement", "🎓 E-Learning", "🎮 Game Character",
            "🧘 Meditation / ASMR", "🔬 Experimental",
        }
        assert set(PRESETS_TTS.keys()) == expected_names


# ──────────────────────────────────────────────────────────────────────────────
# Turbo preset params validation
# ──────────────────────────────────────────────────────────────────────────────

_TURBO_REQUIRED_PARAM_KEYS = {
    "temperature", "top_k", "top_p", "rep_penalty", "min_p", "norm_loudness"
}

_TURBO_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "temperature":  (0.05, 2.0),
    "top_k":        (1,    2000),
    "top_p":        (0.0,  1.0),
    "rep_penalty":  (1.0,  3.0),
    "min_p":        (0.0,  1.0),
}


class TestTurboPresetParams:
    def test_all_turbo_presets_have_required_param_keys(self):
        for name, preset in PRESETS_TURBO.items():
            params = preset.get("params", {})
            missing = _TURBO_REQUIRED_PARAM_KEYS - set(params.keys())
            assert not missing, (
                f"Turbo preset {name!r} params is missing keys: {missing}"
            )

    def test_all_turbo_preset_params_in_range(self):
        """All numeric Turbo params must fall within their validated ranges."""
        for name, preset in PRESETS_TURBO.items():
            params = preset["params"]
            for key, (lo, hi) in _TURBO_PARAM_RANGES.items():
                value = params[key]
                assert lo <= value <= hi, (
                    f"Turbo preset {name!r}: {key}={value} is outside [{lo}, {hi}]"
                )

    def test_all_turbo_norm_loudness_are_bool(self):
        """norm_loudness must be a boolean in every Turbo preset."""
        for name, preset in PRESETS_TURBO.items():
            value = preset["params"]["norm_loudness"]
            assert isinstance(value, bool), (
                f"Turbo preset {name!r}: norm_loudness={value!r} is not a bool"
            )

    def test_default_turbo_preset_params_are_correct(self):
        """Spot-check '🎯 Default' Turbo preset."""
        params = PRESETS_TURBO["🎯 Default"]["params"]
        assert params["temperature"] == pytest.approx(0.80)
        assert params["top_k"] == 1000
        assert params["top_p"] == pytest.approx(0.95)
        assert params["rep_penalty"] == pytest.approx(1.20)
        assert params["min_p"] == pytest.approx(0.0)
        assert params["norm_loudness"] is True

    def test_character_npc_preset_has_max_top_k(self):
        """'🧙 Character / NPC' must use top_k=2000 (documented maximum)."""
        params = PRESETS_TURBO["🧙 Character / NPC"]["params"]
        assert params["top_k"] == 2000

    def test_ivr_preset_has_low_top_k(self):
        """'📞 IVR / Max Reliable' must use top_k=80 (near-greedy for reliability)."""
        params = PRESETS_TURBO["📞 IVR / Max Reliable"]["params"]
        assert params["top_k"] == 80

    def test_character_npc_norm_loudness_is_false(self):
        """Character / NPC turns off norm_loudness to preserve dynamic range."""
        params = PRESETS_TURBO["🧙 Character / NPC"]["params"]
        assert params["norm_loudness"] is False

    def test_turbo_presets_do_not_have_exaggeration(self):
        """Turbo model does not support exaggeration — must NOT be in params."""
        for name, preset in PRESETS_TURBO.items():
            assert "exaggeration" not in preset["params"], (
                f"Turbo preset {name!r} incorrectly contains 'exaggeration' param"
            )

    def test_turbo_presets_do_not_have_cfg_weight(self):
        """Turbo model does not support cfg_weight — must NOT be in params."""
        for name, preset in PRESETS_TURBO.items():
            assert "cfg_weight" not in preset["params"], (
                f"Turbo preset {name!r} incorrectly contains 'cfg_weight' param"
            )

    def test_all_top_k_values_are_integers(self):
        for name, preset in PRESETS_TURBO.items():
            top_k = preset["params"]["top_k"]
            assert isinstance(top_k, int), (
                f"Turbo preset {name!r}: top_k={top_k!r} must be an int"
            )

    def test_known_turbo_preset_names_present(self):
        expected_names = {
            "🎯 Default", "🤖 Voice Agent", "🎙️ Podcast Host",
            "🧙 Character / NPC", "📻 Radio / Promo", "📞 IVR / Max Reliable",
        }
        assert set(PRESETS_TURBO.keys()) == expected_names


# ──────────────────────────────────────────────────────────────────────────────
# get_preset_tts
# ──────────────────────────────────────────────────────────────────────────────

class TestGetPresetTts:
    def test_returns_correct_preset_for_known_name(self):
        result = get_preset_tts("🎯 Default")
        assert result is not None
        assert result == PRESETS_TTS["🎯 Default"]

    def test_returns_none_for_unknown_name(self):
        result = get_preset_tts("🚀 Nonexistent Preset")
        assert result is None

    def test_returns_none_for_empty_string(self):
        result = get_preset_tts("")
        assert result is None

    def test_returns_dict_type(self):
        result = get_preset_tts("📚 Audiobook")
        assert isinstance(result, dict)

    def test_returns_correct_preset_for_each_name(self):
        """Every name in PRESET_TTS_NAMES must resolve to the correct preset."""
        for name in PRESET_TTS_NAMES:
            result = get_preset_tts(name)
            assert result is not None, f"get_preset_tts({name!r}) returned None"
            assert result == PRESETS_TTS[name]

    def test_returns_none_for_turbo_only_preset_name(self):
        """A Turbo-only preset name must return None from get_preset_tts."""
        result = get_preset_tts("🤖 Voice Agent")
        assert result is None

    def test_case_sensitive(self):
        """Lookup must be case-sensitive — partial matches must return None."""
        result = get_preset_tts("default")
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# get_preset_turbo
# ──────────────────────────────────────────────────────────────────────────────

class TestGetPresetTurbo:
    def test_returns_correct_preset_for_known_name(self):
        result = get_preset_turbo("🎯 Default")
        assert result is not None
        assert result == PRESETS_TURBO["🎯 Default"]

    def test_returns_none_for_unknown_name(self):
        result = get_preset_turbo("🚀 Nonexistent Preset")
        assert result is None

    def test_returns_none_for_empty_string(self):
        result = get_preset_turbo("")
        assert result is None

    def test_returns_dict_type(self):
        result = get_preset_turbo("🤖 Voice Agent")
        assert isinstance(result, dict)

    def test_returns_correct_preset_for_each_name(self):
        """Every name in PRESET_TURBO_NAMES must resolve to the correct preset."""
        for name in PRESET_TURBO_NAMES:
            result = get_preset_turbo(name)
            assert result is not None, f"get_preset_turbo({name!r}) returned None"
            assert result == PRESETS_TURBO[name]

    def test_returns_none_for_tts_only_preset_name(self):
        """A TTS-only preset name must return None from get_preset_turbo."""
        result = get_preset_turbo("📚 Audiobook")
        assert result is None

    def test_case_sensitive(self):
        result = get_preset_turbo("default")
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# Module-level purity check
# ──────────────────────────────────────────────────────────────────────────────

def test_presets_module_has_no_torch_import():
    """presets.py must not import torch — domain layer must be framework-free."""
    import importlib
    import sys

    mod_name = "chatterbox_explorer.domain.presets"
    mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
    assert "torch" not in vars(mod), "presets.py must not import torch"


def test_presets_module_has_no_gradio_import():
    """presets.py must not import gradio — domain layer must be framework-free."""
    import importlib
    import sys

    mod_name = "chatterbox_explorer.domain.presets"
    mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
    assert "gradio" not in vars(mod), "presets.py must not import gradio"


def test_presets_module_has_no_chatterbox_import():
    """presets.py must not import chatterbox — domain layer must be framework-free."""
    import importlib
    import sys

    mod_name = "chatterbox_explorer.domain.presets"
    mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
    assert "chatterbox" not in vars(mod), "presets.py must not import chatterbox"
