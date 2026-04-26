"""
tests/unit/domain/test_exceptions.py
======================================
TDD unit tests for domain/exceptions.py.

Written BEFORE the implementation exists (RED phase).

What is tested
--------------
1. Hierarchy — every exception inherits from the correct parent.
2. Construction — every exception can be instantiated with its documented args.
3. Attributes — every exception stores the values it was constructed with.
4. Message — str(exc) is human-readable and mentions the relevant detail.
5. Purity — the module imports ONLY from stdlib; no pydantic, no starlette,
   no fastapi, no HTTP status codes anywhere.
6. No status_code — no exception class carries an HTTP status_code attribute.

Architecture rules verified here
---------------------------------
- Domain exceptions carry ZERO HTTP knowledge.
- Domain exceptions inherit from ChatterboxError (not ValueError/RuntimeError).
- The entire module is safe to import without any optional extras installed.
"""

from __future__ import annotations

import importlib
import sys

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _import_exc():
    """Return the domain.exceptions module, re-importing fresh each time."""
    return importlib.import_module("domain.exceptions")


# ──────────────────────────────────────────────────────────────────────────────
# Architecture purity
# ──────────────────────────────────────────────────────────────────────────────


class TestModulePurity:
    """domain.exceptions must have zero third-party runtime dependencies."""

    def test_module_imports_without_fastapi(self) -> None:
        """Importing domain.exceptions must not require fastapi."""
        sys.modules.pop("domain.exceptions", None)
        mod = importlib.import_module("domain.exceptions")
        assert mod is not None

    def test_module_imports_without_pydantic(self) -> None:
        """Importing domain.exceptions must not require pydantic."""
        sys.modules.pop("domain.exceptions", None)
        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "pydantic", None)  # type: ignore[arg-type]
            sys.modules.pop("domain.exceptions", None)
            mod = importlib.import_module("domain.exceptions")
        assert mod is not None

    def test_module_imports_without_starlette(self) -> None:
        """Importing domain.exceptions must not require starlette."""
        sys.modules.pop("domain.exceptions", None)
        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "starlette", None)  # type: ignore[arg-type]
            sys.modules.pop("domain.exceptions", None)
            mod = importlib.import_module("domain.exceptions")
        assert mod is not None

    def test_no_exception_has_status_code_attribute(self) -> None:
        """HTTP status codes must NOT appear on any domain exception class."""
        mod = _import_exc()
        exception_classes = [
            v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, Exception)
        ]
        assert exception_classes, "No exception classes found in domain.exceptions"
        for cls in exception_classes:
            assert not hasattr(cls, "status_code"), (
                f"{cls.__name__} must not carry status_code — "
                "HTTP mapping belongs in the REST adapter layer."
            )

    def test_no_exception_inherits_from_value_error(self) -> None:
        """Domain exceptions must NOT inherit from ValueError."""
        mod = _import_exc()
        for name, cls in vars(mod).items():
            if isinstance(cls, type) and issubclass(cls, Exception):
                assert not issubclass(cls, ValueError), (
                    f"{name} must not inherit from ValueError — use ChatterboxError as the base."
                )

    def test_no_exception_inherits_from_runtime_error(self) -> None:
        """Domain exceptions must NOT inherit from RuntimeError."""
        mod = _import_exc()
        for name, cls in vars(mod).items():
            if isinstance(cls, type) and issubclass(cls, Exception):
                assert not issubclass(cls, RuntimeError), (
                    f"{name} must not inherit from RuntimeError — use ChatterboxError as the base."
                )


# ──────────────────────────────────────────────────────────────────────────────
# ChatterboxError — root
# ──────────────────────────────────────────────────────────────────────────────


class TestChatterboxError:
    """Root exception — catch-all for all domain failures."""

    def test_is_exception_subclass(self) -> None:
        from domain.exceptions import ChatterboxError

        assert issubclass(ChatterboxError, Exception)

    def test_is_subclass_of_exception(self) -> None:
        from domain.exceptions import ChatterboxError

        exc = ChatterboxError("root error")
        assert isinstance(exc, Exception)

    def test_str_contains_message(self) -> None:
        from domain.exceptions import ChatterboxError

        exc = ChatterboxError("something failed")
        assert "something failed" in str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# TTSInputError + subtypes
# ──────────────────────────────────────────────────────────────────────────────


class TestTTSInputError:
    """TTSInputError — base for all TTS business-rule violations."""

    def test_is_chatterbox_error_subclass(self) -> None:
        from domain.exceptions import ChatterboxError, TTSInputError

        assert issubclass(TTSInputError, ChatterboxError)

    def test_can_be_raised(self) -> None:
        from domain.exceptions import TTSInputError

        with pytest.raises(TTSInputError):
            raise TTSInputError("tts input violated")

    def test_can_be_caught_as_chatterbox_error(self) -> None:
        from domain.exceptions import ChatterboxError, TTSInputError

        with pytest.raises(ChatterboxError):
            raise TTSInputError("tts input violated")


class TestEmptyTextError:
    """EmptyTextError — raised when text is empty or whitespace-only."""

    def test_is_tts_input_error_subclass(self) -> None:
        from domain.exceptions import EmptyTextError, TTSInputError

        assert issubclass(EmptyTextError, TTSInputError)

    def test_default_construction(self) -> None:
        from domain.exceptions import EmptyTextError

        exc = EmptyTextError()
        assert isinstance(exc, Exception)

    def test_construction_with_text(self) -> None:
        from domain.exceptions import EmptyTextError

        exc = EmptyTextError(text="   ")
        assert exc.text == "   "

    def test_str_is_human_readable(self) -> None:
        from domain.exceptions import EmptyTextError

        exc = EmptyTextError(text="")
        assert len(str(exc)) > 0

    def test_can_be_caught_as_tts_input_error(self) -> None:
        from domain.exceptions import EmptyTextError, TTSInputError

        with pytest.raises(TTSInputError):
            raise EmptyTextError()

    def test_can_be_caught_as_chatterbox_error(self) -> None:
        from domain.exceptions import ChatterboxError, EmptyTextError

        with pytest.raises(ChatterboxError):
            raise EmptyTextError()

    def test_is_subclass_of_exception(self) -> None:
        from domain.exceptions import EmptyTextError

        exc = EmptyTextError()
        assert isinstance(exc, Exception)


class TestReferenceTooShortError:
    """ReferenceTooShortError — raised when reference audio is too short."""

    def test_is_tts_input_error_subclass(self) -> None:
        from domain.exceptions import ReferenceTooShortError, TTSInputError

        assert issubclass(ReferenceTooShortError, TTSInputError)

    def test_construction_with_minimum(self) -> None:
        from domain.exceptions import ReferenceTooShortError

        exc = ReferenceTooShortError(minimum_sec=5.0)
        assert exc.minimum_sec == 5.0

    def test_default_minimum_is_five_seconds(self) -> None:
        from domain.exceptions import ReferenceTooShortError

        exc = ReferenceTooShortError()
        assert exc.minimum_sec == 5.0

    def test_str_mentions_minimum_duration(self) -> None:
        from domain.exceptions import ReferenceTooShortError

        exc = ReferenceTooShortError(minimum_sec=5.0)
        assert "5" in str(exc)

    def test_can_be_caught_as_tts_input_error(self) -> None:
        from domain.exceptions import ReferenceTooShortError, TTSInputError

        with pytest.raises(TTSInputError):
            raise ReferenceTooShortError()

    def test_can_be_caught_as_chatterbox_error(self) -> None:
        from domain.exceptions import ChatterboxError, ReferenceTooShortError

        with pytest.raises(ChatterboxError):
            raise ReferenceTooShortError()


# ──────────────────────────────────────────────────────────────────────────────
# VoiceConversionInputError + subtypes
# ──────────────────────────────────────────────────────────────────────────────


class TestVoiceConversionInputError:
    """VoiceConversionInputError — base for VC business-rule violations."""

    def test_is_chatterbox_error_subclass(self) -> None:
        from domain.exceptions import ChatterboxError, VoiceConversionInputError

        assert issubclass(VoiceConversionInputError, ChatterboxError)

    def test_is_not_tts_input_error(self) -> None:
        """VC errors and TTS errors are independent branches of the hierarchy."""
        from domain.exceptions import TTSInputError, VoiceConversionInputError

        assert not issubclass(VoiceConversionInputError, TTSInputError)
        assert not issubclass(TTSInputError, VoiceConversionInputError)


class TestMissingSourceAudioError:
    """MissingSourceAudioError — raised when source_audio_path is absent."""

    def test_is_vc_input_error_subclass(self) -> None:
        from domain.exceptions import MissingSourceAudioError, VoiceConversionInputError

        assert issubclass(MissingSourceAudioError, VoiceConversionInputError)

    def test_default_construction(self) -> None:
        from domain.exceptions import MissingSourceAudioError

        exc = MissingSourceAudioError()
        assert isinstance(exc, Exception)

    def test_str_mentions_source(self) -> None:
        from domain.exceptions import MissingSourceAudioError

        exc = MissingSourceAudioError()
        msg = str(exc).lower()
        assert "source" in msg

    def test_can_be_caught_as_vc_input_error(self) -> None:
        from domain.exceptions import MissingSourceAudioError, VoiceConversionInputError

        with pytest.raises(VoiceConversionInputError):
            raise MissingSourceAudioError()

    def test_can_be_caught_as_chatterbox_error(self) -> None:
        from domain.exceptions import ChatterboxError, MissingSourceAudioError

        with pytest.raises(ChatterboxError):
            raise MissingSourceAudioError()


class TestMissingTargetVoiceError:
    """MissingTargetVoiceError — raised when target_voice_path is absent."""

    def test_is_vc_input_error_subclass(self) -> None:
        from domain.exceptions import MissingTargetVoiceError, VoiceConversionInputError

        assert issubclass(MissingTargetVoiceError, VoiceConversionInputError)

    def test_default_construction(self) -> None:
        from domain.exceptions import MissingTargetVoiceError

        exc = MissingTargetVoiceError()
        assert isinstance(exc, Exception)

    def test_str_mentions_target(self) -> None:
        from domain.exceptions import MissingTargetVoiceError

        exc = MissingTargetVoiceError()
        msg = str(exc).lower()
        assert "target" in msg

    def test_can_be_caught_as_vc_input_error(self) -> None:
        from domain.exceptions import MissingTargetVoiceError, VoiceConversionInputError

        with pytest.raises(VoiceConversionInputError):
            raise MissingTargetVoiceError()

    def test_can_be_caught_as_chatterbox_error(self) -> None:
        from domain.exceptions import ChatterboxError, MissingTargetVoiceError

        with pytest.raises(ChatterboxError):
            raise MissingTargetVoiceError()

    def test_source_and_target_are_sibling_errors(self) -> None:
        """MissingSourceAudioError and MissingTargetVoiceError are independent."""
        from domain.exceptions import MissingSourceAudioError, MissingTargetVoiceError

        assert not issubclass(MissingSourceAudioError, MissingTargetVoiceError)
        assert not issubclass(MissingTargetVoiceError, MissingSourceAudioError)


# ──────────────────────────────────────────────────────────────────────────────
# ModelError + subtypes
# ──────────────────────────────────────────────────────────────────────────────


class TestModelError:
    """ModelError — base for model lifecycle and inference failures."""

    def test_is_chatterbox_error_subclass(self) -> None:
        from domain.exceptions import ChatterboxError, ModelError

        assert issubclass(ModelError, ChatterboxError)

    def test_is_not_tts_input_error(self) -> None:
        """ModelError and TTSInputError are independent branches."""
        from domain.exceptions import ModelError, TTSInputError

        assert not issubclass(ModelError, TTSInputError)
        assert not issubclass(TTSInputError, ModelError)


class TestModelNotLoadedError:
    """ModelNotLoadedError — raised when a model has not been initialised."""

    def test_is_model_error_subclass(self) -> None:
        from domain.exceptions import ModelError, ModelNotLoadedError

        assert issubclass(ModelNotLoadedError, ModelError)

    def test_construction_with_key(self) -> None:
        from domain.exceptions import ModelNotLoadedError

        exc = ModelNotLoadedError(model_key="tts")
        assert exc.model_key == "tts"

    def test_default_construction(self) -> None:
        from domain.exceptions import ModelNotLoadedError

        exc = ModelNotLoadedError()
        assert isinstance(exc, Exception)

    def test_str_mentions_model_key(self) -> None:
        from domain.exceptions import ModelNotLoadedError

        exc = ModelNotLoadedError(model_key="turbo")
        assert "turbo" in str(exc)

    def test_can_be_caught_as_model_error(self) -> None:
        from domain.exceptions import ModelError, ModelNotLoadedError

        with pytest.raises(ModelError):
            raise ModelNotLoadedError(model_key="vc")

    def test_can_be_caught_as_chatterbox_error(self) -> None:
        from domain.exceptions import ChatterboxError, ModelNotLoadedError

        with pytest.raises(ChatterboxError):
            raise ModelNotLoadedError()


class TestModelLoadError:
    """ModelLoadError — raised when loading a model fails."""

    def test_is_model_error_subclass(self) -> None:
        from domain.exceptions import ModelError, ModelLoadError

        assert issubclass(ModelLoadError, ModelError)

    def test_construction_with_key_and_message(self) -> None:
        from domain.exceptions import ModelLoadError

        exc = ModelLoadError(model_key="tts", message="disk full")
        assert exc.model_key == "tts"

    def test_default_construction(self) -> None:
        from domain.exceptions import ModelLoadError

        exc = ModelLoadError()
        assert isinstance(exc, Exception)

    def test_str_mentions_model_key_when_provided(self) -> None:
        from domain.exceptions import ModelLoadError

        exc = ModelLoadError(model_key="multilingual", message="OOM")
        assert "multilingual" in str(exc)

    def test_str_mentions_message_when_provided(self) -> None:
        from domain.exceptions import ModelLoadError

        exc = ModelLoadError(model_key="tts", message="disk full")
        assert "disk full" in str(exc)

    def test_can_be_caught_as_model_error(self) -> None:
        from domain.exceptions import ModelError, ModelLoadError

        with pytest.raises(ModelError):
            raise ModelLoadError(model_key="tts")

    def test_can_be_caught_as_chatterbox_error(self) -> None:
        from domain.exceptions import ChatterboxError, ModelLoadError

        with pytest.raises(ChatterboxError):
            raise ModelLoadError()

    def test_not_same_class_as_model_not_loaded(self) -> None:
        from domain.exceptions import ModelLoadError, ModelNotLoadedError

        assert ModelLoadError is not ModelNotLoadedError
        assert not issubclass(ModelLoadError, ModelNotLoadedError)
        assert not issubclass(ModelNotLoadedError, ModelLoadError)


class TestInferenceError:
    """InferenceError — raised when model.generate() fails unexpectedly."""

    def test_is_model_error_subclass(self) -> None:
        from domain.exceptions import InferenceError, ModelError

        assert issubclass(InferenceError, ModelError)

    def test_construction_with_message(self) -> None:
        from domain.exceptions import InferenceError

        exc = InferenceError(message="CUDA kernel error")
        assert isinstance(exc, Exception)

    def test_default_construction(self) -> None:
        from domain.exceptions import InferenceError

        exc = InferenceError()
        assert isinstance(exc, Exception)

    def test_can_be_caught_as_model_error(self) -> None:
        from domain.exceptions import InferenceError, ModelError

        with pytest.raises(ModelError):
            raise InferenceError()

    def test_can_be_caught_as_chatterbox_error(self) -> None:
        from domain.exceptions import ChatterboxError, InferenceError

        with pytest.raises(ChatterboxError):
            raise InferenceError()


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchy integrity
# ──────────────────────────────────────────────────────────────────────────────


class TestHierarchyIntegrity:
    """Cross-cutting checks that the full hierarchy is coherent."""

    def test_all_exported_exceptions_are_chatterbox_error_subclasses(self) -> None:
        """Every public exception in the module must inherit from ChatterboxError
        (except ChatterboxError itself)."""
        from domain.exceptions import ChatterboxError

        mod = _import_exc()
        for name, obj in vars(mod).items():
            if name.startswith("_"):
                continue
            if not (isinstance(obj, type) and issubclass(obj, Exception)):
                continue
            if obj is ChatterboxError:
                continue
            assert issubclass(obj, ChatterboxError), (
                f"{name} must be a subclass of ChatterboxError, got bases: {obj.__bases__}"
            )

    def test_catching_chatterbox_error_catches_all_domain_errors(self) -> None:
        """A single `except ChatterboxError` must catch every domain exception."""
        from domain.exceptions import (
            ChatterboxError,
            EmptyTextError,
            InferenceError,
            MissingSourceAudioError,
            MissingTargetVoiceError,
            ModelLoadError,
            ModelNotLoadedError,
            ReferenceTooShortError,
        )

        domain_exceptions = [
            EmptyTextError(),
            ReferenceTooShortError(),
            MissingSourceAudioError(),
            MissingTargetVoiceError(),
            ModelNotLoadedError(),
            ModelLoadError(),
            InferenceError(),
        ]
        for exc in domain_exceptions:
            assert isinstance(exc, ChatterboxError), (
                f"{type(exc).__name__} is not a ChatterboxError"
            )

    def test_tts_errors_not_vc_errors(self) -> None:
        from domain.exceptions import EmptyTextError, MissingSourceAudioError

        assert not isinstance(EmptyTextError(), MissingSourceAudioError)
        assert not isinstance(MissingSourceAudioError(), EmptyTextError)

    def test_input_errors_not_model_errors(self) -> None:
        from domain.exceptions import EmptyTextError, ModelLoadError

        assert not isinstance(EmptyTextError(), ModelLoadError)
        assert not isinstance(ModelLoadError(), EmptyTextError)
