"""
tests/unit/adapters/rest/test_rest_exception_handlers.py
==========================================================
TDD unit tests for src/adapters/inbound/rest/exception_handlers.py.

What is tested
--------------
Each handler function is tested in two ways:

1. Directly — call the async handler with a mock Request and a real exception
   instance and assert on the returned Response object.  This isolates the
   handler from the full FastAPI stack and runs without a real server.

2. Through a minimal FastAPI app — register the handler on a tiny app, mount
   a route that raises the target exception, and use TestClient to verify the
   full HTTP contract (status code, body shape, headers).

Why both?
---------
Direct tests are faster and pinpoint failures precisely.
App-level tests catch registration mistakes (wrong exc class, wrong app
method, interaction with default handlers).

Handlers under test
-------------------
tts_input_error_handler
    TTSInputError (EmptyTextError, ReferenceTooShortError) → 422
    Body: {"errors": [{"message": "<exception message>", "path": []}]}
    Logging: none (caller error — silent)

vc_input_error_handler
    VoiceConversionInputError (MissingSourceAudioError, MissingTargetVoiceError) → 422
    Body: {"errors": [{"message": "<exception message>", "path": []}]}
    Logging: none (caller error — silent)

model_error_handler
    ModelError (ModelLoadError, ModelNotLoadedError, InferenceError) → 503
    Body: {"errors": [{"message": "<exception message>", "path": []}]}
    Logging: ERROR level with traceback

chatterbox_error_handler
    ChatterboxError (catch-all) → 500
    Body: {"errors": [{"message": "<exception message>", "path": []}]}
    Logging: ERROR level with traceback

http_exception_handler_with_logging
    StarletteHTTPException → same status code as the exception
    Builds an ErrorResponse directly (no delegation to FastAPI default).
    Logs at ERROR level for status >= 500, silent otherwise.

validation_exception_handler_with_logging
    RequestValidationError → 422 with per-field ErrorDetail list.
    Builds an ErrorResponse directly (no delegation to FastAPI default).
    Logs at DEBUG level.

Architecture rules
------------------
- No real models, no container wiring, no real HTTP server.
- TestClient is used only for the app-level integration sub-tests.
- Handlers are async; tests use pytest-anyio / anyio via TestClient
  (TestClient drives async handlers correctly through its embedded loop).
- No # type: ignore, no # noqa.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError
import pytest
from starlette.exceptions import HTTPException as StarletteHTTPException

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_mock_request(method: str = "POST", path: str = "/test") -> MagicMock:
    """Return a minimal mock of fastapi.Request sufficient for handler tests."""
    req = MagicMock()
    req.method = method
    req.url.path = path
    return req


def _make_minimal_app_with_handler(exc_type: type[Exception], handler) -> FastAPI:
    """
    Build a throwaway FastAPI app that:
      - registers *handler* for *exc_type*
      - exposes GET /raise that raises *exc_type*

    Used to test the full registration + dispatch path.
    """
    app = FastAPI()
    app.add_exception_handler(exc_type, handler)
    return app


# ──────────────────────────────────────────────────────────────────────────────
# tts_input_error_handler
# ──────────────────────────────────────────────────────────────────────────────


class TestTTSInputErrorHandler:
    """tts_input_error_handler(request, exc: TTSInputError) → 422 JSONResponse.

    Covers EmptyTextError, ReferenceTooShortError, and arbitrary subclasses.
    Deliberately silent — TTS input errors are caller mistakes, not server
    faults.
    """

    # ── direct call ───────────────────────────────────────────────────────────

    @pytest.mark.anyio
    async def test_returns_422_for_empty_text_error(self) -> None:
        from adapters.inbound.rest.exception_handlers import tts_input_error_handler
        from domain.exceptions import EmptyTextError

        req = _make_mock_request()
        resp = await tts_input_error_handler(req, EmptyTextError(""))
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_returns_422_for_reference_too_short_error(self) -> None:
        from adapters.inbound.rest.exception_handlers import tts_input_error_handler
        from domain.exceptions import ReferenceTooShortError

        req = _make_mock_request()
        resp = await tts_input_error_handler(req, ReferenceTooShortError())
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_returns_422_for_tts_input_error_subclass(self) -> None:
        from adapters.inbound.rest.exception_handlers import tts_input_error_handler
        from domain.exceptions import TTSInputError

        class _CustomTTSInputError(TTSInputError):
            def __init__(self) -> None:
                super().__init__("custom tts input failure")

        req = _make_mock_request()
        resp = await tts_input_error_handler(req, _CustomTTSInputError())
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_body_contains_detail_key(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import tts_input_error_handler
        from domain.exceptions import EmptyTextError

        req = _make_mock_request()
        resp = await tts_input_error_handler(req, EmptyTextError(""))
        body = json.loads(bytes(resp.body))
        assert "errors" in body

    @pytest.mark.anyio
    async def test_body_detail_matches_exception_message(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import tts_input_error_handler
        from domain.exceptions import EmptyTextError

        exc = EmptyTextError("")
        req = _make_mock_request()
        resp = await tts_input_error_handler(req, exc)
        body = json.loads(bytes(resp.body))
        assert body["errors"][0]["message"] == str(exc)

    @pytest.mark.anyio
    async def test_body_detail_for_reference_too_short_matches_message(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import tts_input_error_handler
        from domain.exceptions import ReferenceTooShortError

        exc = ReferenceTooShortError(minimum_sec=5.0)
        req = _make_mock_request()
        resp = await tts_input_error_handler(req, exc)
        body = json.loads(bytes(resp.body))
        assert body["errors"][0]["message"] == str(exc)

    @pytest.mark.anyio
    async def test_does_not_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TTS input error is a caller mistake — handler must not log it."""
        from adapters.inbound.rest.exception_handlers import tts_input_error_handler
        from domain.exceptions import EmptyTextError

        req = _make_mock_request()
        with caplog.at_level(logging.DEBUG):
            await tts_input_error_handler(req, EmptyTextError(""))
        assert caplog.records == []

    # ── through a real FastAPI app ────────────────────────────────────────────

    def test_app_level_empty_text_error_returns_422(self) -> None:
        from adapters.inbound.rest.exception_handlers import tts_input_error_handler
        from domain.exceptions import EmptyTextError, TTSInputError

        app = _make_minimal_app_with_handler(TTSInputError, tts_input_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise EmptyTextError("")

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.status_code == 422

    def test_app_level_reference_too_short_error_returns_422(self) -> None:
        from adapters.inbound.rest.exception_handlers import tts_input_error_handler
        from domain.exceptions import ReferenceTooShortError, TTSInputError

        app = _make_minimal_app_with_handler(TTSInputError, tts_input_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise ReferenceTooShortError(minimum_sec=5.0)

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.status_code == 422

    def test_app_level_detail_contains_exception_message(self) -> None:
        from adapters.inbound.rest.exception_handlers import tts_input_error_handler
        from domain.exceptions import EmptyTextError, TTSInputError

        exc = EmptyTextError("")
        app = _make_minimal_app_with_handler(TTSInputError, tts_input_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise EmptyTextError("")

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.json()["errors"][0]["message"] == str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# vc_input_error_handler
# ──────────────────────────────────────────────────────────────────────────────


class TestVCInputErrorHandler:
    """vc_input_error_handler(request, exc: VoiceConversionInputError) → 422 JSONResponse.

    Covers MissingSourceAudioError, MissingTargetVoiceError.
    Deliberately silent — VC input errors are caller mistakes, not server
    faults.
    """

    # ── direct call ───────────────────────────────────────────────────────────

    @pytest.mark.anyio
    async def test_returns_422_for_missing_source_audio_error(self) -> None:
        from adapters.inbound.rest.exception_handlers import vc_input_error_handler
        from domain.exceptions import MissingSourceAudioError

        req = _make_mock_request()
        resp = await vc_input_error_handler(req, MissingSourceAudioError())
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_returns_422_for_missing_target_voice_error(self) -> None:
        from adapters.inbound.rest.exception_handlers import vc_input_error_handler
        from domain.exceptions import MissingTargetVoiceError

        req = _make_mock_request()
        resp = await vc_input_error_handler(req, MissingTargetVoiceError())
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_body_contains_detail_key(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import vc_input_error_handler
        from domain.exceptions import MissingSourceAudioError

        req = _make_mock_request()
        resp = await vc_input_error_handler(req, MissingSourceAudioError())
        body = json.loads(bytes(resp.body))
        assert "errors" in body

    @pytest.mark.anyio
    async def test_body_detail_matches_exception_message(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import vc_input_error_handler
        from domain.exceptions import MissingTargetVoiceError

        exc = MissingTargetVoiceError()
        req = _make_mock_request()
        resp = await vc_input_error_handler(req, exc)
        body = json.loads(bytes(resp.body))
        assert body["errors"][0]["message"] == str(exc)

    @pytest.mark.anyio
    async def test_does_not_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """VC input error is a caller mistake — handler must not log it."""
        from adapters.inbound.rest.exception_handlers import vc_input_error_handler
        from domain.exceptions import MissingSourceAudioError

        req = _make_mock_request()
        with caplog.at_level(logging.DEBUG):
            await vc_input_error_handler(req, MissingSourceAudioError())
        assert caplog.records == []

    # ── through a real FastAPI app ────────────────────────────────────────────

    def test_app_level_missing_source_audio_error_returns_422(self) -> None:
        from adapters.inbound.rest.exception_handlers import vc_input_error_handler
        from domain.exceptions import MissingSourceAudioError, VoiceConversionInputError

        app = _make_minimal_app_with_handler(VoiceConversionInputError, vc_input_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise MissingSourceAudioError()

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.status_code == 422

    def test_app_level_missing_target_voice_error_returns_422(self) -> None:
        from adapters.inbound.rest.exception_handlers import vc_input_error_handler
        from domain.exceptions import MissingTargetVoiceError, VoiceConversionInputError

        app = _make_minimal_app_with_handler(VoiceConversionInputError, vc_input_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise MissingTargetVoiceError()

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.status_code == 422

    def test_app_level_detail_contains_exception_message(self) -> None:
        from adapters.inbound.rest.exception_handlers import vc_input_error_handler
        from domain.exceptions import MissingSourceAudioError, VoiceConversionInputError

        exc = MissingSourceAudioError()
        app = _make_minimal_app_with_handler(VoiceConversionInputError, vc_input_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise MissingSourceAudioError()

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.json()["errors"][0]["message"] == str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# model_error_handler
# ──────────────────────────────────────────────────────────────────────────────


class TestModelErrorHandler:
    """model_error_handler(request, exc: ModelError) → 503 JSONResponse.

    Covers ModelLoadError, ModelNotLoadedError, InferenceError.
    Logs at ERROR level because these are infrastructure failures requiring
    operator attention.
    """

    # ── direct call ───────────────────────────────────────────────────────────

    @pytest.mark.anyio
    async def test_returns_503_for_model_load_error(self) -> None:
        from adapters.inbound.rest.exception_handlers import model_error_handler
        from domain.exceptions import ModelLoadError

        req = _make_mock_request()
        resp = await model_error_handler(req, ModelLoadError(model_key="tts", message="OOM"))
        assert resp.status_code == 503

    @pytest.mark.anyio
    async def test_returns_503_for_model_not_loaded_error(self) -> None:
        from adapters.inbound.rest.exception_handlers import model_error_handler
        from domain.exceptions import ModelNotLoadedError

        req = _make_mock_request()
        resp = await model_error_handler(req, ModelNotLoadedError(model_key="turbo"))
        assert resp.status_code == 503

    @pytest.mark.anyio
    async def test_returns_503_for_inference_error(self) -> None:
        from adapters.inbound.rest.exception_handlers import model_error_handler
        from domain.exceptions import InferenceError

        req = _make_mock_request()
        resp = await model_error_handler(req, InferenceError("CUDA device error"))
        assert resp.status_code == 503

    @pytest.mark.anyio
    async def test_body_contains_detail_key(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import model_error_handler
        from domain.exceptions import ModelLoadError

        req = _make_mock_request()
        resp = await model_error_handler(req, ModelLoadError(model_key="tts", message="OOM"))
        body = json.loads(bytes(resp.body))
        assert "errors" in body

    @pytest.mark.anyio
    async def test_body_detail_matches_exception_message(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import model_error_handler
        from domain.exceptions import ModelLoadError

        exc = ModelLoadError(model_key="tts", message="disk full")
        req = _make_mock_request()
        resp = await model_error_handler(req, exc)
        body = json.loads(bytes(resp.body))
        assert body["errors"][0]["message"] == str(exc)

    @pytest.mark.anyio
    async def test_logs_at_error_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Infrastructure failures must be logged at ERROR level."""
        from adapters.inbound.rest.exception_handlers import model_error_handler
        from domain.exceptions import ModelLoadError

        req = _make_mock_request(method="POST", path="/api/v1/tts/generate")
        with caplog.at_level(logging.ERROR):
            await model_error_handler(req, ModelLoadError(model_key="tts", message="OOM"))
        assert any(r.levelno >= logging.ERROR for r in caplog.records)

    @pytest.mark.anyio
    async def test_log_record_contains_method_and_path(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from adapters.inbound.rest.exception_handlers import model_error_handler
        from domain.exceptions import InferenceError

        req = _make_mock_request(method="POST", path="/api/v1/turbo/generate")
        with caplog.at_level(logging.ERROR):
            await model_error_handler(req, InferenceError("CUDA error"))
        combined = " ".join(r.getMessage() for r in caplog.records)
        assert "POST" in combined or "/api/v1/turbo/generate" in combined

    # ── through a real FastAPI app ────────────────────────────────────────────

    def test_app_level_model_load_error_returns_503(self) -> None:
        from adapters.inbound.rest.exception_handlers import model_error_handler
        from domain.exceptions import ModelError, ModelLoadError

        app = _make_minimal_app_with_handler(ModelError, model_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise ModelLoadError(model_key="tts", message="weights missing")

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.status_code == 503

    def test_app_level_inference_error_returns_503(self) -> None:
        from adapters.inbound.rest.exception_handlers import model_error_handler
        from domain.exceptions import InferenceError, ModelError

        app = _make_minimal_app_with_handler(ModelError, model_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise InferenceError("GPU fault")

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.status_code == 503

    def test_app_level_detail_contains_exception_message(self) -> None:
        from adapters.inbound.rest.exception_handlers import model_error_handler
        from domain.exceptions import ModelError, ModelLoadError

        exc = ModelLoadError(model_key="tts", message="weights missing")
        app = _make_minimal_app_with_handler(ModelError, model_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise ModelLoadError(model_key="tts", message="weights missing")

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.json()["errors"][0]["message"] == str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# chatterbox_error_handler
# ──────────────────────────────────────────────────────────────────────────────


class TestChatterboxErrorHandler:
    """chatterbox_error_handler(request, exc: ChatterboxError) → 500 JSONResponse.

    Catch-all for any ChatterboxError not handled by a more specific handler.
    Logs at ERROR level because reaching this handler means a domain exception
    escaped without a dedicated mapping — always requires attention.
    """

    # ── direct call ───────────────────────────────────────────────────────────

    @pytest.mark.anyio
    async def test_returns_500_for_generic_chatterbox_error(self) -> None:
        from adapters.inbound.rest.exception_handlers import chatterbox_error_handler
        from domain.exceptions import ChatterboxError

        req = _make_mock_request()
        resp = await chatterbox_error_handler(req, ChatterboxError("unexpected domain failure"))
        assert resp.status_code == 500

    @pytest.mark.anyio
    async def test_body_contains_detail_key(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import chatterbox_error_handler
        from domain.exceptions import ChatterboxError

        req = _make_mock_request()
        resp = await chatterbox_error_handler(req, ChatterboxError("domain failure"))
        body = json.loads(bytes(resp.body))
        assert "errors" in body

    @pytest.mark.anyio
    async def test_body_detail_matches_exception_message(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import chatterbox_error_handler
        from domain.exceptions import ChatterboxError

        exc = ChatterboxError("something went wrong in the domain")
        req = _make_mock_request()
        resp = await chatterbox_error_handler(req, exc)
        body = json.loads(bytes(resp.body))
        assert body["errors"][0]["message"] == str(exc)

    @pytest.mark.anyio
    async def test_logs_at_error_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unhandled domain errors must be logged at ERROR level."""
        from adapters.inbound.rest.exception_handlers import chatterbox_error_handler
        from domain.exceptions import ChatterboxError

        req = _make_mock_request(method="POST", path="/api/v1/tts/generate")
        with caplog.at_level(logging.ERROR):
            await chatterbox_error_handler(req, ChatterboxError("escaped domain error"))
        assert any(r.levelno >= logging.ERROR for r in caplog.records)

    @pytest.mark.anyio
    async def test_log_record_contains_method_and_path(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from adapters.inbound.rest.exception_handlers import chatterbox_error_handler
        from domain.exceptions import ChatterboxError

        req = _make_mock_request(method="GET", path="/api/v1/models/status")
        with caplog.at_level(logging.ERROR):
            await chatterbox_error_handler(req, ChatterboxError("surprise"))
        combined = " ".join(r.getMessage() for r in caplog.records)
        assert "GET" in combined or "/api/v1/models/status" in combined

    # ── through a real FastAPI app ────────────────────────────────────────────

    def test_app_level_chatterbox_error_returns_500(self) -> None:
        from adapters.inbound.rest.exception_handlers import chatterbox_error_handler
        from domain.exceptions import ChatterboxError

        app = _make_minimal_app_with_handler(ChatterboxError, chatterbox_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise ChatterboxError("unclassified domain failure")

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.status_code == 500

    def test_app_level_detail_contains_exception_message(self) -> None:
        from adapters.inbound.rest.exception_handlers import chatterbox_error_handler
        from domain.exceptions import ChatterboxError

        app = _make_minimal_app_with_handler(ChatterboxError, chatterbox_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise ChatterboxError("unclassified domain failure")

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.json()["errors"][0]["message"] == "unclassified domain failure"


# ──────────────────────────────────────────────────────────────────────────────
# http_exception_handler_with_logging
# ──────────────────────────────────────────────────────────────────────────────


class TestHttpExceptionHandlerWithLogging:
    """
    http_exception_handler_with_logging delegates to FastAPI's default handler
    and additionally logs HTTP errors at ERROR level when status_code >= 500.
    """

    # ── direct call ───────────────────────────────────────────────────────────

    @pytest.mark.anyio
    async def test_delegates_status_code_for_404(self) -> None:
        from adapters.inbound.rest.exception_handlers import (
            http_exception_handler_with_logging,
        )

        req = _make_mock_request()
        exc = StarletteHTTPException(status_code=404, detail="not found")
        resp = await http_exception_handler_with_logging(req, exc)
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_delegates_status_code_for_503(self) -> None:
        from adapters.inbound.rest.exception_handlers import (
            http_exception_handler_with_logging,
        )

        req = _make_mock_request()
        exc = StarletteHTTPException(status_code=503, detail="unavailable")
        resp = await http_exception_handler_with_logging(req, exc)
        assert resp.status_code == 503

    @pytest.mark.anyio
    async def test_does_not_log_for_4xx(self, caplog: pytest.LogCaptureFixture) -> None:
        """Client errors (4xx) are expected — should not produce ERROR log lines."""
        from adapters.inbound.rest.exception_handlers import (
            http_exception_handler_with_logging,
        )

        req = _make_mock_request()
        exc = StarletteHTTPException(status_code=404, detail="not found")
        with caplog.at_level(logging.ERROR):
            await http_exception_handler_with_logging(req, exc)
        assert all(r.levelno < logging.ERROR for r in caplog.records)

    @pytest.mark.anyio
    async def test_logs_at_error_for_5xx(self, caplog: pytest.LogCaptureFixture) -> None:
        """Server errors (5xx) are unexpected — must produce an ERROR log line."""
        from adapters.inbound.rest.exception_handlers import (
            http_exception_handler_with_logging,
        )

        req = _make_mock_request(path="/api/v1/tts/generate")
        exc = StarletteHTTPException(status_code=500, detail="boom")
        with caplog.at_level(logging.ERROR):
            await http_exception_handler_with_logging(req, exc)
        assert any(r.levelno >= logging.ERROR for r in caplog.records)

    @pytest.mark.anyio
    async def test_boundary_500_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        from adapters.inbound.rest.exception_handlers import (
            http_exception_handler_with_logging,
        )

        req = _make_mock_request()
        exc = StarletteHTTPException(status_code=500, detail="internal")
        with caplog.at_level(logging.ERROR):
            await http_exception_handler_with_logging(req, exc)
        assert any(r.levelno >= logging.ERROR for r in caplog.records)

    @pytest.mark.anyio
    async def test_boundary_499_is_not_logged_at_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from adapters.inbound.rest.exception_handlers import (
            http_exception_handler_with_logging,
        )

        req = _make_mock_request()
        exc = StarletteHTTPException(status_code=499, detail="client closed")
        with caplog.at_level(logging.ERROR):
            await http_exception_handler_with_logging(req, exc)
        assert all(r.levelno < logging.ERROR for r in caplog.records)

    # ── through a real FastAPI app ────────────────────────────────────────────

    def test_app_level_404_preserves_detail(self) -> None:
        from adapters.inbound.rest.exception_handlers import (
            http_exception_handler_with_logging,
        )

        app = _make_minimal_app_with_handler(
            StarletteHTTPException, http_exception_handler_with_logging
        )

        @app.get("/raise")
        async def _raise() -> None:
            raise StarletteHTTPException(status_code=404, detail="key not found")

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.status_code == 404
        assert resp.json()["errors"][0]["message"] == "key not found"


# ──────────────────────────────────────────────────────────────────────────────
# validation_exception_handler_with_logging
# ──────────────────────────────────────────────────────────────────────────────


class TestValidationExceptionHandlerWithLogging:
    """
    validation_exception_handler_with_logging delegates to FastAPI's default
    RequestValidationError handler and logs at DEBUG level.
    """

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_request_validation_error() -> RequestValidationError:
        """
        Produce a real RequestValidationError by making Pydantic validate a
        deliberately invalid payload against a model.  We wrap the resulting
        ValidationError in RequestValidationError to match what FastAPI
        produces internally.
        """

        class _Model(BaseModel):
            value: int

        try:
            _Model.model_validate({"value": "not-an-int"})
        except ValidationError as pydantic_exc:
            return RequestValidationError(errors=pydantic_exc.errors())
        raise AssertionError("Expected ValidationError was not raised")  # pragma: no cover

    # ── direct call ───────────────────────────────────────────────────────────

    @pytest.mark.anyio
    async def test_returns_422(self) -> None:
        from adapters.inbound.rest.exception_handlers import (
            validation_exception_handler_with_logging,
        )

        req = _make_mock_request()
        exc = self._make_request_validation_error()
        resp = await validation_exception_handler_with_logging(req, exc)
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_body_contains_errors_list(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import (
            validation_exception_handler_with_logging,
        )

        req = _make_mock_request()
        exc = self._make_request_validation_error()
        resp = await validation_exception_handler_with_logging(req, exc)
        body = json.loads(bytes(resp.body))
        assert "errors" in body
        assert isinstance(body["errors"], list)

    @pytest.mark.anyio
    async def test_logs_at_debug_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Validation errors are common client mistakes — log at DEBUG, not ERROR."""
        from adapters.inbound.rest.exception_handlers import (
            validation_exception_handler_with_logging,
        )

        req = _make_mock_request()
        exc = self._make_request_validation_error()
        with caplog.at_level(logging.DEBUG):
            await validation_exception_handler_with_logging(req, exc)
        assert any(r.levelno == logging.DEBUG for r in caplog.records)

    @pytest.mark.anyio
    async def test_does_not_log_at_error_level(self, caplog: pytest.LogCaptureFixture) -> None:
        from adapters.inbound.rest.exception_handlers import (
            validation_exception_handler_with_logging,
        )

        req = _make_mock_request()
        exc = self._make_request_validation_error()
        with caplog.at_level(logging.ERROR):
            await validation_exception_handler_with_logging(req, exc)
        assert all(r.levelno < logging.ERROR for r in caplog.records)

    # ── through a real FastAPI app ────────────────────────────────────────────

    def test_app_level_bad_body_returns_422_with_detail_list(self) -> None:
        """FastAPI raises RequestValidationError for invalid body schemas."""
        from adapters.inbound.rest.exception_handlers import (
            validation_exception_handler_with_logging,
        )

        class _Body(BaseModel):
            count: int

        app = _make_minimal_app_with_handler(
            RequestValidationError, validation_exception_handler_with_logging
        )

        @app.post("/items")
        async def _endpoint(body: _Body) -> dict:
            return {"count": body.count}

        resp = TestClient(app, raise_server_exceptions=False).post(
            "/items", json={"count": "not-a-number"}
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "errors" in data
        assert isinstance(data["errors"], list)
        assert len(data["errors"]) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Cross-cutting: error response body shape consistency
# ──────────────────────────────────────────────────────────────────────────────


class TestErrorResponseShape:
    """
    All handlers must return a JSON body with at minimum a "detail" key so
    that clients can always parse error responses uniformly.
    """

    @pytest.mark.anyio
    async def test_tts_input_error_response_is_json(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import tts_input_error_handler
        from domain.exceptions import EmptyTextError

        resp = await tts_input_error_handler(_make_mock_request(), EmptyTextError(""))
        json.loads(bytes(resp.body))  # must not raise

    @pytest.mark.anyio
    async def test_model_error_response_is_json(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import model_error_handler
        from domain.exceptions import ModelLoadError

        resp = await model_error_handler(
            _make_mock_request(), ModelLoadError(model_key="tts", message="OOM")
        )
        json.loads(bytes(resp.body))  # must not raise

    @pytest.mark.anyio
    async def test_http_exception_response_is_json(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import (
            http_exception_handler_with_logging,
        )

        exc = StarletteHTTPException(status_code=404, detail="gone")
        resp = await http_exception_handler_with_logging(_make_mock_request(), exc)
        json.loads(bytes(resp.body))  # must not raise

    @pytest.mark.anyio
    async def test_tts_input_error_content_type_is_json(self) -> None:
        from adapters.inbound.rest.exception_handlers import tts_input_error_handler
        from domain.exceptions import EmptyTextError

        resp = await tts_input_error_handler(_make_mock_request(), EmptyTextError(""))
        ct = resp.headers.get("content-type", "")
        assert "application/json" in ct

    @pytest.mark.anyio
    async def test_model_error_content_type_is_json(self) -> None:
        from adapters.inbound.rest.exception_handlers import model_error_handler
        from domain.exceptions import InferenceError

        resp = await model_error_handler(_make_mock_request(), InferenceError("GPU fault"))
        ct = resp.headers.get("content-type", "")
        assert "application/json" in ct

    @pytest.mark.anyio
    async def test_chatterbox_error_content_type_is_json(self) -> None:
        from adapters.inbound.rest.exception_handlers import chatterbox_error_handler
        from domain.exceptions import ChatterboxError

        resp = await chatterbox_error_handler(_make_mock_request(), ChatterboxError("x"))
        ct = resp.headers.get("content-type", "")
        assert "application/json" in ct

    @pytest.mark.anyio
    async def test_vc_input_error_content_type_is_json(self) -> None:
        from adapters.inbound.rest.exception_handlers import vc_input_error_handler
        from domain.exceptions import MissingSourceAudioError

        resp = await vc_input_error_handler(_make_mock_request(), MissingSourceAudioError())
        ct = resp.headers.get("content-type", "")
        assert "application/json" in ct
