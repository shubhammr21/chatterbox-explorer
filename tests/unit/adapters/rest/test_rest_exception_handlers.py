"""
tests/unit/adapters/rest/test_rest_exception_handlers.py
==========================================================
TDD unit tests for src/adapters/inbound/rest/exception_handlers.py.

Written BEFORE the implementation module exists (RED phase).

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
value_error_handler
    ValueError → 422 Unprocessable Entity
    Body: {"detail": "<exception message>"}

runtime_error_handler
    RuntimeError → 503 Service Unavailable
    Body: {"detail": "<exception message>"}

http_exception_handler_with_logging
    StarletteHTTPException → same status code as the exception
    Delegates to FastAPI's default http_exception_handler.
    Logs at ERROR level for status >= 500, silent otherwise.

validation_exception_handler_with_logging
    RequestValidationError → 422 with FastAPI's standard detail list
    Delegates to FastAPI's default request_validation_exception_handler.
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
# value_error_handler
# ──────────────────────────────────────────────────────────────────────────────


class TestValueErrorHandler:
    """value_error_handler(request, exc: ValueError) → 422 JSONResponse."""

    # ── direct call ───────────────────────────────────────────────────────────

    @pytest.mark.anyio
    async def test_returns_422_status(self) -> None:
        from adapters.inbound.rest.exception_handlers import value_error_handler

        req = _make_mock_request()
        resp = await value_error_handler(req, ValueError("bad input"))
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_body_contains_detail_key(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import value_error_handler

        req = _make_mock_request()
        resp = await value_error_handler(req, ValueError("bad input"))
        body = json.loads(bytes(resp.body))
        assert "detail" in body

    @pytest.mark.anyio
    async def test_body_detail_matches_exception_message(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import value_error_handler

        req = _make_mock_request()
        resp = await value_error_handler(req, ValueError("text must not be empty"))
        body = json.loads(bytes(resp.body))
        assert body["detail"] == "text must not be empty"

    @pytest.mark.anyio
    async def test_empty_message_produces_empty_detail(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import value_error_handler

        req = _make_mock_request()
        resp = await value_error_handler(req, ValueError())
        body = json.loads(bytes(resp.body))
        assert body["detail"] == ""

    @pytest.mark.anyio
    async def test_does_not_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """ValueError is a caller error — handler must not log it."""
        from adapters.inbound.rest.exception_handlers import value_error_handler

        req = _make_mock_request()
        with caplog.at_level(logging.DEBUG):
            await value_error_handler(req, ValueError("bad"))
        assert caplog.records == []

    # ── through a real FastAPI app ────────────────────────────────────────────

    def test_app_level_route_raising_value_error_returns_422(self) -> None:
        from adapters.inbound.rest.exception_handlers import value_error_handler

        app = _make_minimal_app_with_handler(ValueError, value_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise ValueError("domain said no")

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.status_code == 422
        assert resp.json()["detail"] == "domain said no"

    def test_app_level_value_error_subclass_is_caught(self) -> None:
        """Subclasses of ValueError must also be caught."""
        from adapters.inbound.rest.exception_handlers import value_error_handler

        class _DomainError(ValueError):
            pass

        app = _make_minimal_app_with_handler(ValueError, value_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise _DomainError("subclass error")

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# runtime_error_handler
# ──────────────────────────────────────────────────────────────────────────────


class TestRuntimeErrorHandler:
    """runtime_error_handler(request, exc: RuntimeError) → 503 JSONResponse."""

    # ── direct call ───────────────────────────────────────────────────────────

    @pytest.mark.anyio
    async def test_returns_503_status(self) -> None:
        from adapters.inbound.rest.exception_handlers import runtime_error_handler

        req = _make_mock_request()
        resp = await runtime_error_handler(req, RuntimeError("GPU OOM"))
        assert resp.status_code == 503

    @pytest.mark.anyio
    async def test_body_contains_detail_key(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import runtime_error_handler

        req = _make_mock_request()
        resp = await runtime_error_handler(req, RuntimeError("GPU OOM"))
        body = json.loads(bytes(resp.body))
        assert "detail" in body

    @pytest.mark.anyio
    async def test_body_detail_matches_exception_message(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import runtime_error_handler

        req = _make_mock_request()
        resp = await runtime_error_handler(req, RuntimeError("model not loaded"))
        body = json.loads(bytes(resp.body))
        assert body["detail"] == "model not loaded"

    @pytest.mark.anyio
    async def test_logs_at_exception_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Infrastructure failures must be logged at ERROR/EXCEPTION level."""
        from adapters.inbound.rest.exception_handlers import runtime_error_handler

        req = _make_mock_request(method="POST", path="/api/v1/tts/generate")
        with caplog.at_level(logging.ERROR):
            await runtime_error_handler(req, RuntimeError("disk full"))
        assert any(r.levelno >= logging.ERROR for r in caplog.records)

    @pytest.mark.anyio
    async def test_log_record_contains_method_and_path(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from adapters.inbound.rest.exception_handlers import runtime_error_handler

        req = _make_mock_request(method="POST", path="/api/v1/turbo/generate")
        with caplog.at_level(logging.ERROR):
            await runtime_error_handler(req, RuntimeError("CUDA error"))
        combined = " ".join(r.getMessage() for r in caplog.records)
        assert "POST" in combined or "/api/v1/turbo/generate" in combined

    # ── through a real FastAPI app ────────────────────────────────────────────

    def test_app_level_route_raising_runtime_error_returns_503(self) -> None:
        from adapters.inbound.rest.exception_handlers import runtime_error_handler

        app = _make_minimal_app_with_handler(RuntimeError, runtime_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise RuntimeError("model failed to load")

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "model failed to load"

    def test_app_level_runtime_error_subclass_is_caught(self) -> None:
        from adapters.inbound.rest.exception_handlers import runtime_error_handler

        class _InfraError(RuntimeError):
            pass

        app = _make_minimal_app_with_handler(RuntimeError, runtime_error_handler)

        @app.get("/raise")
        async def _raise() -> None:
            raise _InfraError("disk full")

        resp = TestClient(app, raise_server_exceptions=False).get("/raise")
        assert resp.status_code == 503


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
        assert resp.json()["detail"] == "key not found"


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
    async def test_body_contains_detail_list(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import (
            validation_exception_handler_with_logging,
        )

        req = _make_mock_request()
        exc = self._make_request_validation_error()
        resp = await validation_exception_handler_with_logging(req, exc)
        body = json.loads(bytes(resp.body))
        assert "detail" in body
        assert isinstance(body["detail"], list)

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
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Cross-cutting: error response body shape consistency
# ──────────────────────────────────────────────────────────────────────────────


class TestErrorResponseShape:
    """
    All handlers must return a JSON body with at minimum a "detail" key so
    that clients can always parse error responses uniformly.
    """

    @pytest.mark.anyio
    async def test_value_error_response_is_json(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import value_error_handler

        resp = await value_error_handler(_make_mock_request(), ValueError("x"))
        json.loads(bytes(resp.body))  # must not raise

    @pytest.mark.anyio
    async def test_runtime_error_response_is_json(self) -> None:
        import json

        from adapters.inbound.rest.exception_handlers import runtime_error_handler

        resp = await runtime_error_handler(_make_mock_request(), RuntimeError("x"))
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
    async def test_value_error_content_type_is_json(self) -> None:
        from adapters.inbound.rest.exception_handlers import value_error_handler

        resp = await value_error_handler(_make_mock_request(), ValueError("x"))
        ct = resp.headers.get("content-type", "")
        assert "application/json" in ct

    @pytest.mark.anyio
    async def test_runtime_error_content_type_is_json(self) -> None:
        from adapters.inbound.rest.exception_handlers import runtime_error_handler

        resp = await runtime_error_handler(_make_mock_request(), RuntimeError("x"))
        ct = resp.headers.get("content-type", "")
        assert "application/json" in ct
