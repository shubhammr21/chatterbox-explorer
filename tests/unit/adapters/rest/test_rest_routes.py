"""
tests/unit/adapters/rest/test_rest_routes.py
============================================
Integration tests for all FastAPI REST route handlers in
src/adapters/inbound/rest/routes.py.

Architecture of test isolation
-------------------------------
* ONE shared ``rest_app`` (FastAPI) and ``rest_client`` (TestClient) are
  created at *module* scope.  ``build_rest_app()`` calls
  ``container.wire(modules=[routes_module])`` which modifies the routes module
  globally; calling it more than once without an intervening ``container.unwire()``
  raises ``AlreadyWiredError``.  The module-scoped ``rest_client`` fixture uses
  ``with TestClient(rest_app) as client:`` which triggers the ASGI lifespan and,
  on exit, triggers the shutdown handler that calls ``container.unwire()``.
  This guarantees that every subsequent test module that also calls
  ``build_rest_app()`` starts from a clean slate.

* Provider overrides are applied per-test via the
  ``with rest_app.container.<provider>.override(mock):`` context manager.
  They are active only for the duration of the ``with`` block — never leaked
  between tests.

* All service mocks are plain ``MagicMock`` objects (synchronous).
  ``TestClient`` drives the ASGI app synchronously, and ``run_in_threadpool``
  runs the mocked sync call in a thread pool — which TestClient's
  synchronous event loop handles correctly.

* No real model weights are loaded.  Every infrastructure provider
  (tts_service, turbo_service, multilingual_service, model_manager_service,
  watermark_service) is replaced by a MagicMock for each test that exercises
  the corresponding route.
"""

from __future__ import annotations

import io
import re
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import numpy as np
import pytest

from domain.models import (
    AppConfig,
    AudioResult,
    MemoryStats,
    ModelStatus,
    WatermarkResult,
)

# ─────────────────────────────────────────────────────────────────────────────
# Module-scoped fixtures  (ONE app + ONE client for the entire module)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rest_app():
    """Build and wire the FastAPI app exactly once for the whole module.

    ``watermark_available=False`` keeps the container configuration simple
    and avoids any perth/librosa import side-effects in the test process.
    The real ``PerThWatermarkDetector`` is never exercised here; each
    watermark test overrides ``watermark_service`` with a MagicMock.
    """
    from bootstrap import build_rest_app

    return build_rest_app(watermark_available=False)


@pytest.fixture(scope="module")
def rest_client(rest_app):
    """Open the ASGI lifespan for the entire module, yield the TestClient,
    then close it (triggering container.unwire()) when all tests finish.
    """
    with TestClient(rest_app) as client:
        yield client


# ─────────────────────────────────────────────────────────────────────────────
# Test-local factory helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fake_audio() -> AudioResult:
    """Return a 1-second silence AudioResult — sufficient for WAV encoding."""
    return AudioResult(
        sample_rate=24000,
        samples=np.zeros(24000, dtype=np.float32),
    )


def _fake_model_statuses() -> list[ModelStatus]:
    """Return a single ModelStatus snapshot used by model-management tests."""
    return [
        ModelStatus(
            key="tts",
            display_name="Standard TTS",
            class_name="ChatterboxTTS",
            description="desc",
            params="500M",
            size_gb=1.4,
            in_memory=False,
            on_disk=True,
        )
    ]


def _fake_memory_stats() -> MemoryStats:
    """Return a CPU-only MemoryStats snapshot (device fields are None)."""
    return MemoryStats(
        sys_total_gb=16.0,
        sys_used_gb=8.0,
        sys_avail_gb=8.0,
        sys_percent=50.0,
        proc_rss_gb=2.0,
    )


def _fake_wav_bytes() -> bytes:
    """Build a minimal but valid WAV file in memory using scipy."""
    from scipy.io import wavfile

    buf = io.BytesIO()
    wavfile.write(buf, 24000, np.zeros(240, dtype=np.int16))
    return buf.getvalue()


def _fake_watermark_result(
    verdict: str = "detected",
    score: float = 0.95,
    available: bool = True,
) -> WatermarkResult:
    return WatermarkResult(
        score=score,
        verdict=verdict,
        message=f"verdict={verdict}",
        available=available,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoint  GET /api/v1/health
# ─────────────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """GET /api/v1/health — immediate, no inference, depends only on app_config."""

    def test_returns_200(self, rest_client, rest_app):
        mock_cfg = AppConfig(device="cpu", watermark_available=False)
        with rest_app.container.app_config.override(mock_cfg):
            resp = rest_client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_body_status_is_ok(self, rest_client, rest_app):
        mock_cfg = AppConfig(device="cpu", watermark_available=False)
        with rest_app.container.app_config.override(mock_cfg):
            resp = rest_client.get("/api/v1/health")
        assert resp.json()["status"] == "ok"

    def test_body_device_matches_config(self, rest_client, rest_app):
        mock_cfg = AppConfig(device="cpu", watermark_available=False)
        with rest_app.container.app_config.override(mock_cfg):
            resp = rest_client.get("/api/v1/health")
        assert resp.json()["device"] == "cpu"

    def test_response_has_x_request_id_header(self, rest_client, rest_app):
        mock_cfg = AppConfig(device="cpu", watermark_available=False)
        with rest_app.container.app_config.override(mock_cfg):
            resp = rest_client.get("/api/v1/health")
        assert "X-Request-Id" in resp.headers

    def test_x_request_id_is_uuid4(self, rest_client, rest_app):
        _UUID4_RE = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        )
        mock_cfg = AppConfig(device="cpu", watermark_available=False)
        with rest_app.container.app_config.override(mock_cfg):
            resp = rest_client.get("/api/v1/health")
        request_id = resp.headers["X-Request-Id"]
        assert _UUID4_RE.match(request_id), f"Not a UUID4: {request_id!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Standard TTS  POST /api/v1/tts/generate
# ─────────────────────────────────────────────────────────────────────────────


class TestTTSGenerateEndpoint:
    """POST /api/v1/tts/generate — happy path + error translation."""

    def test_valid_request_returns_200(self, rest_client, rest_app):
        mock_tts = MagicMock()
        mock_tts.generate.return_value = _fake_audio()
        with rest_app.container.tts_service.override(mock_tts):
            resp = rest_client.post("/api/v1/tts/generate", json={"text": "Hello"})
        assert resp.status_code == 200

    def test_valid_request_content_type_is_audio_wav(self, rest_client, rest_app):
        mock_tts = MagicMock()
        mock_tts.generate.return_value = _fake_audio()
        with rest_app.container.tts_service.override(mock_tts):
            resp = rest_client.post("/api/v1/tts/generate", json={"text": "Hello"})
        assert "audio/wav" in resp.headers["content-type"]

    def test_valid_request_body_starts_with_riff(self, rest_client, rest_app):
        mock_tts = MagicMock()
        mock_tts.generate.return_value = _fake_audio()
        with rest_app.container.tts_service.override(mock_tts):
            resp = rest_client.post("/api/v1/tts/generate", json={"text": "Hello"})
        assert resp.content[:4] == b"RIFF"

    def test_empty_text_returns_422_without_calling_service(self, rest_client, rest_app):
        """Pydantic min_length=1 rejects empty text before the handler runs."""
        mock_tts = MagicMock()
        with rest_app.container.tts_service.override(mock_tts):
            resp = rest_client.post("/api/v1/tts/generate", json={"text": ""})
        assert resp.status_code == 422
        mock_tts.generate.assert_not_called()

    def test_missing_text_field_returns_422(self, rest_client, rest_app):
        resp = rest_client.post("/api/v1/tts/generate", json={})
        assert resp.status_code == 422

    def test_service_value_error_returns_422(self, rest_client, rest_app):
        mock_tts = MagicMock()
        mock_tts.generate.side_effect = ValueError("empty text")
        with rest_app.container.tts_service.override(mock_tts):
            resp = rest_client.post("/api/v1/tts/generate", json={"text": "Hello"})
        assert resp.status_code == 422

    def test_service_runtime_error_returns_503(self, rest_client, rest_app):
        mock_tts = MagicMock()
        mock_tts.generate.side_effect = RuntimeError("OOM")
        with rest_app.container.tts_service.override(mock_tts):
            resp = rest_client.post("/api/v1/tts/generate", json={"text": "Hello"})
        assert resp.status_code == 503

    def test_service_called_with_domain_request(self, rest_client, rest_app):
        """The route must translate the JSON body into a TTSRequest and pass it to the service."""
        from domain.models import TTSRequest

        mock_tts = MagicMock()
        mock_tts.generate.return_value = _fake_audio()
        with rest_app.container.tts_service.override(mock_tts):
            rest_client.post("/api/v1/tts/generate", json={"text": "Check args"})
        call_args = mock_tts.generate.call_args
        assert call_args is not None
        req = call_args.args[0]
        assert isinstance(req, TTSRequest)
        assert req.text == "Check args"


# ─────────────────────────────────────────────────────────────────────────────
# Turbo TTS  POST /api/v1/turbo/generate
# ─────────────────────────────────────────────────────────────────────────────


class TestTurboGenerateEndpoint:
    """POST /api/v1/turbo/generate — happy path + error translation."""

    def test_valid_request_returns_200(self, rest_client, rest_app):
        mock_turbo = MagicMock()
        mock_turbo.generate.return_value = _fake_audio()
        with rest_app.container.turbo_service.override(mock_turbo):
            resp = rest_client.post("/api/v1/turbo/generate", json={"text": "Hello"})
        assert resp.status_code == 200

    def test_valid_request_content_type_is_audio_wav(self, rest_client, rest_app):
        mock_turbo = MagicMock()
        mock_turbo.generate.return_value = _fake_audio()
        with rest_app.container.turbo_service.override(mock_turbo):
            resp = rest_client.post("/api/v1/turbo/generate", json={"text": "Hello"})
        assert "audio/wav" in resp.headers["content-type"]

    def test_valid_request_body_starts_with_riff(self, rest_client, rest_app):
        mock_turbo = MagicMock()
        mock_turbo.generate.return_value = _fake_audio()
        with rest_app.container.turbo_service.override(mock_turbo):
            resp = rest_client.post("/api/v1/turbo/generate", json={"text": "Hello"})
        assert resp.content[:4] == b"RIFF"

    def test_empty_text_returns_422(self, rest_client, rest_app):
        resp = rest_client.post("/api/v1/turbo/generate", json={"text": ""})
        assert resp.status_code == 422

    def test_service_value_error_returns_422(self, rest_client, rest_app):
        mock_turbo = MagicMock()
        mock_turbo.generate.side_effect = ValueError("bad reference audio")
        with rest_app.container.turbo_service.override(mock_turbo):
            resp = rest_client.post("/api/v1/turbo/generate", json={"text": "Hello"})
        assert resp.status_code == 422

    def test_service_runtime_error_returns_503(self, rest_client, rest_app):
        mock_turbo = MagicMock()
        mock_turbo.generate.side_effect = RuntimeError("OOM")
        with rest_app.container.turbo_service.override(mock_turbo):
            resp = rest_client.post("/api/v1/turbo/generate", json={"text": "Hello"})
        assert resp.status_code == 503

    def test_service_called_with_domain_request(self, rest_client, rest_app):
        from domain.models import TurboTTSRequest

        mock_turbo = MagicMock()
        mock_turbo.generate.return_value = _fake_audio()
        with rest_app.container.turbo_service.override(mock_turbo):
            rest_client.post("/api/v1/turbo/generate", json={"text": "Turbo check"})
        req = mock_turbo.generate.call_args.args[0]
        assert isinstance(req, TurboTTSRequest)
        assert req.text == "Turbo check"


# ─────────────────────────────────────────────────────────────────────────────
# Multilingual TTS  POST /api/v1/multilingual/generate
# ─────────────────────────────────────────────────────────────────────────────


class TestMultilingualGenerateEndpoint:
    """POST /api/v1/multilingual/generate — happy path + error translation."""

    def test_valid_request_returns_200(self, rest_client, rest_app):
        mock_mtl = MagicMock()
        mock_mtl.generate.return_value = _fake_audio()
        with rest_app.container.multilingual_service.override(mock_mtl):
            resp = rest_client.post(
                "/api/v1/multilingual/generate",
                json={"text": "Bonjour", "language": "fr"},
            )
        assert resp.status_code == 200

    def test_valid_request_content_type_is_audio_wav(self, rest_client, rest_app):
        mock_mtl = MagicMock()
        mock_mtl.generate.return_value = _fake_audio()
        with rest_app.container.multilingual_service.override(mock_mtl):
            resp = rest_client.post(
                "/api/v1/multilingual/generate",
                json={"text": "Bonjour", "language": "fr"},
            )
        assert "audio/wav" in resp.headers["content-type"]

    def test_valid_request_body_starts_with_riff(self, rest_client, rest_app):
        mock_mtl = MagicMock()
        mock_mtl.generate.return_value = _fake_audio()
        with rest_app.container.multilingual_service.override(mock_mtl):
            resp = rest_client.post(
                "/api/v1/multilingual/generate",
                json={"text": "Hola", "language": "es"},
            )
        assert resp.content[:4] == b"RIFF"

    def test_empty_text_returns_422(self, rest_client, rest_app):
        resp = rest_client.post(
            "/api/v1/multilingual/generate",
            json={"text": "", "language": "en"},
        )
        assert resp.status_code == 422

    def test_service_value_error_returns_422(self, rest_client, rest_app):
        mock_mtl = MagicMock()
        mock_mtl.generate.side_effect = ValueError("bad input")
        with rest_app.container.multilingual_service.override(mock_mtl):
            resp = rest_client.post(
                "/api/v1/multilingual/generate",
                json={"text": "Hello", "language": "en"},
            )
        assert resp.status_code == 422

    def test_service_runtime_error_returns_503(self, rest_client, rest_app):
        mock_mtl = MagicMock()
        mock_mtl.generate.side_effect = RuntimeError("device OOM")
        with rest_app.container.multilingual_service.override(mock_mtl):
            resp = rest_client.post(
                "/api/v1/multilingual/generate",
                json={"text": "Hello", "language": "en"},
            )
        assert resp.status_code == 503

    def test_service_called_with_domain_request_including_language(self, rest_client, rest_app):
        from domain.models import MultilingualTTSRequest

        mock_mtl = MagicMock()
        mock_mtl.generate.return_value = _fake_audio()
        with rest_app.container.multilingual_service.override(mock_mtl):
            rest_client.post(
                "/api/v1/multilingual/generate",
                json={"text": "Guten Tag", "language": "de"},
            )
        req = mock_mtl.generate.call_args.args[0]
        assert isinstance(req, MultilingualTTSRequest)
        assert req.text == "Guten Tag"
        assert req.language == "de"


# ─────────────────────────────────────────────────────────────────────────────
# Model Status  GET /api/v1/models/status
# ─────────────────────────────────────────────────────────────────────────────


class TestModelStatusEndpoint:
    """GET /api/v1/models/status — returns JSON list of ModelStatusResponse."""

    def test_returns_200(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        with rest_app.container.model_manager_service.override(mock_manager):
            resp = rest_client.get("/api/v1/models/status")
        assert resp.status_code == 200

    def test_body_is_a_list(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        with rest_app.container.model_manager_service.override(mock_manager):
            resp = rest_client.get("/api/v1/models/status")
        assert isinstance(resp.json(), list)

    def test_list_has_correct_length(self, rest_client, rest_app):
        mock_manager = MagicMock()
        statuses = _fake_model_statuses()
        mock_manager.get_all_status.return_value = statuses
        with rest_app.container.model_manager_service.override(mock_manager):
            resp = rest_client.get("/api/v1/models/status")
        assert len(resp.json()) == len(statuses)

    def test_item_contains_key(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        with rest_app.container.model_manager_service.override(mock_manager):
            item = rest_client.get("/api/v1/models/status").json()[0]
        assert item["key"] == "tts"

    def test_item_contains_display_name(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        with rest_app.container.model_manager_service.override(mock_manager):
            item = rest_client.get("/api/v1/models/status").json()[0]
        assert item["display_name"] == "Standard TTS"

    def test_item_contains_in_memory(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        with rest_app.container.model_manager_service.override(mock_manager):
            item = rest_client.get("/api/v1/models/status").json()[0]
        assert item["in_memory"] is False

    def test_item_contains_on_disk(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        with rest_app.container.model_manager_service.override(mock_manager):
            item = rest_client.get("/api/v1/models/status").json()[0]
        assert item["on_disk"] is True

    def test_empty_registry_returns_empty_list(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = []
        with rest_app.container.model_manager_service.override(mock_manager):
            resp = rest_client.get("/api/v1/models/status")
        assert resp.json() == []


# ─────────────────────────────────────────────────────────────────────────────
# Model Memory  GET /api/v1/models/memory
# ─────────────────────────────────────────────────────────────────────────────


class TestModelMemoryEndpoint:
    """GET /api/v1/models/memory — returns MemoryStatsResponse JSON."""

    def test_returns_200(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_memory_stats.return_value = _fake_memory_stats()
        with rest_app.container.model_manager_service.override(mock_manager):
            resp = rest_client.get("/api/v1/models/memory")
        assert resp.status_code == 200

    def test_body_contains_sys_total_gb(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_memory_stats.return_value = _fake_memory_stats()
        with rest_app.container.model_manager_service.override(mock_manager):
            data = rest_client.get("/api/v1/models/memory").json()
        assert data["sys_total_gb"] == 16.0

    def test_body_contains_sys_used_gb(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_memory_stats.return_value = _fake_memory_stats()
        with rest_app.container.model_manager_service.override(mock_manager):
            data = rest_client.get("/api/v1/models/memory").json()
        assert data["sys_used_gb"] == 8.0

    def test_body_device_driver_gb_is_none_on_cpu(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_memory_stats.return_value = _fake_memory_stats()
        with rest_app.container.model_manager_service.override(mock_manager):
            data = rest_client.get("/api/v1/models/memory").json()
        assert data["device_driver_gb"] is None

    def test_body_device_max_gb_is_none_on_cpu(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_memory_stats.return_value = _fake_memory_stats()
        with rest_app.container.model_manager_service.override(mock_manager):
            data = rest_client.get("/api/v1/models/memory").json()
        assert data["device_max_gb"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Model Load  POST /api/v1/models/{key}/load
# ─────────────────────────────────────────────────────────────────────────────


class TestModelLoadEndpoint:
    """POST /api/v1/models/{key}/load — 200 / 404 / 503."""

    def test_known_key_returns_200(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        mock_manager.load.return_value = "Model tts loaded successfully"
        with rest_app.container.model_manager_service.override(mock_manager):
            resp = rest_client.post("/api/v1/models/tts/load")
        assert resp.status_code == 200

    def test_known_key_body_contains_message(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        mock_manager.load.return_value = "Model tts loaded successfully"
        with rest_app.container.model_manager_service.override(mock_manager):
            data = rest_client.post("/api/v1/models/tts/load").json()
        assert "message" in data
        assert data["message"] == "Model tts loaded successfully"

    def test_unknown_key_returns_404(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        with rest_app.container.model_manager_service.override(mock_manager):
            resp = rest_client.post("/api/v1/models/unknown_model/load")
        assert resp.status_code == 404

    def test_unknown_key_does_not_call_load(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        with rest_app.container.model_manager_service.override(mock_manager):
            rest_client.post("/api/v1/models/does_not_exist/load")
        mock_manager.load.assert_not_called()

    def test_service_runtime_error_returns_503(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        mock_manager.load.side_effect = RuntimeError("OOM during model load")
        with rest_app.container.model_manager_service.override(mock_manager):
            resp = rest_client.post("/api/v1/models/tts/load")
        assert resp.status_code == 503

    def test_service_called_with_correct_key(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        mock_manager.load.return_value = "ok"
        with rest_app.container.model_manager_service.override(mock_manager):
            rest_client.post("/api/v1/models/tts/load")
        mock_manager.load.assert_called_once_with("tts")


# ─────────────────────────────────────────────────────────────────────────────
# Model Unload  POST /api/v1/models/{key}/unload
# ─────────────────────────────────────────────────────────────────────────────


class TestModelUnloadEndpoint:
    """POST /api/v1/models/{key}/unload — 200 / 404."""

    def test_known_key_returns_200(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        mock_manager.unload.return_value = "Model tts unloaded"
        with rest_app.container.model_manager_service.override(mock_manager):
            resp = rest_client.post("/api/v1/models/tts/unload")
        assert resp.status_code == 200

    def test_known_key_body_contains_message(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        mock_manager.unload.return_value = "Model tts unloaded"
        with rest_app.container.model_manager_service.override(mock_manager):
            data = rest_client.post("/api/v1/models/tts/unload").json()
        assert "message" in data

    def test_unknown_key_returns_404(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        with rest_app.container.model_manager_service.override(mock_manager):
            resp = rest_client.post("/api/v1/models/no_such_model/unload")
        assert resp.status_code == 404

    def test_unknown_key_does_not_call_unload(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        with rest_app.container.model_manager_service.override(mock_manager):
            rest_client.post("/api/v1/models/no_such_model/unload")
        mock_manager.unload.assert_not_called()

    def test_service_called_with_correct_key(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        mock_manager.unload.return_value = "ok"
        with rest_app.container.model_manager_service.override(mock_manager):
            rest_client.post("/api/v1/models/tts/unload")
        mock_manager.unload.assert_called_once_with("tts")


# ─────────────────────────────────────────────────────────────────────────────
# Watermark Detection  POST /api/v1/watermark/detect
# ─────────────────────────────────────────────────────────────────────────────


class TestWatermarkDetectEndpoint:
    """POST /api/v1/watermark/detect — multipart file upload → WatermarkResponse."""

    def test_upload_returns_200(self, rest_client, rest_app):
        mock_wm = MagicMock()
        mock_wm.detect.return_value = _fake_watermark_result("detected", 0.95)
        with rest_app.container.watermark_service.override(mock_wm):
            resp = rest_client.post(
                "/api/v1/watermark/detect",
                files={"audio": ("clip.wav", _fake_wav_bytes(), "audio/wav")},
            )
        assert resp.status_code == 200

    def test_body_contains_score(self, rest_client, rest_app):
        mock_wm = MagicMock()
        mock_wm.detect.return_value = _fake_watermark_result("detected", 0.95)
        with rest_app.container.watermark_service.override(mock_wm):
            data = rest_client.post(
                "/api/v1/watermark/detect",
                files={"audio": ("clip.wav", _fake_wav_bytes(), "audio/wav")},
            ).json()
        assert "score" in data
        assert data["score"] == pytest.approx(0.95)

    def test_body_contains_verdict(self, rest_client, rest_app):
        mock_wm = MagicMock()
        mock_wm.detect.return_value = _fake_watermark_result("detected", 0.95)
        with rest_app.container.watermark_service.override(mock_wm):
            data = rest_client.post(
                "/api/v1/watermark/detect",
                files={"audio": ("clip.wav", _fake_wav_bytes(), "audio/wav")},
            ).json()
        assert data["verdict"] == "detected"

    def test_body_contains_available(self, rest_client, rest_app):
        mock_wm = MagicMock()
        mock_wm.detect.return_value = _fake_watermark_result("detected", 0.95, available=True)
        with rest_app.container.watermark_service.override(mock_wm):
            data = rest_client.post(
                "/api/v1/watermark/detect",
                files={"audio": ("clip.wav", _fake_wav_bytes(), "audio/wav")},
            ).json()
        assert data["available"] is True

    def test_unavailable_returns_200_with_unavailable_verdict(self, rest_client, rest_app):
        """When the detector is not installed, the service returns verdict='unavailable'
        and the endpoint must still return HTTP 200 (not an error)."""
        mock_wm = MagicMock()
        mock_wm.detect.return_value = _fake_watermark_result("unavailable", 0.0, available=False)
        with rest_app.container.watermark_service.override(mock_wm):
            resp = rest_client.post(
                "/api/v1/watermark/detect",
                files={"audio": ("clip.wav", _fake_wav_bytes(), "audio/wav")},
            )
        assert resp.status_code == 200
        assert resp.json()["verdict"] == "unavailable"

    def test_not_detected_verdict(self, rest_client, rest_app):
        mock_wm = MagicMock()
        mock_wm.detect.return_value = _fake_watermark_result("not_detected", 0.02)
        with rest_app.container.watermark_service.override(mock_wm):
            data = rest_client.post(
                "/api/v1/watermark/detect",
                files={"audio": ("clip.wav", _fake_wav_bytes(), "audio/wav")},
            ).json()
        assert data["verdict"] == "not_detected"

    def test_service_called_with_a_file_path(self, rest_client, rest_app):
        """The route writes the upload to a temp file and passes its path to
        watermark.detect().  The path must be a non-empty string."""
        mock_wm = MagicMock()
        mock_wm.detect.return_value = _fake_watermark_result()
        with rest_app.container.watermark_service.override(mock_wm):
            rest_client.post(
                "/api/v1/watermark/detect",
                files={"audio": ("clip.wav", _fake_wav_bytes(), "audio/wav")},
            )
        call_args = mock_wm.detect.call_args
        assert call_args is not None
        path_arg = call_args.args[0]
        assert isinstance(path_arg, str)
        assert len(path_arg) > 0

    def test_missing_file_returns_422(self, rest_client, rest_app):
        """Omitting the multipart 'audio' field must yield a 422 from FastAPI."""
        resp = rest_client.post("/api/v1/watermark/detect")
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Middleware  (X-Request-Id on every response)
# ─────────────────────────────────────────────────────────────────────────────


class TestRequestIdMiddleware:
    """RequestLoggingMiddleware must add X-Request-Id to every response."""

    _UUID4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")

    def test_health_response_has_x_request_id(self, rest_client, rest_app):
        mock_cfg = AppConfig(device="cpu", watermark_available=False)
        with rest_app.container.app_config.override(mock_cfg):
            resp = rest_client.get("/api/v1/health")
        assert "X-Request-Id" in resp.headers

    def test_x_request_id_is_valid_uuid4(self, rest_client, rest_app):
        mock_cfg = AppConfig(device="cpu", watermark_available=False)
        with rest_app.container.app_config.override(mock_cfg):
            resp = rest_client.get("/api/v1/health")
        rid = resp.headers["X-Request-Id"]
        assert self._UUID4_RE.match(rid), f"Not a UUID4: {rid!r}"

    def test_two_requests_get_different_request_ids(self, rest_client, rest_app):
        mock_cfg = AppConfig(device="cpu", watermark_available=False)
        with rest_app.container.app_config.override(mock_cfg):
            r1 = rest_client.get("/api/v1/health")
            r2 = rest_client.get("/api/v1/health")
        assert r1.headers["X-Request-Id"] != r2.headers["X-Request-Id"]

    def test_tts_error_response_also_has_x_request_id(self, rest_client, rest_app):
        """Middleware must add X-Request-Id even when the handler returns 422."""
        resp = rest_client.post("/api/v1/tts/generate", json={"text": ""})
        assert resp.status_code == 422
        assert "X-Request-Id" in resp.headers

    def test_model_status_response_has_x_request_id(self, rest_client, rest_app):
        mock_manager = MagicMock()
        mock_manager.get_all_status.return_value = _fake_model_statuses()
        with rest_app.container.model_manager_service.override(mock_manager):
            resp = rest_client.get("/api/v1/models/status")
        assert "X-Request-Id" in resp.headers
