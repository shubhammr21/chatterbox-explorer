"""
tests/unit/adapters/rest/test_rest_benchmark.py
===============================================
Benchmark tests for FastAPI REST routing overhead.

All services are replaced by instant MagicMock objects so the measured
latency reflects *only* routing, serialisation, middleware, and framework
overhead — not model inference.

Fixture strategy
----------------
* ``bench_app`` and ``bench_client`` are module-scoped so the app is built
  and wired exactly once per module.  The TestClient's ASGI lifespan
  (``with TestClient(bench_app) as client:``) triggers ``container.unwire()``
  on teardown, which keeps the routes module clean for any subsequently
  executed test modules that also call ``build_rest_app()``.

* ``benchmark`` is function-scoped — provided automatically by
  pytest-benchmark for every test function.

Provider override pattern
-------------------------
The override context manager must be active during ALL benchmark iterations::

    with bench_app.container.some_provider.override(mock):
        benchmark(lambda: bench_client.get("/api/v1/..."))
    assert benchmark.stats["median"] < threshold

``benchmark(callable)`` runs the callable many times inside the ``with``
block, populates ``benchmark.stats``, and returns.  The assertion is safe
to perform after the ``with`` block because ``stats`` is already frozen.

Thresholds
----------
health          50 ms  — no inference, no serialisation overhead beyond JSON
tts/generate    50 ms  — WAV encoding of 24 000-sample silence is fast
models/status  100 ms  — JSON list serialisation, slightly more generous
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import numpy as np
import pytest

from domain.models import AppConfig, AudioResult, MemoryStats, ModelStatus

# ─────────────────────────────────────────────────────────────────────────────
# Module-scoped fixtures  (one app + one client for the entire module)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bench_app():
    """Build and wire the FastAPI app exactly once for all benchmarks in this module.

    ``watermark_available=False`` avoids any perth/librosa import side-effects.
    """
    from bootstrap import build_rest_app

    return build_rest_app(watermark_available=False)


@pytest.fixture(scope="module")
def bench_client(bench_app):
    """Open the ASGI lifespan for the module, yield the TestClient, then close it.

    Closing triggers the lifespan shutdown handler which calls
    ``container.unwire()``, leaving the routes module clean for the next test
    module that calls ``build_rest_app()``.
    """
    with TestClient(bench_app) as client:
        yield client


# ─────────────────────────────────────────────────────────────────────────────
# Instant mock factories
# ─────────────────────────────────────────────────────────────────────────────


def _instant_audio() -> AudioResult:
    """1-second silence — fast to encode, sufficient for a WAV response."""
    return AudioResult(
        sample_rate=24000,
        samples=np.zeros(24000, dtype=np.float32),
    )


def _instant_model_statuses() -> list[ModelStatus]:
    return [
        ModelStatus(
            key="tts",
            display_name="Standard TTS",
            class_name="ChatterboxTTS",
            description="Benchmark stub",
            params="500M",
            size_gb=1.4,
            in_memory=False,
            on_disk=True,
        )
    ]


def _instant_memory_stats() -> MemoryStats:
    return MemoryStats(
        sys_total_gb=16.0,
        sys_used_gb=8.0,
        sys_avail_gb=8.0,
        sys_percent=50.0,
        proc_rss_gb=2.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.benchmark(group="rest-routes")
def test_benchmark_health(benchmark, bench_client, bench_app):
    """Measure GET /api/v1/health round-trip latency.

    This endpoint does no I/O and no inference — the measured overhead is
    purely routing + middleware + JSON serialisation.  Expected median < 50 ms
    even on CI-grade hardware.
    """
    mock_cfg = AppConfig(device="cpu", watermark_available=False)

    with bench_app.container.app_config.override(mock_cfg):
        benchmark(lambda: bench_client.get("/api/v1/health"))

    assert benchmark.stats["median"] < 0.050, (
        f"health endpoint median latency {benchmark.stats['median'] * 1000:.2f} ms "
        f"exceeded the 50 ms threshold"
    )


@pytest.mark.benchmark(group="rest-routes")
def test_benchmark_tts_generate(benchmark, bench_client, bench_app):
    """Measure POST /api/v1/tts/generate round-trip latency with instant mock.

    The mock service returns immediately; the benchmark captures routing,
    schema validation, WAV encoding of 24 000-sample silence, and the
    audio/wav Response construction.  Expected median < 50 ms.
    """
    mock_tts = MagicMock()
    mock_tts.generate.return_value = _instant_audio()

    with bench_app.container.tts_service.override(mock_tts):
        benchmark(lambda: bench_client.post("/api/v1/tts/generate", json={"text": "benchmark"}))

    assert benchmark.stats["median"] < 0.050, (
        f"tts/generate endpoint median latency {benchmark.stats['median'] * 1000:.2f} ms "
        f"exceeded the 50 ms threshold"
    )


@pytest.mark.benchmark(group="rest-routes")
def test_benchmark_models_status(benchmark, bench_client, bench_app):
    """Measure GET /api/v1/models/status round-trip latency with instant mock.

    The mock manager returns a single ModelStatus; the benchmark captures
    routing, JSON serialisation of the list response, and middleware overhead.
    A slightly more generous 100 ms threshold is used because the list
    serialisation path is marginally heavier than plain JSON primitives.
    """
    mock_manager = MagicMock()
    mock_manager.get_all_status.return_value = _instant_model_statuses()

    with bench_app.container.model_manager_service.override(mock_manager):
        benchmark(lambda: bench_client.get("/api/v1/models/status"))

    assert benchmark.stats["median"] < 0.100, (
        f"models/status endpoint median latency {benchmark.stats['median'] * 1000:.2f} ms "
        f"exceeded the 100 ms threshold"
    )
