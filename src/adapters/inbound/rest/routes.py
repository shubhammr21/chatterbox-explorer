"""
src/adapters/inbound/rest/routes.py
========================================================
FastAPI route handlers for the Chatterbox TTS REST adapter.

Architecture contract
---------------------
- All service dependencies are injected via dependency-injector's @inject +
  Depends(Provide[AppContainer.*]) pattern.
- Decorator order is load-bearing: @router.*(…) → @inject → async def.
  Reversing @inject and @router silently breaks injection.
- ALL blocking sync service calls are offloaded to a thread pool via:
      await run_in_threadpool(service.method, args)
  This keeps the asyncio event loop free during 5-30 s inference operations.
- GPU/CPU-bound calls are additionally gated by inference_semaphore to
  prevent concurrent model execution that would saturate the device.
- Error translation:
      ValueError     → HTTP 422 Unprocessable Entity (caller input error)
      RuntimeError   → HTTP 503 Service Unavailable  (infrastructure failure)
      unknown key    → HTTP 404 Not Found
- VC and watermark endpoints accept UploadFile; bytes are written to a temp
  file before passing the path to the domain service. Cleanup in finally.
- This module is imported INSIDE build_rest_app() — safely after all compat
  patches applied in cli.main() have fired.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING, cast

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

from adapters.inbound.rest.concurrency import inference_semaphore
from adapters.inbound.rest.schemas import (
    HealthResponse,
    MemoryStatsResponse,
    MessageResponse,
    ModelStatusResponse,
    MultilingualRequestSchema,
    TTSRequestSchema,
    TurboRequestSchema,
    WatermarkResponse,
    audio_result_to_wav_bytes,
)
from infrastructure.container import AppContainer

if TYPE_CHECKING:
    from domain.models import AppConfig
    from domain.types import ModelKey
    from ports.input import (
        IModelManagerService,
        IMultilingualTTSService,
        ITTSService,
        ITurboTTSService,
        IVoiceConversionService,
        IWatermarkService,
    )

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["health"],
)
@inject
async def health(
    config: AppConfig = Depends(Provide[AppContainer.app_config]),
) -> HealthResponse:
    """Return service health and the active compute device.

    This endpoint has NO blocking operations and responds immediately even
    when an inference request is running on the GPU/CPU.
    """
    return HealthResponse(status="ok", device=config.device)


# ──────────────────────────────────────────────────────────────────────────────
# Standard TTS
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/tts/generate",
    summary="Generate TTS audio (one-shot)",
    response_class=Response,
    responses={
        200: {"content": {"audio/wav": {}}, "description": "WAV audio file"},
        422: {"description": "Input validation error (empty text, bad params)"},
        503: {"description": "Model or infrastructure failure"},
    },
    tags=["tts"],
)
@inject
async def generate_tts(
    body: TTSRequestSchema,
    tts: ITTSService = Depends(Provide[AppContainer.tts_service]),
) -> Response:
    """Generate a complete Standard TTS audio clip in one shot.

    tts.generate() is a blocking synchronous call (5-30 s on CPU, 1-5 s on GPU).
    It is offloaded to a thread pool via run_in_threadpool so the event loop
    stays free during inference. inference_semaphore ensures only one inference
    runs at a time on the shared GPU/CPU device.

    Voice cloning via reference audio is not supported in v1 of the REST API
    (ref_audio_path is always None). Support will be added in v2 via UploadFile.
    """
    try:
        async with inference_semaphore:
            result = await run_in_threadpool(tts.generate, body.to_domain())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        log.exception("TTS generation failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    wav = audio_result_to_wav_bytes(result)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Turbo TTS
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/turbo/generate",
    summary="Generate Turbo TTS audio (one-shot)",
    response_class=Response,
    responses={
        200: {"content": {"audio/wav": {}}, "description": "WAV audio file"},
        422: {"description": "Input validation error (empty text, reference audio too short)"},
        503: {"description": "Model or infrastructure failure"},
    },
    tags=["turbo"],
)
@inject
async def generate_turbo(
    body: TurboRequestSchema,
    turbo: ITurboTTSService = Depends(Provide[AppContainer.turbo_service]),
) -> Response:
    """Generate a complete Turbo TTS audio clip in one shot.

    Turbo is faster and lower-VRAM than Standard TTS. Supports paralinguistic
    tags ([laugh], [sigh], [clears throat], etc.) but does not support
    exaggeration or CFG weight controls.
    """
    try:
        async with inference_semaphore:
            result = await run_in_threadpool(turbo.generate, body.to_domain())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        log.exception("Turbo TTS generation failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    wav = audio_result_to_wav_bytes(result)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output_turbo.wav"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Multilingual TTS
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/multilingual/generate",
    summary="Generate multilingual TTS audio (one-shot)",
    response_class=Response,
    responses={
        200: {"content": {"audio/wav": {}}, "description": "WAV audio file"},
        422: {"description": "Input validation error (empty text, unknown language)"},
        503: {"description": "Model or infrastructure failure"},
    },
    tags=["multilingual"],
)
@inject
async def generate_multilingual(
    body: MultilingualRequestSchema,
    mtl: IMultilingualTTSService = Depends(Provide[AppContainer.multilingual_service]),
) -> Response:
    """Generate a complete multilingual TTS audio clip in one shot.

    Supports 23 languages with zero-shot cross-language voice cloning.
    Pass the ISO 639-1 language code in the ``language`` field (e.g. ``"fr"``
    for French). Set ``cfg_weight=0`` when the reference speaker's language
    differs from the target to minimise accent bleed.
    """
    try:
        async with inference_semaphore:
            result = await run_in_threadpool(mtl.generate, body.to_domain())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        log.exception("Multilingual TTS generation failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    wav = audio_result_to_wav_bytes(result)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output_multilingual.wav"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Voice Conversion
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/vc/convert",
    summary="Convert source audio to target voice",
    response_class=Response,
    responses={
        200: {"content": {"audio/wav": {}}, "description": "WAV audio file"},
        422: {"description": "Input validation error (missing audio files)"},
        503: {"description": "Model or infrastructure failure"},
    },
    tags=["voice-conversion"],
)
@inject
async def convert_voice(
    source_audio: UploadFile = File(
        ...,
        description="Audio file whose speech content to preserve.",
    ),
    target_voice: UploadFile = File(
        ...,
        description="Reference audio for the desired voice identity.",
    ),
    vc: IVoiceConversionService = Depends(Provide[AppContainer.vc_service]),
) -> Response:
    """Convert source audio to sound like the target voice.

    The speech content (words, timing, prosody) comes from ``source_audio``.
    The voice identity (timbre, accent, character) comes from ``target_voice``.
    No text prompt is required.

    Both files must be in a format supported by torchaudio (WAV, MP3, FLAC, etc.).
    The converted audio is returned as a 16-bit mono WAV file.
    """
    from domain.models import VoiceConversionRequest

    src_path: str | None = None
    tgt_path: str | None = None

    try:
        # await upload.read() is async I/O — no threadpool needed here.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src_tmp:
            src_tmp.write(await source_audio.read())
            src_path = src_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tgt_tmp:
            tgt_tmp.write(await target_voice.read())
            tgt_path = tgt_tmp.name

        req = VoiceConversionRequest(
            source_audio_path=src_path,
            target_voice_path=tgt_path,
        )
        # vc.convert() is blocking inference — offload to thread pool.
        async with inference_semaphore:
            result = await run_in_threadpool(vc.convert, req)

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        log.exception("Voice conversion failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    finally:
        for path in (src_path, tgt_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    log.warning("Failed to delete temp file: %s", path)

    wav = audio_result_to_wav_bytes(result)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=converted.wav"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model Management
# ──────────────────────────────────────────────────────────────────────────────


@router.get(
    "/models/status",
    response_model=list[ModelStatusResponse],
    summary="Get status of all models",
    tags=["models"],
)
@inject
async def get_model_status(
    manager: IModelManagerService = Depends(Provide[AppContainer.model_manager_service]),
) -> list[ModelStatusResponse]:
    """Return the current load/disk status of every registered Chatterbox model.

    get_all_status() performs disk-cache probes (filesystem I/O) — offloaded
    to a thread pool so the event loop stays free.
    """
    statuses = await run_in_threadpool(manager.get_all_status)
    return [ModelStatusResponse.from_domain(s) for s in statuses]


@router.get(
    "/models/memory",
    response_model=MemoryStatsResponse,
    summary="Get system and device memory statistics",
    tags=["models"],
)
@inject
async def get_memory_stats(
    manager: IModelManagerService = Depends(Provide[AppContainer.model_manager_service]),
) -> MemoryStatsResponse:
    """Return current system RAM and GPU/MPS memory usage.

    get_memory_stats() calls psutil and torch memory APIs — offloaded to a
    thread pool to avoid blocking the event loop.
    """
    stats = await run_in_threadpool(manager.get_memory_stats)
    return MemoryStatsResponse.from_domain(stats)


@router.post(
    "/models/{key}/load",
    response_model=MessageResponse,
    summary="Load a model into memory",
    responses={
        200: {"description": "Model loaded (or already in memory)"},
        404: {"description": "Unknown model key"},
        503: {"description": "Model failed to load (OOM, missing weights, etc.)"},
    },
    tags=["models"],
)
@inject
async def load_model(
    key: str,
    manager: IModelManagerService = Depends(Provide[AppContainer.model_manager_service]),
) -> MessageResponse:
    """Load the model identified by ``key`` into GPU/CPU memory.

    Valid keys: ``tts``, ``turbo``, ``multilingual``, ``vc``.

    manager.load() downloads weights and initialises the model on the device —
    this can take seconds and is blocking I/O; offloaded to a thread pool.
    """
    known_keys = {s.key for s in await run_in_threadpool(manager.get_all_status)}
    if key not in known_keys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown model key {key!r}. Valid keys: {sorted(known_keys)}",
        )
    try:
        async with inference_semaphore:
            message = await run_in_threadpool(manager.load, cast("ModelKey", key))
    except RuntimeError as exc:
        log.exception("Model load failed: %s", key)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    return MessageResponse(message=message)


@router.post(
    "/models/{key}/unload",
    response_model=MessageResponse,
    summary="Unload a model from memory",
    responses={
        200: {"description": "Model unloaded (or was not loaded)"},
        404: {"description": "Unknown model key"},
    },
    tags=["models"],
)
@inject
async def unload_model(
    key: str,
    manager: IModelManagerService = Depends(Provide[AppContainer.model_manager_service]),
) -> MessageResponse:
    """Unload the model identified by ``key``, flushing device memory.

    manager.unload() synchronizes GPU cache and runs gc.collect() — blocking;
    offloaded to a thread pool.
    """
    known_keys = {s.key for s in await run_in_threadpool(manager.get_all_status)}
    if key not in known_keys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown model key {key!r}. Valid keys: {sorted(known_keys)}",
        )
    async with inference_semaphore:
        message = await run_in_threadpool(manager.unload, cast("ModelKey", key))
    return MessageResponse(message=message)


# ──────────────────────────────────────────────────────────────────────────────
# Watermark Detection
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/watermark/detect",
    response_model=WatermarkResponse,
    summary="Detect PerTh watermark in audio",
    responses={
        200: {"description": "Detection result (available or unavailable)"},
        422: {"description": "Missing or unreadable audio file"},
    },
    tags=["watermark"],
)
@inject
async def detect_watermark(
    audio: UploadFile = File(
        ...,
        description="Audio file to check for a Chatterbox PerTh watermark.",
    ),
    watermark: IWatermarkService = Depends(Provide[AppContainer.watermark_service]),
) -> WatermarkResponse:
    """Detect whether the uploaded audio carries a Chatterbox PerTh watermark.

    Returns the raw detection score, a human-readable verdict, and a flag
    indicating whether the detection library is available. When the library
    is not installed the verdict will be ``"unavailable"`` with score 0.0.

    Verdict values:
    - ``detected``     — watermark found above threshold
    - ``not_detected`` — no watermark signal found
    - ``inconclusive`` — score is in the ambiguous range
    - ``unavailable``  — detection library not installed
    """
    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        # watermark.detect() runs neural inference — offload to thread pool.
        async with inference_semaphore:
            result = await run_in_threadpool(watermark.detect, tmp_path)

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                log.warning("Failed to delete temp file: %s", tmp_path)

    return WatermarkResponse.from_domain(result)
