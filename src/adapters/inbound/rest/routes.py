"""
src/adapters/inbound/rest/routes.py
========================================================
FastAPI REST routes — stub showing where to add REST endpoints.

Install dependencies first:
    uv add --optional rest "fastapi>=0.100.0" "uvicorn[standard]>=0.20.0"

Architecture note
-----------------
The REST adapter is a *primary* adapter — a driving adapter that sits on the
same hexagonal boundary as the Gradio adapter.  Both adapters accept the same
domain request value-objects (TTSRequest, TurboTTSRequest, etc.) and call the
same service ports (ITTSService, ITurboTTSService, etc.).

No domain changes are required to add REST:
    Gradio UI  ──►  GradioHandlers  ──►  ITTSService  ──►  (model)
    REST API   ──►  FastAPI routes  ──►  ITTSService  ──►  (model)

The domain layer is blissfully unaware of which adapter is driving it.

Dependency injection for FastAPI
---------------------------------
FastAPI uses Depends() for DI.  The pattern mirrors the Gradio adapter's
constructor injection, except services are resolved per-request via a
get_*_service() provider function rather than once at startup:

    def get_tts_service() -> ITTSService:
        # Return the singleton service wired up at application startup.
        return _app_state.tts_service

    @router.post("/tts/generate")
    async def generate_tts(
        request: TTSRequest,
        tts_service: ITTSService = Depends(get_tts_service),
    ):
        ...

In practice, for a single-process deployment the services are stateful
singletons (the models live in GPU memory), so Depends(get_tts_service)
should resolve to the same object on every request.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Uncomment the block below once FastAPI and uvicorn are installed:
#
#     uv add --optional rest "fastapi>=0.100.0" "uvicorn[standard]>=0.20.0"
# ---------------------------------------------------------------------------

# from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
# from fastapi.responses import StreamingResponse, JSONResponse
# from domain.models import (
#     TTSRequest,
#     TurboTTSRequest,
#     MultilingualTTSRequest,
#     VoiceConversionRequest,
#     AudioResult,
#     WatermarkResult,
# )
# from ports.input import (
#     ITTSService,
#     ITurboTTSService,
#     IMultilingualTTSService,
#     IVoiceConversionService,
#     IModelManagerService,
#     IWatermarkService,
# )
#
#
# router = APIRouter(prefix="/api/v1", tags=["chatterbox"])
#
#
# # ── Dependency providers ──────────────────────────────────────────────────
# # Replace with your actual application-state accessor once wired up.
#
# def get_tts_service() -> ITTSService:
#     raise NotImplementedError("Wire ITTSService via app startup state.")
#
# def get_turbo_service() -> ITurboTTSService:
#     raise NotImplementedError("Wire ITurboTTSService via app startup state.")
#
# def get_mtl_service() -> IMultilingualTTSService:
#     raise NotImplementedError("Wire IMultilingualTTSService via app startup state.")
#
# def get_vc_service() -> IVoiceConversionService:
#     raise NotImplementedError("Wire IVoiceConversionService via app startup state.")
#
# def get_manager_service() -> IModelManagerService:
#     raise NotImplementedError("Wire IModelManagerService via app startup state.")
#
# def get_watermark_service() -> IWatermarkService:
#     raise NotImplementedError("Wire IWatermarkService via app startup state.")
#
#
# # ── TTS endpoints ─────────────────────────────────────────────────────────
#
# @router.post("/tts/generate", summary="Generate TTS audio (one-shot)")
# async def generate_tts(
#     request: TTSRequest,
#     tts_service: ITTSService = Depends(get_tts_service),
# ):
#     """Generate a complete TTS audio clip in one shot.
#
#     Uses the same TTSService and TTSRequest domain model as the Gradio adapter —
#     no duplication of generation logic.
#
#     Returns the audio as a WAV byte stream (audio/wav).
#     """
#     result: AudioResult = tts_service.generate(request)
#     wav_bytes = _audio_result_to_wav_bytes(result)
#     return StreamingResponse(
#         iter([wav_bytes]),
#         media_type="audio/wav",
#         headers={"Content-Disposition": "attachment; filename=output.wav"},
#     )
#
#
# @router.post("/tts/stream", summary="Stream TTS audio sentence-by-sentence")
# async def stream_tts(
#     request: TTSRequest,
#     tts_service: ITTSService = Depends(get_tts_service),
# ):
#     """Stream TTS audio sentence-by-sentence.
#
#     Each chunk in the response is a raw PCM byte sequence for one cumulative
#     audio segment.  Clients should buffer and concatenate chunks, or replace
#     the previous segment with each successive (cumulative) one.
#     """
#     def _audio_generator():
#         for chunk in tts_service.generate_stream(request):
#             yield chunk.samples.tobytes()
#
#     return StreamingResponse(_audio_generator(), media_type="audio/wav")
#
#
# # ── Turbo TTS endpoints ───────────────────────────────────────────────────
#
# @router.post("/turbo/generate", summary="Generate Turbo TTS audio (one-shot)")
# async def generate_turbo(
#     request: TurboTTSRequest,
#     turbo_service: ITurboTTSService = Depends(get_turbo_service),
# ):
#     """Generate a complete Turbo TTS audio clip in one shot.
#
#     Turbo is faster and lower-VRAM than Standard TTS, supports paralinguistic
#     tags ([laugh], [sigh], etc.), but does not support exaggeration or CFG weight.
#     """
#     result: AudioResult = turbo_service.generate(request)
#     wav_bytes = _audio_result_to_wav_bytes(result)
#     return StreamingResponse(
#         iter([wav_bytes]),
#         media_type="audio/wav",
#         headers={"Content-Disposition": "attachment; filename=output_turbo.wav"},
#     )
#
#
# @router.post("/turbo/stream", summary="Stream Turbo TTS audio sentence-by-sentence")
# async def stream_turbo(
#     request: TurboTTSRequest,
#     turbo_service: ITurboTTSService = Depends(get_turbo_service),
# ):
#     """Stream Turbo TTS audio sentence-by-sentence."""
#     def _audio_generator():
#         for chunk in turbo_service.generate_stream(request):
#             yield chunk.samples.tobytes()
#
#     return StreamingResponse(_audio_generator(), media_type="audio/wav")
#
#
# # ── Multilingual TTS endpoints ────────────────────────────────────────────
#
# @router.post("/multilingual/generate", summary="Generate multilingual TTS audio")
# async def generate_multilingual(
#     request: MultilingualTTSRequest,
#     mtl_service: IMultilingualTTSService = Depends(get_mtl_service),
# ):
#     """Generate a complete multilingual TTS audio clip in one shot.
#
#     Supports 23 languages with zero-shot cross-language voice cloning.
#     Set cfg_weight=0 when the reference speaker's language differs from the
#     target language to minimise accent bleed.
#     """
#     result: AudioResult = mtl_service.generate(request)
#     wav_bytes = _audio_result_to_wav_bytes(result)
#     return StreamingResponse(
#         iter([wav_bytes]),
#         media_type="audio/wav",
#         headers={"Content-Disposition": "attachment; filename=output_mtl.wav"},
#     )
#
#
# # ── Voice Conversion endpoint ─────────────────────────────────────────────
#
# @router.post("/vc/convert", summary="Convert source audio to target voice")
# async def convert_voice(
#     source_audio: UploadFile = File(..., description="Audio file whose content to preserve"),
#     target_voice: UploadFile = File(..., description="Reference audio for the target voice identity"),
#     vc_service: IVoiceConversionService = Depends(get_vc_service),
# ):
#     """Convert source audio to sound like the target voice.
#
#     The speech content (words, timing, prosody) comes from source_audio.
#     The voice identity (timbre, accent, character) comes from target_voice.
#     No text prompt is needed.
#     """
#     import tempfile, os
#
#     # Write uploads to temp files so the service can load them via torchaudio.
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src_tmp:
#         src_tmp.write(await source_audio.read())
#         src_path = src_tmp.name
#
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tgt_tmp:
#         tgt_tmp.write(await target_voice.read())
#         tgt_path = tgt_tmp.name
#
#     try:
#         vc_request = VoiceConversionRequest(
#             source_audio_path=src_path,
#             target_voice_path=tgt_path,
#         )
#         result: AudioResult = vc_service.convert(vc_request)
#         wav_bytes = _audio_result_to_wav_bytes(result)
#         return StreamingResponse(
#             iter([wav_bytes]),
#             media_type="audio/wav",
#             headers={"Content-Disposition": "attachment; filename=converted.wav"},
#         )
#     finally:
#         os.unlink(src_path)
#         os.unlink(tgt_path)
#
#
# # ── Model Manager endpoints ───────────────────────────────────────────────
#
# @router.get("/models/status", summary="Get status of all models")
# async def get_model_status(
#     manager: IModelManagerService = Depends(get_manager_service),
# ):
#     """Return the current load/disk status of every registered model."""
#     statuses = manager.get_all_status()
#     return JSONResponse([
#         {
#             "key":          s.key,
#             "display_name": s.display_name,
#             "class_name":   s.class_name,
#             "description":  s.description,
#             "params":       s.params,
#             "size_gb":      s.size_gb,
#             "in_memory":    s.in_memory,
#             "on_disk":      s.on_disk,
#         }
#         for s in statuses
#     ])
#
#
# @router.post("/models/{key}/load", summary="Load a model into memory")
# async def load_model(
#     key: str,
#     manager: IModelManagerService = Depends(get_manager_service),
# ):
#     """Load the model identified by *key* into GPU/CPU memory."""
#     try:
#         message = manager.load(key)
#         return {"status": "ok", "message": message}
#     except RuntimeError as e:
#         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
#
#
# @router.post("/models/{key}/unload", summary="Unload a model from memory")
# async def unload_model(
#     key: str,
#     manager: IModelManagerService = Depends(get_manager_service),
# ):
#     """Unload the model identified by *key*, flushing device memory."""
#     message = manager.unload(key)
#     return {"status": "ok", "message": message}
#
#
# @router.get("/models/memory", summary="Get system and device memory statistics")
# async def get_memory_stats(
#     manager: IModelManagerService = Depends(get_manager_service),
# ):
#     """Return current RAM and GPU/MPS memory usage."""
#     stats = manager.get_memory_stats()
#     return {
#         "sys_total_gb":      stats.sys_total_gb,
#         "sys_used_gb":       stats.sys_used_gb,
#         "sys_avail_gb":      stats.sys_avail_gb,
#         "sys_percent":       stats.sys_percent,
#         "proc_rss_gb":       stats.proc_rss_gb,
#         "device_name":       stats.device_name,
#         "device_driver_gb":  stats.device_driver_gb,
#         "device_max_gb":     stats.device_max_gb,
#     }
#
#
# # ── Watermark endpoint ────────────────────────────────────────────────────
#
# @router.post("/watermark/detect", summary="Detect PerTh watermark in audio")
# async def detect_watermark(
#     audio: UploadFile = File(..., description="Audio file to check for a PerTh watermark"),
#     watermark_service: IWatermarkService = Depends(get_watermark_service),
# ):
#     """Detect whether the uploaded audio carries a Chatterbox PerTh watermark.
#
#     Returns the raw detection score and verdict so clients can apply their
#     own threshold logic if needed.
#     """
#     import tempfile, os
#
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#         tmp.write(await audio.read())
#         tmp_path = tmp.name
#
#     try:
#         result: WatermarkResult = watermark_service.detect(tmp_path)
#         return {
#             "score":     result.score,
#             "verdict":   result.verdict,
#             "message":   result.message,
#             "available": result.available,
#         }
#     finally:
#         os.unlink(tmp_path)
#
#
# # ── Internal helpers ──────────────────────────────────────────────────────
#
# def _audio_result_to_wav_bytes(result: AudioResult) -> bytes:
#     """Encode an AudioResult (float32 numpy array) as in-memory WAV bytes.
#
#     Uses scipy.io.wavfile for lightweight WAV encoding without requiring
#     torchaudio in the REST adapter layer.
#     """
#     import io
#     import numpy as np
#     from scipy.io import wavfile
#
#     arr = result.samples.astype(np.float32)
#     peak = float(np.abs(arr).max())
#     if peak > 1.0:
#         arr = arr / peak
#     int16_arr = np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)
#
#     buf = io.BytesIO()
#     wavfile.write(buf, result.sample_rate, int16_arr)
#     return buf.getvalue()
