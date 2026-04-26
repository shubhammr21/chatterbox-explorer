"""
src/adapters/inbound/rest/exception_handlers.py
=================================================
Global exception handlers for the Chatterbox TTS REST adapter.

Why global handlers instead of per-route try/except
----------------------------------------------------
The routes previously duplicated the same 4-line try/except pattern for
ValueError and RuntimeError in every handler (TTS, Turbo, Multilingual,
VC, Watermark, Model load).  Registering centralised handlers:

  1. Eliminates repetition — error translation is declared once.
  2. Catches errors from any layer the route calls (services, DI providers,
     thread-pool callables) without requiring the route to enumerate them.
  3. Keeps route bodies focused on the happy path only.
  4. Preserves try/finally in routes that need cleanup (VC, Watermark) —
     Python's finally block still runs before the exception propagates to
     the handler layer.

Execution order when, e.g., ValueError is raised inside a TTS route:

    1. Route body raises ValueError
    2. (any finally blocks in the route execute — temp file cleanup etc.)
    3. Exception propagates up through run_in_threadpool / inference_semaphore
    4. FastAPI's ExceptionMiddleware catches it
    5. value_error_handler is called → returns JSONResponse(422)
    6. Response flows outward through RequestLoggingMiddleware (logs 422)
    7. CorrelationIdMiddleware appends X-Request-ID to the error response

X-Request-ID on error responses
--------------------------------
CorrelationIdMiddleware is the outermost middleware in the stack.  It hooks
the ASGI http.response.start message and appends X-Request-ID to every
response that flows through it — including error responses produced by these
handlers.  Do NOT add X-Request-ID manually here; doing so would cause the
header to be duplicated.

Handler signature contract (Starlette protocol)
-----------------------------------------------
Starlette's add_exception_handler() requires the handler to be typed as:

    (Request, Exception) -> Response | Awaitable[Response]

The second parameter must be ``Exception`` (the base class), not a specific
subtype.  This is correct by Liskov substitution / contravariance: a handler
registered for ValueError must still satisfy the protocol that expects any
Exception.  We therefore type all handlers with ``exc: Exception`` and use
isinstance() narrowing inside the body where subtype attributes are needed.

Registration
------------
All handlers are registered in bootstrap.build_rest_app() via
app.add_exception_handler(), NOT via the @app.exception_handler() decorator,
because the handlers are defined in this separate module and the decorator
form would require importing the app instance at definition time.

Handler registration order
--------------------------
  app.add_exception_handler(TTSInputError,              tts_input_error_handler)
  app.add_exception_handler(VoiceConversionInputError,  vc_input_error_handler)
  app.add_exception_handler(ModelError,                 model_error_handler)
  app.add_exception_handler(ChatterboxError,            chatterbox_error_handler)
  app.add_exception_handler(StarletteHTTPException,     http_exception_handler_with_logging)
  app.add_exception_handler(RequestValidationError,     validation_exception_handler_with_logging)

Starlette resolves handlers by walking the exception's MRO and picking the
most-specific registered type.  Register against StarletteHTTPException (not
fastapi.HTTPException) per FastAPI docs so that exceptions raised by Starlette
internals (405 Method Not Allowed, middleware-level 400s, etc.) are also
handled.

Logging policy
--------------
  TTSInputError             — caller input error; NOT logged (noise, client's fault)
  VoiceConversionInputError — caller input error; NOT logged (noise, client's fault)
  ModelError                — infrastructure failure; logged at ERROR with traceback
  ChatterboxError           — unexpected domain error; logged at ERROR with traceback
  HTTPException 4xx         — expected client error; NOT logged at ERROR
  HTTPException 5xx         — unexpected server error; logged at ERROR
  RequestValidationError    — bad request schema; logged at DEBUG only
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

if TYPE_CHECKING:
    from fastapi import Request
    from starlette.responses import Response

log = logging.getLogger(__name__)


async def tts_input_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Translate ``TTSInputError`` (``EmptyTextError``, ``ReferenceTooShortError``) → 422.

    These are expected caller errors — empty text, reference audio too short,
    etc.  They are deliberately NOT logged because they are the client's fault
    and would generate noise in the error log.

    Args:
        request: The incoming HTTP request (unused directly, required by protocol).
        exc: The exception that was raised; will be a ``TTSInputError`` instance.

    Returns:
        A ``JSONResponse`` with status 422 and body ``{"detail": "<message>"}``.
    """
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def vc_input_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Translate ``VoiceConversionInputError`` → 422.

    These are expected caller errors — missing source or target audio paths.
    Deliberately NOT logged for the same reason as ``tts_input_error_handler``.

    Args:
        request: The incoming HTTP request (unused directly, required by protocol).
        exc: The exception that was raised; will be a ``VoiceConversionInputError`` instance.

    Returns:
        A ``JSONResponse`` with status 422 and body ``{"detail": "<message>"}``.
    """
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def model_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Translate ``ModelError`` (``ModelLoadError``, ``ModelNotLoadedError``, ``InferenceError``) → 503.

    These are unexpected infrastructure failures — model load failure, GPU OOM,
    inference crash, etc.  The full traceback is logged at ERROR level because
    these require operator attention.

    Args:
        request: The incoming HTTP request.
        exc: The exception that was raised; will be a ``ModelError`` instance.

    Returns:
        A ``JSONResponse`` with status 503 and body ``{"detail": "<message>"}``.
    """
    log.error(
        "Model error at %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=exc,
    )
    return JSONResponse(status_code=503, content={"detail": str(exc)})


async def chatterbox_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for any ``ChatterboxError`` not handled by a more specific handler.

    Provides a safety net for new domain exception subtypes that have not yet
    been assigned a dedicated handler.  Logged at ERROR because any unhandled
    domain exception is unexpected at the adapter boundary.

    Args:
        request: The incoming HTTP request.
        exc: The exception that was raised; will be a ``ChatterboxError`` instance.

    Returns:
        A ``JSONResponse`` with status 500 and body ``{"detail": "<message>"}``.
    """
    log.error(
        "Unhandled domain error at %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=exc,
    )
    return JSONResponse(status_code=500, content={"detail": str(exc)})


async def http_exception_handler_with_logging(
    request: Request,
    exc: Exception,
) -> Response:
    """Wrap FastAPI's default ``HTTPException`` handler with selective logging.

    Delegates entirely to ``fastapi.exception_handlers.http_exception_handler``
    for response construction so the response body and header format is
    identical to what FastAPI produces by default.

    Logging policy:
      - ``status_code >= 500``: logged at ERROR — unexpected server faults.
      - ``status_code < 500``: not logged at ERROR — 4xx are expected client
        mistakes (bad key, method not allowed, etc.) and would generate noise.

    Registered against ``starlette.exceptions.HTTPException`` (not
    ``fastapi.HTTPException``) so that exceptions raised by Starlette
    internals (e.g. 405 Method Not Allowed from the router) are also caught.

    The ``exc`` parameter is typed as ``Exception`` to satisfy the Starlette
    handler protocol.  An isinstance check narrows to ``StarletteHTTPException``
    before accessing ``.status_code`` and delegating to the default handler.

    Args:
        request: The incoming HTTP request.
        exc: The exception that was raised; will be a ``StarletteHTTPException``.

    Returns:
        The same ``Response`` that FastAPI's default handler would return.
    """
    http_exc = exc if isinstance(exc, StarletteHTTPException) else StarletteHTTPException(500)
    if http_exc.status_code >= 500:
        log.error(
            "HTTP %d at %s %s: %s",
            http_exc.status_code,
            request.method,
            request.url.path,
            http_exc.detail,
        )
    return await http_exception_handler(request, http_exc)


async def validation_exception_handler_with_logging(
    request: Request,
    exc: Exception,
) -> Response:
    """Wrap FastAPI's default ``RequestValidationError`` handler with DEBUG logging.

    Delegates entirely to
    ``fastapi.exception_handlers.request_validation_exception_handler`` for
    response construction — the response body is the standard FastAPI 422
    payload with a ``"detail"`` list of per-field Pydantic error objects.

    Logging policy: DEBUG only.  Validation errors are common client mistakes
    (wrong field type, missing required field) and do not indicate a server
    problem.  Logging at DEBUG makes them visible during local development
    while keeping production logs clean.

    The ``exc`` parameter is typed as ``Exception`` to satisfy the Starlette
    handler protocol.  An isinstance check narrows to ``RequestValidationError``
    before accessing ``.errors()`` and delegating to the default handler.

    Args:
        request: The incoming HTTP request.
        exc: The exception that was raised; will be a ``RequestValidationError``.

    Returns:
        The same 422 ``Response`` that FastAPI's default handler would return.
    """
    val_exc = exc if isinstance(exc, RequestValidationError) else RequestValidationError([])
    log.debug(
        "Request validation error at %s %s: %s",
        request.method,
        request.url.path,
        val_exc.errors(),
    )
    return await request_validation_exception_handler(request, val_exc)
