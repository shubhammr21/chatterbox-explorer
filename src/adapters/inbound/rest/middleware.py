"""
src/adapters/inbound/rest/middleware.py
=========================================
HTTP request/response middleware for the FastAPI REST adapter.

RequestLoggingMiddleware
    Emits one structured log record per request containing:
        request_id  — UUID4 string, also returned as X-Request-Id response header
        method      — HTTP verb
        path        — URL path (no query string)
        status_code — integer HTTP status
        duration_ms — wall-clock time from first byte received to response sent

    Uses logging.getLogger("chatterbox.access") so the log record can be
    routed to a separate handler or suppressed independently of the root logger.

    The request_id is attached to request.state so route handlers can include
    it in their own log records if needed.

Architecture note
-----------------
BaseHTTPMiddleware adds one asyncio context switch per request (~microseconds).
For a TTS server where inference takes 1-30 seconds per call, this overhead
is completely negligible. Only for sub-millisecond latency APIs would you
consider a raw ASGI middleware instead.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
import uuid

from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Callable

    from starlette.requests import Request
    from starlette.responses import Response

access_log = logging.getLogger("chatterbox.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Emit one structured access-log record per request.

    Attaches a UUID4 request_id to request.state and to the X-Request-Id
    response header so requests can be correlated across log lines and
    client-side retries.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            access_log.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": duration_ms,
                },
            )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        access_log.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["X-Request-Id"] = request_id
        return response
