"""
src/adapters/inbound/rest/middleware.py
=========================================
HTTP request/response middleware for the FastAPI REST adapter.

RequestLoggingMiddleware
------------------------
Emits one structured access-log record per request containing:

    method      -- HTTP verb (GET, POST, ...)
    path        -- URL path, no query string
    status_code -- integer HTTP response status
    duration_ms -- wall-clock milliseconds from first byte received to
                   response headers sent

Correlation ID
--------------
This middleware has NO responsibility for generating or attaching a
correlation / request ID.  That is entirely delegated to
``CorrelationIdMiddleware`` from the ``asgi-correlation-id`` library, which
must be registered as the **outermost** middleware (added last via
``app.add_middleware``).  By the time ``RequestLoggingMiddleware.dispatch()``
is called, the ``correlation_id`` ContextVar is already set.

The ``CorrelationIdFilter`` attached to the JSON log handler (configured in
``logging_config.configure_json()``) automatically injects
``record.correlation_id`` into every ``LogRecord`` that flows through the
handler -- including the access-log records emitted here.  No manual
``correlation_id.get()`` call is needed in this class.

The ``X-Request-ID`` response header is also set by ``CorrelationIdMiddleware``
-- this class does **not** touch response headers.

Middleware stack (request order, outermost to innermost):

    CorrelationIdMiddleware   -- sets ContextVar + X-Request-ID header
    RequestLoggingMiddleware  -- measures timing, emits access log
    route handler             -- business logic

Architecture note
-----------------
``BaseHTTPMiddleware`` adds one asyncio context switch per request
(~microseconds overhead).  For a TTS server where inference takes 1-30
seconds, this is completely negligible.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Callable

    from starlette.requests import Request
    from starlette.responses import Response

access_log = logging.getLogger("chatterbox.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Emit one structured access-log record per HTTP request.

    The ``CorrelationIdFilter`` on the JSON handler enriches every emitted
    ``LogRecord`` with ``correlation_id`` automatically -- this class is
    intentionally unaware of correlation IDs.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            access_log.error(
                "Request failed",
                extra={
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
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
