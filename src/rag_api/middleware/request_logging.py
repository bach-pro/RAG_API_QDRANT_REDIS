from __future__ import annotations

import logging
from time import perf_counter

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, logger_name: str = "rag_api") -> None:
        super().__init__(app)
        self.logger = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next) -> Response:
        start = perf_counter()
        request_id = getattr(request.state, "request_id", "-")

        try:
            response = await call_next(request)
        except Exception:
            duration = perf_counter() - start
            self.logger.exception(
                "%s %s 500 %.2fs request_id=%s",
                request.method,
                request.url.path,
                duration,
                request_id,
            )
            raise

        duration = perf_counter() - start
        self.logger.info(
            "%s %s %s %.2fs request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration,
            request_id,
        )
        return response
