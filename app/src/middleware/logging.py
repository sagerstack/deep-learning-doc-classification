import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.src.config import ENVIRONMENT


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            environment=ENVIRONMENT,
            request_id=str(uuid.uuid4()),
            method=request.method,
            path=request.url.path,
        )
        response = await call_next(request)
        return response
