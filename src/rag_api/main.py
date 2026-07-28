from __future__ import annotations

from fastapi import FastAPI

from rag_api.core.logging import configure_api_logging
from rag_api.dependencies.services import get_api_settings
from rag_api.middleware.request_id import RequestIDMiddleware
from rag_api.middleware.request_logging import RequestLoggingMiddleware
from rag_api.routes import chat, health, ingest


def create_app() -> FastAPI:
    configure_api_logging()
    settings = get_api_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(ingest.router)
    return app


app = create_app()
