from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_api.core.settings import ApiSettings
from rag_api.dependencies.services import get_api_settings, get_rag_service
from rag_api.schemas.common import HealthResponse
from rag_app.services import RagService


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    settings: ApiSettings = Depends(get_api_settings),
    service: RagService = Depends(get_rag_service),
) -> HealthResponse:
    try:
        count = service.count()
    except Exception as exc:
        return HealthResponse(
            status="degraded",
            app=settings.app_name,
            version=settings.app_version,
            collection_name=settings.app_config.collection_name,
            vector_count=None,
            chroma_count=None,
            error=str(exc),
        )

    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        collection_name=settings.app_config.collection_name,
        vector_count=count,
        chroma_count=count,
    )
