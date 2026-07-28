from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    collection_name: str
    vector_count: int | None = None
    chroma_count: int | None = None
    error: str | None = None
