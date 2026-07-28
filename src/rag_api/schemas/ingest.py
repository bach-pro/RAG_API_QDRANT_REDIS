from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestDocumentInput(BaseModel):
    id: str | None = None
    title: str | None = None
    text: str = Field(..., min_length=1)
    source_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestRequest(BaseModel):
    reset: bool = False
    documents: list[IngestDocumentInput] = Field(default_factory=list, min_length=1)


class IngestJobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]


class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    message: str = ""
    documents_loaded: int | None = None
    chunks_indexed: int | None = None
    collection_count: int | None = None
    error: str | None = None
