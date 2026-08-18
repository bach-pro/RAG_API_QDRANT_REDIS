from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_api.schemas.common import BotId


class IngestDocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    text: str = Field(..., min_length=1)
    source_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: BotId
    doc_id: str | None = Field(
        default=None,
        max_length=256,
        description="Existing document ID to replace; omit it to generate a new ID.",
    )
    documents: list[IngestDocumentInput] = Field(default_factory=list, min_length=1)

    @field_validator("doc_id", mode="before")
    @classmethod
    def normalize_empty_doc_id(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value


class IngestJobResponse(BaseModel):
    job_id: str
    bot_id: str
    doc_id: str | None = None
    doc_ids: list[str] | None = None
    status: Literal["queued", "running", "completed", "failed"]


class IngestJobStatusResponse(BaseModel):
    job_id: str
    bot_id: str
    doc_id: str | None = None
    doc_ids: list[str] | None = None
    status: Literal["queued", "running", "completed", "failed"]
    message: str = ""
    documents_loaded: int | None = None
    chunks_indexed: int | None = None
    collection_count: int | None = None
    replaced_existing: bool | None = None
    error: str | None = None


class BotDeleteResponse(BaseModel):
    bot_id: str
    chunks_deleted: int
