from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rag_api.schemas.common import BotId


class ChatRequest(BaseModel):
    bot_id: BotId
    question: str = Field(..., min_length=1)
    conversation_id: str | None = Field(default=None, max_length=128)
    mode: str = "Auto Router"
    top_k: int = Field(5, ge=1, le=20)
    fetch_k: int = Field(20, ge=1, le=50)
    mmr_lambda: float = Field(0.5, ge=0.0, le=1.0)


class SourceResponse(BaseModel):
    id: str
    title: str | None = None
    document_id: str | None = None
    source_type: str | None = None
    score: float | None = None
    text_preview: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    bot_id: str
    answer: str
    mode: str
    sources: list[SourceResponse]
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None
