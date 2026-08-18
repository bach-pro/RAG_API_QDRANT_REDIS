from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, model_validator

from rag_api.schemas.common import BotId


class ChatRequest(BaseModel):
    bot_id: BotId
    question: str = Field(..., min_length=1)
    conversation_id: str | None = Field(default=None, max_length=128)
    mode: str = "Auto Router"
    top_k: int = Field(5, ge=1, le=20)
    fetch_k: int = Field(20, ge=1, le=50)
    mmr_lambda: float = Field(0.5, ge=0.0, le=1.0)


class ChatStopRequest(BaseModel):
    bot_id: BotId
    conversation_id: str = Field(..., min_length=1, max_length=128)

    @model_validator(mode="before")
    @classmethod
    def parse_json_string(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("body must be a valid JSON object or JSON string") from exc


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
