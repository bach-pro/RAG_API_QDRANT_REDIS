from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, StringConstraints


BOT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"
BotId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=BOT_ID_PATTERN,
    ),
]


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    collection_name: str
    vector_count: int | None = None
    chroma_count: int | None = None
    error: str | None = None
