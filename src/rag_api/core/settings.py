from __future__ import annotations

import os
from dataclasses import dataclass, field

from rag_app.config import AppConfig


@dataclass(frozen=True)
class ApiSettings:
    app_name: str = os.getenv("RAG_API_NAME", "Multi-source RAG API")
    app_version: str = os.getenv("RAG_API_VERSION", "0.1.0")
    app_config: AppConfig = field(default_factory=AppConfig)
