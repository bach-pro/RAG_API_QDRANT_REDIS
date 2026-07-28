from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


@dataclass(frozen=True)
class AppConfig:
    qdrant_url: str = os.getenv("RAG_QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("RAG_QDRANT_API_KEY", "")
    collection_name: str = os.getenv("RAG_COLLECTION_NAME", "documents")
    embedding_provider: str = os.getenv("RAG_EMBEDDING_PROVIDER", "ollama")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "embeddinggemma:latest")
    local_ollama_host: str = os.getenv("RAG_LOCAL_OLLAMA_HOST", "http://localhost:11434")
    redis_url: str = os.getenv("RAG_REDIS_URL", "redis://localhost:6379/0")
    chat_memory_ttl_seconds: int = int(os.getenv("RAG_CHAT_MEMORY_TTL_SECONDS", "3600"))
    chat_memory_max_messages: int = int(os.getenv("RAG_CHAT_MEMORY_MAX_MESSAGES", "20"))

    answer_provider: str = os.getenv("RAG_ANSWER_PROVIDER", "ollama")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "")
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "10s")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    google_model: str = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash")
    router_ollama_host: str = os.getenv(
        "RAG_ROUTER_OLLAMA_HOST",
        os.getenv("RAG_LOCAL_OLLAMA_HOST", "http://localhost:11434"),
    )
    router_ollama_api_key: str = os.getenv(
        "RAG_ROUTER_OLLAMA_API_KEY",
        os.getenv("OLLAMA_API_KEY", ""),
    )
    router_model: str = os.getenv("RAG_ROUTER_MODEL", "qwen3:1.7b")

    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1024"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "128"))
    max_upload_files: int = int(os.getenv("RAG_MAX_UPLOAD_FILES", "10"))
    max_upload_file_bytes: int = int(
        os.getenv("RAG_MAX_UPLOAD_FILE_BYTES", str(20 * 1024 * 1024))
    )
    max_extracted_chars: int = int(os.getenv("RAG_MAX_EXTRACTED_CHARS", "2000000"))
    default_k: int = int(os.getenv("RAG_TOP_K", "5"))
    fetch_k: int = int(os.getenv("RAG_FETCH_K", "20"))
    rrf_k: int = int(os.getenv("RAG_RRF_K", "60"))
    mmr_lambda: float = float(os.getenv("RAG_MMR_LAMBDA", "0.5"))

    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    top_p: float = float(os.getenv("OLLAMA_TOP_P", "0.9"))
    num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

    @property
    def uses_direct_ollama_cloud(self) -> bool:
        host = self.ollama_host.rstrip("/").lower()
        return host in {"https://ollama.com", "https://www.ollama.com"}

    def with_overrides(self, **kwargs: object) -> "AppConfig":
        values = self.__dict__.copy()
        values.update(kwargs)
        return AppConfig(**values)
