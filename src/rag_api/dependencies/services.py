from __future__ import annotations

from functools import lru_cache

from rag_app.memory import RedisChatMemory
from rag_app.services import RagService

from rag_api.core.settings import ApiSettings
from rag_api.services.ingest_request_store import IngestRequestStore
from rag_api.services.job_service import JobService, RedisJobService


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    return ApiSettings()


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return RagService(get_api_settings().app_config)


@lru_cache(maxsize=1)
def get_job_service() -> JobService:
    settings = get_api_settings()
    return RedisJobService(
        settings.app_config.redis_url,
        queue_name=settings.ingest_queue_name,
    )


@lru_cache(maxsize=1)
def get_ingest_request_store() -> IngestRequestStore:
    return IngestRequestStore(get_api_settings().ingest_data_dir)


@lru_cache(maxsize=1)
def get_chat_memory() -> RedisChatMemory:
    config = get_api_settings().app_config
    return RedisChatMemory(
        redis_url=config.redis_url,
        ttl_seconds=config.chat_memory_ttl_seconds,
        max_messages=config.chat_memory_max_messages,
    )
