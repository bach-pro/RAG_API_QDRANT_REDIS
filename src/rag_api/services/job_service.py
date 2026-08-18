from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Condition, Lock
from typing import Literal
from uuid import uuid4

import redis


JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class IngestJobState:
    job_id: str
    bot_id: str
    doc_id: str | None
    status: JobStatus
    doc_ids: list[str] | None = None
    message: str = ""
    documents_loaded: int | None = None
    chunks_indexed: int | None = None
    collection_count: int | None = None
    replaced_existing: bool | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class JobService:
    """In-memory implementation used by tests and local composition."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestJobState] = {}
        self._lock = Lock()
        self._queue_condition = Condition()
        self._queue: deque[str] = deque()

    def create_ingest_job(
        self,
        bot_id: str,
        doc_id: str | None = None,
        *,
        job_id: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> IngestJobState:
        with self._lock:
            now = datetime.now(timezone.utc)
            job = IngestJobState(
                job_id=job_id or str(uuid4()),
                bot_id=bot_id,
                doc_id=doc_id,
                status="queued",
                doc_ids=doc_ids,
                message="Ingest job queued.",
                created_at=now,
                updated_at=now,
            )
            self._jobs[job.job_id] = job
        return job

    def enqueue(self, job_id: str) -> None:
        with self._queue_condition:
            self._queue.append(job_id)
            self._queue_condition.notify()

    def dequeue(self, timeout: int = 0) -> str | None:
        with self._queue_condition:
            if not self._queue:
                self._queue_condition.wait(timeout=None if timeout == 0 else timeout)
            return self._queue.popleft() if self._queue else None

    def get(self, job_id: str) -> IngestJobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **updates: object) -> IngestJobState:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)
            job.updated_at = datetime.now(timezone.utc)
            return job

class RedisJobService(JobService):
    """Redis-backed state shared by the API producer and ingest worker."""

    def __init__(
        self,
        redis_url: str,
        *,
        queue_name: str = "ingest_queue",
        key_prefix: str = "rag:ingest",
        socket_timeout: float = 30.0,
        socket_connect_timeout: float = 5.0,
    ) -> None:
        self.client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
        )
        self.queue_name = queue_name
        self.key_prefix = key_prefix.rstrip(":")
        self.job_key_prefix = f"{self.key_prefix}:jobs:"

    def _job_key(self, job_id: str) -> str:
        return f"{self.job_key_prefix}{job_id}"

    def create_ingest_job(
        self,
        bot_id: str,
        doc_id: str | None = None,
        *,
        job_id: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> IngestJobState:
        job_id = job_id or str(uuid4())
        now = datetime.now(timezone.utc)
        job = IngestJobState(
            job_id=job_id,
            bot_id=bot_id,
            doc_id=doc_id,
            status="queued",
            doc_ids=doc_ids,
            message="Ingest job queued.",
            created_at=now,
            updated_at=now,
        )

        self.client.hset(self._job_key(job_id), mapping=self._serialize_job(job))
        return job

    def enqueue(self, job_id: str) -> None:
        self.client.rpush(self.queue_name, job_id)

    def dequeue(self, timeout: int = 0) -> str | None:
        item = self.client.blpop(self.queue_name, timeout=timeout)
        if item is None:
            return None
        return item[1]

    def get(self, job_id: str) -> IngestJobState | None:
        values = self.client.hgetall(self._job_key(job_id))
        if not values:
            return None
        return self._deserialize_job(values)

    def update(self, job_id: str, **updates: object) -> IngestJobState:
        if not self.client.exists(self._job_key(job_id)):
            raise KeyError(job_id)

        updates["updated_at"] = datetime.now(timezone.utc)
        encoded = {key: self._encode_value(value) for key, value in updates.items()}
        self.client.hset(self._job_key(job_id), mapping=encoded)

        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    @classmethod
    def _serialize_job(cls, job: IngestJobState) -> dict[str, str | int]:
        return {field: cls._encode_value(value) for field, value in vars(job).items()}

    @staticmethod
    def _encode_value(value: object) -> str | int:
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @classmethod
    def _deserialize_job(cls, values: dict[str, str]) -> IngestJobState:
        def optional_int(field: str) -> int | None:
            value = values.get(field, "")
            return int(value) if value else None

        replaced = values.get("replaced_existing", "")
        return IngestJobState(
            job_id=values["job_id"],
            bot_id=values["bot_id"],
            doc_id=values.get("doc_id") or None,
            status=values["status"],  # type: ignore[arg-type]
            doc_ids=(json.loads(values["doc_ids"]) if values.get("doc_ids") else None),
            message=values.get("message", ""),
            documents_loaded=optional_int("documents_loaded"),
            chunks_indexed=optional_int("chunks_indexed"),
            collection_count=optional_int("collection_count"),
            replaced_existing=(bool(int(replaced)) if replaced else None),
            error=values.get("error") or None,
            created_at=cls._parse_datetime(values.get("created_at")),
            updated_at=cls._parse_datetime(values.get("updated_at")),
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None
