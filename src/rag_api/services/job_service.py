from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Literal
from uuid import uuid4


JobStatus = Literal["queued", "running", "completed", "failed"]
ACTIVE_JOB_STATUSES = {"queued", "running"}


@dataclass
class IngestJobState:
    job_id: str
    status: JobStatus
    message: str = ""
    documents_loaded: int | None = None
    chunks_indexed: int | None = None
    collection_count: int | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ActiveIngestJobError(RuntimeError):
    def __init__(self, active_job: IngestJobState) -> None:
        super().__init__("An ingest job is already active.")
        self.active_job = active_job


class JobService:
    def __init__(self) -> None:
        self._jobs: dict[str, IngestJobState] = {}
        self._lock = Lock()

    def create_ingest_job(self) -> IngestJobState:
        with self._lock:
            active_job = self._active_ingest_job_unlocked()
            if active_job is not None:
                raise ActiveIngestJobError(active_job)

            now = datetime.now(timezone.utc)
            job = IngestJobState(
                job_id=str(uuid4()),
                status="queued",
                message="Ingest job queued.",
                created_at=now,
                updated_at=now,
            )
            self._jobs[job.job_id] = job
        return job

    def get_active_ingest_job(self) -> IngestJobState | None:
        with self._lock:
            return self._active_ingest_job_unlocked()

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

    def _active_ingest_job_unlocked(self) -> IngestJobState | None:
        for job in self._jobs.values():
            if job.status in ACTIVE_JOB_STATUSES:
                return job
        return None
