from __future__ import annotations

import pytest
import redis

from rag_api import worker as worker_module
from rag_api.services.job_service import RedisJobService


def test_redis_job_service_socket_timeout_is_longer_than_worker_blpop(monkeypatch):
    captured: dict[str, object] = {}
    fake_client = object()

    def fake_from_url(redis_url: str, **kwargs: object) -> object:
        captured["redis_url"] = redis_url
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr(redis.Redis, "from_url", fake_from_url)

    service = RedisJobService("redis://redis:6379/0")

    assert service.client is fake_client
    assert captured["socket_timeout"] == 30.0
    assert captured["socket_connect_timeout"] == 5.0


def test_worker_retries_after_redis_timeout(monkeypatch):
    class FakeJobService:
        queue_name = "ingest_queue"

        def __init__(self) -> None:
            self.dequeue_calls = 0

        def dequeue(self, timeout: int) -> None:
            self.dequeue_calls += 1
            if self.dequeue_calls == 1:
                raise redis.TimeoutError("temporary timeout")
            raise KeyboardInterrupt

    job_service = FakeJobService()
    sleep_calls: list[float] = []
    monkeypatch.setattr(worker_module, "get_job_service", lambda: job_service)
    monkeypatch.setattr(worker_module, "get_rag_service", object)
    monkeypatch.setattr(worker_module, "get_ingest_request_store", object)
    monkeypatch.setattr(worker_module, "IngestJobProcessor", lambda **_kwargs: object())
    monkeypatch.setattr(worker_module.time, "sleep", sleep_calls.append)

    with pytest.raises(KeyboardInterrupt):
        worker_module.run_worker(dequeue_timeout=5, reconnect_delay=0.25)

    assert job_service.dequeue_calls == 2
    assert sleep_calls == [0.25]


def test_redis_job_service_queues_a_second_job_while_first_is_running(monkeypatch):
    class FakeRedis:
        def __init__(self) -> None:
            self.hashes: dict[str, dict[str, str]] = {}
            self.queues: dict[str, list[str]] = {}

        def hset(self, key: str, *, mapping: dict[str, object]) -> None:
            values = self.hashes.setdefault(key, {})
            values.update({field: str(value) for field, value in mapping.items()})

        def hgetall(self, key: str) -> dict[str, str]:
            return dict(self.hashes.get(key, {}))

        def exists(self, key: str) -> bool:
            return key in self.hashes

        def rpush(self, queue: str, job_id: str) -> None:
            self.queues.setdefault(queue, []).append(job_id)

        def blpop(self, queue: str, *, timeout: int) -> tuple[str, str] | None:
            del timeout
            values = self.queues.get(queue, [])
            return (queue, values.pop(0)) if values else None

    fake_redis = FakeRedis()
    monkeypatch.setattr(redis.Redis, "from_url", lambda *_args, **_kwargs: fake_redis)
    service = RedisJobService("redis://redis:6379/0")

    first = service.create_ingest_job("bot-a", "doc-1")
    service.enqueue(first.job_id)
    assert service.dequeue(timeout=1) == first.job_id
    service.update(first.job_id, status="running")

    second = service.create_ingest_job("bot-a", "doc-2")
    service.enqueue(second.job_id)

    assert second.status == "queued"
    assert service.dequeue(timeout=1) == second.job_id
