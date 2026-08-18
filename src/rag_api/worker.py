from __future__ import annotations

import logging
import time

import redis

from rag_api.core.logging import configure_api_logging
from rag_api.dependencies.services import (
    get_ingest_request_store,
    get_job_service,
    get_rag_service,
)
from rag_api.services.ingest_processor import IngestJobProcessor


logger = logging.getLogger("rag_api")


def run_worker(*, dequeue_timeout: int = 5, reconnect_delay: float = 1.0) -> None:
    job_service = get_job_service()
    processor = IngestJobProcessor(
        service=get_rag_service(),
        job_service=job_service,
        request_store=get_ingest_request_store(),
    )
    logger.info("Ingest worker listening on queue %s.", job_service.queue_name)

    while True:
        try:
            job_id = job_service.dequeue(timeout=dequeue_timeout)
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            logger.warning("Redis queue unavailable; retrying in %.1fs: %s", reconnect_delay, exc)
            time.sleep(reconnect_delay)
            continue
        if job_id is not None:
            processor.process(job_id)


def main() -> None:
    configure_api_logging()
    try:
        run_worker()
    except KeyboardInterrupt:
        logger.info("Ingest worker stopped.")


if __name__ == "__main__":
    main()
