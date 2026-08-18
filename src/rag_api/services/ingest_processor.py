from __future__ import annotations

import logging
from uuid import uuid4

from rag_api.services.ingest_request_store import IngestRequestStore, StoredIngestRequest
from rag_api.services.job_service import JobService
from rag_app.data_loader import load_uploaded_documents, load_uploaded_files
from rag_app.services import IngestResult, ProgressCallback, RagService


logger = logging.getLogger("rag_api")


def _document_records_for_ingest(request: StoredIngestRequest) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    documents = request.get("documents", [])
    doc_id = request.get("doc_id")
    for index, document in enumerate(documents):
        if doc_id is None:
            internal_id = f"api-{uuid4()}"
        elif len(documents) == 1:
            internal_id = doc_id
        else:
            internal_id = f"{doc_id}::{index}"
        records.append({**document, "id": internal_id})
    return records


def _progress_message(event: dict[str, object]) -> str | None:
    event_name = event.get("event")
    if event_name == "documents_loaded":
        return f"Loaded {event.get('documents')} documents."
    if event_name == "chunks_prepared":
        return f"Prepared {event.get('chunks')} chunks."
    if event_name == "index_wait_start":
        return "Waiting for vector index lock."
    if event_name == "index_start":
        return "Vector indexing started."
    if event_name == "replace_start":
        return f"Replacing existing document '{event.get('doc_id')}'."
    if event_name == "replace_done":
        return f"Existing document '{event.get('doc_id')}' removed."
    if event_name == "batch_start":
        return (
            f"Indexing chunk batch {event.get('start')}-{event.get('end')} "
            f"of {event.get('total')}."
        )
    if event_name == "batch_done":
        return (
            f"Indexed chunk batch {event.get('start')}-{event.get('end')} "
            f"of {event.get('total')}."
        )
    return None


class IngestJobProcessor:
    def __init__(
        self,
        *,
        service: RagService,
        job_service: JobService,
        request_store: IngestRequestStore,
    ) -> None:
        self.service = service
        self.job_service = job_service
        self.request_store = request_store

    def process(self, job_id: str) -> None:
        job = self.job_service.get(job_id)
        if job is None:
            logger.error("Ignoring unknown ingest job %s.", job_id)
            return
        if job.status != "queued":
            logger.warning("Ignoring ingest job %s with status %s.", job_id, job.status)
            return

        self.job_service.update(job_id, status="running", message="Ingest job started.")

        def progress_callback(event: dict[str, object]) -> None:
            updates: dict[str, object] = {}
            message = _progress_message(event)
            if message is not None:
                updates["message"] = message
            event_name = event.get("event")
            if event_name == "documents_loaded":
                updates["documents_loaded"] = event.get("documents")
            elif event_name == "batch_done":
                updates["chunks_indexed"] = event.get("end")
            if updates:
                self.job_service.update(job_id, **updates)

        try:
            request = self.request_store.load(job_id)
            request_kind = request.get("kind")
            if request_kind == "documents":
                documents = load_uploaded_documents(
                    _document_records_for_ingest(request),
                    source_label="api_upload",
                )
                result = self.service.ingest_documents(
                    bot_id=job.bot_id,
                    documents=documents,
                    doc_id=job.doc_id,
                    progress_callback=progress_callback,
                )
            elif request_kind == "files":
                if job.doc_ids is None:
                    documents = load_uploaded_files(
                        self.request_store.load_files(job_id, request),
                        source_label="file_upload",
                        max_extracted_chars=self.service.config.max_extracted_chars,
                    )
                    result = self.service.ingest_documents(
                        bot_id=job.bot_id,
                        documents=documents,
                        doc_id=job.doc_id,
                        progress_callback=progress_callback,
                    )
                else:
                    result = self._ingest_files_with_document_ids(
                        job_id=job_id,
                        bot_id=job.bot_id,
                        doc_ids=job.doc_ids,
                        request=request,
                        progress_callback=progress_callback,
                    )
            else:
                raise ValueError(f"Unsupported ingest request kind: {request_kind!r}")
        except Exception as exc:
            logger.exception("Ingest job %s failed.", job_id)
            self.job_service.update(
                job_id,
                status="failed",
                message="Ingest job failed.",
                error=f"{exc.__class__.__name__}: {exc}",
            )
        else:
            self.job_service.update(
                job_id,
                status="completed",
                message="Ingest job completed.",
                documents_loaded=result.documents_loaded,
                chunks_indexed=result.chunks_indexed,
                collection_count=result.collection_count,
                replaced_existing=result.replaced_existing,
                error=None,
            )
        finally:
            self.request_store.delete(job_id)

    def _ingest_files_with_document_ids(
        self,
        *,
        job_id: str,
        bot_id: str,
        doc_ids: list[str],
        request: StoredIngestRequest,
        progress_callback: ProgressCallback,
    ) -> IngestResult:
        stored_files = request.get("files", [])
        if len(stored_files) != len(doc_ids):
            raise ValueError("Stored files and doc_ids do not have the same length.")

        documents_loaded = 0
        chunks_indexed = 0
        collection_count = 0
        replaced_existing = False

        for index, (file_info, doc_id) in enumerate(zip(stored_files, doc_ids, strict=True)):
            if file_info.get("doc_id") != doc_id:
                raise ValueError("Stored file doc_id does not match the ingest job.")
            filename = file_info["filename"]
            self.job_service.update(
                job_id,
                message=f"Processing file {index + 1}/{len(stored_files)}: {filename}.",
            )
            documents = load_uploaded_files(
                [self.request_store.load_file(job_id, file_info)],
                source_label="file_upload",
                max_extracted_chars=self.service.config.max_extracted_chars,
            )

            loaded_before = documents_loaded
            chunks_before = chunks_indexed

            def file_progress_callback(event: dict[str, object]) -> None:
                adjusted_event = dict(event)
                if event.get("event") == "documents_loaded":
                    adjusted_event["documents"] = loaded_before + int(event["documents"])
                elif event.get("event") in {"batch_start", "batch_done"}:
                    for field in ("start", "end", "total"):
                        adjusted_event[field] = chunks_before + int(event[field])
                progress_callback(adjusted_event)

            file_result = self.service.ingest_documents(
                bot_id=bot_id,
                documents=documents,
                doc_id=doc_id,
                progress_callback=file_progress_callback,
            )
            documents_loaded += file_result.documents_loaded
            chunks_indexed += file_result.chunks_indexed
            collection_count = file_result.collection_count
            replaced_existing = replaced_existing or file_result.replaced_existing

        return IngestResult(
            documents_loaded=documents_loaded,
            chunks_indexed=chunks_indexed,
            collection_count=collection_count,
            replaced_existing=replaced_existing,
        )
