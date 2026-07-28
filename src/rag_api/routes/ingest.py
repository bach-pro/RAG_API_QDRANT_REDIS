from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from rag_api.dependencies.services import get_job_service, get_rag_service
from rag_api.schemas.ingest import (
    DocumentIngestRequest,
    IngestJobResponse,
    IngestJobStatusResponse,
)
from rag_api.services.job_service import ActiveIngestJobError, IngestJobState, JobService
from rag_app.data_loader import (
    SUPPORTED_FILE_EXTENSIONS,
    UploadedFileData,
    load_uploaded_documents,
    load_uploaded_files,
)
from rag_app.services import RagService


router = APIRouter(prefix="/v1", tags=["ingest"])
logger = logging.getLogger("rag_api")


def _job_to_status_response(job: IngestJobState) -> IngestJobStatusResponse:
    return IngestJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        message=job.message,
        documents_loaded=job.documents_loaded,
        chunks_indexed=job.chunks_indexed,
        collection_count=job.collection_count,
        error=job.error,
    )


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
    if event_name == "reset_start":
        return "Resetting vector collection."
    if event_name == "reset_done":
        return "Vector collection reset completed."
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


def _run_document_ingest_job(
    *,
    job_id: str,
    request: DocumentIngestRequest,
    service: RagService,
    job_service: JobService,
) -> None:
    job_service.update(job_id, status="running", message="Ingest job started.")

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
            job_service.update(job_id, **updates)

    try:
        documents = load_uploaded_documents(
            [document.model_dump() for document in request.documents],
            source_label="api_upload",
        )
        result = service.ingest_documents(
            documents=documents,
            reset=request.reset,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        logger.exception("Ingest job %s failed.", job_id)
        job_service.update(
            job_id,
            status="failed",
            message="Ingest job failed.",
            error=f"{exc.__class__.__name__}: {exc}",
        )
        return

    job_service.update(
        job_id,
        status="completed",
        message="Ingest job completed.",
        documents_loaded=result.documents_loaded,
        chunks_indexed=result.chunks_indexed,
        collection_count=result.collection_count,
        error=None,
    )


def _run_file_ingest_job(
    *,
    job_id: str,
    files: list[UploadedFileData],
    reset: bool,
    service: RagService,
    job_service: JobService,
) -> None:
    job_service.update(job_id, status="running", message="Extracting uploaded files.")

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
            job_service.update(job_id, **updates)

    try:
        documents = load_uploaded_files(
            files,
            source_label="file_upload",
            max_extracted_chars=service.config.max_extracted_chars,
        )
        result = service.ingest_documents(
            documents=documents,
            reset=reset,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        logger.exception("File ingest job %s failed.", job_id)
        job_service.update(
            job_id,
            status="failed",
            message="File ingest job failed.",
            error=f"{exc.__class__.__name__}: {exc}",
        )
        return

    job_service.update(
        job_id,
        status="completed",
        message="File ingest job completed.",
        documents_loaded=result.documents_loaded,
        chunks_indexed=result.chunks_indexed,
        collection_count=result.collection_count,
        error=None,
    )


@router.post("/ingest/documents", response_model=IngestJobResponse)
def start_document_ingest(
    request: DocumentIngestRequest,
    background_tasks: BackgroundTasks,
    service: RagService = Depends(get_rag_service),
    job_service: JobService = Depends(get_job_service),
) -> IngestJobResponse:
    try:
        job = job_service.create_ingest_job()
    except ActiveIngestJobError as exc:
        active_job = exc.active_job
        raise HTTPException(
            status_code=409,
            detail={
                "message": "An ingest job is already active.",
                "job_id": active_job.job_id,
                "status": active_job.status,
            },
        ) from exc

    background_tasks.add_task(
        _run_document_ingest_job,
        job_id=job.job_id,
        request=request,
        service=service,
        job_service=job_service,
    )
    return IngestJobResponse(job_id=job.job_id, status=job.status)


@router.post("/ingest/files", response_model=IngestJobResponse)
async def start_file_ingest(
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File(description="PDF or DOCX files")],
    reset: Annotated[bool, Form()] = False,
    service: RagService = Depends(get_rag_service),
    job_service: JobService = Depends(get_job_service),
) -> IngestJobResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF or DOCX file is required.")
    if len(files) > service.config.max_upload_files:
        raise HTTPException(
            status_code=400,
            detail=f"At most {service.config.max_upload_files} files are allowed per request.",
        )

    buffered_files: list[UploadedFileData] = []
    for upload in files:
        filename = Path((upload.filename or "").replace("\x00", "")).name.strip()
        extension = Path(filename).suffix.casefold()
        if extension not in SUPPORTED_FILE_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_FILE_EXTENSIONS))
            await upload.close()
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file '{filename or 'unnamed'}'. Supported types: {supported}.",
            )
        content = await upload.read(service.config.max_upload_file_bytes + 1)
        await upload.close()
        if not content:
            raise HTTPException(status_code=400, detail=f"Uploaded file '{filename}' is empty.")
        if len(content) > service.config.max_upload_file_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File '{filename}' exceeds the {service.config.max_upload_file_bytes}-byte limit."
                ),
            )
        buffered_files.append(
            UploadedFileData(
                filename=filename,
                content=content,
                content_type=upload.content_type,
            )
        )

    try:
        job = job_service.create_ingest_job()
    except ActiveIngestJobError as exc:
        active_job = exc.active_job
        raise HTTPException(
            status_code=409,
            detail={
                "message": "An ingest job is already active.",
                "job_id": active_job.job_id,
                "status": active_job.status,
            },
        ) from exc

    background_tasks.add_task(
        _run_file_ingest_job,
        job_id=job.job_id,
        files=buffered_files,
        reset=reset,
        service=service,
        job_service=job_service,
    )
    return IngestJobResponse(job_id=job.job_id, status=job.status)


@router.get("/ingest/{job_id}", response_model=IngestJobStatusResponse)
def get_ingest_job(
    job_id: str,
    job_service: JobService = Depends(get_job_service),
) -> IngestJobStatusResponse:
    job = job_service.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found.")
    return _job_to_status_response(job)
