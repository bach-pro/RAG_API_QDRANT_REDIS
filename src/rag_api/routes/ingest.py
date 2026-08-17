from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from rag_api.dependencies.services import (
    get_ingest_request_store,
    get_job_service,
    get_rag_service,
)
from rag_api.schemas.common import BotId
from rag_api.schemas.ingest import (
    BotDeleteResponse,
    DocumentIngestRequest,
    IngestJobResponse,
    IngestJobStatusResponse,
)
from rag_api.services.ingest_request_store import IngestRequestStore
from rag_api.services.job_service import ActiveIngestJobError, IngestJobState, JobService
from rag_app.data_loader import SUPPORTED_FILE_EXTENSIONS, UploadedFileData
from rag_app.services import RagService


router = APIRouter(prefix="/v1", tags=["ingest"])
logger = logging.getLogger("rag_api")


def _new_document_id() -> str:
    return f"doc-{uuid4()}"


def _job_to_status_response(job: IngestJobState) -> IngestJobStatusResponse:
    return IngestJobStatusResponse(
        job_id=job.job_id,
        bot_id=job.bot_id,
        doc_id=job.doc_id,
        doc_ids=job.doc_ids,
        status=job.status,
        message=job.message,
        documents_loaded=job.documents_loaded,
        chunks_indexed=job.chunks_indexed,
        collection_count=job.collection_count,
        replaced_existing=job.replaced_existing,
        error=job.error,
    )


def _active_job_response(exc: ActiveIngestJobError) -> HTTPException:
    active_job = exc.active_job
    return HTTPException(
        status_code=409,
        detail={
            "message": "An ingest job is already active.",
            "job_id": active_job.job_id,
            "bot_id": active_job.bot_id,
            "status": active_job.status,
        },
    )


def _create_and_enqueue_job(
    *,
    job_id: str,
    bot_id: str,
    doc_id: str | None,
    doc_ids: list[str] | None,
    job_service: JobService,
    request_store: IngestRequestStore,
) -> IngestJobState:
    try:
        job = job_service.create_ingest_job(
            bot_id,
            doc_id,
            job_id=job_id,
            doc_ids=doc_ids,
        )
    except ActiveIngestJobError as exc:
        request_store.delete(job_id)
        raise _active_job_response(exc) from exc
    except Exception:
        request_store.delete(job_id)
        raise

    try:
        job_service.enqueue(job_id)
    except Exception as exc:
        logger.exception("Could not enqueue ingest job %s.", job_id)
        request_store.delete(job_id)
        job_service.update(
            job_id,
            status="failed",
            message="Ingest job could not be queued.",
            error=f"{exc.__class__.__name__}: {exc}",
        )
        raise HTTPException(status_code=503, detail="Ingest queue is unavailable.") from exc
    return job


@router.post("/ingest/documents", response_model=IngestJobResponse)
def start_document_ingest(
    request: DocumentIngestRequest,
    job_service: JobService = Depends(get_job_service),
    request_store: IngestRequestStore = Depends(get_ingest_request_store),
) -> IngestJobResponse:
    if request.doc_id is None:
        request = request.model_copy(update={"doc_id": _new_document_id()})
    job_id = str(uuid4())
    request_store.save_documents(job_id, request)
    job = _create_and_enqueue_job(
        job_id=job_id,
        bot_id=request.bot_id,
        doc_id=request.doc_id,
        doc_ids=None,
        job_service=job_service,
        request_store=request_store,
    )
    return IngestJobResponse(
        job_id=job.job_id,
        bot_id=job.bot_id,
        doc_id=job.doc_id,
        doc_ids=job.doc_ids,
        status=job.status,
    )


@router.post("/ingest/files", response_model=IngestJobResponse)
async def start_file_ingest(
    files: Annotated[
        list[UploadFile],
        File(
            description="PDF or DOCX files",
            json_schema_extra={"items": {"type": "string", "format": "binary"}},
        ),
    ],
    bot_id: Annotated[BotId, Form()],
    doc_id: Annotated[
        str | None,
        Form(
            max_length=256,
            description="Existing document/group ID to replace; omit it to generate IDs.",
        ),
    ] = None,
    doc_ids: Annotated[
        list[str] | None,
        Form(
            description=(
                "Existing document IDs to replace, in the same order as files; "
                "omit them to generate IDs."
            )
        ),
    ] = None,
    service: RagService = Depends(get_rag_service),
    job_service: JobService = Depends(get_job_service),
    request_store: IngestRequestStore = Depends(get_ingest_request_store),
) -> IngestJobResponse:
    doc_id = doc_id.strip() if doc_id is not None else None
    if not doc_id:
        doc_id = None

    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF or DOCX file is required.")
    if len(files) > service.config.max_upload_files:
        raise HTTPException(
            status_code=400,
            detail=f"At most {service.config.max_upload_files} files are allowed per request.",
        )

    if doc_ids is not None:
        doc_ids = [value.strip() for value in doc_ids]
        if not doc_ids or all(not value for value in doc_ids):
            doc_ids = None
        elif any(not value for value in doc_ids):
            raise HTTPException(status_code=422, detail="Every doc_ids item must not be empty.")
        
    if doc_id is not None and doc_ids:
        raise HTTPException(status_code=422, detail="Use either doc_id or doc_ids, not both.")
    if doc_id is None and doc_ids is None:
        if len(files) == 1:
            doc_id = _new_document_id()
        else:
            doc_ids = [_new_document_id() for _ in files]
    if doc_ids is not None:
        if any(len(value) > 256 for value in doc_ids):
            raise HTTPException(
                status_code=422,
                detail="Every doc_ids item must be at most 256 characters.",
            )
        if len(doc_ids) != len(files):
            raise HTTPException(
                status_code=422,
                detail="doc_ids must contain exactly one ID for each uploaded file.",
            )
        if len(set(doc_ids)) != len(doc_ids):
            raise HTTPException(
                status_code=422,
                detail="doc_ids must be unique within one upload request.",
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

    job_id = str(uuid4())
    request_store.save_files(
        job_id,
        bot_id=bot_id,
        doc_id=doc_id,
        doc_ids=doc_ids,
        files=buffered_files,
    )
    job = _create_and_enqueue_job(
        job_id=job_id,
        bot_id=bot_id,
        doc_id=doc_id,
        doc_ids=doc_ids,
        job_service=job_service,
        request_store=request_store,
    )
    return IngestJobResponse(
        job_id=job.job_id,
        bot_id=job.bot_id,
        doc_id=job.doc_id,
        doc_ids=job.doc_ids,
        status=job.status,
    )


@router.delete("/bots/{bot_id}", response_model=BotDeleteResponse)
def delete_bot(
    bot_id: BotId,
    service: RagService = Depends(get_rag_service),
) -> BotDeleteResponse:
    result = service.delete_bot(bot_id)
    return BotDeleteResponse(
        bot_id=result.bot_id,
        chunks_deleted=result.chunks_deleted,
    )


@router.get("/ingest/{job_id}", response_model=IngestJobStatusResponse)
def get_ingest_job(
    job_id: str,
    job_service: JobService = Depends(get_job_service),
) -> IngestJobStatusResponse:
    job = job_service.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found.")
    return _job_to_status_response(job)
