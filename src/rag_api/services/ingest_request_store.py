from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

from rag_api.schemas.ingest import DocumentIngestRequest
from rag_app.data_loader import UploadedFileData


class StoredFile(TypedDict):
    filename: str
    stored_name: str
    content_type: str | None
    doc_id: str | None


class StoredIngestRequest(TypedDict, total=False):
    kind: Literal["documents", "files"]
    bot_id: str
    doc_id: str | None
    doc_ids: list[str] | None
    documents: list[dict[str, Any]]
    files: list[StoredFile]


class IngestRequestStore:
    """Disk handoff shared by the API and worker processes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save_documents(self, job_id: str, request: DocumentIngestRequest) -> None:
        self._save(
            job_id,
            {
                "kind": "documents",
                "bot_id": request.bot_id,
                "doc_id": request.doc_id,
                "documents": [document.model_dump(mode="json") for document in request.documents],
            },
            files=[],
        )

    def save_files(
        self,
        job_id: str,
        *,
        bot_id: str,
        doc_id: str | None,
        doc_ids: list[str] | None,
        files: list[UploadedFileData],
    ) -> None:
        stored_files: list[StoredFile] = []
        file_payloads: list[tuple[str, bytes]] = []
        for index, uploaded_file in enumerate(files):
            suffix = Path(uploaded_file.filename).suffix.casefold()
            stored_name = f"{index:04d}{suffix}"
            stored_files.append(
                {
                    "filename": uploaded_file.filename,
                    "stored_name": stored_name,
                    "content_type": uploaded_file.content_type,
                    "doc_id": doc_ids[index] if doc_ids is not None else None,
                }
            )
            file_payloads.append((stored_name, uploaded_file.content))

        self._save(
            job_id,
            {
                "kind": "files",
                "bot_id": bot_id,
                "doc_id": doc_id,
                "doc_ids": doc_ids,
                "files": stored_files,
            },
            files=file_payloads,
        )

    def load(self, job_id: str) -> StoredIngestRequest:
        manifest_path = self._job_dir(job_id) / "request.json"
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            return json.load(manifest_file)

    def load_files(self, job_id: str, request: StoredIngestRequest) -> list[UploadedFileData]:
        return [
            self.load_file(job_id, file_info)
            for file_info in request.get("files", [])
        ]

    def load_file(self, job_id: str, file_info: StoredFile) -> UploadedFileData:
        return UploadedFileData(
            filename=file_info["filename"],
            content=(self._job_dir(job_id) / file_info["stored_name"]).read_bytes(),
            content_type=file_info.get("content_type"),
        )

    def delete(self, job_id: str) -> None:
        shutil.rmtree(self._job_dir(job_id), ignore_errors=True)

    def _save(
        self,
        job_id: str,
        request: StoredIngestRequest,
        *,
        files: list[tuple[str, bytes]],
    ) -> None:
        job_dir = self._job_dir(job_id)
        temporary_dir = self.root / f".{job_id}.{uuid4().hex}.tmp"
        temporary_dir.mkdir(parents=False, exist_ok=False)
        try:
            for stored_name, content in files:
                (temporary_dir / stored_name).write_bytes(content)
            with (temporary_dir / "request.json").open("w", encoding="utf-8") as manifest_file:
                json.dump(request, manifest_file, ensure_ascii=False)
            temporary_dir.rename(job_dir)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

    def _job_dir(self, job_id: str) -> Path:
        path = (self.root / job_id).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Invalid ingest job ID.") from exc
        if path == self.root:
            raise ValueError("Invalid ingest job ID.")
        return path
