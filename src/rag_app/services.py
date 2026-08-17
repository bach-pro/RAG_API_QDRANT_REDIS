from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

from .config import AppConfig
from .data_loader import chunk_documents
from .models import KnowledgeDocument, RagResponse, RagStreamResponse
from .rag import answer_question, stream_answer_question
from .retrievers import BM25Index
from .vector_store import QdrantKnowledgeStore


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class IngestResult:
    documents_loaded: int
    chunks_indexed: int
    collection_count: int
    replaced_existing: bool = False


@dataclass(frozen=True)
class BotDeleteResult:
    bot_id: str
    chunks_deleted: int


class RagService:
    """Framework-free owner for shared RAG resources and workflows."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._store: QdrantKnowledgeStore | None = None
        self._bm25_by_bot: dict[str, BM25Index] = {}
        self._lock = RLock()

    @property
    def store(self) -> QdrantKnowledgeStore:
        with self._lock:
            if self._store is None:
                self._store = self._create_store()
            return self._store

    def _create_store(self, *, reset: bool = False) -> QdrantKnowledgeStore:
        return QdrantKnowledgeStore(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
            collection_name=self.config.collection_name,
            embedding_provider=self.config.embedding_provider,
            embedding_model=self.config.embedding_model,
            embedding_host=self.config.local_ollama_host,
            keep_alive=self.config.ollama_keep_alive,
            reset=reset,
        )

    def reset_store(self) -> None:
        with self._lock:
            if self._store is not None:
                self._store.close()
            self._store = self._create_store(reset=True)
            self._bm25_by_bot.clear()

    def bm25_for(self, bot_id: str) -> BM25Index:
        with self._lock:
            if bot_id not in self._bm25_by_bot:
                self.refresh_retrieval_cache(bot_id)
            index = self._bm25_by_bot.get(bot_id)
            if index is None:
                raise RuntimeError("BM25 cache could not be initialized.")
            return index

    def count(self, bot_id: str | None = None) -> int:
        with self._lock:
            return self.store.count(bot_id=bot_id)

    def refresh_retrieval_cache(self, bot_id: str) -> None:
        with self._lock:
            self._bm25_by_bot[bot_id] = BM25Index.from_store(self.store, bot_id=bot_id)

    def ingest_documents(
        self,
        *,
        bot_id: str,
        documents: list[KnowledgeDocument],
        doc_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IngestResult:
        if not documents:
            raise ValueError("documents must not be empty.")
        bot_id = bot_id.strip()
        if not bot_id:
            raise ValueError("bot_id must not be empty.")
        doc_id = doc_id.strip() if doc_id is not None else None
        if not doc_id:
            doc_id = None
        scoped_documents = [
            KnowledgeDocument(
                id=document.id,
                text=document.text,
                metadata={
                    **document.metadata,
                    "bot_id": bot_id,
                    **({"document_id": doc_id} if doc_id is not None else {}),
                },
            )
            for document in documents
        ]
        return self._ingest_prepared_documents(
            bot_id=bot_id,
            documents=scoped_documents,
            doc_id=doc_id,
            progress_callback=progress_callback,
        )

    def _ingest_prepared_documents(
        self,
        *,
        bot_id: str,
        documents: list[KnowledgeDocument],
        doc_id: str | None,
        progress_callback: ProgressCallback | None,
    ) -> IngestResult:
        if progress_callback is not None:
            progress_callback({"event": "documents_loaded", "documents": len(documents)})

        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        if progress_callback is not None:
            progress_callback({"event": "chunks_prepared", "chunks": len(chunks)})
            progress_callback({"event": "index_wait_start"})

        with self._lock:
            if progress_callback is not None:
                progress_callback({"event": "index_start"})

            replaced_existing = False
            if doc_id is not None:
                replaced_existing = self.store.document_exists(doc_id, bot_id)
            if replaced_existing:
                if progress_callback is not None:
                    progress_callback({"event": "replace_start", "doc_id": doc_id})
                self.store.delete_by_document_id(doc_id, bot_id)
                self._bm25_by_bot.pop(bot_id, None)
                if progress_callback is not None:
                    progress_callback({"event": "replace_done", "doc_id": doc_id})

            collection_count = self.store.ingest(
                chunks,
                bot_id=bot_id,
                progress_callback=progress_callback,
            )
            self.refresh_retrieval_cache(bot_id)
            return IngestResult(
                documents_loaded=len(documents),
                chunks_indexed=len(chunks),
                collection_count=collection_count,
                replaced_existing=replaced_existing,
            )

    def delete_bot(self, bot_id: str) -> BotDeleteResult:
        bot_id = bot_id.strip()
        if not bot_id:
            raise ValueError("bot_id must not be empty.")
        with self._lock:
            chunks_deleted = self.store.count(bot_id=bot_id)
            if chunks_deleted:
                self.store.delete_by_bot_id(bot_id)
            self._bm25_by_bot.pop(bot_id, None)
            return BotDeleteResult(bot_id=bot_id, chunks_deleted=chunks_deleted)

    def answer(
        self,
        *,
        bot_id: str,
        question: str,
        mode: str = "Auto Router",
        top_k: int = 5,
        fetch_k: int = 20,
        mmr_lambda: float = 0.5,
        history: list[dict[str, Any]] | None = None,
    ) -> RagResponse:
        with self._lock:
            return answer_question(
                config=self.config,
                store=self.store,
                bm25=self.bm25_for(bot_id),
                bot_id=bot_id,
                question=question,
                mode=mode,
                k=top_k,
                fetch_k=fetch_k,
                lambda_mult=mmr_lambda,
                history=history or [],
            )

    def answer_stream(
        self,
        *,
        bot_id: str,
        question: str,
        mode: str = "Auto Router",
        top_k: int = 5,
        fetch_k: int = 20,
        mmr_lambda: float = 0.5,
        history: list[dict[str, Any]] | None = None,
    ) -> RagStreamResponse:
        with self._lock:
            return stream_answer_question(
                config=self.config,
                store=self.store,
                bm25=self.bm25_for(bot_id),
                bot_id=bot_id,
                question=question,
                mode=mode,
                k=top_k,
                fetch_k=fetch_k,
                lambda_mult=mmr_lambda,
                history=history or [],
            )
