from __future__ import annotations

from rag_app.config import AppConfig
from rag_app.models import KnowledgeDocument, RagResponse, RetrievedDocument
from rag_app.services import RagService
import rag_app.services as service_module


class FakeStore:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reset_called = False
        self.close_called = False
        self.ingested = []
        self.ingested_bot_id = None
        self.deleted_bot_ids = []
        self.document_exists_calls = []
        self.deleted_document_ids = []
        FakeStore.instances.append(self)

    def count(self, bot_id=None):
        return 12

    def delete_by_bot_id(self, bot_id):
        self.deleted_bot_ids.append(bot_id)

    def document_exists(self, document_id, bot_id):
        self.document_exists_calls.append((document_id, bot_id))
        return document_id == "policy-v1"

    def delete_by_document_id(self, document_id, bot_id):
        self.deleted_document_ids.append((document_id, bot_id))

    def reset(self):
        self.reset_called = True

    def close(self):
        self.close_called = True

    def ingest(self, documents, bot_id, progress_callback=None):
        self.ingested = list(documents)
        self.ingested_bot_id = bot_id
        if progress_callback is not None:
            progress_callback({"event": "batch_done", "start": 0, "end": len(documents), "total": len(documents)})
        return 42


class FakeBM25Index:
    built_from = []

    @classmethod
    def from_store(cls, store, bot_id):
        cls.built_from.append((store, bot_id))
        return cls()


def test_rag_service_reuses_store_and_builds_bm25(monkeypatch):
    FakeStore.instances.clear()
    FakeBM25Index.built_from.clear()
    monkeypatch.setattr(service_module, "QdrantKnowledgeStore", FakeStore)
    monkeypatch.setattr(service_module, "BM25Index", FakeBM25Index)

    service = RagService(AppConfig())

    assert service.count() == 12
    assert service.store is service.store
    assert len(FakeStore.instances) == 1

    service.refresh_retrieval_cache("bot-a")

    assert FakeBM25Index.built_from == [(service.store, "bot-a")]


def test_rag_service_ingest_uses_shared_workflow_and_refreshes_cache(monkeypatch):
    FakeStore.instances.clear()
    FakeBM25Index.built_from.clear()
    monkeypatch.setattr(service_module, "QdrantKnowledgeStore", FakeStore)
    monkeypatch.setattr(service_module, "BM25Index", FakeBM25Index)

    document = KnowledgeDocument(id="DOC-1", text="source", metadata={"bot_id": "bot-other"})
    chunk = KnowledgeDocument(id="DOC-1::0", text="chunk", metadata={})
    prepared_documents = []

    def fake_chunk_documents(documents, chunk_size, chunk_overlap):
        prepared_documents.extend(documents)
        return [chunk]

    monkeypatch.setattr(service_module, "chunk_documents", fake_chunk_documents)

    events = []
    service = RagService(AppConfig())
    original_store = service.store
    result = service.ingest_documents(
        bot_id="bot-a",
        documents=[document],
        doc_id=" policy-v1 ",
        progress_callback=events.append,
    )

    assert result.documents_loaded == 1
    assert result.chunks_indexed == 1
    assert result.collection_count == 42
    assert result.replaced_existing is True
    assert prepared_documents[0].metadata["bot_id"] == "bot-a"
    assert prepared_documents[0].metadata["document_id"] == "policy-v1"
    assert len(FakeStore.instances) == 1
    assert FakeStore.instances[0] is original_store
    assert original_store.close_called is False
    assert original_store.deleted_bot_ids == []
    assert original_store.document_exists_calls == [("policy-v1", "bot-a")]
    assert original_store.deleted_document_ids == [("policy-v1", "bot-a")]
    assert service.store is original_store
    assert service.store.reset_called is False
    assert service.store.ingested == [chunk]
    assert service.store.ingested_bot_id == "bot-a"
    assert FakeBM25Index.built_from == [(service.store, "bot-a")]
    assert [event["event"] for event in events] == [
        "documents_loaded",
        "chunks_prepared",
        "index_wait_start",
        "index_start",
        "replace_start",
        "replace_done",
        "batch_done",
    ]


def test_rag_service_normal_ingest_does_not_check_or_delete_document(monkeypatch):
    FakeStore.instances.clear()
    FakeBM25Index.built_from.clear()
    monkeypatch.setattr(service_module, "QdrantKnowledgeStore", FakeStore)
    monkeypatch.setattr(service_module, "BM25Index", FakeBM25Index)
    monkeypatch.setattr(
        service_module,
        "chunk_documents",
        lambda documents, chunk_size, chunk_overlap: list(documents),
    )

    service = RagService(AppConfig())
    result = service.ingest_documents(
        bot_id="bot-a",
        documents=[KnowledgeDocument(id="DOC-1", text="source")],
        doc_id="   ",
    )

    assert result.replaced_existing is False
    assert service.store.document_exists_calls == []
    assert service.store.deleted_document_ids == []
    assert "document_id" not in service.store.ingested[0].metadata


def test_rag_service_delete_bot_clears_only_bot_data_and_cache(monkeypatch):
    FakeStore.instances.clear()
    FakeBM25Index.built_from.clear()
    monkeypatch.setattr(service_module, "QdrantKnowledgeStore", FakeStore)
    monkeypatch.setattr(service_module, "BM25Index", FakeBM25Index)

    service = RagService(AppConfig())
    service.refresh_retrieval_cache("bot-a")
    service.refresh_retrieval_cache("bot-b")

    result = service.delete_bot("bot-a")

    assert result.bot_id == "bot-a"
    assert result.chunks_deleted == 12
    assert service.store.deleted_bot_ids == ["bot-a"]
    assert "bot-a" not in service._bm25_by_bot
    assert "bot-b" in service._bm25_by_bot


def test_rag_service_answer_delegates_to_answer_question(monkeypatch):
    FakeStore.instances.clear()
    FakeBM25Index.built_from.clear()
    monkeypatch.setattr(service_module, "QdrantKnowledgeStore", FakeStore)
    monkeypatch.setattr(service_module, "BM25Index", FakeBM25Index)

    captured = {}

    def fake_answer_question(**kwargs):
        captured.update(kwargs)
        return RagResponse(
            answer="answer",
            sources=[RetrievedDocument(id="DOC-1::0", text="text")],
            mode=kwargs["mode"],
        )

    monkeypatch.setattr(service_module, "answer_question", fake_answer_question)

    service = RagService(AppConfig())
    response = service.answer(
        bot_id="bot-a",
        question="question",
        mode="Semantic",
        top_k=3,
        fetch_k=8,
    )

    assert response.answer == "answer"
    assert captured["store"] is service.store
    assert captured["bm25"] is service.bm25_for("bot-a")
    assert captured["bot_id"] == "bot-a"
    assert captured["question"] == "question"
    assert captured["mode"] == "Semantic"
    assert captured["k"] == 3
    assert captured["fetch_k"] == 8
