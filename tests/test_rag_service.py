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
        FakeStore.instances.append(self)

    def count(self):
        return 12

    def reset(self):
        self.reset_called = True

    def close(self):
        self.close_called = True

    def ingest(self, documents, progress_callback=None):
        self.ingested = list(documents)
        if progress_callback is not None:
            progress_callback({"event": "batch_done", "start": 0, "end": len(documents), "total": len(documents)})
        return 42


class FakeBM25Index:
    built_from = []

    @classmethod
    def from_store(cls, store):
        cls.built_from.append(store)
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

    service.refresh_retrieval_cache()

    assert FakeBM25Index.built_from == [service.store]


def test_rag_service_ingest_uses_shared_workflow_and_refreshes_cache(monkeypatch):
    FakeStore.instances.clear()
    FakeBM25Index.built_from.clear()
    monkeypatch.setattr(service_module, "QdrantKnowledgeStore", FakeStore)
    monkeypatch.setattr(service_module, "BM25Index", FakeBM25Index)

    document = KnowledgeDocument(id="DOC-1", text="source", metadata={})
    chunk = KnowledgeDocument(id="DOC-1::0", text="chunk", metadata={})
    monkeypatch.setattr(
        service_module,
        "chunk_documents",
        lambda documents, chunk_size, chunk_overlap: [chunk],
    )

    events = []
    service = RagService(AppConfig())
    original_store = service.store
    result = service.ingest_documents(
        documents=[document],
        reset=True,
        progress_callback=events.append,
    )

    assert result.documents_loaded == 1
    assert result.chunks_indexed == 1
    assert result.collection_count == 42
    assert len(FakeStore.instances) == 2
    assert FakeStore.instances[0] is original_store
    assert original_store.close_called is True
    assert FakeStore.instances[1].kwargs["reset"] is True
    assert service.store is FakeStore.instances[1]
    assert service.store.reset_called is False
    assert service.store.ingested == [chunk]
    assert FakeBM25Index.built_from == [service.store]
    assert [event["event"] for event in events] == [
        "documents_loaded",
        "chunks_prepared",
        "index_wait_start",
        "index_start",
        "reset_start",
        "reset_done",
        "batch_done",
    ]


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
    response = service.answer(question="question", mode="Semantic", top_k=3, fetch_k=8)

    assert response.answer == "answer"
    assert captured["store"] is service.store
    assert captured["bm25"] is service.bm25
    assert captured["question"] == "question"
    assert captured["mode"] == "Semantic"
    assert captured["k"] == 3
    assert captured["fetch_k"] == 8
