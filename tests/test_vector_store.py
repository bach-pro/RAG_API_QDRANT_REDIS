from __future__ import annotations

from rag_app.models import KnowledgeDocument
from rag_app.vector_store import QdrantKnowledgeStore
import rag_app.vector_store as vector_store_module


class FakeEmbeddingFunction:
    def __call__(self, input: list[str]) -> list[list[float]]:
        if isinstance(input, str):
            input = [input]
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if "alpha" in lowered else 0.0,
            1.0 if "beta" in lowered else 0.0,
            1.0,
        ]


def test_qdrant_store_ingests_queries_and_filters(monkeypatch):
    monkeypatch.setattr(
        vector_store_module,
        "build_embedding_function",
        lambda *args, **kwargs: FakeEmbeddingFunction(),
    )
    store = QdrantKnowledgeStore(
        url=":memory:",
        collection_name="test_documents",
        embedding_provider="fake",
        embedding_model="fake",
    )

    count = store.ingest(
        [
            KnowledgeDocument(
                id="DOC-1::0",
                text="alpha policy text",
                metadata={"document_id": "DOC-1", "title": "Alpha"},
            ),
            KnowledgeDocument(
                id="DOC-2::0",
                text="beta handbook text",
                metadata={"document_id": "DOC-2", "title": "Beta"},
            ),
        ]
    )

    assert count == 2
    assert store.count() == 2
    assert {doc.id for doc in store.get_all_documents()} == {"DOC-1::0", "DOC-2::0"}

    results = store.query("alpha", k=1, include_embeddings=True)
    assert results[0].id == "DOC-1::0"
    assert results[0].embedding

    exact = store.get_by_document_id("DOC-2")
    assert [doc.id for doc in exact] == ["DOC-2::0"]
