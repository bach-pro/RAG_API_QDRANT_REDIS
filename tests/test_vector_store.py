from __future__ import annotations

from rag_app.models import KnowledgeDocument
from rag_app.retrievers import BM25Index
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


def test_qdrant_store_isolates_ingest_query_exact_lookup_and_delete_by_bot(monkeypatch):
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

    alpha_count = store.ingest(
        [
            KnowledgeDocument(
                id="DOC-1::0",
                text="alpha policy text",
                metadata={"document_id": "1", "title": "Alpha"},
            ),
            KnowledgeDocument(
                id="DOC-2::0",
                text="alpha handbook text",
                metadata={"document_id": "2", "title": "Alpha handbook"},
            ),
        ],
        bot_id="bot-alpha",
    )
    beta_count = store.ingest(
        [
            KnowledgeDocument(
                id="DOC-1::0",
                text="beta policy text",
                metadata={"document_id": "1", "title": "Beta"},
            ),
            KnowledgeDocument(
                id="DOC-3::0",
                text="beta handbook text",
                metadata={"document_id": "3", "title": "Beta handbook"},
            ),
        ],
        bot_id="bot-beta",
    )

    assert alpha_count == 2
    assert beta_count == 2
    assert store.count() == 4
    assert store.count(bot_id="bot-alpha") == 2
    assert store.count(bot_id="bot-beta") == 2
    assert {doc.id for doc in store.get_all_documents(bot_id="bot-alpha")} == {
        "DOC-1::0",
        "DOC-2::0",
    }

    alpha_results = store.query(
        "alpha",
        bot_id="bot-alpha",
        k=2,
        include_embeddings=True,
    )
    beta_results = store.query("alpha", bot_id="bot-beta", k=2)
    assert alpha_results[0].embedding
    assert {doc.metadata["bot_id"] for doc in alpha_results} == {"bot-alpha"}
    assert {doc.metadata["bot_id"] for doc in beta_results} == {"bot-beta"}
    assert all("alpha" not in doc.text for doc in beta_results)

    alpha_bm25 = BM25Index.from_store(store, bot_id="bot-alpha")
    beta_bm25 = BM25Index.from_store(store, bot_id="bot-beta")
    assert {doc.metadata["bot_id"] for doc in alpha_bm25.search("policy", k=2)} == {
        "bot-alpha"
    }
    assert {doc.metadata["bot_id"] for doc in beta_bm25.search("policy", k=2)} == {
        "bot-beta"
    }

    exact = store.get_by_document_id("2", bot_id="bot-alpha")
    assert [doc.id for doc in exact] == ["DOC-2::0"]
    assert store.get_by_document_id("2", bot_id="bot-beta") == []
    assert store.document_exists("2", bot_id="bot-alpha") is True

    store.delete_by_document_id("2", bot_id="bot-alpha")

    assert store.document_exists("2", bot_id="bot-alpha") is False
    assert store.count(bot_id="bot-alpha") == 1
    assert store.count(bot_id="bot-beta") == 2

    store.delete_by_bot_id("bot-alpha")

    assert store.count(bot_id="bot-alpha") == 0
    assert store.count(bot_id="bot-beta") == 2
