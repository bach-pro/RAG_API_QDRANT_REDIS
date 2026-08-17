from __future__ import annotations

from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid5

from .data_loader import sanitize_metadata
from .models import KnowledgeDocument, RetrievedDocument


class OllamaEmbeddingFunction:
    """Embedding function backed by local Ollama."""

    def __init__(
        self,
        model_name: str = "embeddinggemma:latest",
        host: str = "http://localhost:11434",
        batch_size: int = 32,
        keep_alive: str = "10s",
    ) -> None:
        from ollama import Client

        self.model_name = model_name
        self.host = host
        self.batch_size = batch_size
        self.keep_alive = keep_alive
        self.client = Client(host=host)

    def __call__(self, input: list[str]) -> list[list[float]]:
        if isinstance(input, str):
            input = [input]
        embeddings: list[list[float]] = []
        for start in range(0, len(input), self.batch_size):
            batch = input[start : start + self.batch_size]
            response = self.client.embed(
                model=self.model_name,
                input=batch,
                keep_alive=self.keep_alive,
            )
            batch_embeddings = (
                response["embeddings"]
                if isinstance(response, dict)
                else response.embeddings
            )
            embeddings.extend([list(embedding) for embedding in batch_embeddings])
        return embeddings

    def name(self) -> str:
        return f"ollama-{self.model_name}"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)


class SentenceTransformerEmbeddingFunction:
    """Small adapter for optional sentence-transformers embeddings."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "SentenceTransformer embeddings need the optional dependency set: "
                "run `uv sync --extra multilingual`, then rebuild the index."
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(list(input), normalize_embeddings=True)
        return [list(map(float, embedding)) for embedding in embeddings]

    def name(self) -> str:
        return f"sentence-transformers-{self.model_name}"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)


def build_embedding_function(
    provider: str,
    model_name: str,
    host: str = "http://localhost:11434",
    keep_alive: str = "10s",
):
    provider = (provider or "ollama").strip().lower()
    if provider == "ollama":
        return OllamaEmbeddingFunction(
            model_name=model_name,
            host=host,
            keep_alive=keep_alive,
        )
    if provider == "sentence-transformers":
        return SentenceTransformerEmbeddingFunction(model_name)
    raise ValueError(
        f"Unsupported embedding provider '{provider}'. Use 'ollama' or "
        "'sentence-transformers'."
    )


class QdrantKnowledgeStore:
    def __init__(
        self,
        url: str,
        collection_name: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_host: str = "http://localhost:11434",
        api_key: str | None = None,
        keep_alive: str = "10s",
        reset: bool = False,
    ) -> None:
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        self.embedding_function = build_embedding_function(
            embedding_provider,
            embedding_model,
            embedding_host,
            keep_alive,
        )
        self._use_payload_index = url != ":memory:"
        if url == ":memory:":
            self.client = QdrantClient(location=url, api_key=api_key or None)
        else:
            self.client = QdrantClient(url=url, api_key=api_key or None)
        self.vector_size = self._detect_vector_size()
        self._payload_index_ready = not self._use_payload_index
        if reset:
            self.reset()
        else:
            self._ensure_collection()

    def _detect_vector_size(self) -> int:
        embedding = self.embedding_function(["dimension probe"])[0]
        if not embedding:
            raise RuntimeError("Embedding provider returned an empty vector.")
        return len(embedding)

    def _ensure_collection(self) -> None:
        from qdrant_client.http import models as qmodels

        collections = self.client.get_collections().collections
        if not any(collection.name == self.collection_name for collection in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        if not self._payload_index_ready:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="bot_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            self._payload_index_ready = True

    @staticmethod
    def _point_id(bot_id: str, document_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"{bot_id}\x00{document_id}"))

    @staticmethod
    def _bot_filter(bot_id: str):
        from qdrant_client.http import models as qmodels

        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="bot_id",
                    match=qmodels.MatchValue(value=bot_id),
                )
            ]
        )

    @staticmethod
    def _document_filter(bot_id: str, document_id: str):
        from qdrant_client.http import models as qmodels

        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="bot_id",
                    match=qmodels.MatchValue(value=bot_id),
                )
            ],
            should=[
                qmodels.FieldCondition(
                    key=key,
                    match=qmodels.MatchValue(value=str(document_id)),
                )
                for key in (
                    "metadata.document_id",
                    "metadata.doc_id",
                    "metadata.source_doc_id",
                )
            ],
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._payload_index_ready = not self._use_payload_index
        self._ensure_collection()

    def count(self, bot_id: str | None = None) -> int:
        self._ensure_collection()
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=self._bot_filter(bot_id) if bot_id is not None else None,
            exact=True,
        )
        return int(result.count)

    def delete_by_bot_id(self, bot_id: str) -> None:
        from qdrant_client.http import models as qmodels

        self._ensure_collection()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(filter=self._bot_filter(bot_id)),
            wait=True,
        )

    def document_exists(self, document_id: str, bot_id: str) -> bool:
        self._ensure_collection()
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=self._document_filter(bot_id, document_id),
            exact=True,
        )
        return int(result.count) > 0

    def delete_by_document_id(self, document_id: str, bot_id: str) -> None:
        from qdrant_client.http import models as qmodels

        self._ensure_collection()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=self._document_filter(bot_id, document_id)
            ),
            wait=True,
        )

    def ingest(
        self,
        documents: list[KnowledgeDocument],
        bot_id: str,
        batch_size: int = 100,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> int:
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "batch_start",
                        "start": start,
                        "end": start + len(batch),
                        "total": len(documents),
                    }
                )
            self._upsert_batch(batch, bot_id=bot_id)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "batch_done",
                        "start": start,
                        "end": start + len(batch),
                        "total": len(documents),
                    }
                )
        return self.count(bot_id=bot_id)

    def _upsert_batch(self, batch: list[KnowledgeDocument], *, bot_id: str) -> None:
        from qdrant_client.http import models as qmodels

        vectors = self.embedding_function([doc.text for doc in batch])
        points = []
        for doc, vector in zip(batch, vectors):
            metadata = dict(doc.metadata)
            metadata["bot_id"] = bot_id
            points.append(
                qmodels.PointStruct(
                    id=self._point_id(bot_id, doc.id),
                    vector=list(vector),
                    payload={
                        "id": doc.id,
                        "bot_id": bot_id,
                        "text": doc.text,
                        "metadata": sanitize_metadata(metadata),
                    },
                )
            )
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def query(
        self,
        query_text: str,
        bot_id: str,
        k: int = 5,
        include_embeddings: bool = False,
    ) -> list[RetrievedDocument]:
        self._ensure_collection()
        query_vector = self.embedding_function([query_text])[0]
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._bot_filter(bot_id),
            limit=max(1, k),
            with_payload=True,
            with_vectors=include_embeddings,
        )
        return [
            _point_to_document(point, include_embedding=include_embeddings)
            for point in result.points
        ]

    def get_all_documents(self, bot_id: str) -> list[RetrievedDocument]:
        self._ensure_collection()
        documents: list[RetrievedDocument] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._bot_filter(bot_id),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            documents.extend(_point_to_document(point) for point in points)
            if offset is None:
                break
        return documents

    def get_by_document_id(self, document_id: str, bot_id: str) -> list[RetrievedDocument]:
        self._ensure_collection()
        points: list[Any] = []
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._document_filter(bot_id, document_id),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if offset is None:
                break
        documents: list[RetrievedDocument] = []
        for point in points:
            document = _point_to_document(point)
            document.score = 1.0
            documents.append(document)
        return documents


def _point_to_document(point: Any, include_embedding: bool = False) -> RetrievedDocument:
    payload = point.payload or {}
    metadata = dict(payload.get("metadata") or {})
    if payload.get("bot_id") is not None:
        metadata["bot_id"] = str(payload["bot_id"])
    score = getattr(point, "score", None)
    embedding = None
    if include_embedding:
        vector = getattr(point, "vector", None)
        if isinstance(vector, dict):
            vector = next(iter(vector.values()), None)
        if vector is not None:
            embedding = list(vector)

    return RetrievedDocument(
        id=str(payload.get("id") or point.id),
        text=str(payload.get("text") or ""),
        metadata=metadata,
        score=float(score) if score is not None else None,
        distance=(1.0 - float(score)) if score is not None else None,
        embedding=embedding,
    )
