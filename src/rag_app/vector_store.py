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
        if url == ":memory:":
            self.client = QdrantClient(location=url, api_key=api_key or None)
        else:
            self.client = QdrantClient(url=url, api_key=api_key or None)
        self.vector_size = self._detect_vector_size()
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
        if any(collection.name == self.collection_name for collection in collections):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )

    @staticmethod
    def _point_id(document_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, document_id))

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._ensure_collection()

    def count(self) -> int:
        self._ensure_collection()
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )
        return int(result.count)

    def ingest(
        self,
        documents: list[KnowledgeDocument],
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
            self._upsert_batch(batch)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "batch_done",
                        "start": start,
                        "end": start + len(batch),
                        "total": len(documents),
                    }
                )
        return self.count()

    def _upsert_batch(self, batch: list[KnowledgeDocument]) -> None:
        from qdrant_client.http import models as qmodels

        vectors = self.embedding_function([doc.text for doc in batch])
        points = [
            qmodels.PointStruct(
                id=self._point_id(doc.id),
                vector=list(vector),
                payload={
                    "id": doc.id,
                    "text": doc.text,
                    "metadata": sanitize_metadata(doc.metadata),
                },
            )
            for doc, vector in zip(batch, vectors)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def query(
        self,
        query_text: str,
        k: int = 5,
        include_embeddings: bool = False,
    ) -> list[RetrievedDocument]:
        self._ensure_collection()
        query_vector = self.embedding_function([query_text])[0]
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=max(1, k),
            with_payload=True,
            with_vectors=include_embeddings,
        )
        return [
            _point_to_document(point, include_embedding=include_embeddings)
            for point in result.points
        ]

    def get_all_documents(self) -> list[RetrievedDocument]:
        self._ensure_collection()
        documents: list[RetrievedDocument] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            documents.extend(_point_to_document(point) for point in points)
            if offset is None:
                break
        return documents

    def get_by_document_id(self, document_id: str) -> list[RetrievedDocument]:
        from qdrant_client.http import models as qmodels

        self._ensure_collection()
        points: list[Any] = []
        for key in ("metadata.document_id", "metadata.doc_id", "metadata.source_doc_id"):
            batch, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key=key,
                            match=qmodels.MatchValue(value=str(document_id)),
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)

        seen: set[str] = set()
        documents: list[RetrievedDocument] = []
        for point in points:
            document = _point_to_document(point)
            if document.id in seen:
                continue
            seen.add(document.id)
            document.score = 1.0
            documents.append(document)
        return documents


def _point_to_document(point: Any, include_embedding: bool = False) -> RetrievedDocument:
    payload = point.payload or {}
    metadata = dict(payload.get("metadata") or {})
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
