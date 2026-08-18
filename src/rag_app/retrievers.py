from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from .models import RetrievedDocument
from .vector_store import QdrantKnowledgeStore


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


class BM25Index:
    def __init__(self, documents: list[RetrievedDocument]) -> None:
        self.documents = documents
        self._tokenized = [tokenize(doc.text) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized) if documents else None

    @classmethod
    def from_store(cls, store: QdrantKnowledgeStore, bot_id: str) -> "BM25Index":
        return cls(store.get_all_documents(bot_id=bot_id))

    def search(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        if not self.documents or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][: max(1, k)]
        results: list[RetrievedDocument] = []
        for idx in order:
            score = float(scores[idx])
            if score <= 0 and results:
                continue
            results.append(replace(self.documents[int(idx)], score=score))
        return results


def semantic_search(
    store: QdrantKnowledgeStore,
    query: str,
    bot_id: str,
    k: int = 5,
) -> list[RetrievedDocument]:
    return store.query(query, bot_id=bot_id, k=k)


def bm25_search(index: BM25Index, query: str, k: int = 5) -> list[RetrievedDocument]:
    return index.search(query, k=k)


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[RetrievedDocument]],
    limit: int = 5,
    rrf_k: int = 60,
) -> list[RetrievedDocument]:
    scores: dict[str, float] = {}
    docs: dict[str, RetrievedDocument] = {}
    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs, start=1):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rrf_k + rank)
            docs[doc.id] = doc

    ordered_ids = sorted(scores, key=scores.get, reverse=True)
    return [replace(docs[doc_id], score=scores[doc_id]) for doc_id in ordered_ids[:limit]]


def hybrid_rrf_search(
    store: QdrantKnowledgeStore,
    bm25: BM25Index,
    query: str,
    bot_id: str,
    k: int = 5,
    fetch_k: int = 10,
    rrf_k: int = 60,
) -> list[RetrievedDocument]:
    semantic = semantic_search(store, query, bot_id=bot_id, k=fetch_k)
    sparse = bm25_search(bm25, query, k=fetch_k)
    return reciprocal_rank_fusion([semantic, sparse], limit=k, rrf_k=rrf_k)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0 or math.isnan(denominator):
        return 0.0
    return float(np.dot(a, b) / denominator)


def mmr_search(
    store: QdrantKnowledgeStore,
    query: str,
    bot_id: str,
    k: int = 5,
    fetch_k: int = 12,
    lambda_mult: float = 0.5,
) -> list[RetrievedDocument]:
    candidates = store.query(
        query,
        bot_id=bot_id,
        k=max(fetch_k, k),
        include_embeddings=True,
    )
    candidates = [doc for doc in candidates if doc.embedding]
    if not candidates:
        return semantic_search(store, query, bot_id=bot_id, k=k)

    query_embedding = np.array(store.embedding_function([query])[0], dtype=np.float32)
    doc_embeddings = [np.array(doc.embedding, dtype=np.float32) for doc in candidates]
    query_scores = [_cosine(query_embedding, emb) for emb in doc_embeddings]

    selected: list[int] = []
    remaining = set(range(len(candidates)))
    while remaining and len(selected) < k:
        best_idx = None
        best_score = -float("inf")
        for idx in remaining:
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    _cosine(doc_embeddings[idx], doc_embeddings[chosen])
                    for chosen in selected
                )
            score = lambda_mult * query_scores[idx] - (1.0 - lambda_mult) * diversity_penalty
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [
        replace(candidates[idx], score=float(query_scores[idx]))
        for idx in selected
    ]
