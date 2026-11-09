"""Dense searcher."""

from __future__ import annotations

from .embeddings import EmbeddingProvider
from .index import VectorIndex
from .models import DenseSearchResult
from .similarity import cosine_similarity


class DenseSearcher:
    def __init__(self, index: VectorIndex, embedder: EmbeddingProvider):
        self.index = index
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[DenseSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self.index.size == 0:
            return []
        qvec = self.embedder.embed(query)
        results = []
        embeddings = self.index.embeddings()
        for cid, chunk in self.index.items():
            emb = embeddings[cid]
            score = cosine_similarity(qvec, emb)
            results.append((cid, chunk, score))
        results.sort(key=lambda x: (-x[2], x[0]))
        out = []
        for cid, chunk, score in results[:top_k]:
            out.append(DenseSearchResult(chunk_id=cid, document_id=chunk.document_id, score=score, chunk=chunk, metadata=dict(chunk.metadata)))
        return out
