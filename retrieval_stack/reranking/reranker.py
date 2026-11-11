"""Reranking engine."""

from __future__ import annotations

from typing import Sequence

from .models import RerankedResult
from .provider import RerankerProvider


class Reranker:
    def __init__(self, provider: RerankerProvider):
        self.provider = provider

    def rerank(self, query: str, candidates: Sequence[object], top_k: int = 5, candidate_count: int | None = None) -> list[RerankedResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if candidate_count is not None and candidate_count <= 0:
            raise ValueError("candidate_count must be greater than 0")
        if not candidates:
            return []
        items = list(candidates)
        if candidate_count is not None:
            items = items[:candidate_count]
       
        texts = []
        for item in items:
            chunk = getattr(item, "chunk", None)
            if chunk is None:
                raise ValueError("candidate must have chunk")
            texts.append(chunk.content)

        scores = self.provider.score(query, texts)
        if len(scores) != len(items):
            raise ValueError("reranker returned incorrect number of scores")

        results = []
        for item, score in zip(items, scores):
            chunk = item.chunk
            hybrid = item if hasattr(item, "rrf_score") else None
            results.append(
                RerankedResult(
                    chunk_id=getattr(item, "chunk_id", chunk.id),
                    chunk=chunk,
                    relevance_score=float(score),
                    rrf_score=getattr(item, "rrf_score", 0.0) if hybrid else 0.0,
                    sources=list(getattr(item, "sources", []) or []),
                    bm25_rank=getattr(item, "bm25_rank", None),
                    dense_rank=getattr(item, "dense_rank", None),
                    metadata=dict(getattr(item, "metadata", {}) or getattr(chunk, "metadata", {}) or {}),
                    hybrid_result=hybrid,
                )
            )

        results.sort(key=lambda r: (-r.relevance_score, r.chunk_id))
        return results[:top_k]
