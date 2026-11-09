"""Hybrid searcher."""

from __future__ import annotations

from .rrf import reciprocal_rank_fusion


class HybridSearcher:
    def __init__(self, bm25_searcher, dense_searcher, rrf_k: int = 60):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")
        self.bm25_searcher = bm25_searcher
        self.dense_searcher = dense_searcher
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        bm25_top_k: int | None = None,
        dense_top_k: int | None = None,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        b_k = top_k if bm25_top_k is None else bm25_top_k
        d_k = top_k if dense_top_k is None else dense_top_k
        bm25_results = self.bm25_searcher.search(query, top_k=b_k)
        dense_results = self.dense_searcher.search(query, top_k=d_k)
        combined = reciprocal_rank_fusion(
            [bm25_results, dense_results],
            source_names=["bm25", "dense"],
            k=self.rrf_k,
        )
        return combined[:top_k]
