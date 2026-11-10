"""Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Sequence

from retrieval_stack.chunking.models import Chunk

from .models import HybridSearchResult


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[object]],
    source_names: Sequence[str] | None = None,
    k: int = 60,
) -> list[HybridSearchResult]:
    if k <= 0:
        raise ValueError("RRF constant k must be greater than 0")
    if source_names is not None and len(source_names) != len(ranked_lists):
        raise ValueError("source_names length must match ranked_lists")
    names = list(source_names) if source_names is not None else [f"source_{i}" for i in range(len(ranked_lists))]

    scores: dict[str, float] = {} # e.g {"A": 0.03252}
    chunks: dict[str, Chunk] = {} # e.g {"A": chunk_object_A}
    ranks: dict[str, dict[str, int]] = {} # e.g {"A": {"bm25": 1, "dense": 2}}
    sources: dict[str, list[str]] = {} # e.g {"A": ["bm25", "dense"]}

    for name, results in zip(names, ranked_lists):
        seen: set[str] = set()
        rank = 0
        for item in results:
            chunk_id = getattr(item, "chunk_id", None)
            chunk = getattr(item, "chunk", None)
            if chunk_id is None or chunk is None:
                raise ValueError("each result must have chunk_id and chunk")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

            chunks.setdefault(chunk_id, chunk)
            ranks.setdefault(chunk_id, {})[name] = rank

            if name not in sources.setdefault(chunk_id, []):
                sources[chunk_id].append(name)

    combined = []

    for chunk_id, score in scores.items():
        chunk = chunks[chunk_id]
        combined.append(
            HybridSearchResult(
                chunk_id=chunk_id,
                chunk=chunk,
                rrf_score=score,
                sources=list(sources.get(chunk_id, [])),
                bm25_rank=ranks.get(chunk_id, {}).get("bm25"),
                dense_rank=ranks.get(chunk_id, {}).get("dense"),
                metadata=dict(getattr(chunk, "metadata", {}) or {}),
            )
        )

    # sort the hybrid search results by RRF score, highest first. 
    # If two results have the same score, sort by chunk ID.
    combined.sort(key=lambda r: (-r.rrf_score, r.chunk_id)) 
    return combined
