"""BM25 searcher."""

from __future__ import annotations

from .index import InvertedIndex
from .models import SearchResult
from .scorer import BM25Scorer
from .tokenizer import Tokenizer


class BM25Searcher:
    def __init__(self, index: InvertedIndex, scorer: BM25Scorer | None = None, tokenizer: Tokenizer | None = None):
        self.index = index
        self.scorer = scorer or BM25Scorer()
        self.tokenizer = tokenizer or Tokenizer()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            return []
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return []
        candidates = self.index.get_candidates(tokens)
        results = []
        for cid in candidates:
            score = self.scorer.score(self.index, tokens, cid)
            if score <= 0:
                continue
            chunk = self.index.chunks[cid]
            results.append(SearchResult(chunk_id=cid, document_id=chunk.document_id, score=score, chunk=chunk, metadata=dict(chunk.metadata)))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
