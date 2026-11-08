"""BM25 scorer."""

from __future__ import annotations

import math

class BM25Scorer:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0")
        if not (0 <= b <= 1):
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b

    def idf(self, n: int, df: int) -> float:
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, index, query_tokens: list[str], chunk_id: str) -> float:
        n = index.total_docs
        avgdl = index.avg_doc_length or 1.0
        dl = index.doc_lengths.get(chunk_id, 0)
        total = 0.0
        for term in set(query_tokens):
            tf = index.term_freq.get(term, {}).get(chunk_id, 0)
            if tf == 0:
                continue
            df = index.doc_freq.get(term, 0)
            idf = self.idf(n, df)   # IDF — Rare words get higher scores
            denom = tf + self.k1 * (1 - self.b + self.b * (dl / avgdl))
            total += idf * (tf * (self.k1 + 1)) / denom
        return total
