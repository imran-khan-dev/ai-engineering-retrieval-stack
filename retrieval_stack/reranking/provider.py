"""Cross-encoder provider interface and adapter."""

from __future__ import annotations

from typing import Protocol, Sequence


class RerankerProvider(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class SentenceTransformerCrossEncoder:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]
