"""Evaluation metrics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _relevant_set(judgments: Mapping[str, int]) -> set[str]:
    return {cid for cid, grade in judgments.items() if grade > 0}


def precision_at_k(retrieved_ids: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    _validate_k(k)
    top = list(retrieved_ids)[:k]
    if k == 0:
        return 0.0
    relevant = _relevant_set(judgments)
    hits = sum(1 for cid in top if cid in relevant)
    return hits / k


def recall_at_k(retrieved_ids: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    _validate_k(k)
    relevant = _relevant_set(judgments)
    if not relevant:
        return 0.0
    top = list(retrieved_ids)[:k]
    hits = sum(1 for cid in top if cid in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved_ids: Sequence[str], judgments: Mapping[str, int]) -> float:
    relevant = _relevant_set(judgments)
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved_ids: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    _validate_k(k)
    top = list(retrieved_ids)[:k]
    dcg = 0.0
    for i, cid in enumerate(top, start=1):
        grade = int(judgments.get(cid, 0))
        if grade <= 0:
            continue
        gain = (2 ** grade) - 1
        dcg += gain / math.log2(i + 1)
    return dcg


def idcg_at_k(judgments: Mapping[str, int], k: int) -> float:
    _validate_k(k)
    grades = sorted([int(g) for g in judgments.values() if g > 0], reverse=True)[:k]
    idcg = 0.0
    for i, grade in enumerate(grades, start=1):
        gain = (2 ** grade) - 1
        idcg += gain / math.log2(i + 1)
    return idcg


def ndcg_at_k(retrieved_ids: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    _validate_k(k)
    ideal = idcg_at_k(judgments, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(retrieved_ids, judgments, k) / ideal
