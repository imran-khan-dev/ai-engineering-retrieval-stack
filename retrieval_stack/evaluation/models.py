"""Evaluation result models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryEvaluationResult(BaseModel):
    query_id: str
    query_text: str
    method: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    precision_at_k: dict[int, float] = Field(default_factory=dict)
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    reciprocal_rank: float = 0.0
    ndcg_at_k: dict[int, float] = Field(default_factory=dict)


class AggregateEvaluationResult(BaseModel):
    method: str
    num_queries: int
    avg_precision_at_k: dict[int, float] = Field(default_factory=dict)
    avg_recall_at_k: dict[int, float] = Field(default_factory=dict)
    mrr: float = 0.0
    avg_ndcg_at_k: dict[int, float] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    k_values: list[int] = Field(default_factory=list)
    per_query: list[QueryEvaluationResult] = Field(default_factory=list)
    aggregate: list[AggregateEvaluationResult] = Field(default_factory=list)
