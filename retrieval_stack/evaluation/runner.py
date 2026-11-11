"""Evaluation runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .dataset import EvaluationDataset
from .metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from .models import AggregateEvaluationResult, EvaluationReport, QueryEvaluationResult


SearchFn = Callable[[str, int], Sequence[object]]

def _extract_chunk_id(item: object) -> str:
    cid = getattr(item, "chunk_id", None)
    if cid is not None:
        return str(cid)
    chunk = getattr(item, "chunk", None)
    if chunk is not None and getattr(chunk, "id", None) is not None:
        return str(chunk.id)
    raise ValueError("result item must expose chunk_id or chunk.id")


def evaluate_search_fn(
    method: str,
    search_fn: SearchFn,
    dataset: EvaluationDataset,
    k_values: Sequence[int],
) -> tuple[list[QueryEvaluationResult], AggregateEvaluationResult]:
    ks = list(k_values)
    for k in ks:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k values must be positive integers")
    per_query: list[QueryEvaluationResult] = []
    max_k = max(ks) if ks else 0

    for q in dataset.queries:
        results = search_fn(q.query_text, max_k) if max_k > 0 else []
        ids = [_extract_chunk_id(r) for r in results]
        p = {k: precision_at_k(ids, q.relevance_judgments, k) for k in ks}
        r = {k: recall_at_k(ids, q.relevance_judgments, k) for k in ks}
        n = {k: ndcg_at_k(ids, q.relevance_judgments, k) for k in ks}
        rr = reciprocal_rank(ids, q.relevance_judgments)
        
        per_query.append(
            QueryEvaluationResult(
                query_id=q.query_id,
                query_text=q.query_text,
                method=method,
                retrieved_chunk_ids=ids,
                precision_at_k=p,
                recall_at_k=r,
                reciprocal_rank=rr,
                ndcg_at_k=n,
            )
        )

    n_queries = len(dataset.queries)
    def avg(vals: list[float]) -> float:
        return sum(vals) / n_queries if n_queries else 0.0

    agg = AggregateEvaluationResult(
        method=method,
        num_queries=n_queries,
        avg_precision_at_k={k: avg([x.precision_at_k[k] for x in per_query]) for k in ks},
        avg_recall_at_k={k: avg([x.recall_at_k[k] for x in per_query]) for k in ks},
        mrr=avg([x.reciprocal_rank for x in per_query]),
        avg_ndcg_at_k={k: avg([x.ndcg_at_k[k] for x in per_query]) for k in ks},
    )
    return per_query, agg


def evaluate_methods(
    search_fns: Mapping[str, SearchFn],
    dataset: EvaluationDataset,
    k_values: Sequence[int],
) -> EvaluationReport:
    per_query: list[QueryEvaluationResult] = []
    aggregate: list[AggregateEvaluationResult] = []
    for method, fn in search_fns.items():
        pq, ag = evaluate_search_fn(method, fn, dataset, k_values)
        per_query.extend(pq)
        aggregate.append(ag)
    return EvaluationReport(k_values=list(k_values), per_query=per_query, aggregate=aggregate)


def evaluate_pipeline(pipeline, dataset: EvaluationDataset, k_values: Sequence[int]) -> EvaluationReport:
    max_k = max(k_values) if k_values else 0

    def bm25_fn(query: str, top_k: int):
        return pipeline.bm25_searcher.search(query, top_k=top_k)

    def dense_fn(query: str, top_k: int):
        if pipeline.dense_searcher is None:
            return []
        return pipeline.dense_searcher.search(query, top_k=top_k)

    def hybrid_fn(query: str, top_k: int):
        if pipeline.hybrid_searcher is None:
            return []
        return pipeline.hybrid_searcher.search(query, top_k=top_k)

    def rerank_fn(query: str, top_k: int):
        resp = pipeline.search(query, bm25_top_k=max_k, dense_top_k=max_k, hybrid_top_k=max_k, top_k=top_k)
        return resp.results

    return evaluate_methods(
        {
            "bm25": bm25_fn,
            "dense": dense_fn,
            "hybrid": hybrid_fn,
            "hybrid_rerank": rerank_fn,
        },
        dataset,
        k_values,
    )


def export_report_json(report: EvaluationReport, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")


def export_report_csv(report: EvaluationReport, path: str | Path) -> None:
    ks = report.k_values
    fields = ["method", "num_queries", "mrr"]
    for k in ks:
        fields += [f"precision@{k}", f"recall@{k}", f"ndcg@{k}"]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for a in report.aggregate:
            row = {"method": a.method, "num_queries": a.num_queries, "mrr": a.mrr}
            for k in ks:
                row[f"precision@{k}"] = a.avg_precision_at_k.get(k, 0.0)
                row[f"recall@{k}"] = a.avg_recall_at_k.get(k, 0.0)
                row[f"ndcg@{k}"] = a.avg_ndcg_at_k.get(k, 0.0)
            w.writerow(row)


def comparison_table(report: EvaluationReport, k: int | None = None) -> str:
    if not report.aggregate:
        return ""
    use_k = k if k is not None else (report.k_values[0] if report.k_values else None)
    header = f"| Method | Precision@{use_k} | Recall@{use_k} | MRR | NDCG@{use_k} |"
    sep = "|---|---:|---:|---:|---:|"
    rows = [header, sep]
    for a in report.aggregate:
        p = a.avg_precision_at_k.get(use_k, 0.0) if use_k else 0.0
        r = a.avg_recall_at_k.get(use_k, 0.0) if use_k else 0.0
        n = a.avg_ndcg_at_k.get(use_k, 0.0) if use_k else 0.0
        rows.append(f"| {a.method} | {p:.4f} | {r:.4f} | {a.mrr:.4f} | {n:.4f} |")
    return "\n".join(rows)
