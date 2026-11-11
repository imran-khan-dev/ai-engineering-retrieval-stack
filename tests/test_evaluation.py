import json

import pytest

from retrieval_stack.evaluation import (
    EvaluationDataset,
    EvaluationQuery,
    comparison_table,
    dcg_at_k,
    evaluate_methods,
    evaluate_search_fn,
    example_dataset,
    export_report_csv,
    export_report_json,
    idcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class R:
    def __init__(self, chunk_id):
        self.chunk_id = chunk_id


def test_precision_recall_known():
    retrieved = ["c1", "c2", "c3", "c4"]
    judgments = {"c1": 3, "c3": 1}
    assert precision_at_k(retrieved, judgments, 2) == pytest.approx(0.5)
    assert precision_at_k(retrieved, judgments, 4) == pytest.approx(0.5)
    assert recall_at_k(retrieved, judgments, 2) == pytest.approx(0.5)
    assert recall_at_k(retrieved, judgments, 4) == pytest.approx(1.0)


def test_mrr():
    judgments = {"c2": 1}
    assert reciprocal_rank(["c1", "c2", "c3"], judgments) == pytest.approx(0.5)
    assert reciprocal_rank(["c2", "c1"], judgments) == pytest.approx(1.0)
    assert reciprocal_rank(["c1", "c3"], judgments) == 0.0


def test_dcg_ndcg():
    retrieved = ["c1", "c2", "c3"]
    judgments = {"c1": 3, "c3": 2}
    # DCG: (2^3-1)/log2(2) + 0 + (2^2-1)/log2(4)
    expected_dcg = (7 / 1.0) + (3 / 2.0)
    assert dcg_at_k(retrieved, judgments, 3) == pytest.approx(expected_dcg)
    # ideal: grades [3,2] at ranks 1,2
    expected_idcg = (7 / 1.0) + (3 / __import__("math").log2(3))
    assert idcg_at_k(judgments, 3) == pytest.approx(expected_idcg)
    assert ndcg_at_k(retrieved, judgments, 3) == pytest.approx(expected_dcg / expected_idcg)
    # reorder
    assert ndcg_at_k(["c3", "c2", "c1"], judgments, 3) == pytest.approx(((3 / 1.0) + (7 / 2.0)) / expected_idcg)


def test_ndcg_zero_ideal():
    assert ndcg_at_k(["c1"], {}, 5) == 0.0


def test_empty_results():
    judgments = {"c1": 1}
    assert precision_at_k([], judgments, 5) == 0.0
    assert recall_at_k([], judgments, 5) == 0.0
    assert reciprocal_rank([], judgments) == 0.0
    assert ndcg_at_k([], judgments, 5) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["c1"], {"c1": 1}, 0)


def test_no_relevant_judgments_recall():
    assert recall_at_k(["c1"], {}, 5) == 0.0


def test_dataset_validation():
    ds = EvaluationDataset(queries=[EvaluationQuery(query_id="q1", query_text="hi", relevance_judgments={"c1": 2})])
    assert len(ds.queries) == 1
    with pytest.raises(Exception):
        EvaluationDataset(queries=[EvaluationQuery(query_id="q1", query_text="a", relevance_judgments={}), EvaluationQuery(query_id="q1", query_text="b", relevance_judgments={})])
    with pytest.raises(Exception):
        EvaluationQuery(query_id="", query_text="a", relevance_judgments={})
    with pytest.raises(Exception):
        EvaluationQuery(query_id="q1", query_text="", relevance_judgments={})
    with pytest.raises(Exception):
        EvaluationQuery(query_id="q1", query_text="a", relevance_judgments={"c": -1})


def test_example_dataset():
    ds = example_dataset()
    assert len(ds.queries) == 2


def test_evaluate_search_fn_and_methods():
    ds = EvaluationDataset(queries=[
        EvaluationQuery(query_id="q1", query_text="a", relevance_judgments={"c1": 3, "c2": 1}),
        EvaluationQuery(query_id="q2", query_text="b", relevance_judgments={"c3": 2}),
    ])
    def fn(q, top_k):
        if q == "a":
            return [R("c2"), R("c1")]
        return [R("x"), R("c3")]
    per, agg = evaluate_search_fn("bm25", fn, ds, [1, 2])
    assert len(per) == 2
    assert per[0].retrieved_chunk_ids == ["c2", "c1"]
    assert per[0].precision_at_k[1] == pytest.approx(1.0)
    assert per[1].reciprocal_rank == pytest.approx(0.5)
    assert agg.num_queries == 2
    report = evaluate_methods({"bm25": fn, "dense": fn}, ds, [2])
    assert len(report.aggregate) == 2
    assert len(report.per_query) == 4


def test_export_and_comparison(tmp_path):
    ds = example_dataset()
    report = evaluate_methods({"bm25": lambda q, k: [R("C1")]}, ds, [5])
    j = tmp_path / "r.json"
    c = tmp_path / "r.csv"
    export_report_json(report, j)
    export_report_csv(report, c)
    data = json.loads(j.read_text())
    assert data["aggregate"][0]["method"] == "bm25"
    assert c.read_text().splitlines()[0].startswith("method,")
    table = comparison_table(report, 5)
    assert "BM25" in table.upper() or "bm25" in table
    assert "Precision@5" in table
