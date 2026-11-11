# Retrieval Stack

## Milestone 1: Document Ingestion Foundation

### Setup

```bash
pip install -r requirements.txt
```

### Supported formats
- PDF (.pdf)
- TXT (.txt)

### Component responsibilities
- `loader.py`: file validation and reading.
- `extractors.py`: text extraction (PyMuPDF for PDF, plain read for TXT).
- `normalizer.py`: whitespace/line-ending normalization.
- `models.py`: standardized `Document` Pydantic model.
- `service.py`: `DocumentIngestor` orchestrates loading, extraction, normalization.

### Usage

```python
from retrieval_stack.ingestion import DocumentIngestor

ingestor = DocumentIngestor()
doc = ingestor.ingest("path/to/file.pdf")
print(doc.content)
print(doc.metadata)
```

## Milestone 2: Chunking Engine

### Strategies
- fixed: token-based fixed size with overlap
- recursive: separator-based recursive splitting

### Usage

```python
from retrieval_stack.chunking import ChunkConfig, ChunkingService

cfg = ChunkConfig(chunk_size=200, chunk_overlap=20)
svc = ChunkingService(cfg, strategy="recursive")
chunks = svc.chunk_document(doc)
```

## Milestone 5: Hybrid Search (RRF)

Combines BM25 (keyword) and Dense (semantic) rankings using Reciprocal Rank Fusion: `RRF = Σ 1/(k + rank)`. Uses ranks only (not raw scores). Default `k=60`.

```python
from retrieval_stack.hybrid import HybridSearcher
h = HybridSearcher(bm25_searcher, dense_searcher, rrf_k=60)
results = h.search("query", top_k=5, bm25_top_k=20, dense_top_k=20)
```

Example: BM25 `[A,B]`, Dense `[B,C]` with k=60 → `B=1/62+1/61`, `A=1/61`, `C=1/62`.

## Milestone 7: Retrieval Pipeline Orchestrator

`RetrievalPipeline` coordinates ingestion, chunking, BM25, dense indexing, hybrid search, and reranking.

```python
from retrieval_stack import RetrievalPipeline
from retrieval_stack.chunking import ChunkConfig, ChunkingService

# Provide injectable/fake embedder and reranker for tests; use real adapters in production.
pipeline = RetrievalPipeline(
    chunker=ChunkingService(ChunkConfig(chunk_size=200, chunk_overlap=20)),
    embedder=my_embedder,
    reranker=my_reranker,
)
chunks = pipeline.index_documents(["docs/a.pdf", "docs/b.txt"])
response = pipeline.search(
    "recover account",
    bm25_top_k=20,
    dense_top_k=20,
    hybrid_top_k=20,
    top_k=5,
)
```

- Indexing: ingest -> chunk -> stage BM25/vector builds and batch embeddings -> atomic swap on success.
- Querying: hybrid search over existing indexes -> rerank candidates -> return `SearchResponse`.
- Repeated searches reuse indexes and do not regenerate document embeddings (query embedding only). In-memory only.

## Milestone 8: Retrieval Evaluation Framework

Measures retrieval quality with human-provided (ground-truth) relevance judgments (graded) using Precision@K, Recall@K, MRR, and NDCG@K. Evaluates BM25, Dense, Hybrid, and Hybrid+Reranking over the same queries/K; metrics use ranking positions and chunk IDs only (no mixing of raw scores). Example judgments are illustrative, not experimentally validated.

```python
from retrieval_stack.evaluation import EvaluationDataset, EvaluationQuery, evaluate_pipeline, comparison_table, export_report_json

dataset = EvaluationDataset(queries=[
    EvaluationQuery(query_id="q1", query_text="reset password", relevance_judgments={"C1": 3, "C3": 1}),
])
report = evaluate_pipeline(pipeline, dataset, k_values=[5])
print(comparison_table(report, 5))
export_report_json(report, "eval.json")
```

Edge cases: invalid/non-positive K raises; no relevant judgments -> recall 0; ideal DCG 0 -> NDCG 0; empty results score 0 where applicable. Small manual benchmarks are limited and may not generalize.

