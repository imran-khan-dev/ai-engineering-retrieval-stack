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

