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

### Tests
pytest
