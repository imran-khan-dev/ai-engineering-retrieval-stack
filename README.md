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

### Tests

```bash
pytest
```
