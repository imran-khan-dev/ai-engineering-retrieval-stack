"""Document ingestion service."""

from __future__ import annotations

import uuid
from pathlib import Path

from .errors import ExtractionError, UnsupportedFileTypeError
from .extractors import extract_text_from_pdf, extract_text_from_txt
from .loader import validate_path
from .models import Document
from .normalizer import normalize_text


class DocumentIngestor:
    def ingest(self, path: str | Path) -> Document:
        p = validate_path(path)
        suffix = p.suffix.lower()
        if suffix == ".txt":
            raw = extract_text_from_txt(p)
            page_count = None
        elif suffix == ".pdf":
            raw, page_count = extract_text_from_pdf(p)
        else:
            raise UnsupportedFileTypeError(f"Unsupported file type: {suffix}")

        content = normalize_text(raw)
        if not content:
            raise ExtractionError(f"No extractable text found in file: {p}")

        metadata: dict = {}
        if page_count is not None:
            metadata["page_count"] = page_count

        return Document(
            id=str(uuid.uuid4()),
            filename=p.name,
            file_type=suffix.lstrip("."),
            content=content,
            metadata=metadata,
        )
