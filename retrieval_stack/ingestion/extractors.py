"""Text extraction from supported file types."""

from __future__ import annotations

from pathlib import Path

import fitz

from .errors import ExtractionError


def extract_text_from_txt(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return p.read_text(encoding="latin-1")
        except Exception as e:
            raise ExtractionError(f"Failed to extract text from TXT: {path}") from e
    except Exception as e:
        raise ExtractionError(f"Failed to extract text from TXT: {path}") from e


def extract_text_from_pdf(path: str | Path) -> tuple[str, int]:
    p = Path(path)
    try:
        doc = fitz.open(str(p))
    except Exception as e:
        raise ExtractionError(f"Failed to open PDF: {path}") from e
    try:
        pages = []
        for page in doc:
            pages.append(page.get_text())
        page_count = doc.page_count
        text = "\n\n".join(pages)
        return text, page_count
    except Exception as e:
        raise ExtractionError(f"Failed to extract text from PDF: {path}") from e
    finally:
        try:
            doc.close()
        except Exception:
            pass
