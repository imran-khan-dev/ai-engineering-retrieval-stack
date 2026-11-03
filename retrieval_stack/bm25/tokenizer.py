"""BM25 tokenizer."""

from __future__ import annotations

import re
import unicodedata


class Tokenizer:
    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        # split on non-alphanumeric (unicode letters/digits)
        return [t for t in re.findall(r"\w+", text, flags=re.UNICODE) if t]
