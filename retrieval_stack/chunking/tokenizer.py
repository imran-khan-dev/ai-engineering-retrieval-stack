"""Tokenizer abstraction."""

from __future__ import annotations

import tiktoken


# class Tokenizer:
#     def __init__(self, encoding: str = "cl100k_base"):
#         self.encoding_name = encoding
#         self._enc = tiktoken.get_encoding(encoding)

#     def encode(self, text: str) -> list[int]:
#         return self._enc.encode(text)

#     def decode(self, tokens: list[int]) -> str:
#         return self._enc.decode(tokens)


class Tokenizer:
    def __init__(self, encoding: str = "cl100k_base"):
        self.encoding_name = encoding
        self._enc = tiktoken.get_encoding(encoding)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)

    def encode_with_offsets(
        self, text: str
    ) -> tuple[list[int], list[int]]:
        tokens = self.encode(text)
        _, offsets = self._enc.decode_with_offsets(tokens)
        return tokens, offsets