from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lightweight tokenizer for RU/EN; lowercase alnum/underscore words."""
    if not text:
        return []
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def chunk_text(text: str, chunk_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    """Split text into overlapping character windows.

    This is intentionally simple and provider-agnostic.
    """
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    n = len(text)
    if chunk_chars <= 0:
        return [text]
    overlap_chars = max(0, min(overlap_chars, chunk_chars - 1))
    out: list[str] = []
    start = 0
    while start < n:
        end = min(n, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= n:
            break
        start = end - overlap_chars
    return out


@dataclass
class Attachment:
    kind: str  # 'image'
    path: str

