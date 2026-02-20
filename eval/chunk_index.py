"""
Chunk index for citation verification.
Builds in-memory lookup from (book_id, start_line, end_line) to chunk.
"""

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOKS_DIR = PROJECT_ROOT / "books"
CHUNK_FILES = [
    "apology_chunks.json",
    "meno_chunks.json",
    "gorgias_chunks.json",
    "republic_chunks.json",
]


def _section_key(chunk: dict[str, Any]) -> tuple[str, str | int]:
    """Return (book_id, thematic_division) for diversity metrics."""
    book = (chunk.get("book_id") or "?").lower()
    div = chunk.get("thematic_division", "?")
    return (book, div)


def load_chunk_index(books_dir: Path | None = None) -> "ChunkIndex":
    """Load all chunks and build index."""
    books_dir = books_dir or BOOKS_DIR
    return ChunkIndex(books_dir)


class ChunkIndex:
    """
    Index for lookup by (book_id, start_line, end_line).
    Supports fallback to (book_id, start_line) when exact range not found.
    """

    def __init__(self, books_dir: Path = BOOKS_DIR):
        self.books_dir = Path(books_dir)
        self._by_exact: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._by_book_start: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._all_chunks: list[dict[str, Any]] = []
        self._chunk_key_set: set[tuple[str, int, int, int | None]] = set()
        self._build()

    def _build(self) -> None:
        for name in CHUNK_FILES:
            path = self.books_dir / name
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for c in data:
                c_copy = {k: v for k, v in c.items() if k != "embedding"}
                book = (c_copy.get("book_id") or "?").lower()
                start = c_copy.get("start_line")
                end = c_copy.get("end_line")
                chunk_id = c_copy.get("chunk_id")
                if start is None or end is None:
                    continue
                try:
                    s, e = int(start), int(end)
                except (TypeError, ValueError):
                    continue
                key_exact = (book, s, e)
                self._by_exact[key_exact] = c_copy
                key_start = (book, s)
                if key_start not in self._by_book_start:
                    self._by_book_start[key_start] = []
                self._by_book_start[key_start].append(c_copy)
                self._all_chunks.append(c_copy)
                self._chunk_key_set.add((book, s, e, chunk_id))

    def get(
        self,
        book_id: str,
        start_line: int,
        end_line: int,
    ) -> dict[str, Any] | None:
        """Look up chunk by (book_id, start_line, end_line). Fallback to (book_id, start_line)."""
        book = book_id.lower().strip()
        try:
            s, e = int(start_line), int(end_line)
        except (TypeError, ValueError):
            return None
        key = (book, s, e)
        if key in self._by_exact:
            return self._by_exact[key]
        key_start = (book, s)
        if key_start in self._by_book_start:
            candidates = self._by_book_start[key_start]
            for c in candidates:
                ce = c.get("end_line")
                if ce is not None and int(ce) == e:
                    return c
            return candidates[0]
        return None

    def chunk_key(self, chunk: dict[str, Any]) -> tuple[str, int, int]:
        """Unique key for a chunk: (book_id, start_line, end_line)."""
        book = (chunk.get("book_id") or "?").lower()
        s = chunk.get("start_line", 0)
        e = chunk.get("end_line", 0)
        try:
            return (book, int(s), int(e))
        except (TypeError, ValueError):
            return (book, 0, 0)

    def section_key(self, chunk: dict[str, Any]) -> tuple[str, str | int]:
        """Section key for diversity: (book_id, thematic_division)."""
        return _section_key(chunk)
