"""
Chunk index for citation verification.
Builds in-memory lookup from (book_id, volume_id, start_line, end_line) to chunk.
"""

import json
import re
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


def parse_book_volume(file_or_book: str) -> tuple[str, str]:
    """
    Parse citation file/book string into (book_id, volume_id) for lookup.
    E.g. 'Republic II' -> ('republic', 'II'), 'Republic 2' -> ('republic', 'II'),
    'Meno' -> ('meno', ''). Handles Roman numerals I–X and Arabic 1–10.
    """
    s = (file_or_book or "").strip()
    if not s:
        return ("", "")
    lower = s.lower()
    # Multi-volume: "Republic II", "Republic 2", "republic book 2", "Apology I"
    # Match "Republic II" or "Republic 2" or "republic book II"
    m = re.match(r"^(republic|apology|meno|gorgias)\s+(?:book\s+)?(II|III|IV|V|VI|VII|VIII|IX|X|I|\d+)$", lower, re.I)
    if m:
        book = m.group(1).lower()
        vol = m.group(2).upper()
        if vol.isdigit():
            romans = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
            vol = romans[int(vol)] if 0 <= int(vol) <= 10 else vol
        return (book, vol)
    # Single token or "BookName" only
    base = re.sub(r"\s+(?:book\s+)?(II|III|IV|V|VI|VII|VIII|IX|X|I|\d+)$", "", lower, flags=re.I).strip()
    if base in ("republic", "apology", "meno", "gorgias"):
        return (base, "")
    # Use whole string as book_id (lowercase), no volume
    return (lower, "")


class ChunkIndex:
    """
    Index for lookup by (book_id, volume_id, start_line, end_line).
    """

    def __init__(self, books_dir: Path = BOOKS_DIR):
        self.books_dir = Path(books_dir)
        self._by_exact: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        self._all_chunks: list[dict[str, Any]] = []
        self._chunk_key_set: set[tuple[str, str, int, int, int | None]] = set()
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
                vol = (c_copy.get("volume_id") or "").strip().upper()
                start = c_copy.get("start_line")
                end = c_copy.get("end_line")
                chunk_id = c_copy.get("chunk_id")
                if start is None or end is None:
                    continue
                try:
                    s, e = int(start), int(end)
                except (TypeError, ValueError):
                    continue
                key_exact = (book, vol, s, e)
                self._by_exact[key_exact] = c_copy
                self._all_chunks.append(c_copy)
                self._chunk_key_set.add((book, vol, s, e, chunk_id))

    def get(
        self,
        book_id: str,
        start_line: int,
        end_line: int,
        volume_id: str = "",
    ) -> dict[str, Any] | None:
        """Look up chunk by (book_id, volume_id, start_line, end_line)."""
        book = book_id.lower().strip()
        vol = (volume_id or "").strip().upper()
        try:
            s, e = int(start_line), int(end_line)
        except (TypeError, ValueError):
            return None
        key = (book, vol, s, e)
        return self._by_exact.get(key)

    def chunk_key(self, chunk: dict[str, Any]) -> tuple[str, str, int, int]:
        """Unique key for a chunk: (book_id, volume_id, start_line, end_line)."""
        book = (chunk.get("book_id") or "?").lower()
        vol = (chunk.get("volume_id") or "").strip()
        s = chunk.get("start_line", 0)
        e = chunk.get("end_line", 0)
        try:
            return (book, vol, int(s), int(e))
        except (TypeError, ValueError):
            return (book, vol, 0, 0)

    def section_key(self, chunk: dict[str, Any]) -> tuple[str, str | int]:
        """Section key for diversity: (book_id, thematic_division)."""
        return _section_key(chunk)
