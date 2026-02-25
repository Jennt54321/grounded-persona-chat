import re
from typing import Any


def make_citation_from_chunk(chunk: dict[str, Any]) -> str:
    """Format chunk as citation string file:start-end. Includes volume when present (e.g. Republic II:573-600, Apology I:1-50)."""
    book_raw = chunk.get("book_id") or "?"
    book = book_raw.title() if book_raw else "?"
    vol = (chunk.get("volume_id") or "").strip()
    s = chunk.get("start_line")
    e = chunk.get("end_line")
    try:
        start, end = int(s), int(e)
    except (TypeError, ValueError):
        return ""
    file_part = f"{book} {vol}" if vol else book
    return f"{file_part}:{start}-{end}"