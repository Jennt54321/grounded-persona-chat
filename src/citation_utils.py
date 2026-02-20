"""
Citation utilities: auto-cite fallback when model fails to produce valid citations.
"""

import re
from typing import Any

# Reuse strict citation logic - avoid circular import by using local regex
_CITATION_RE = re.compile(r'^([^:]+):(\d+)-(\d+)$', re.IGNORECASE)


def _has_valid_citation_in_text(text: str) -> bool:
    """Check if text contains at least one citation in file:start-end format."""
    # [file:start-end]
    if re.search(r'\[[^:\]]+:\d+-\d+\]', text):
        return True
    # file:start-end (in JSON or plain)
    if _CITATION_RE.search(text):
        return True
    return False


def _has_valid_citation_in_data(data: dict[str, Any]) -> bool:
    """Check if parsed JSON has at least one quote with valid citation format."""
    quotes = data.get("quotes")
    if not isinstance(quotes, list):
        return False
    for q in quotes:
        if isinstance(q, dict):
            cite = (q.get("citation") or "").strip()
            if _CITATION_RE.match(cite):
                return True
    return False


def make_citation_from_chunk(chunk: dict[str, Any]) -> str:
    """Format chunk as citation string file:start-end. Public for template builder."""
    book = chunk.get("book_id") or "?"
    s = chunk.get("start_line")
    e = chunk.get("end_line")
    try:
        return f"{book}:{int(s)}-{int(e)}"
    except (TypeError, ValueError):
        return ""


def apply_auto_cite_to_data(data: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    If data has no valid citations and chunks is non-empty, add a quote from chunks[0].
    Returns modified data (mutates and returns same dict).
    """
    if not chunks:
        return data
    if _has_valid_citation_in_data(data):
        return data
    if "quotes" not in data:
        data["quotes"] = []
    if not isinstance(data["quotes"], list):
        data["quotes"] = []
    cite = make_citation_from_chunk(chunks[0])
    if not cite:
        return data
    text = (chunks[0].get("text") or "")[:200]
    if len((chunks[0].get("text") or "")) > 200:
        text += "..."
    data["quotes"].insert(0, {
        "text": text,
        "citation": cite,
        "relation_to_question": "(auto-cited from highest-ranked passage)",
        "value_system": "",
    })
    return data


def apply_auto_cite(raw: str, chunks: list[dict[str, Any]]) -> str:
    """
    If raw response has no valid citation and chunks is non-empty, append citation from chunks[0].
    Use for plain-text responses. For JSON flow, use apply_auto_cite_to_data instead.
    """
    if not chunks:
        return raw
    if _has_valid_citation_in_text(raw):
        return raw
    cite = make_citation_from_chunk(chunks[0])
    if not cite:
        return raw
    suffix = f" [{cite}]"
    return (raw or "").rstrip() + suffix
