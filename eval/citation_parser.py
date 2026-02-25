"""
Extract citations from model response.
Supports:
- Strict format: [file:start-end] or file:start-end (e.g. Apology:1-50)
- Legacy: [Book, Speaker(s), lines 1-2] or [Book, lines 1-2]
Also extracts quoted text adjacent to citations.
"""

import json
import re
from dataclasses import dataclass
from typing import List


@dataclass
class StrictCitation:
    """Strict citation format: file:start-end."""
    file: str
    start_line: int
    end_line: int


# Strict format: [file:start-end] or file:start-end (file may include volume e.g. "Republic II")
STRICT_CITATION_BRACKETS_RE = re.compile(r'\[([^:\]]+):(\d+)-(\d+)\]')
STRICT_CITATION_PLAIN_RE = re.compile(r'(?<![\[\w])([^:]+):(\d+)-(\d+)(?![\]\d-])')


@dataclass
class ParsedCitation:
    """A parsed citation from model output."""
    book: str
    speaker: str
    start_line: int
    end_line: int
    raw: str
    quoted_text: str | None = None


# Matches [Book, Speaker(s), lines X-Y]
# Speaker can be "Socrates" or "Socrates, Glaucon" - last comma before "lines" separates speakers from "lines X-Y"
CITATION_RE = re.compile(
    r'\[([^,\]]+),\s*([^\[\]]+?),\s*lines\s+(\d+)\s*-\s*(\d+)\]',
    re.IGNORECASE
)

# Simpler fallback: [Book, lines X-Y] (no speaker)
CITATION_RE_FALLBACK = re.compile(
    r'\[([^,\]]+),\s*lines\s+(\d+)\s*-\s*(\d+)\]',
    re.IGNORECASE
)

# Quoted text: "something in quotes"
QUOTED_RE = re.compile(r'"([^"]+)"')


def parse_citations_strict(response: str) -> List[StrictCitation]:
    """
    Extract citations in strict format file:start-end.
    Works on: JSON with quotes[].citation, or raw text with [file:start-end] / file:start-end.
    """
    results: List[StrictCitation] = []
    seen: set[tuple[str, int, int]] = set()

    # Try JSON extraction first
    text = (response or "").strip()
    start = text.find('{')
    if start >= 0:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start : i + 1])
                        quotes = data.get("quotes") or []
                        for q in quotes:
                            if isinstance(q, dict):
                                cite = (q.get("citation") or "").strip()
                                m = re.match(r'^([^:]+):(\d+)-(\d+)$', cite, re.IGNORECASE)
                                if m:
                                    f, s, e = m.group(1).strip(), int(m.group(2)), int(m.group(3))
                                    key = (f.lower(), s, e)
                                    if key not in seen:
                                        seen.add(key)
                                        results.append(StrictCitation(file=f, start_line=s, end_line=e))
                        if results:
                            return results
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

    # Regex fallback: [file:start-end]
    for m in STRICT_CITATION_BRACKETS_RE.finditer(response):
        f, s, e = m.group(1).strip(), int(m.group(2)), int(m.group(3))
        key = (f.lower(), s, e)
        if key not in seen:
            seen.add(key)
            results.append(StrictCitation(file=f, start_line=s, end_line=e))

    # Regex fallback: file:start-end (plain)
    for m in STRICT_CITATION_PLAIN_RE.finditer(response):
        f, s, e = m.group(1).strip(), int(m.group(2)), int(m.group(3))
        key = (f.lower(), s, e)
        if key not in seen:
            seen.add(key)
            results.append(StrictCitation(file=f, start_line=s, end_line=e))

    return results


def strict_citations_from_data(data: dict) -> List[StrictCitation]:
    """Extract StrictCitations from parsed JSON data (quotes array)."""
    results: List[StrictCitation] = []
    seen: set[tuple[str, int, int]] = set()
    quotes = data.get("quotes") or []
    for q in quotes:
        if not isinstance(q, dict):
            continue
        cite = (q.get("citation") or "").strip()
        m = re.match(r'^([^:]+):(\d+)-(\d+)$', cite, re.IGNORECASE)
        if m:
            f, s, e = m.group(1).strip(), int(m.group(2)), int(m.group(3))
            key = (f.lower(), s, e)
            if key not in seen:
                seen.add(key)
                results.append(StrictCitation(file=f, start_line=s, end_line=e))
    return results


def strict_to_parsed(sc: StrictCitation) -> ParsedCitation:
    """Convert StrictCitation to ParsedCitation for legacy metrics."""
    return ParsedCitation(
        book=sc.file,
        speaker="?",
        start_line=sc.start_line,
        end_line=sc.end_line,
        raw=f"[{sc.file}:{sc.start_line}-{sc.end_line}]",
        quoted_text=None,
    )


def parse_citations(response: str) -> List[ParsedCitation]:
    """
    Extract all citations from model response.
    Returns list of ParsedCitation with book, speaker, start_line, end_line, and optionally quoted_text.
    """
    results: List[ParsedCitation] = []

    for m in CITATION_RE.finditer(response):
        book = m.group(1).strip()
        speaker_part = m.group(2).strip()
        start_line = int(m.group(3))
        end_line = int(m.group(4))
        raw = m.group(0)
        quoted = _extract_quoted_near(response, m.start(), m.end())
        results.append(ParsedCitation(
            book=book,
            speaker=speaker_part,
            start_line=start_line,
            end_line=end_line,
            raw=raw,
            quoted_text=quoted,
        ))

    for m in CITATION_RE_FALLBACK.finditer(response):
        if any(
            r.book.lower() == m.group(1).strip().lower()
            and r.start_line == int(m.group(2))
            and r.end_line == int(m.group(3))
            for r in results
        ):
            continue
        book = m.group(1).strip()
        start_line = int(m.group(2))
        end_line = int(m.group(3))
        raw = m.group(0)
        quoted = _extract_quoted_near(response, m.start(), m.end())
        results.append(ParsedCitation(
            book=book,
            speaker="?",
            start_line=start_line,
            end_line=end_line,
            raw=raw,
            quoted_text=quoted,
        ))

    return results


def _extract_quoted_near(text: str, start: int, end: int, window: int = 200) -> str | None:
    """Extract first quoted string within window chars before or after citation."""
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    snippet = text[lo:hi]
    for m in QUOTED_RE.finditer(snippet):
        q = m.group(1).strip()
        if len(q) >= 10:
            return q
    return None


def normalize_text_for_match(s: str) -> str:
    """Normalize text for substring/quote matching: lowercase, collapse whitespace, strip punctuation."""
    s = (s or "").lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return s.strip()
