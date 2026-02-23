# Mechanical Citation Verification — Line-by-Line Code Explanation

This document explains every line of code related to **Mechanical Citation Verification**: extracting citations via regex, validating that cited line ranges overlap retrieved chunks using interval arithmetic, and flagging hallucinations.

---

## Overview of the Method

1. **Step 1** — Enforce citation format: `[file:start-end]`
2. **Step 2** — Extract citations via regex (file, start_line, end_line)
3. **Step 3** — Interval overlap check: cited range must overlap at least one retrieved chunk
4. **Step 4** — Flag hallucination when citation does not overlap

---

## 1. `src/citation_utils.py` — Citation Format & Auto-Cite

### Lines 1–9

```python
"""
Citation utilities: auto-cite fallback when model fails to produce valid citations.
"""

import re
from typing import Any

# Reuse strict citation logic - avoid circular import by using local regex
_CITATION_RE = re.compile(r'^([^:]+):(\d+)-(\d+)$', re.IGNORECASE)
```

| Line | Explanation |
|------|-------------|
| 1–3 | Module docstring: this file handles fallback when the model outputs no valid citation. |
| 5 | Import `re` for regex. |
| 6 | Import `Any` for type hints. |
| 9 | Regex to match `file:start-end` from start to end of string: `([^:]+)` = file (non-colon chars), `(\d+)-(\d+)` = start and end line numbers. `re.IGNORECASE` for case-insensitive book names. |

### Lines 12–21

```python
def _has_valid_citation_in_text(text: str) -> bool:
    """Check if text contains at least one citation in file:start-end format."""
    # [file:start-end]
    if re.search(r'\[[^:\]]+:\d+-\d+\]', text):
        return True
    # file:start-end (in JSON or plain)
    if _CITATION_RE.search(text):
        return True
    return False
```

| Line | Explanation |
|------|-------------|
| 12 | Declare function to check if text has at least one valid citation. |
| 14–16 | Regex `\[[^:\]]+:\d+-\d+\]` matches `[file:start-end]` anywhere. `[^:\]]+` = chars inside brackets. If found → True. |
| 17–19 | `_CITATION_RE` matches `file:start-end` anywhere. If found → True. |
| 20 | Otherwise, no valid citation → False. |

### Lines 24–34

```python
def _has_valid_citation_in_data(data: dict[str, Any]) -> bool:
    """Check if parsed JSON has at least one quote with valid citation format."""
    quotes = data.get("quotes") or []
    if not isinstance(quotes, list):
        return False
    for q in quotes:
        if isinstance(q, dict):
            cite = (q.get("citation") or "").strip()
            if _CITATION_RE.match(cite):
                return True
    return False
```

| Line | Explanation |
|------|-------------|
| 24 | Check if parsed JSON contains at least one valid citation. |
| 25 | Get `quotes` list from JSON; default to empty. |
| 26–27 | If `quotes` is not a list, return False. |
| 28–32 | For each quote, get `citation` and match against `file:start-end`. If any match → True. |
| 33 | No valid citation found → False. |

### Lines 36–44

```python
def make_citation_from_chunk(chunk: dict[str, Any]) -> str:
    """Format chunk as citation string file:start-end. Public for template builder."""
    book = chunk.get("book_id") or "?"
    s = chunk.get("start_line")
    e = chunk.get("end_line")
    try:
        return f"{book}:{int(s)}-{int(e)}"
    except (TypeError, ValueError):
        return ""
```

| Line | Explanation |
|------|-------------|
| 36 | Create citation string from a chunk dict. |
| 37 | Docstring: enforces `file:start-end` format for the template. |
| 38 | Use `book_id` or `"?"`. |
| 39–40 | Read `start_line` and `end_line`. |
| 41–42 | Format as `book:start-end`; int conversion validates numeric. |
| 43–44 | On invalid types, return empty string. |

---

## 2. `eval/citation_parser.py` — Regex Extraction

### Lines 14–24

```python
@dataclass
class StrictCitation:
    """Strict citation format: file:start-end."""
    file: str
    start_line: int
    end_line: int


# Strict format: [file:start-end] or file:start-end
STRICT_CITATION_BRACKETS_RE = re.compile(r'\[([^:\]]+):(\d+)-(\d+)\]')
STRICT_CITATION_PLAIN_RE = re.compile(r'(?<![\[\w])([A-Za-z]+):(\d+)-(\d+)(?![\]\d-])')
```

| Line | Explanation |
|------|-------------|
| 14–15 | `@dataclass` creates `StrictCitation` with fields. |
| 16 | Docstring: strict format is `file:start-end`. |
| 17–19 | Fields: file name, start line, end line. |
| 23 | `\[([^:\]]+):(\d+)-(\d+)\]` matches `[file:start-end]`. Group 1 = file, 2 = start, 3 = end. |
| 24 | `(?<![\[\w])` = no `[` or word char before; `([A-Za-z]+):(\d+)-(\d+)` = file and range; `(?![\]\d-])` = no `]`, digit, or `-` after. Matches plain `file:start-end` outside brackets. |

### Lines 55–110 — `parse_citations_strict()`

```python
def parse_citations_strict(response: str) -> List[StrictCitation]:
    """..."""
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
```

| Line | Explanation |
|------|-------------|
| 55–56 | Function signature: string → list of `StrictCitation`. |
| 60–61 | Output list and `seen` to deduplicate (file, start, end). |
| 64–65 | Normalize text and find first `{`. |
| 66 | If JSON-like content exists… |
| 67–74 | Bracket-matching loop: track depth; when depth returns to 0, we have a complete JSON object. |
| 75 | Parse substring as JSON. |
| 76 | Get `quotes` or empty list. |
| 77–86 | For each dict quote, get `citation`, match `^([^:]+):(\d+)-(\d+)$`, extract file, start, end; deduplicate and append. |
| 87–88 | If any citation found, return immediately. |
| 89–91 | On parse error, ignore and exit loop. |
| 92 | Exit after first balanced JSON block. |
| 94–99 | If no JSON: use `STRICT_CITATION_BRACKETS_RE` to find `[file:start-end]`, extract (file, start, end), deduplicate, append. |
| 102–108 | Fallback: use `STRICT_CITATION_PLAIN_RE` for plain `file:start-end`, same logic. |
| 109 | Return all collected citations. |

### Lines 112–128 — `strict_citations_from_data()`

```python
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
```

| Line | Explanation |
|------|-------------|
| 112 | Extract `StrictCitation` list from already-parsed JSON. |
| 114–115 | Initialize results and seen set. |
| 116 | Get `quotes` list. |
| 117–127 | For each dict quote, parse citation with `^([^:]+):(\d+)-(\d+)$`, deduplicate, append. |
| 128 | Return list. |

---

## 3. `eval/metrics.py` — Interval Overlap & Hallucination Flagging

### Lines 13–16 — Interval arithmetic

```python
def intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Two intervals [a_start, a_end] and [b_start, b_end] overlap iff a_start <= b_end and b_start <= a_end."""
    return a_start <= b_end and b_start <= a_end
```

| Line | Explanation |
|------|-------------|
| 13 | Function takes two intervals as (start, end). |
| 14 | Docstring: standard interval overlap rule. |
| 15 | Intervals overlap when neither is entirely to the left of the other: `a_start <= b_end` and `b_start <= a_end`. |

### Lines 18–32 — Citation vs. retrieved chunks

```python
def citation_overlaps_retrieved(citation: StrictCitation, retrieved_chunks: list[dict[str, Any]]) -> bool:
    """Check if citation range overlaps at least one retrieved chunk (same book, interval overlap)."""
    file_lower = citation.file.lower().strip()
    for c in retrieved_chunks:
        book = (c.get("book_id") or "").lower().strip()
        if book != file_lower:
            continue
        try:
            s = int(c.get("start_line", 0))
            e = int(c.get("end_line", 0))
        except (TypeError, ValueError):
            continue
        if intervals_overlap(citation.start_line, citation.end_line, s, e):
            return True
    return False
```

| Line | Explanation |
|------|-------------|
| 18 | Function: does the citation overlap any retrieved chunk? |
| 19 | Docstring: same book and line-range overlap required. |
| 20 | Normalize citation file name for comparison. |
| 21 | Loop over each retrieved chunk. |
| 22 | Get chunk book. |
| 23–24 | Skip chunks from other books. |
| 25–28 | Read chunk start/end; skip on invalid values. |
| 29 | Use `intervals_overlap` for cited vs. chunk line ranges. |
| 30 | If any overlap → citation is grounded. |
| 31 | No overlap found → citation not grounded. |

### Lines 35–52 — Hallucination verification

```python
def verify_citations_against_retrieved(
    parsed_citations: list[StrictCitation],
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[list[StrictCitation], list[StrictCitation], float]:
    """
    Verify each citation overlaps at least one retrieved chunk.
    Returns (passed_list, hallucinated_list, hallucination_rate).
    """
    passed: list[StrictCitation] = []
    hallucinated: list[StrictCitation] = []
    for pc in parsed_citations:
        if citation_overlaps_retrieved(pc, retrieved_chunks):
            passed.append(pc)
        else:
            hallucinated.append(pc)
    total = len(parsed_citations)
    rate = len(hallucinated) / total if total else 0.0
    return passed, hallucinated, rate
```

| Line | Explanation |
|------|-------------|
| 35–38 | Function: verify all citations against retrieved chunks. |
| 39–41 | Docstring: each citation must overlap some chunk; returns passed, hallucinated, rate. |
| 42–43 | Lists for grounded and hallucinated citations. |
| 44–48 | For each citation: if it overlaps at least one chunk → passed; else → hallucinated. |
| 49 | Total number of citations. |
| 50 | Hallucination rate = hallucinated / total; 0 when total is 0. |
| 51 | Return passed list, hallucinated list, and rate. |

---

## 4. `eval/run_eval.py` — Integration into Evaluation

### Lines 23–28 — Imports

```python
from eval.metrics import (
    compute_citation_validity,
    compute_retrieval_diversity,
    compute_citation_diversity,
    compute_similarity,
    verify_citations_against_retrieved,
)
```

| Line | Explanation |
|------|-------------|
| 23–28 | Import `verify_citations_against_retrieved` for Mechanical Citation Verification. |

### Lines 38–54 — Resolving retrieved chunks

```python
def _chunks_from_keys(
    chunk_index: ChunkIndex,
    keys: list,
) -> list[dict]:
    """Resolve retrieved_chunk_keys to full chunk dicts (with text)."""
    chunks: list[dict] = []
    for k in keys:
        if isinstance(k, (list, tuple)) and len(k) >= 3:
            book, start, end = str(k[0]), int(k[1]), int(k[2])
        elif isinstance(k, dict):
            book = str(k.get("file", k.get("book_id", "")))
            start = int(k.get("start_line", 0))
            end = int(k.get("end_line", 0))
        else:
            continue
        c = chunk_index.get(book, start, end)
        if c:
            chunks.append(c)
    return chunks
```

| Line | Explanation |
|------|-------------|
| 38–40 | Convert chunk keys into full chunk dicts. |
| 43 | Result list. |
| 44–51 | Parse key as `(book, start, end)` or dict; otherwise skip. |
| 52–54 | Look up chunk by (book, start, end); append if found. |
| 55 | Return retrieved chunks for overlap check. |

### Lines 98–99 — Load path: retrieve chunks

```python
            keys = r.get("retrieved_chunk_keys", [])
            chunks = _chunks_from_keys(chunk_index, keys)
```

| Line | Explanation |
|------|-------------|
| 98 | Get keys of chunks that were retrieved for this question. |
| 99 | Resolve to full chunk dicts for verification. |

### Lines 105–115 — Rebuild parsed citations from stored results

```python
            # Rebuild strict_parsed from parsed_citations
            pcs = r.get("parsed_citations", [])
            strict_parsed = [
                StrictCitation(
                    file=pc.get("file", ""),
                    start_line=int(pc.get("start_line", 0)),
                    end_line=int(pc.get("end_line", 0)),
                )
                for pc in pcs
                if pc.get("file") and pc.get("start_line") is not None and pc.get("end_line") is not None
            ]
```

| Line | Explanation |
|------|-------------|
| 105–106 | Read stored `parsed_citations`. |
| 107–115 | Build `StrictCitation` list from stored (file, start_line, end_line). |

### Lines 128–135 — Hallucination check (load path)

```python
            validity = compute_citation_validity(parsed, chunk_index)
            _, _, halluc_rate = verify_citations_against_retrieved(strict_parsed, chunks)

            result_item = dict(r)
            result_item["A1_existence_rate"] = validity["A1_existence_rate"]
            result_item["A2_quote_match_rate"] = validity["A2_quote_match_rate"]
            result_item["A3_fabrication_rate"] = validity["A3_fabrication_rate"]
            result_item["A4_hallucination_rate"] = halluc_rate
```

| Line | Explanation |
|------|-------------|
| 128 | Other citation validity metrics. |
| 129 | Call `verify_citations_against_retrieved(strict_parsed, chunks)`. Only `halluc_rate` is used. |
| 132 | Store `A4_hallucination_rate` = fraction of citations not overlapping retrieved chunks. |

### Lines 157–204 — Full pipeline: generation and verification

```python
            chunks = retriever.search(q, top_k=top_k)
            all_retrieved.append(chunks)

            data, gen_errors = generate_per_citation(q, chunks)
            ...
            strict_parsed = (
                strict_citations_from_data(data) if data else parse_citations_strict(response)
            )
            ...
            _, _, halluc_rate = verify_citations_against_retrieved(strict_parsed, chunks)
            ...
            result_item = {
                ...
                "A4_hallucination_rate": halluc_rate,
            }
```

| Line | Explanation |
|------|-------------|
| 157–158 | Retrieve top-k chunks and store them. |
| 162–163 | Parse citations from model output (JSON or plain text). |
| 185 | Run Mechanical Citation Verification. |
| 202 | Attach `A4_hallucination_rate` to the result. |

### Lines 231–232, 240, 291 — Aggregation and reporting

```python
    a4_mean = sum(all_hallucination_rates) / len(all_hallucination_rates) if all_hallucination_rates else 0.0
```

| Line | Explanation |
|------|-------------|
| 231 | Mean hallucination rate across all questions. |

```python
        "A4_hallucination_rate": round(a4_mean, 4),
```

| Line | Explanation |
|------|-------------|
| 240 | Add mean A4 rate to summary. |

```python
        f"- **A4 Hallucination Rate** (citation range does not overlap retrieved chunks): {summary['A4_hallucination_rate']:.2%}",
```

| Line | Explanation |
|------|-------------|
| 291 | Human-readable report: "citation range does not overlap retrieved chunks". |

---

## Summary Flow

1. **Format** — `make_citation_from_chunk` enforces `file:start-end` when building prompts.
2. **Extract** — `parse_citations_strict` and `strict_citations_from_data` use regex (and JSON) to get file, start_line, end_line.
3. **Overlap** — `intervals_overlap` implements interval arithmetic; `citation_overlaps_retrieved` checks each citation against retrieved chunks.
4. **Flag** — `verify_citations_against_retrieved` splits citations into passed/hallucinated and computes `A4_hallucination_rate`.
