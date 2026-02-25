"""
Parse model output (JSON or plain-text) and render to bullet format.
Batch value pipeline: model outputs a JSON array of value strings in passage order; Python merges by index into template (chunks).
"""

import json
import re
from typing import Any

from src.conversation import build_quote_template

# citation format: file:start-end (e.g. Apology:1-50)
CITATION_FORMAT_RE = re.compile(r'^([^:]+):(\d+)-(\d+)$', re.IGNORECASE)

# Strip "Passage N" or "Passage N:" prefix from model value and capitalize first letter
PASSAGE_PREFIX_RE = re.compile(r'^\s*Passage\s+\d+\s*:?\s*', re.IGNORECASE)


def _normalize_value_text(s: str) -> str:
    """Remove leading 'Passage N' / 'Passage N:' and capitalize first character."""
    if not s or not s.strip():
        return s.strip()
    t = PASSAGE_PREFIX_RE.sub("", s).strip()
    if not t:
        return t
    return t[0].upper() + t[1:] if len(t) > 1 else t.upper()


def parse_batch_values_json(raw: str, expected_count: int) -> list[str] | None:
    """
    Parse batch model output as a JSON array of value strings (one per passage, in order).
    Strips markdown code fences if present; on failure returns None.
    Returns list of value strings of length expected_count (padded or trimmed), or None.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    start = text.find("[")
    if start == -1:
        return None
    end = text.rfind("]")
    if end == -1 or end < start:
        return None
    segment = text[start : end + 1]
    try:
        data = json.loads(segment)
    except (json.JSONDecodeError, TypeError):
        # Salvage truncated JSON: try closing the last string and the array
        for suffix in ['"]', '"]]']:
            try:
                data = json.loads(segment + suffix)
                if isinstance(data, list):
                    break
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            return None
    if not isinstance(data, list):
        return None
    result: list[str] = []
    for item in data:
        if item is None:
            result.append("")
        elif isinstance(item, str):
            result.append(_normalize_value_text(item.strip()))
        else:
            result.append(_normalize_value_text(str(item).strip()))
    while len(result) < expected_count:
        result.append("")
    return result[:expected_count]

def merge_model_into_template(
    template: dict,
    parsed_quotes: list[dict[str, str]] | None,
) -> dict:
    """
    Merge parsed relation_to_question and value_system into template.
    Template has text/citation from chunks; parsed_quotes has model's analysis.
    Match by index. Preserves other template keys (e.g. refusal).
    """
    if not parsed_quotes:
        return template
    template_quotes = template.get("quotes") or []
    merged = []
    for i, tq in enumerate(template_quotes):
        mq = parsed_quotes[i] if i < len(parsed_quotes) and isinstance(parsed_quotes[i], dict) else None
        merged_quote = dict(tq)
        if mq:
            val = (mq.get("value_system") or "").strip()
            if val:
                merged_quote["value_system"] = val
                merged_quote["value"] = val  # template uses "value"; keep in sync
        merged.append(merged_quote)
    return {**template, "quotes": merged}


def normalize_citation(citation: str) -> str:
    """Normalize citation to [file:start-end] for display."""
    c = (citation or "").strip()
    if not c:
        return ""
    # If already [file:start-end], keep as-is; else wrap
    if c.startswith('[') and c.endswith(']'):
        return c
    if CITATION_FORMAT_RE.match(c):
        return f"[{c}]"
    return c


def render_quotes_to_bullets(data: dict[str, Any]) -> str:
    """
    Render parsed JSON to bullet format (value only):
    • Quote 1 [Apology:1-50]: "text..."
      - <value>
    If refusal is present and quotes is empty, return refusal text.
    """
    refusal = (data.get("refusal") or "").strip()
    quotes = data.get("quotes")
    if not isinstance(quotes, list):
        return refusal if refusal else ""
    if not quotes:
        return refusal if refusal else ""

    lines: list[str] = []
    for i, q in enumerate(quotes, 1):
        if not isinstance(q, dict):
            continue
        text = (q.get("text") or "").strip()
        citation = (q.get("citation") or "").strip()
        value_sys = (q.get("value_system") or q.get("value") or "").strip()

        cite_display = normalize_citation(citation)
        quote_line = f"• Quote {i}"
        if cite_display:
            quote_line += f" {cite_display}"
        if text:
            quote_line += f': "{text}"'
        lines.append(quote_line)

        if value_sys:
            lines.append(f"  - {value_sys}")
        else:
            lines.append("  - (分析失敗，請查看伺服器日誌以確認原因：LLM 錯誤或 JSON 解析失敗)")

        if i < len(quotes):
            lines.append("")  # blank between quotes

    return "\n".join(lines).rstrip()


def process_response(raw: str, chunks: list | None = None) -> str:
    """
    Parse raw model plain-text response, merge relation/value into template, render to bullets.
    Template has text/citation from chunks; model output is parsed for Relation/Value per quote.
    """
    merged = process_response_to_data(raw, chunks)
    if not merged:
        return raw
    rendered = render_quotes_to_bullets(merged)
    return rendered if rendered else raw


def process_response_to_data(raw: str, chunks: list | None = None) -> dict | None:
    """
    Parse raw model plain-text response and merge into template.
    Returns merged dict with "quotes" (each with text, citation, relation_to_question, value_system).
    Used by eval pipeline for citation extraction.
    """
    template = build_quote_template(chunks) if chunks else {"quotes": []}
    quotes = template.get("quotes") or []
    if not quotes:
        return template
    expected_count = len(quotes)
    parsed_values = parse_batch_values_json(raw, expected_count)
    # parse_batch_values_json returns list[str]; merge_model_into_template expects list[dict]
    parsed_quotes = (
        [{"value_system": v, "value": v} for v in parsed_values]
        if parsed_values is not None
        else None
    )
    merged = merge_model_into_template(template, parsed_quotes)
    return merged
