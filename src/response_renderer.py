"""
Parse model plain-text output and render to bullet format.
Model outputs Relation/Value per quote; Python assembles into template (chunks).
"""

import re
from typing import Any

from src.conversation import build_quote_template

# citation format: file:start-end (e.g. Apology:1-50)
CITATION_FORMAT_RE = re.compile(r'^([^:]+):(\d+)-(\d+)$', re.IGNORECASE)

# Plain-text format: "Quote N:" then "Relation:" and "Value:"
_QUOTE_BLOCK_RE = re.compile(r'Quote\s+(\d+)\s*:\s*(.*?)(?=Quote\s+\d+\s*:|\Z)', re.IGNORECASE | re.DOTALL)
_RELATION_LINE_RE = re.compile(r'Relation\s*:\s*(.+?)(?=Value\s*:|$)', re.IGNORECASE | re.DOTALL)
_VALUE_LINE_RE = re.compile(r'Value\s*:\s*(.+?)(?=Relation\s*:|Quote\s+\d+\s*:|$)', re.IGNORECASE | re.DOTALL)


def parse_plain_text_response(raw: str) -> list[dict[str, str]]:
    """
    Parse model plain-text output. Extract relation_to_question and value_system
    for each "Quote N:" block. Returns list of {relation_to_question, value_system} by index.
    """
    if not raw or not raw.strip():
        return []
    text = raw.strip()

    result: list[dict[str, str]] = []
    for m in _QUOTE_BLOCK_RE.finditer(text):
        block = m.group(2).strip()
        rel_m = _RELATION_LINE_RE.search(block)
        val_m = _VALUE_LINE_RE.search(block)
        rel = (rel_m.group(1).strip() if rel_m else "").strip()
        val = (val_m.group(1).strip() if val_m else "").strip()
        result.append({
            "relation_to_question": rel,
            "value_system": val,
        })
    return result


# Fallback: numbered format "1.\nRelation:" or "1)\nRelation:"
_NUMERIC_BLOCK_RE = re.compile(r'(?:^|\n)\s*(\d+)[\.\)]\s*(.*?)(?=(?:^|\n)\s*\d+[\.\)]\s*|\Z)', re.DOTALL)


def _parse_plain_text_fallback(raw: str) -> list[dict[str, str]]:
    """Fallback when Quote N: format fails - try "1. Relation: ... Value: ..." blocks."""
    if not raw or not raw.strip():
        return []
    result = []
    for m in _NUMERIC_BLOCK_RE.finditer(raw):
        block = m.group(2).strip()
        rel_m = _RELATION_LINE_RE.search(block)
        val_m = _VALUE_LINE_RE.search(block)
        rel = (rel_m.group(1).strip() if rel_m else "").strip()
        val = (val_m.group(1).strip() if val_m else "").strip()
        result.append({
            "relation_to_question": rel,
            "value_system": val,
        })
    return result


def _parse_model_output(raw: str) -> list[dict[str, str]]:
    """
    Parse model plain-text output. Tries "Quote N:" format first, then "1." fallback.
    Returns list of {relation_to_question, value_system} by quote index.
    """
    parsed = parse_plain_text_response(raw)
    if not parsed:
        parsed = _parse_plain_text_fallback(raw)
    return parsed


def merge_model_into_template(
    template: dict,
    parsed_quotes: list[dict[str, str]] | None,
) -> dict:
    """
    Merge parsed relation_to_question and value_system into template.
    Template has text/citation from chunks; parsed_quotes has model's analysis.
    Match by index.
    """
    if not parsed_quotes:
        return template
    template_quotes = template.get("quotes") or []
    merged = []
    for i, tq in enumerate(template_quotes):
        mq = parsed_quotes[i] if i < len(parsed_quotes) and isinstance(parsed_quotes[i], dict) else None
        merged_quote = dict(tq)
        if mq:
            rel = (mq.get("relation_to_question") or "").strip()
            val = (mq.get("value_system") or "").strip()
            if rel:
                merged_quote["relation_to_question"] = rel
            if val:
                merged_quote["value_system"] = val
        merged.append(merged_quote)
    return {"quotes": merged}


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
    Render parsed JSON to bullet format:
    • Quote 1 [Apology:1-50]: "text..."
      - 與提問的關聯：...
      - 反映的價值觀：...
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
        relation = (q.get("relation_to_question") or "").strip()
        value_sys = (q.get("value_system") or "").strip()

        cite_display = normalize_citation(citation)
        quote_line = f"• Quote {i}"
        if cite_display:
            quote_line += f" {cite_display}"
        if text:
            quote_line += f': "{text}"'
        lines.append(quote_line)

        if relation:
            lines.append(f"  - 與提問的關聯：{relation}")
        if value_sys:
            lines.append(f"  - 反映的價值觀：{value_sys}")

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
    if not template.get("quotes"):
        return template
    parsed_quotes = _parse_model_output(raw)
    merged = merge_model_into_template(template, parsed_quotes)
    return merged
