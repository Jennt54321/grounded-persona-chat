# Source Module Architecture (`src/`)

This document explains how the Python modules in the `src/` folder interact with each other and provides line-by-line explanations of their functions.

---

## Overview: Module Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              app/main.py (API layer)                             │
│   Uses: Retriever, generate/generate_stream, process_response                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                    │                    │                    │
                    ▼                    ▼                    ▼
┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
│     retriever.py      │  │   conversation.py     │  │  response_renderer.py │
│  - load_chunks()      │  │ - build_quote_template│  │ - parse_plain_text    │
│  - Retriever.search() │  │ - build_messages()    │  │ - merge_model_into_   │
│  - retrieve()         │  │ - generate()          │  │   template()          │
└───────────────────────┘  │ - generate_stream()   │  │ - render_quotes_to_   │
          │                └───────────────────────┘  │   bullets()           │
          │                           │               │ - process_response()  │
          │                           │               └───────────────────────┘
          │                           │                              │
          │                           ▼                              │
          │                ┌───────────────────────┐                 │
          │                │   citation_utils.py   │                 │
          │                │ - make_citation_      │◄────────────────┘
          └───────────────►│   from_chunk()        │   (build_quote_template
                           │ - apply_auto_cite_    │    uses citation_utils)
                           │   to_data()           │
                           └───────────────────────┘
```

**Data flow for a typical chat request:**
1. **retriever.py** → Takes user query, returns top-k relevant chunks from Plato's dialogues
2. **conversation.py** → Builds prompt with chunks + citations, sends to LLM (Ollama/Phi-3.5), gets Relation/Value per quote
3. **citation_utils.py** → Used by conversation.py to format citations (file:start-end)
4. **response_renderer.py** → Parses model output, merges into template from chunks, renders to bullet format

---

## 1. `citation_utils.py`

Low-level utilities for citation formatting. Used by **conversation.py** and **response_renderer.py** (indirectly via conversation's `build_quote_template`). Also used by **eval/run_eval.py** for auto-cite fallback.

### Line-by-line explanation

```python
# Lines 1-6: Imports
import re
from typing import Any

_CITATION_RE = re.compile(r'^([^:]+):(\d+)-(\d+)$', re.IGNORECASE)
```
- `re`: Regular expression module for parsing citation strings.
- `typing.Any`: Type hint for flexible dict values.
- `_CITATION_RE`: Regex that matches citations like `Apology:1-50` or `Republic:100-200`. Captures: (1) book name, (2) start line, (3) end line.

---

#### `_has_valid_citation_in_text(text: str) -> bool`

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
- **Line 16**: Check if text contains bracketed citation `[Apology:1-50]`.
- **Line 19**: Check if text contains unbracketed citation `Apology:1-50`.
- **Line 21**: Return `False` if no valid citation found.

---

#### `_has_valid_citation_in_data(data: dict) -> bool`

```python
def _has_valid_citation_in_data(data: dict[str, Any]) -> bool:
    quotes = data.get("quotes")
    if not isinstance(quotes, list):
        return False
    for q in quotes:
        if isinstance(q, dict):
            cite = (q.get("citation") or "").strip()
            if _CITATION_RE.match(cite):
                return True
    return False
```
- **Line 26-28**: Ensure `data["quotes"]` exists and is a list.
- **Line 29-33**: Loop over each quote; if any quote has a `citation` matching `file:start-end`, return `True`.
- **Line 34**: Return `False` if none match.

---

#### `make_citation_from_chunk(chunk: dict) -> str`

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
- **Line 38-39**: Get `book_id`, default `"?"` if missing.
- **Line 40-41**: Get `start_line` and `end_line`.
- **Line 42-45**: Format as `Apology:1-50`, or `""` if conversion fails.

---

#### `apply_auto_cite_to_data(data, chunks) -> dict`

```python
def apply_auto_cite_to_data(data: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
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
```
- **Line 52-56**: If no chunks or data already has valid citations, return unchanged.
- **Line 57-60**: Ensure `data["quotes"]` is a list.
- **Line 61-63**: Create citation from first (top-ranked) chunk; bail if it fails.
- **Line 64-67**: Take first 200 chars of chunk text, add "..." if truncated.
- **Line 68-74**: Insert a new quote at the front with that text, citation, and a placeholder relation.
- **Line 75**: Return modified `data`.

---

#### `apply_auto_cite(raw, chunks) -> str`

```python
def apply_auto_cite(raw: str, chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return raw
    if _has_valid_citation_in_text(raw):
        return raw
    cite = make_citation_from_chunk(chunks[0])
    if not cite:
        return raw
    suffix = f" [{cite}]"
    return (raw or "").rstrip() + suffix
```
- **Line 80-82**: If no chunks, return raw response.
- **Line 83-84**: If raw text already has a valid citation, return it.
- **Line 85-87**: Build citation from first chunk; otherwise return raw.
- **Line 88-89**: Append ` [Apology:1-50]` to the end of raw and return.

---

## 2. `retriever.py`

Semantic search over Plato chunk embeddings. Loads chunk JSONs, encodes queries with BGE, returns top-k chunks by similarity. Used by **app/main.py** and **eval/run_eval.py**.

### Line-by-line explanation

```python
# Lines 1-25: Imports and constants
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOKS_DIR = PROJECT_ROOT / "books"
CHUNK_FILES = [
    "apology_chunks.json",
    "meno_chunks.json",
    "gorgias_chunks.json",
    "republic_chunks.json",
]
MODEL_ID = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 8

MIN_CHUNK_TEXT_LEN = 50
```
- **Line 14**: `PROJECT_ROOT` = directory above `src/`.
- **Line 15**: Path to `books/` containing chunk JSONs.
- **Line 16-21**: Chunk file names for each dialogue.
- **Line 22**: BGE embedding model from Hugging Face.
- **Line 23**: BGE-specific query prefix for better retrieval.
- **Line 24**: Number of chunks to return.
- **Line 26**: Minimum character length; shorter chunks are skipped.

---

#### `load_chunks(books_dir: Path) -> List[Dict[str, Any]]`

```python
def load_chunks(books_dir: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for name in CHUNK_FILES:
        path = books_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for c in data:
            emb = c.get("embedding")
            text = (c.get("text") or "").strip()
            if emb is not None and len(text) >= MIN_CHUNK_TEXT_LEN:
                c_copy = dict(c)
                c_copy["embedding"] = np.array(emb, dtype=np.float32)
                chunks.append(c_copy)
    return chunks
```
- **Line 31**: Initialize empty list.
- **Line 32-36**: Loop over each chunk file; skip if missing.
- **Line 35**: Load JSON.
- **Line 36-38**: For each chunk, get embedding and text.
- **Line 39**: Require embedding and sufficient text length.
- **Line 40-42**: Copy chunk, convert embedding to `np.float32`, append.
- **Line 44**: Return all loaded chunks.

---

#### `load_model()`

```python
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID)
```
- Lazy import to avoid loading BGE at module load.
- Returns the BGE model instance.

---

#### `Retriever` class

```python
class Retriever:
    def __init__(self, books_dir: Path = BOOKS_DIR):
        self.books_dir = books_dir
        self._chunks: List[Dict[str, Any]] | None = None
        self._embeddings: np.ndarray | None = None
        self._model = None
```
- **Line 54**: Store path to books directory.
- **Line 55-57**: Lazy-loaded: chunks, stacked embedding matrix, model.

---

#### `_ensure_loaded()`

```python
def _ensure_loaded(self):
    if self._chunks is None:
        self._chunks = load_chunks(self.books_dir)
        if not self._chunks:
            self._embeddings = np.array([]).reshape(0, 768)
        else:
            embs = [c["embedding"] for c in self._chunks]
            self._embeddings = np.stack(embs)
    if self._model is None:
        self._model = load_model()
```
- **Line 60-66**: If chunks not loaded: load them, and either create empty (768-dim) array or stack embeddings.
- **Line 67-68**: If model not loaded, load BGE.

---

#### `search(query, books_dir=None) -> List[Dict[str, Any]]`

```python
def search(self, query: str, books_dir: Path | None = None) -> List[Dict[str, Any]]:
    self._ensure_loaded()
    prefixed = BGE_QUERY_PREFIX + query
    q_emb = self._model.encode(
        [prefixed],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    q_vec = np.asarray(q_emb, dtype=np.float32)
    scores = np.dot(self._embeddings, q_vec)
    top_indices = np.argsort(scores)[::-1][:TOP_K]
    results = []
    for i in top_indices:
        c = dict(self._chunks[i])
        text = (c.get("text") or "").strip()
        if len(text) < MIN_CHUNK_TEXT_LEN:
            continue
        c["embedding"] = c["embedding"].tolist()
        c["score"] = float(scores[i])
        results.append(c)
    return results
```
- **Line 76**: Ensure chunks and model are loaded.
- **Line 77**: Add BGE query prefix.
- **Line 78-82**: Encode query and take first (only) vector.
- **Line 83**: Convert to float32 numpy array.
- **Line 84**: Dot product with all chunk embeddings for similarity scores.
- **Line 85**: Indices of top `TOP_K` scores (descending).
- **Line 86-94**: For each top index: copy chunk, skip short text, convert embedding to list, add `score`, append.
- **Line 95**: Return ranked list of chunks with scores.

---

#### `retrieve(query, books_dir=None) -> List[Dict[str, Any]]`

```python
def retrieve(query: str, books_dir: Path | None = None) -> List[Dict[str, Any]]:
    r = Retriever(books_dir=books_dir or BOOKS_DIR)
    return r.search(query)
```
- Thin wrapper: create `Retriever`, call `search`, return results.

---

## 3. `conversation.py`

Builds prompts with retrieved chunks and calls the LLM (Ollama/Phi-3.5). The model outputs Relation and Value per quote; this module provides the template and API. Uses **citation_utils.make_citation_from_chunk**.

### Line-by-line explanation

```python
# Imports and constants
from typing import List, Dict, Any, Iterator
from src.citation_utils import make_citation_from_chunk

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_ID = "phi3.5"
MAX_CHUNK_TEXT_LEN = 400

SYSTEM_PROMPT = """..."""
```
- **Line 8**: Import citation formatter for chunks.
- **Line 10-11**: Ollama URL and model name.
- **Line 14**: Truncate chunks to 400 chars in prompts.
- **Line 16**: System prompt instructing the model to output Relation/Value per passage.

---

#### `build_quote_template(chunks) -> dict`

```python
def build_quote_template(chunks: List[Dict[str, Any]]) -> dict:
    quotes = []
    for c in chunks:
        text = (c.get("text") or "").strip()
        if len(text) < 20:
            continue
        if len(text) > MAX_CHUNK_TEXT_LEN:
            text = text[:MAX_CHUNK_TEXT_LEN] + "..."
        citation = make_citation_from_chunk(c)
        quotes.append({
            "text": text,
            "citation": citation,
            "relation_to_question": "",
            "value_system": "",
        })
    return {"quotes": quotes}
```
- **Line 35-36**: Build list of quote dicts.
- **Line 37-39**: Get text, skip very short chunks.
- **Line 40-41**: Truncate long text.
- **Line 42**: Use `citation_utils.make_citation_from_chunk` for citation string.
- **Line 43-48**: Append dict with text, citation, and empty relation/value for the model to fill.
- **Line 49**: Return `{"quotes": [...]}`.

---

#### `_format_passages_for_prompt(chunks) -> tuple[str, int]`

```python
def _format_passages_for_prompt(chunks: List[Dict[str, Any]]) -> tuple[str, int]:
    lines = []
    for i, c in enumerate(chunks, 1):
        text = (c.get("text") or "").strip()
        if len(text) < 20:
            continue
        if len(text) > MAX_CHUNK_TEXT_LEN:
            text = text[:MAX_CHUNK_TEXT_LEN] + "..."
        citation = make_citation_from_chunk(c)
        lines.append(f"Passage {i} [{citation}]:\n{text}")
    return "\n\n".join(lines), len(lines)
```
- **Line 54-63**: Similar to template: for each chunk, format text and citation.
- **Line 62**: Create `"Passage 1 [Apology:1-50]:\ntext..."`.
- **Line 63**: Return joined text and passage count.

---

#### `build_messages(user_message, chunks, history) -> List[Dict]`

```python
def build_messages(user_message: str, chunks: List[Dict[str, Any]], history: List[Dict[str, str]] | None = None) -> List[Dict[str, str]]:
    passages, num_passages = _format_passages_for_prompt(chunks)
    user_content = f"""Question: {user_message}
Passages to analyze:
{passages}
For each passage above, provide Relation and Value in the format:
Quote 1:
Relation: ...
Value: ...
(Continue for all {num_passages} passages.)"""
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for h in history:
            messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": user_content})
    return messages
```
- **Line 72**: Get passages text and count.
- **Line 73-88**: Build user message: question + passages + instructions.
- **Line 89**: Start with system prompt.
- **Line 90-92**: Append history if provided.
- **Line 93-94**: Append current user content, return messages list.

---

#### `generate(user_message, chunks, history, base_url, model) -> str`

```python
def generate(...) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="ollama")
    messages = build_messages(user_message, chunks, history)
    resp = client.chat.completions.create(model=model, messages=messages)
    content = resp.choices[0].message.content or ""
    return content
```
- Uses OpenAI-compatible client against Ollama.
- Builds messages and calls chat completions API.
- Returns the assistant text (Relation/Value per quote).

---

#### `generate_stream(...) -> Iterator[str]`

```python
def generate_stream(...) -> Iterator[str]:
    ...
    stream = client.chat.completions.create(..., stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content
```
- Same setup as `generate`, but with `stream=True`.
- Yields content tokens as they arrive.

---

## 4. `response_renderer.py`

Parses model plain-text output (Relation/Value per quote), merges it into the quote template, and renders to bullet format. Uses **conversation.build_quote_template**.

### Line-by-line explanation

```python
import re
from typing import Any

from src.conversation import build_quote_template

CITATION_FORMAT_RE = re.compile(r'^([^:]+):(\d+)-(\d+)$', re.IGNORECASE)
_QUOTE_BLOCK_RE = re.compile(r'Quote\s+(\d+)\s*:\s*(.*?)(?=Quote\s+\d+\s*:|\Z)', ...)
_RELATION_LINE_RE = re.compile(r'Relation\s*:\s*(.+?)(?=Value\s*:|$)', ...)
_VALUE_LINE_RE = re.compile(r'Value\s*:\s*(.+?)(?=Relation\s*:|Quote\s+\d+\s*:|$)', ...)
```
- **Line 9**: Import template builder from `conversation`.
- **Line 12**: Citation format `file:start-end`.
- **Lines 15-17**: Regexes to find Quote blocks and Relation/Value lines.

---

#### `parse_plain_text_response(raw) -> list[dict]`

```python
def parse_plain_text_response(raw: str) -> list[dict[str, str]]:
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
```
- **Line 25-27**: Handle empty input.
- **Line 30**: Match each `"Quote N:"` block.
- **Line 31**: Extract block content.
- **Line 32-35**: Extract Relation and Value from block.
- **Line 36-39**: Append dict with relation and value strings.
- **Line 40**: Return list of parsed quotes (by index).

---

#### `_parse_plain_text_fallback(raw) -> list[dict]`

- Same idea as `parse_plain_text_response`, but uses `"1."` / `"1)"` instead of `"Quote N:"`.
- Used when primary format fails.

---

#### `_parse_model_output(raw) -> list[dict]`

```python
def _parse_model_output(raw: str) -> list[dict[str, str]]:
    parsed = parse_plain_text_response(raw)
    if not parsed:
        parsed = _parse_plain_text_fallback(raw)
    return parsed
```
- Try Quote format first, fall back to numeric format.
- Returns list of `{relation_to_question, value_system}` per quote.

---

#### `merge_model_into_template(template, parsed_quotes) -> dict`

```python
def merge_model_into_template(template: dict, parsed_quotes: list[dict[str, str]] | None) -> dict:
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
```
- **Line 86-88**: If nothing parsed, return template unchanged.
- **Line 89-90**: Get template quotes.
- **Line 91-99**: For each template quote, match with same index in parsed_quotes; copy over relation/value if present.
- **Line 100**: Return merged `{"quotes": [...]}`.

---

#### `normalize_citation(citation) -> str`

```python
def normalize_citation(citation: str) -> str:
    c = (citation or "").strip()
    if not c:
        return ""
    if c.startswith('[') and c.endswith(']'):
        return c
    if CITATION_FORMAT_RE.match(c):
        return f"[{c}]"
    return c
```
- Ensures citation is `[file:start-end]` for display (wraps if needed).

---

#### `render_quotes_to_bullets(data) -> str`

```python
def render_quotes_to_bullets(data: dict[str, Any]) -> str:
    refusal = (data.get("refusal") or "").strip()
    quotes = data.get("quotes")
    if not isinstance(quotes, list):
        return refusal if refusal else ""
    if not quotes:
        return refusal if refusal else ""
    lines: list[str] = []
    for i, q in enumerate(quotes, 1):
        ...
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
        ...
    return "\n".join(lines).rstrip()
```
- If no quotes, return refusal or empty.
- For each quote: build bullet line with citation and text, then indented relation and value lines (Chinese labels).
- Join with newlines.

---

#### `process_response(raw, chunks=None) -> str`

```python
def process_response(raw: str, chunks: list | None = None) -> str:
    merged = process_response_to_data(raw, chunks)
    if not merged:
        return raw
    rendered = render_quotes_to_bullets(merged)
    return rendered if rendered else raw
```
- Delegates to `process_response_to_data`, then `render_quotes_to_bullets`.
- Falls back to raw if merging/rendering yields nothing.

---

#### `process_response_to_data(raw, chunks=None) -> dict | None`

```python
def process_response_to_data(raw: str, chunks: list | None = None) -> dict | None:
    template = build_quote_template(chunks) if chunks else {"quotes": []}
    if not template.get("quotes"):
        return template
    parsed_quotes = _parse_model_output(raw)
    merged = merge_model_into_template(template, parsed_quotes)
    return merged
```
- **Line 178**: Build template from chunks via `conversation.build_quote_template`, or empty.
- **Line 179-180**: Return template if no quotes.
- **Line 181**: Parse model output for Relation/Value.
- **Line 182**: Merge parsed values into template.
- **Line 183**: Return merged dict (used by eval and `process_response`).

---

## 5. `__init__.py`

```python
"""Talk to the People - retrieval and conversation modules."""
```
- Package docstring only; no exports.

---

## External Usage Summary

| Consumer        | Imports from src                                             |
|----------------|---------------------------------------------------------------|
| `app/main.py`  | `Retriever`, `generate`, `generate_stream`, `process_response` |
| `eval/run_eval.py` | `Retriever`, `generate`, `process_response_to_data`, `render_quotes_to_bullets`, `apply_auto_cite_to_data` |
| `scripts/generate_questions.py` | `Retriever` |

---

## End-to-end flow for a chat request

1. **app/main.py** receives the user message.
2. **retriever.search(query)** returns top-8 chunks with text, metadata, embeddings, scores.
3. **conversation.generate(message, chunks, history)**:
   - Builds template via `build_quote_template(chunks)` using `citation_utils.make_citation_from_chunk`.
   - Builds messages with passages and instructions.
   - Calls Ollama; model returns plain text with Relation/Value per quote.
4. **response_renderer.process_response(raw, chunks)**:
   - Calls `build_quote_template(chunks)` again.
   - Parses Relation/Value from raw with `_parse_model_output`.
   - Merges into template with `merge_model_into_template`.
   - Renders with `render_quotes_to_bullets` and returns bullet-formatted string.
5. The formatted string is sent back to the client.
