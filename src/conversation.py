"""
Conversation module: prompt + Ollama/Phi-3.5 for grounded responses.
Model outputs plain text (Relation/Value per quote); Python assembles JSON.
"""

from typing import List, Dict, Any, Iterator

from src.citation_utils import make_citation_from_chunk

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_ID = "phi3.5"

MAX_CHUNK_TEXT_LEN = 400

SYSTEM_PROMPT = """You are a conversational assistant grounded ONLY in Plato's dialogues (Apology, Meno, Gorgias, Republic).

Your task: For each passage below, write Relation and Value in this exact format:

Quote 1:
Relation: [how this passage relates to the user's question]
Value: [values or beliefs reflected, e.g. justice, virtue, knowledge]

Quote 2:
Relation: ...
Value: ...

RULES:
- Use exactly "Quote N:", "Relation:", and "Value:" as labels.
- One block per passage. Base your analysis ONLY on the passages given.
- Write plain text only, no JSON or special formatting."""


def build_quote_template(chunks: List[Dict[str, Any]]) -> dict:
    """Build JSON template with text and citation pre-filled from chunks. Model fills relation_to_question and value_system."""
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


def _format_passages_for_prompt(chunks: List[Dict[str, Any]]) -> tuple[str, int]:
    """Format chunks as numbered passages for the model to analyze. Returns (text, passage_count)."""
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


def build_messages(
    user_message: str,
    chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]] | None = None,
) -> List[Dict[str, str]]:
    """Build OpenAI-format messages for Ollama. Model returns Relation/Value per passage."""
    passages, num_passages = _format_passages_for_prompt(chunks)
    user_content = f"""Question: {user_message}

Passages to analyze:

{passages}

For each passage above, provide Relation and Value in the format:
Quote 1:
Relation: ...
Value: ...

Quote 2:
Relation: ...
Value: ...

(Continue for all {num_passages} passages.)"""
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for h in history:
            messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": user_content})
    return messages


def generate(
    user_message: str,
    chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]] | None = None,
    base_url: str = OLLAMA_BASE_URL,
    model: str = MODEL_ID,
) -> str:
    """Generate a single response (non-streaming)."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="ollama")
    messages = build_messages(user_message, chunks, history)
    resp = client.chat.completions.create(model=model, messages=messages)
    content = resp.choices[0].message.content or ""
    return content


def generate_stream(
    user_message: str,
    chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]] | None = None,
    base_url: str = OLLAMA_BASE_URL,
    model: str = MODEL_ID,
) -> Iterator[str]:
    """Generate response token by token (streaming)."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="ollama")
    messages = build_messages(user_message, chunks, history)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content
