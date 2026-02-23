"""
Conversation module: prompt + LLM for grounded responses.
Supports Ollama (Phi-3.5) or Hugging Face (microsoft/Phi-3.5-mini-instruct).
Model outputs plain text (Relation/Value per quote); Python assembles JSON.
"""

import os
from typing import List, Dict, Any, Iterator

from src.citation_utils import make_citation_from_chunk

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_ID = "phi3.5"
HF_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

# "ollama" or "huggingface"; set LLM_BACKEND=huggingface for Colab
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")

MAX_CHUNK_TEXT_LEN = None  # None = no truncation; set int to limit chars per chunk

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
        if MAX_CHUNK_TEXT_LEN is not None and len(text) > MAX_CHUNK_TEXT_LEN:
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
        if MAX_CHUNK_TEXT_LEN is not None and len(text) > MAX_CHUNK_TEXT_LEN:
            text = text[:MAX_CHUNK_TEXT_LEN] + "..."
        citation = make_citation_from_chunk(c)
        lines.append(f"Passage {i} [{citation}]:\n{text}")
    return "\n\n".join(lines), len(lines)


def _format_single_chunk(chunk: Dict[str, Any]) -> tuple[str, str] | None:
    """Format a single chunk for prompt. Returns (text, citation) or None if too short."""
    text = (chunk.get("text") or "").strip()
    if len(text) < 20:
        return None
    if MAX_CHUNK_TEXT_LEN is not None and len(text) > MAX_CHUNK_TEXT_LEN:
        text = text[:MAX_CHUNK_TEXT_LEN] + "..."
    citation = make_citation_from_chunk(chunk)
    return text, citation


def _is_context_length_error(exc: BaseException) -> bool:
    """Check if exception indicates token/context length exceeded."""
    msg = str(exc).lower()
    return any(
        k in msg
        for k in ("context", "token", "length exceeded", "maximum context", "context_length_exceeded")
    )


# ----- Hugging Face backend -----

_HF_PIPELINE = None


def _load_hf_pipeline(model_id: str = HF_MODEL_ID):
    """Lazy-load Hugging Face pipeline with 4-bit quantization for Colab T4."""
    global _HF_PIPELINE
    if _HF_PIPELINE is not None:
        return _HF_PIPELINE
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _HF_PIPELINE = (model, tokenizer)
    return _HF_PIPELINE


def _hf_chat_completion(messages: List[Dict[str, str]], model_id: str = HF_MODEL_ID) -> str:
    """Run chat completion via Hugging Face Phi-3.5."""
    model, tokenizer = _load_hf_pipeline(model_id)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,  # Avoid DynamicCache.from_legacy_cache compatibility issues with transformers 4.48+
    )
    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return reply.strip()


RELATION_SYSTEM_PROMPT = """You are a conversational assistant grounded ONLY in Plato's dialogues (Apology, Meno, Gorgias, Republic).

Your task: Given a user question, a cited passage, and the values/beliefs the passage reflects, explain in 1-2 sentences how this passage (and its values) relates to the question.
Reply with plain text only. No labels or special formatting."""

VALUE_SYSTEM_PROMPT = """You are a conversational assistant grounded ONLY in Plato's dialogues (Apology, Meno, Gorgias, Republic).

Your task: Consider ONLY the cited passage. State in 1-2 sentences what values or beliefs this passage reflects (e.g., justice, virtue, knowledge).
Reply with plain text only. No labels or special formatting."""


def generate_value(
    chunk: Dict[str, Any],
    base_url: str = OLLAMA_BASE_URL,
    model: str = MODEL_ID,
    errors_out: list | None = None,
) -> str:
    """Ask model: what values/beliefs does this passage reflect? Uses ONLY the cited chunk, no question.
    Returns plain text, no parsing needed. On context/token error, appends to errors_out and returns ""."""
    formatted = _format_single_chunk(chunk)
    if not formatted:
        return ""
    text, citation = formatted
    user_content = f"""Passage [{citation}]:
{text}

What values or beliefs does this passage reflect? Reply in 1-2 sentences."""
    messages = [{"role": "system", "content": VALUE_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    try:
        if LLM_BACKEND == "huggingface":
            content = _hf_chat_completion(messages, model_id=HF_MODEL_ID)
            return content
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="ollama")
        resp = client.chat.completions.create(model=model, messages=messages)
        content = (resp.choices[0].message.content or "").strip()
        return content
    except Exception as e:
        if errors_out is not None and _is_context_length_error(e):
            errors_out.append({"stage": "value", "citation": citation, "error": str(e)})
            return ""
        raise


def generate_relation(
    question: str,
    chunk: Dict[str, Any],
    value: str,
    base_url: str = OLLAMA_BASE_URL,
    model: str = MODEL_ID,
    errors_out: list | None = None,
) -> str:
    """Ask model: how does this passage relate to the question? Uses question, citation, and the value generated before.
    Returns plain text, no parsing needed. On context/token error, appends to errors_out and returns ""."""
    formatted = _format_single_chunk(chunk)
    if not formatted:
        return ""
    text, citation = formatted
    user_content = f"""Question: {question}

Passage [{citation}]:
{text}

Values/beliefs this passage reflects: {value}

How does this passage (and its values) relate to the question? Reply in 1-2 sentences."""
    messages = [{"role": "system", "content": RELATION_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    try:
        if LLM_BACKEND == "huggingface":
            content = _hf_chat_completion(messages, model_id=HF_MODEL_ID)
            return content
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="ollama")
        resp = client.chat.completions.create(model=model, messages=messages)
        content = (resp.choices[0].message.content or "").strip()
        return content
    except Exception as e:
        if errors_out is not None and _is_context_length_error(e):
            errors_out.append({"stage": "relation", "citation": citation, "error": str(e)})
            return ""
        raise


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
    messages = build_messages(user_message, chunks, history)
    if LLM_BACKEND == "huggingface":
        return _hf_chat_completion(messages, model_id=HF_MODEL_ID)
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="ollama")
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
    """Generate response token by token (streaming). HF backend yields full response at once."""
    messages = build_messages(user_message, chunks, history)
    if LLM_BACKEND == "huggingface":
        content = _hf_chat_completion(messages, model_id=HF_MODEL_ID)
        yield content
        return
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="ollama")
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def generate_per_citation(
    user_message: str,
    chunks: List[Dict[str, Any]],
    base_url: str = OLLAMA_BASE_URL,
    model: str = MODEL_ID,
) -> tuple[dict, list[dict]]:
    """
    For-loop each chunk: first generate_value (passage only), then generate_relation (question + citation + value).
    Returns (data, errors) where data is {"quotes": [...]} and errors is list of
    {stage, citation, error} for context/token limit failures.
    """
    quotes = []
    errors: list[dict] = []
    for c in chunks:
        formatted = _format_single_chunk(c)
        if not formatted:
            continue
        text, citation = formatted
        template_item = {
            "text": text,
            "citation": citation,
            "relation_to_question": "",
            "value_system": "",
        }
        template_item["value_system"] = generate_value(
            c, base_url=base_url, model=model, errors_out=errors
        )
        template_item["relation_to_question"] = generate_relation(
            user_message,
            c,
            template_item["value_system"],
            base_url=base_url,
            model=model,
            errors_out=errors,
        )
        quotes.append(template_item)
    return {"quotes": quotes}, errors


def generate_per_citation_stream(
    user_message: str,
    chunks: List[Dict[str, Any]],
    base_url: str = OLLAMA_BASE_URL,
    model: str = MODEL_ID,
) -> Iterator[tuple[str, int | dict]]:
    """
    Same as generate_per_citation but yields ("status", i, total) after each citation,
    then ("done", {"quotes": [...], "errors": [...]}) at the end.
    """
    valid_chunks = [c for c in chunks if _format_single_chunk(c)]
    total = len(valid_chunks)
    quotes = []
    errors: list[dict] = []
    completed = 0
    for c in valid_chunks:
        text, citation = _format_single_chunk(c)  # guaranteed non-None
        template_item = {
            "text": text,
            "citation": citation,
            "relation_to_question": "",
            "value_system": "",
        }
        template_item["value_system"] = generate_value(
            c, base_url=base_url, model=model, errors_out=errors
        )
        template_item["relation_to_question"] = generate_relation(
            user_message,
            c,
            template_item["value_system"],
            base_url=base_url,
            model=model,
            errors_out=errors,
        )
        quotes.append(template_item)
        completed += 1
        yield ("status", completed, total)
    yield ("done", {"quotes": quotes, "errors": errors})
