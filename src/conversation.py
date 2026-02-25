"""
Conversation module: prompt + LLM for grounded responses.
Uses Hugging Face (Qwen2.5-3B-Instruct) for local inference.
Model receives question + passages in order and outputs a JSON array of value strings only; parser merges by index.
"""

import logging
from typing import List, Dict, Any, Iterator

logger = logging.getLogger(__name__)

from src.citation_utils import make_citation_from_chunk

HF_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

def build_quote_template(chunks: List[Dict[str, Any]]) -> dict:
    """Build JSON template with text and citation pre-filled from chunks. Model fills value."""
    quotes = []
    for c in chunks:
        text = (c.get("text") or "").strip()
        if len(text) < 20:
            continue
        citation = make_citation_from_chunk(c)
        quotes.append({
            "text": text,
            "citation": citation,
            "value": "",
        })
    return {"quotes": quotes}


def _format_passages_for_prompt(chunks: List[Dict[str, Any]]) -> tuple[str, int]:
    """Format chunks as numbered passages for the model to analyze. Returns (text, passage_count)."""
    lines = []
    for i, c in enumerate(chunks, 1):
        text = (c.get("text") or "").strip()
        if len(text) < 20:
            continue
        citation = make_citation_from_chunk(c)
        lines.append(f"Passage {i} [{citation}]:\n{text}")
    return "\n\n".join(lines), len(lines)


def _format_single_chunk(chunk: Dict[str, Any]) -> tuple[str, str] | None:
    """Format a single chunk for prompt. Returns (text, citation) or None if too short."""
    text = (chunk.get("text") or "").strip()
    if len(text) < 20:
        return None
    citation = make_citation_from_chunk(chunk)
    return text, citation


# ----- Hugging Face backend -----

_HF_PIPELINE = None

def _load_hf_pipeline(model_id: str = HF_MODEL_ID):
    """
    One loader for both:
    - Colab T4 (CUDA): 4-bit bitsandbytes
    - Mac (MPS): fp16 on MPS (no bitsandbytes)
    - CPU fallback: fp32
    """
    global _HF_PIPELINE
    if _HF_PIPELINE is not None:
        return _HF_PIPELINE

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Make padding safe for generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        # ----- Colab T4 path -----
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    elif torch.backends.mps.is_available():
        # ----- Mac Apple Silicon path -----
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map={"": "mps"},   # force everything on MPS
            trust_remote_code=True,
        )

    else:
        # ----- CPU fallback -----
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True,
        )

    model.eval()
    _HF_PIPELINE = (model, tokenizer)
    device = next(model.parameters()).device
    logger.info("Hugging Face model loaded successfully: %s on %s", model_id, device)
    return _HF_PIPELINE


def _hf_chat_completion(messages: List[Dict[str, str]], model_id: str = HF_MODEL_ID) -> str:
    import torch

    model, tokenizer = _load_hf_pipeline(model_id)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt")

    # Move tensors to the same device as the model (works for CUDA/MPS/CPU)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Allow enough tokens for 5 passages × 2-3 line value_system; truncation causes parse failure
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # use_cache=True,  # optional; you can enable later if no compat issues
    )

    reply = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return reply.strip()


# Batch value generation: model receives question + paragraphs in order and outputs a JSON array of values only.
BATCH_VALUE_PROMPT = """You are a conversational assistant grounded ONLY in Plato's dialogues (Apology, Meno, Gorgias, Republic).

You will receive a question and then several passages in order (Passage 1, Passage 2, ...). For each passage, write a 2-3 sentence (2-3 line) analysis of what values or beliefs that passage reflects (e.g., justice, virtue, knowledge), considering the user's question. Each array element should be substantive: multiple sentences or lines, not a single short phrase.

Output a single valid JSON array of strings: one string per passage, in the same order as the passages. Do not add any text before or after the JSON. No markdown, no code fences, no commentary."""


def build_messages_batch_values(
    user_message: str,
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build messages: question once, then passages in order; model outputs a JSON array of values in the same order."""
    valid_chunks = [c for c in chunks if _format_single_chunk(c)]
    passages, num_passages = _format_passages_for_prompt(valid_chunks)
    user_content = f"""Question: {user_message}

Passages (analyze each in order and output exactly {num_passages} values as a JSON array of strings):

{passages}

Output a JSON array of exactly {num_passages} strings: the value analysis for Passage 1, then Passage 2, and so on."""
    return [
        {"role": "system", "content": BATCH_VALUE_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_values_batch_stream(
    user_message: str,
    chunks: List[Dict[str, Any]],
) -> Iterator[tuple[str, int | dict]]:
    """
    One LLM call for all passages; model returns a JSON array of value strings in passage order.
    Parser merges values by index into template quotes.
    Yields ("status", current, total) then ("done", {"quotes": [...], "errors": [...]}).
    """
    valid_chunks = [c for c in chunks if _format_single_chunk(c)]
    if not valid_chunks:
        yield ("done", {"quotes": [], "errors": []})
        return

    template_quotes = []
    for c in valid_chunks:
        text, citation = _format_single_chunk(c)  # guaranteed non-None
        template_quotes.append({
            "text": text,
            "citation": citation,
            "value": "",
        })
    expected_count = len(template_quotes)
    yield ("status", 0, expected_count)  # started

    try:
        messages = build_messages_batch_values(user_message, valid_chunks)
        raw = _hf_chat_completion(messages, model_id=HF_MODEL_ID)
    except Exception as e:
        logger.warning("generate_values_batch_stream: LLM call failed: %s", e, exc_info=True)
        errors = [{"stage": "value", "error": str(e)}]
        yield ("status", expected_count, expected_count)
        yield ("done", {"quotes": template_quotes, "errors": errors})
        return

    from src.response_renderer import parse_batch_values_json
    parsed_values = parse_batch_values_json(raw or "", expected_count)
    if parsed_values is None or len(parsed_values) != expected_count:
        # Log at WARNING so it appears with default log level; include raw snippet to trace
        raw_preview = (raw or "").strip()
        if len(raw_preview) > 1200:
            raw_preview = raw_preview[:1200] + "\n... [truncated, total %d chars]" % len((raw or "").strip())
        logger.warning(
            "generate_values_batch_stream: parse failed or count mismatch (got %s, expected %s). Raw output snippet:\n%s",
            len(parsed_values) if parsed_values else 0,
            expected_count,
            raw_preview,
        )
        errors = [{"stage": "value", "error": "parse failed or count mismatch"}]
        yield ("status", expected_count, expected_count)
        yield ("done", {"quotes": template_quotes, "errors": errors})
        return

    for i, val in enumerate(parsed_values):
        if i < len(template_quotes):
            v = (val or "").strip()
            template_quotes[i]["value"] = v
            template_quotes[i]["value_system"] = v
    yield ("status", expected_count, expected_count)
    yield ("done", {"quotes": template_quotes, "errors": []})


def ensure_model_loaded(model_id: str = HF_MODEL_ID) -> bool:
    """
    Load the Hugging Face pipeline if not already loaded. Use this to verify
    the model loads successfully (e.g. at startup or in a test script).
    Returns True if loaded (or already loaded), raises on failure.
    """
    _load_hf_pipeline(model_id)
    return True


if __name__ == "__main__":
    # Verify Hugging Face model loads: python -m src.conversation
    logging.basicConfig(level=logging.INFO)
    try:
        ensure_model_loaded()
        print("OK: Hugging Face model loaded successfully.")
    except Exception as e:
        print("FAIL: Could not load Hugging Face model:", e)
        raise
