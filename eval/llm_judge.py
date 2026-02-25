"""
LLM-as-a-judge: relevancy and faithfulness for RAG responses.
Uses the same HF model as conversation (Qwen2.5-3B-Instruct) for consistency.
Scores 1-5.
"""

import json
import re
from typing import Any

# Use same backend as conversation to avoid extra deps
from src.conversation import _hf_chat_completion

JUDGE_SYSTEM_PROMPT = """You are an evaluator for a Q&A system grounded in Plato's texts (Apology, Gorgias, Meno, Republic).

You will be given:
1. A user QUESTION
2. The system's RESPONSE (citing passages and their values)
3. The CITED PASSAGES (exact text that was cited)

Score two dimensions from 1 to 5 (integer only):

RELEVANCY (1-5): How relevant is the response (and the chosen citations) to the question?
- 1: Not relevant; citations do not address the question.
- 3: Partially relevant; some connection but incomplete.
- 5: Highly relevant; citations directly address the question.

FAITHFULNESS (1-5): How faithful is the response to the cited passages? Are the stated "values" supported by the cited text?
- 1: Not faithful; values or claims contradict or are unsupported by the passages.
- 3: Partially faithful; some support but overclaims or minor inaccuracies.
- 5: Fully faithful; values and claims are clearly supported by the cited text.

You MUST reply with exactly two lines in this format (no other text):
Relevancy: <1-5>
Faithfulness: <1-5>"""


def _parse_judge_reply(reply: str) -> dict[str, int | None]:
    """Parse judge output to relevancy and faithfulness scores (1-5)."""
    out: dict[str, int | None] = {"relevancy": None, "faithfulness": None}
    if not reply or not reply.strip():
        return out
    text = reply.strip()
    rel_m = re.search(r"Relevancy\s*:\s*(\d+)", text, re.IGNORECASE)
    faith_m = re.search(r"Faithfulness\s*:\s*(\d+)", text, re.IGNORECASE)
    if rel_m:
        v = int(rel_m.group(1))
        out["relevancy"] = max(1, min(5, v))
    if faith_m:
        v = int(faith_m.group(1))
        out["faithfulness"] = max(1, min(5, v))
    return out


def score_relevancy_faithfulness(
    question: str,
    response: str,
    cited_texts: list[str],
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    LLM-as-a-judge: score relevancy and faithfulness (1-5).
    question: user question
    response: full system response (with citations and values)
    cited_texts: list of cited passage texts (in order)
    Returns dict with relevancy, faithfulness (int 1-5 or None), and raw_reply.
    """
    cited_block = "\n\n".join(
        f"[Passage {i+1}]:\n{(t or '').strip()[:800]}"
        for i, t in enumerate(cited_texts)
    )
    if not cited_block.strip():
        cited_block = "(No passages provided)"
    user_content = f"""QUESTION:
{question}

RESPONSE:
{(response or '').strip()[:2000]}

CITED PASSAGES:
{cited_block}

Score Relevancy and Faithfulness (1-5). Reply with exactly:
Relevancy: <1-5>
Faithfulness: <1-5>"""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        raw = _hf_chat_completion(messages, model_id=model_id or "Qwen/Qwen2.5-3B-Instruct")
        parsed = _parse_judge_reply(raw)
        return {
            "relevancy": parsed["relevancy"],
            "faithfulness": parsed["faithfulness"],
            "raw_reply": raw,
        }
    except Exception as e:
        return {
            "relevancy": None,
            "faithfulness": None,
            "raw_reply": "",
            "error": str(e),
        }


def run_judge_on_results(
    results: list[dict],
    model_id: str | None = None,
) -> tuple[list[dict], dict[str, float]]:
    """
    Run LLM judge on each result; add relevancy/faithfulness to each item.
    results: list of per-question dicts with question, response, cited_texts (or parsed_citations used to resolve text).
    Expects each item to have at least: question, response, and cited_texts (list of strings).
    Returns (results_with_scores, summary) where summary has mean relevancy and mean faithfulness.
    """
    relevancies: list[float] = []
    faithfulnesses: list[float] = []
    out_results = []
    for r in results:
        q = r.get("question", "")
        response = r.get("response", "")
        cited_texts = r.get("cited_texts") or []
        judge = score_relevancy_faithfulness(q, response, cited_texts, model_id=model_id)
        new_r = dict(r)
        new_r["judge_relevancy"] = judge["relevancy"]
        new_r["judge_faithfulness"] = judge["faithfulness"]
        new_r["judge_raw_reply"] = judge.get("raw_reply", "")
        if judge.get("error"):
            new_r["judge_error"] = judge["error"]
        out_results.append(new_r)
        if judge["relevancy"] is not None:
            relevancies.append(float(judge["relevancy"]))
        if judge["faithfulness"] is not None:
            faithfulnesses.append(float(judge["faithfulness"]))
    summary = {
        "judge_relevancy_mean": sum(relevancies) / len(relevancies) if relevancies else 0.0,
        "judge_faithfulness_mean": sum(faithfulnesses) / len(faithfulnesses) if faithfulnesses else 0.0,
        "judge_n_scored": len(relevancies),
    }
    return out_results, summary
