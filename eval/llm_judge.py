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

JUDGE_SYSTEM_PROMPT = JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a Q&A system grounded in Plato's texts (Apology, Gorgias, Meno, Republic).

You will be given:
1. A user QUESTION
2. The system's RESPONSE (which includes cited passages and their stated values)
3. The CITED PASSAGES (the exact source text)

Evaluate TWO dimensions from 1 to 5 using the detailed rubric below.

--------------------------------
RELEVANCY (1–5)
--------------------------------
Definition: How well the response AND its chosen citations address the user's question.

Score carefully:

5 — Highly Relevant
- The response directly answers the question.
- The selected passages are clearly appropriate and tightly aligned with the question.
- No major missing aspects of the question.
- Minimal or no off-topic content.

4 — Mostly Relevant
- The response addresses the question correctly.
- Citations are generally appropriate.
- May have minor omissions or slight verbosity/off-topic content.
- Overall still clearly useful for answering the question.

3 — Partially Relevant
- The response is only partially aligned with the question.
- Some citations are weakly related OR important aspects of the question are missing.
- Contains noticeable but not fatal irrelevance.

2 — Weakly Relevant
- The response shows limited understanding of the question.
- Citations are mostly loosely related or poorly chosen.
- Major parts of the question are not addressed.

1 — Not Relevant
- The response fails to answer the question.
- Citations do not meaningfully relate to the question.
- Mostly off-topic or incorrect focus.

--------------------------------
FAITHFULNESS (1–5)
--------------------------------
Definition: How well the response's claims and stated "values" are supported by the cited passages.

IMPORTANT RULE:
The response must NOT add claims that are not supported by the cited text.

Score carefully:

5 — Fully Faithful
- All key claims are clearly supported by the cited passages.
- The stated "values" accurately reflect the passage meaning.
- No hallucinations or unsupported inferences.
- Paraphrasing is accurate and conservative.

4 — Mostly Faithful
- The main claims are supported.
- Minor overinterpretation or slight wording drift may exist.
- No major contradictions with the text.

3 — Partially Faithful
- Some claims are supported but others are weakly grounded.
- Noticeable overinterpretation OR mild unsupported additions.
- No direct contradiction, but grounding is incomplete.

2 — Weakly Faithful
- Multiple claims are poorly supported.
- Clear overreach beyond what the passages justify.
- Possible minor contradictions or misreadings.

1 — Not Faithful
- Claims contradict the cited text OR
- Major hallucinations not supported by passages OR
- The stated values misrepresent the passage meaning.

--------------------------------

Scoring rules:
- Be strict and avoid giving 5 unless the evidence is clearly strong.
- Use the full scale when appropriate.
- Output integers only.

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
