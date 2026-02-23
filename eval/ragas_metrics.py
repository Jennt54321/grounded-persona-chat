"""
RAGAS metrics: Contextual Precision, Recall, Relevancy; Answer Relevancy; Faithfulness.
Uses Prometheus 2 via Ollama as default LLM-as-judge, BGE for embeddings.
"""

import os
from typing import Any

# Default config
RAGAS_LLM_MODEL = os.environ.get("RAGAS_LLM_MODEL", "tensortemplar/prometheus2:7b")
RAGAS_CHUNK_MAX_CHARS = int(os.environ.get("RAGAS_CHUNK_MAX_CHARS", "600"))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def _truncate_chunks(chunks: list[dict[str, Any]], max_chars: int = RAGAS_CHUNK_MAX_CHARS) -> list[str]:
    """Extract and truncate chunk texts for RAGAS context."""
    texts: list[str] = []
    for c in chunks:
        t = (c.get("text") or "").strip()
        if t:
            texts.append(t[:max_chars] + ("..." if len(t) > max_chars else ""))
    return texts


def _get_llm(model: str | None = None, base_url: str | None = None):
    """Create RAGAS LLM from Ollama (Prometheus 2)."""
    from openai import OpenAI
    from ragas.llms import llm_factory

    model = model or RAGAS_LLM_MODEL
    base_url = base_url or OLLAMA_BASE_URL
    client = OpenAI(api_key="ollama", base_url=base_url)
    return llm_factory(model, provider="openai", client=client)


def _get_embeddings():
    """Create RAGAS embeddings using BGE (same as retriever)."""
    try:
        from ragas.embeddings import HuggingfaceEmbeddings

        return HuggingfaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    except Exception:
        try:
            from ragas.embeddings.base import embedding_factory

            return embedding_factory("huggingface", model="BAAI/bge-base-en-v1.5")
        except Exception:
            from langchain_huggingface import HuggingFaceEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper

            return LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
            )


def compute_ragas_metrics(
    questions: list[dict],
    per_question_results: list[dict],
    all_retrieved: list[list[dict]],
    *,
    llm_model: str | None = None,
    chunk_max_chars: int = RAGAS_CHUNK_MAX_CHARS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Compute RAGAS metrics for each question and aggregate.
    Returns (per_question_scores, summary_dict).
    """
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        ContextRelevance,
        ContextUtilization,
        Faithfulness,
    )

    llm = _get_llm(model=llm_model)
    embeddings = _get_embeddings()

    faithfulness_scorer = Faithfulness(llm=llm)
    answer_relevancy_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_relevance_scorer = ContextRelevance(llm=llm)
    context_precision_ref_scorer = ContextPrecision(llm=llm)
    context_utilization_scorer = ContextUtilization(llm=llm)
    context_recall_scorer = ContextRecall(llm=llm)

    qid_to_question = {q.get("id", i): q for i, q in enumerate(questions)}
    per_question_scores: list[dict[str, Any]] = []
    precision_vals: list[float] = []
    recall_vals: list[float] = []
    relevancy_vals: list[float] = []
    answer_relevancy_vals: list[float] = []
    faithfulness_vals: list[float] = []

    for i, result in enumerate(per_question_results):
        qid = result.get("id", i + 1)
        question = result.get("question", "")
        response = result.get("response", "") or ""
        chunks = all_retrieved[i] if i < len(all_retrieved) else []
        q_item = qid_to_question.get(qid, questions[i] if i < len(questions) else {})
        reference = q_item.get("reference") if isinstance(q_item, dict) else None

        contexts = _truncate_chunks(chunks, chunk_max_chars)
        scores: dict[str, Any] = {
            "contextual_precision": None,
            "contextual_recall": None,
            "contextual_relevancy": None,
            "answer_relevancy": None,
            "faithfulness": None,
        }

        if not contexts:
            per_question_scores.append(scores)
            continue

        # Context Relevancy (always)
        try:
            r = context_relevance_scorer.score(
                user_input=question,
                retrieved_contexts=contexts,
            )
            v = float(r) if hasattr(r, "__float__") else float(getattr(r, "value", r))
            scores["contextual_relevancy"] = round(v, 4)
            relevancy_vals.append(v)
        except Exception as e:
            scores["contextual_relevancy_error"] = str(e)

        # Context Precision: use reference if available, else ContextUtilization (response-based)
        if reference:
            try:
                r = context_precision_ref_scorer.score(
                    user_input=question,
                    reference=reference,
                    retrieved_contexts=contexts,
                )
                v = float(r) if hasattr(r, "__float__") else float(getattr(r, "value", r))
                scores["contextual_precision"] = round(v, 4)
                precision_vals.append(v)
            except Exception as e:
                scores["contextual_precision_error"] = str(e)
        else:
            try:
                r = context_utilization_scorer.score(
                    user_input=question,
                    response=response,
                    retrieved_contexts=contexts,
                )
                v = float(r) if hasattr(r, "__float__") else float(getattr(r, "value", r))
                scores["contextual_precision"] = round(v, 4)
                precision_vals.append(v)
            except Exception as e:
                scores["contextual_precision_error"] = str(e)

        # Context Recall (only with reference)
        if reference:
            try:
                r = context_recall_scorer.score(
                    user_input=question,
                    retrieved_contexts=contexts,
                    reference=reference,
                )
                v = float(r) if hasattr(r, "__float__") else float(getattr(r, "value", r))
                scores["contextual_recall"] = round(v, 4)
                recall_vals.append(v)
            except Exception as e:
                scores["contextual_recall_error"] = str(e)

        # Answer Relevancy
        if response:
            try:
                r = answer_relevancy_scorer.score(
                    user_input=question,
                    response=response,
                )
                v = float(r) if hasattr(r, "__float__") else float(getattr(r, "value", r))
                scores["answer_relevancy"] = round(v, 4)
                answer_relevancy_vals.append(v)
            except Exception as e:
                scores["answer_relevancy_error"] = str(e)

        # Faithfulness
        if response and contexts:
            try:
                r = faithfulness_scorer.score(
                    user_input=question,
                    response=response,
                    retrieved_contexts=contexts,
                )
                v = float(r) if hasattr(r, "__float__") else float(getattr(r, "value", r))
                scores["faithfulness"] = round(v, 4)
                faithfulness_vals.append(v)
            except Exception as e:
                scores["faithfulness_error"] = str(e)

        per_question_scores.append(scores)

    def mean(lst: list[float]) -> float | None:
        return round(sum(lst) / len(lst), 4) if lst else None

    summary = {
        "ragas_contextual_precision_mean": mean(precision_vals),
        "ragas_contextual_recall_mean": mean(recall_vals) if recall_vals else None,
        "ragas_contextual_relevancy_mean": mean(relevancy_vals),
        "ragas_answer_relevancy_mean": mean(answer_relevancy_vals),
        "ragas_faithfulness_mean": mean(faithfulness_vals),
    }

    return per_question_scores, summary
