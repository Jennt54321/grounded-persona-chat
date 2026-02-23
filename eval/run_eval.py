#!/usr/bin/env python3
"""
Evaluation pipeline: retrieve, generate, parse citations, compute metrics.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.citation_parser import (
    ParsedCitation,
    StrictCitation,
    parse_citations_strict,
    strict_citations_from_data,
    strict_to_parsed,
)
from eval.chunk_index import ChunkIndex
from eval.metrics import (
    compute_citation_validity,
    compute_retrieval_diversity,
    compute_citation_diversity,
    compute_similarity,
    verify_citations_against_retrieved,
)
from src.retriever import Retriever
from src.conversation import generate_per_citation
from src.response_renderer import render_quotes_to_bullets
from src.citation_utils import apply_auto_cite_to_data


TOP_K = 8


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


def run_eval(
    questions_path: Path | None = None,
    output_dir: Path | None = None,
    books_dir: Path | None = None,
    top_k: int = TOP_K,
    auto_cite: bool = False,
    limit: int | None = None,
    ragas: bool = True,
    ragas_llm: str | None = None,
    from_results: Path | None = None,
    from_retrieval: Path | None = None,
    rerank: bool = True,
) -> None:
    books_dir = books_dir or PROJECT_ROOT / "books"
    output_dir = Path(output_dir or PROJECT_ROOT / "eval" / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_index = ChunkIndex(books_dir)
    all_retrieved: list[list[dict]] = []
    all_parsed: list[list[ParsedCitation]] = []
    all_strict_parsed: list[list[StrictCitation]] = []
    all_questions: list[str] = []
    all_cited_texts: list[list[str]] = []
    per_question_results: list[dict] = []
    all_hallucination_rates: list[float] = []
    questions: list[dict] = []

    if from_results is not None:
        # Load existing results; skip retrieval and generation
        data = json.loads(Path(from_results).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
        elif isinstance(data, list):
            results = data
        else:
            raise ValueError("from_results must be eval_results.json or eval_full.json")
        if limit is not None:
            results = results[:limit]

        print(f"Loading {len(results)} results from {from_results}...")
        for r in results:
            keys = r.get("retrieved_chunk_keys", [])
            chunks = _chunks_from_keys(chunk_index, keys)
            all_retrieved.append(chunks)

            q = r.get("question", "")
            response = r.get("response", "")
            all_questions.append(q)

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
            parsed = [strict_to_parsed(sc) for sc in strict_parsed]
            all_parsed.append(parsed)
            all_strict_parsed.append(strict_parsed)

            cited_texts = []
            for sc in strict_parsed:
                c = chunk_index.get(sc.file, sc.start_line, sc.end_line)
                if c:
                    t = (c.get("text") or "").strip()
                    if t:
                        cited_texts.append(t)
            all_cited_texts.append(cited_texts)

            validity = compute_citation_validity(parsed, chunk_index)
            _, _, halluc_rate = verify_citations_against_retrieved(strict_parsed, chunks)

            result_item = dict(r)
            result_item["A1_existence_rate"] = validity["A1_existence_rate"]
            result_item["A2_quote_match_rate"] = validity["A2_quote_match_rate"]
            result_item["A3_fabrication_rate"] = validity["A3_fabrication_rate"]
            result_item["A4_hallucination_rate"] = halluc_rate
            per_question_results.append(result_item)
            all_hallucination_rates.append(halluc_rate)
            questions.append({"id": r.get("id"), "question": q, "reference": r.get("reference")})

        retriever = Retriever(books_dir=books_dir)
        retriever._ensure_loaded()
    elif from_retrieval is not None:
        # Load retrieval checkpoint; skip Stage 1, run Stage 2 (generation) and beyond
        retrieval_list = json.loads(Path(from_retrieval).read_text(encoding="utf-8"))
        if not isinstance(retrieval_list, list):
            raise ValueError("from_retrieval must be eval_retrieval.json (list of {id, question, retrieved_chunk_keys})")
        if limit is not None:
            retrieval_list = retrieval_list[:limit]

        print(f"Loading retrieval from {from_retrieval} ({len(retrieval_list)} questions)...")
        for r in retrieval_list:
            keys = r.get("retrieved_chunk_keys", [])
            chunks = _chunks_from_keys(chunk_index, keys)
            all_retrieved.append(chunks)
            questions.append({
                "id": r.get("id"),
                "question": r.get("question", ""),
                "book_hint": r.get("book_hint"),
            })

        retriever = Retriever(books_dir=books_dir)
        retriever._ensure_loaded()

        # Stage 2: Generation (skip Stage 1)
        print("Stage 2: Generation (using loaded retrieval)...")
        for i, q_item in enumerate(questions):
            q = q_item.get("question", q_item) if isinstance(q_item, dict) else str(q_item)
            qid = q_item.get("id", i + 1) if isinstance(q_item, dict) else i + 1
            chunks = all_retrieved[i]
            print(f"  [{i+1}/{len(questions)}] {q[:60]}...")

            data, gen_errors = generate_per_citation(q, chunks)
            if auto_cite and data and chunks:
                data = apply_auto_cite_to_data(data, chunks)
            response = render_quotes_to_bullets(data) if data else ""

            strict_parsed = (
                strict_citations_from_data(data) if data else parse_citations_strict(response)
            )
            parsed = [strict_to_parsed(sc) for sc in strict_parsed]
            all_parsed.append(parsed)
            all_strict_parsed.append(strict_parsed)
            all_questions.append(q)

            cited_texts: list[str] = []
            for sc in strict_parsed:
                chunk = chunk_index.get(sc.file, sc.start_line, sc.end_line)
                if chunk:
                    t = (chunk.get("text") or "").strip()
                    if t:
                        cited_texts.append(t)
            all_cited_texts.append(cited_texts)

            validity = compute_citation_validity(parsed, chunk_index)
            _, _, halluc_rate = verify_citations_against_retrieved(strict_parsed, chunks)
            all_hallucination_rates.append(halluc_rate)

            result_item = {
                "id": qid,
                "question": q,
                "book_hint": q_item.get("book_hint") if isinstance(q_item, dict) else None,
                "response": response,
                "parsed_citations": [
                    {"file": sc.file, "start_line": sc.start_line, "end_line": sc.end_line}
                    for sc in strict_parsed
                ],
                "retrieved_chunk_keys": [chunk_index.chunk_key(c) for c in chunks],
                "A1_existence_rate": validity["A1_existence_rate"],
                "A2_quote_match_rate": validity["A2_quote_match_rate"],
                "A3_fabrication_rate": validity["A3_fabrication_rate"],
                "A4_hallucination_rate": halluc_rate,
            }
            if data and data.get("quotes"):
                result_item["quotes"] = [
                    {
                        "text": qt.get("text", ""),
                        "citation": qt.get("citation", ""),
                        "relation_to_question": qt.get("relation_to_question", ""),
                        "value_system": qt.get("value_system", ""),
                    }
                    for qt in data["quotes"]
                ]
            if gen_errors:
                result_item["generation_errors"] = gen_errors
            per_question_results.append(result_item)
    else:
        # Full pipeline: retrieve + generate
        questions_path = questions_path or PROJECT_ROOT / "eval" / "questions.json"
        questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
        if not isinstance(questions, list):
            questions = [questions]
        if limit is not None:
            questions = questions[:limit]

        retriever = Retriever(books_dir=books_dir)
        retriever._ensure_loaded()

        # Stage 1: Retrieval only (bi 50 -> cross 8 when rerank, else bi top_k)
        print("Stage 1: Retrieval...")
        retrieval_results: list[dict] = []
        for i, q_item in enumerate(questions):
            q = q_item.get("question", q_item) if isinstance(q_item, dict) else str(q_item)
            qid = q_item.get("id", i + 1) if isinstance(q_item, dict) else i + 1
            print(f"  [{i+1}/{len(questions)}] {q[:60]}...")
            if rerank:
                chunks, _ = retriever.search_with_rerank(q, bi_top_k=50, final_top_k=top_k)
            else:
                chunks = retriever.search(q, top_k=top_k)
            all_retrieved.append(chunks)
            retrieval_results.append({
                "id": qid,
                "question": q,
                "retrieved_chunk_keys": [chunk_index.chunk_key(c) for c in chunks],
            })

        retrieval_path = output_dir / "eval_retrieval.json"
        retrieval_path.write_text(
            json.dumps(retrieval_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved retrieval checkpoint: {retrieval_path}")

        # Stage 2: Generation (Ollama relevance/value per citation)
        print("Stage 2: Generation...")
        for i, q_item in enumerate(questions):
            q = q_item.get("question", q_item) if isinstance(q_item, dict) else str(q_item)
            qid = q_item.get("id", i + 1) if isinstance(q_item, dict) else i + 1
            chunks = all_retrieved[i]
            print(f"  [{i+1}/{len(questions)}] {q[:60]}...")

            data, gen_errors = generate_per_citation(q, chunks)
            if auto_cite and data and chunks:
                data = apply_auto_cite_to_data(data, chunks)
            response = render_quotes_to_bullets(data) if data else ""

            strict_parsed = (
                strict_citations_from_data(data) if data else parse_citations_strict(response)
            )
            parsed = [strict_to_parsed(sc) for sc in strict_parsed]
            all_parsed.append(parsed)
            all_strict_parsed.append(strict_parsed)
            all_questions.append(q)

            cited_texts: list[str] = []
            for sc in strict_parsed:
                chunk = chunk_index.get(sc.file, sc.start_line, sc.end_line)
                if chunk:
                    t = (chunk.get("text") or "").strip()
                    if t:
                        cited_texts.append(t)
            all_cited_texts.append(cited_texts)

            validity = compute_citation_validity(parsed, chunk_index)
            _, _, halluc_rate = verify_citations_against_retrieved(strict_parsed, chunks)
            all_hallucination_rates.append(halluc_rate)

            result_item = {
                "id": qid,
                "question": q,
                "book_hint": q_item.get("book_hint") if isinstance(q_item, dict) else None,
                "response": response,
                "parsed_citations": [
                    {"file": sc.file, "start_line": sc.start_line, "end_line": sc.end_line}
                    for sc in strict_parsed
                ],
                "retrieved_chunk_keys": [chunk_index.chunk_key(c) for c in chunks],
                "A1_existence_rate": validity["A1_existence_rate"],
                "A2_quote_match_rate": validity["A2_quote_match_rate"],
                "A3_fabrication_rate": validity["A3_fabrication_rate"],
                "A4_hallucination_rate": halluc_rate,
            }
            if data and data.get("quotes"):
                result_item["quotes"] = [
                    {
                        "text": qt.get("text", ""),
                        "citation": qt.get("citation", ""),
                        "relation_to_question": qt.get("relation_to_question", ""),
                        "value_system": qt.get("value_system", ""),
                    }
                    for qt in data["quotes"]
                ]
            if gen_errors:
                result_item["generation_errors"] = gen_errors
            per_question_results.append(result_item)

    # Stage 2 checkpoint: retrieval + generation + A validity (before RAGAS)
    pre_ragas_path = output_dir / "eval_pre_ragas.json"
    pre_ragas_path.write_text(
        json.dumps(per_question_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved pre-RAGAS results (relevance/value/validity): {pre_ragas_path}")

    ragas_summary: dict = {}
    # RAGAS metrics (Prometheus 2 via Ollama)
    if ragas:
        try:
            from eval.ragas_metrics import compute_ragas_metrics

            print("\nComputing RAGAS metrics...")
            ragas_scores, ragas_summary = compute_ragas_metrics(
                questions,
                per_question_results,
                all_retrieved,
                llm_model=ragas_llm,
            )
            for i, r in enumerate(per_question_results):
                if i < len(ragas_scores):
                    r.update(ragas_scores[i])
        except Exception as e:
            print(f"\nRAGAS failed: {e}")
            print("Pre-RAGAS results (relevance, value, validity) saved. Re-run with --from-results to retry RAGAS.")

    b1 = compute_retrieval_diversity(all_retrieved, chunk_index)
    b2 = compute_citation_diversity(all_parsed, chunk_index)
    c_sims = compute_similarity(all_questions, all_cited_texts, retriever)
    c_mean = sum(c_sims) / len(c_sims) if c_sims else 0.0

    all_validity = compute_citation_validity(
        [p for citations in all_parsed for p in citations],
        chunk_index,
    )
    a4_mean = sum(all_hallucination_rates) / len(all_hallucination_rates) if all_hallucination_rates else 0.0

    questions_with_errors = [r for r in per_question_results if r.get("generation_errors")]
    total_generation_errors = sum(len(r.get("generation_errors", [])) for r in per_question_results)

    summary = {
        "A1_citation_existence_rate": all_validity["A1_existence_rate"],
        "A2_exact_quote_match_rate": all_validity["A2_quote_match_rate"],
        "A3_fabrication_rate": all_validity["A3_fabrication_rate"],
        "A4_hallucination_rate": round(a4_mean, 4),
        **b1,
        **b2,
        "C_q_citation_similarity_mean": round(c_mean, 4),
        "n_questions": len(questions),
        "n_questions_with_generation_errors": len(questions_with_errors),
        "total_generation_errors": total_generation_errors,
    }
    if ragas:
        summary.update(ragas_summary)

    results_path = output_dir / "eval_results.json"
    summary_path = output_dir / "eval_summary.json"
    full_path = output_dir / "eval_full.json"
    report_path = output_dir / "eval_report.md"

    eval_full = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_questions": len(questions),
            "top_k": top_k,
            "auto_cite": auto_cite,
            "ragas": ragas,
            "rerank": rerank,
            "from_results": str(from_results) if from_results else None,
            "questions_path": str(questions_path) if (questions_path and not from_results) else None,
        },
        "summary": summary,
        "results": per_question_results,
    }
    full_path.write_text(
        json.dumps(eval_full, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    results_path.write_text(
        json.dumps(per_question_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_lines = [
        "# RAG Evaluation Report",
        "",
        "## A. Citation Validity",
        f"- **A1 Citation Existence Rate**: {summary['A1_citation_existence_rate']:.2%}",
        f"- **A2 Exact Quote Match Rate**: {summary['A2_exact_quote_match_rate']:.2%}",
        f"- **A3 Fabrication Rate**: {summary['A3_fabrication_rate']:.2%}",
        f"- **A4 Hallucination Rate** (citation range does not overlap retrieved chunks): {summary['A4_hallucination_rate']:.2%}",
        "",
        "## B. Diversity",
        "### B1. Retrieval Diversity",
        f"- **Unique retrieved chunks**: {summary['B1a_unique_retrieved_chunks']}",
        f"- **Unique sections**: {summary['B1b_unique_sections']}",
        f"- **Rank-1 unique count**: {summary['B1c_rank1_unique_count']}",
        f"- **Rank-1 entropy**: {summary['B1c_rank1_entropy']}",
        "### B2. Citation Diversity",
        f"- **Unique model citations**: {summary['B2a_unique_citations']}",
        f"- **Citation entropy**: {summary['B2b_citation_entropy']}",
        f"- **Citation reuse rate**: {summary['B2c_citation_reuse_rate']}",
        "",
        "## C. Similarity",
        f"- **Q-Citation similarity (BGE) mean**: {summary['C_q_citation_similarity_mean']}",
        "",
        "## D. Generation (LLM)",
        f"- **Questions with token/context errors**: {summary['n_questions_with_generation_errors']}",
        f"- **Total generation errors** (context length exceeded, etc.): {summary['total_generation_errors']}",
        "",
    ]
    if "ragas_contextual_precision_mean" in summary:
        cp = summary.get("ragas_contextual_precision_mean")
        cr = summary.get("ragas_contextual_recall_mean")
        crel = summary.get("ragas_contextual_relevancy_mean")
        ar = summary.get("ragas_answer_relevancy_mean")
        fth = summary.get("ragas_faithfulness_mean")
        report_lines.extend([
            "## E. RAGAS Metrics (Prometheus 2)",
            f"- **Contextual Precision**: {cp}" if cp is not None else "- **Contextual Precision**: N/A",
            f"- **Contextual Recall**: {cr}" if cr is not None else "- **Contextual Recall**: N/A (no reference answers)",
            f"- **Contextual Relevancy**: {crel}" if crel is not None else "- **Contextual Relevancy**: N/A",
            f"- **Answer Relevancy**: {ar}" if ar is not None else "- **Answer Relevancy**: N/A",
            f"- **Faithfulness**: {fth}" if fth is not None else "- **Faithfulness**: N/A",
            "",
        ])
    report_lines.append(f"**N questions**: {summary['n_questions']}")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    retrieval_checkpoint = output_dir / "eval_retrieval.json"
    if retrieval_checkpoint.exists():
        print(f"Retrieval checkpoint: {retrieval_checkpoint}")
    print(f"Pre-RAGAS checkpoint: {pre_ragas_path}")
    print(f"Full output: {full_path}")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--questions", "-q", type=Path, default=PROJECT_ROOT / "eval" / "questions.json")
    parser.add_argument("--output", "-o", type=Path, default=PROJECT_ROOT / "eval" / "results")
    parser.add_argument("--books-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--auto-cite", action="store_true", help="Apply auto-cite fallback before metrics")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Run only first N questions (for quick testing)")
    parser.add_argument("--ragas", action="store_true", default=True, help="Enable RAGAS metrics (default: True)")
    parser.add_argument("--no-ragas", dest="ragas", action="store_false", help="Disable RAGAS metrics")
    parser.add_argument("--ragas-llm", type=str, default=None, help="RAGAS judge model (default: tensortemplar/prometheus2:7b via Ollama)")
    parser.add_argument(
        "--from-results",
        "-f",
        type=Path,
        default=None,
        help="Skip retrieval+generation; load existing eval_results.json or eval_full.json and run metrics only",
    )
    parser.add_argument(
        "--from-retrieval",
        type=Path,
        default=None,
        help="Skip Stage 1; load eval_retrieval.json and continue with generation + metrics",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=True,
        help="Use Bi-encoder (top 50) + Cross-encoder rerank to final top_k (default: True)",
    )
    parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Use Bi-encoder only, no Cross-encoder reranking",
    )
    args = parser.parse_args()

    run_eval(
        questions_path=args.questions,
        output_dir=args.output,
        books_dir=args.books_dir,
        top_k=args.top_k,
        auto_cite=args.auto_cite,
        limit=args.limit,
        ragas=args.ragas,
        ragas_llm=args.ragas_llm,
        from_results=args.from_results,
        from_retrieval=args.from_retrieval,
        rerank=args.rerank,
    )


if __name__ == "__main__":
    main()
