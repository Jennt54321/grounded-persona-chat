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
from src.conversation import generate
from src.response_renderer import process_response_to_data, render_quotes_to_bullets
from src.citation_utils import apply_auto_cite_to_data


TOP_K = 8


def run_eval(
    questions_path: Path,
    output_dir: Path,
    books_dir: Path | None = None,
    top_k: int = TOP_K,
    auto_cite: bool = False,
    limit: int | None = None,
) -> None:
    books_dir = books_dir or PROJECT_ROOT / "books"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        questions = [questions]
    if limit is not None:
        questions = questions[:limit]

    retriever = Retriever(books_dir=books_dir)
    retriever._ensure_loaded()
    chunk_index = ChunkIndex(books_dir)

    all_retrieved: list[list[dict]] = []
    all_parsed: list[list[ParsedCitation]] = []
    all_strict_parsed: list[list[StrictCitation]] = []
    all_questions: list[str] = []
    all_cited_texts: list[list[str]] = []
    per_question_results: list[dict] = []
    all_hallucination_rates: list[float] = []

    for i, q_item in enumerate(questions):
        q = q_item.get("question", q_item) if isinstance(q_item, dict) else str(q_item)
        qid = q_item.get("id", i + 1) if isinstance(q_item, dict) else i + 1
        print(f"[{i+1}/{len(questions)}] {q[:60]}...")

        chunks = retriever.search(q, top_k=top_k)
        all_retrieved.append(chunks)

        raw_response = generate(q, chunks, history=None)
        data = process_response_to_data(raw_response, chunks)
        if auto_cite and data and chunks:
            data = apply_auto_cite_to_data(data, chunks)
        response = render_quotes_to_bullets(data) if data else raw_response

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

        per_question_results.append({
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
        })

    b1 = compute_retrieval_diversity(all_retrieved, chunk_index)
    b2 = compute_citation_diversity(all_parsed, chunk_index)
    c_sims = compute_similarity(all_questions, all_cited_texts, retriever)
    c_mean = sum(c_sims) / len(c_sims) if c_sims else 0.0

    all_validity = compute_citation_validity(
        [p for citations in all_parsed for p in citations],
        chunk_index,
    )
    a4_mean = sum(all_hallucination_rates) / len(all_hallucination_rates) if all_hallucination_rates else 0.0

    summary = {
        "A1_citation_existence_rate": all_validity["A1_existence_rate"],
        "A2_exact_quote_match_rate": all_validity["A2_quote_match_rate"],
        "A3_fabrication_rate": all_validity["A3_fabrication_rate"],
        "A4_hallucination_rate": round(a4_mean, 4),
        **b1,
        **b2,
        "C_q_citation_similarity_mean": round(c_mean, 4),
        "n_questions": len(questions),
    }

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
            "questions_path": str(questions_path),
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
        f"**N questions**: {summary['n_questions']}",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nFull output: {full_path}")
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
    args = parser.parse_args()

    run_eval(
        questions_path=args.questions,
        output_dir=args.output,
        books_dir=args.books_dir,
        top_k=args.top_k,
        auto_cite=args.auto_cite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
