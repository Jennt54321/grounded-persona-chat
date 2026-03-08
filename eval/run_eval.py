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

# Sentinel: --from-results with no path means use default output dir's eval_results_checkpoint.json
_DEFAULT_FROM_RESULTS = object()

from eval.citation_parser import (
    ParsedCitation,
    StrictCitation,
    parse_citations_strict,
    strict_citations_from_data,
    strict_to_parsed,
)
from eval.chunk_index import ChunkIndex, parse_book_volume
from eval.metrics import (
    compute_citation_validity,
    compute_retrieval_diversity,
    compute_citation_diversity,
    compute_similarity,
    verify_citations_against_retrieved,
)
from eval.llm_judge import run_judge_on_results
from src.retriever import Retriever
from src.conversation import generate_values_batch_stream
from src.response_renderer import render_quotes_to_bullets


TOP_K = 5


def _chunks_from_keys(
    chunk_index: ChunkIndex,
    keys: list,
) -> list[dict]:
    """Resolve retrieved_chunk_keys to full chunk dicts (with text). Keys are 4-tuples (book, volume_id, start, end) or dicts with volume_id."""
    chunks: list[dict] = []
    for k in keys:
        vol = ""
        if isinstance(k, (list, tuple)) and len(k) >= 4:
            book, vol, start, end = str(k[0]), str(k[1]) if k[1] is not None else "", int(k[2]), int(k[3])
        elif isinstance(k, dict):
            book = str(k.get("file", k.get("book_id", "")))
            start = int(k.get("start_line", 0))
            end = int(k.get("end_line", 0))
            vol = str(k.get("volume_id", "") or "")
        else:
            continue
        c = chunk_index.get(book, start, end, volume_id=vol)
        if c:
            chunks.append(c)
    return chunks


def run_eval(
    questions_path: Path | None = None,
    output_dir: Path | None = None,
    books_dir: Path | None = None,
    top_k: int = TOP_K,
    limit: int | None = None,
    from_results: Path | None = None,
    from_retrieval: Path | None = None,
    rerank: bool = True,
    rerun_retrieval: bool = False,
    retrieval_only: bool = False,
    batch: bool = False,
    run_judge: bool = True,
) -> None:
    books_dir = books_dir or PROJECT_ROOT / "books"
    output_dir = Path(output_dir or PROJECT_ROOT / "eval" / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Questions-specific retrieval cache (e.g. questions_life_retrieval.json)
    effective_questions_path = questions_path or PROJECT_ROOT / "eval" / "questions_life.json"
    questions_stem = effective_questions_path.stem
    out_retrieval = output_dir / f"{questions_stem}_retrieval.json"
    out_prefix = questions_stem
    out_checkpoint = output_dir / f"{out_prefix}_results_checkpoint.json"
    out_results = output_dir / f"{out_prefix}_results.json"
    out_summary = output_dir / f"{out_prefix}_summary.json"
    out_full = output_dir / f"{out_prefix}_full.json"
    out_report = output_dir / f"{out_prefix}_report.md"

    chunk_index = ChunkIndex(books_dir)
    all_retrieved: list[list[dict]] = []
    all_parsed: list[list[ParsedCitation]] = []
    all_strict_parsed: list[list[StrictCitation]] = []
    all_questions: list[str] = []
    all_cited_texts: list[list[str]] = []
    per_question_results: list[dict] = []
    all_hallucination_rates: list[float] = []
    questions: list[dict] = []
    retriever = None  # set in each branch below; ensure bound before compute_similarity

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

            # Prefer cited text from result's quotes (so --from-results works without books/)
            cited_texts = []
            quotes = r.get("quotes") or []
            if quotes:
                for q in quotes:
                    if isinstance(q, dict):
                        t = (q.get("text") or "").strip()
                        if t:
                            cited_texts.append(t)
            if not cited_texts:
                for sc in strict_parsed:
                    book_id, vol_id = parse_book_volume(sc.file)
                    c = chunk_index.get(book_id or sc.file, sc.start_line, sc.end_line, volume_id=vol_id or None)
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
            result_item["cited_texts"] = cited_texts
            # Ensure parsed_citations include actual cited text (for self-contained JSON)
            pcs_out = result_item.get("parsed_citations") or []
            for j, pc in enumerate(pcs_out):
                if isinstance(pc, dict) and j < len(cited_texts):
                    pc["text"] = cited_texts[j]
            per_question_results.append(result_item)
            all_hallucination_rates.append(halluc_rate)
            questions.append({"id": r.get("id"), "question": q, "reference": r.get("reference")})

        retriever = Retriever(books_dir=books_dir)
        retriever._ensure_loaded()
    else:
        # Not from_results: resolve questions, then either load retrieval (explicit or cache) or run Stage 1
        questions_path_resolved = questions_path or PROJECT_ROOT / "eval" / "questions_life.json"
        questions = json.loads(Path(questions_path_resolved).read_text(encoding="utf-8"))
        if not isinstance(questions, list):
            questions = [questions]
        if limit is not None:
            questions = questions[:limit]

        retrieval_list = None
        if from_retrieval is not None:
            retrieval_list = json.loads(Path(from_retrieval).read_text(encoding="utf-8"))
            if not isinstance(retrieval_list, list):
                raise ValueError("from_retrieval must be list of {id, question, retrieved_chunk_keys})")
            if limit is not None:
                retrieval_list = retrieval_list[:limit]
            print(f"Loading retrieval from {from_retrieval} ({len(retrieval_list)} questions)...")
        elif not rerun_retrieval and out_retrieval.exists():
            cache_list = json.loads(out_retrieval.read_text(encoding="utf-8"))
            if isinstance(cache_list, list) and len(cache_list) == len(questions):
                retrieval_list = cache_list
                print(f"Using cached retrieval from {out_retrieval} ({len(retrieval_list)} questions)...")

        if retrieval_list is not None:
            questions = []  # use retrieval list as source of truth so lengths match
            for r in retrieval_list:
                keys = r.get("retrieved_chunk_keys", [])
                chunks = _chunks_from_keys(chunk_index, keys)
                all_retrieved.append(chunks)
                questions.append({
                    "id": r.get("id"),
                    "question": r.get("question", ""),
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
                print(f"  Found {len(chunks)} passages. Analyzing...", flush=True)

                data = None
                gen_errors = []
                stream_fn = generate_values_batch_stream
                for event in stream_fn(q, chunks):
                    if event[0] == "status":
                        _, cit_i, total = event
                        print(f"    Citation {cit_i}/{total} done...", flush=True)
                    elif event[0] == "done":
                        payload = event[1]
                        data = {"quotes": payload.get("quotes", [])}
                        gen_errors = payload.get("errors", [])
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
                    book_id, vol_id = parse_book_volume(sc.file)
                    chunk = chunk_index.get(book_id or sc.file, sc.start_line, sc.end_line, volume_id=vol_id or None)
                    if chunk:
                        t = (chunk.get("text") or "").strip()
                        if t:
                            cited_texts.append(t)
                all_cited_texts.append(cited_texts)

                validity = compute_citation_validity(parsed, chunk_index)
                _, _, halluc_rate = verify_citations_against_retrieved(strict_parsed, chunks)
                all_hallucination_rates.append(halluc_rate)

                parsed_citations_with_text = []
                for j, sc in enumerate(strict_parsed):
                    _, vol_id = parse_book_volume(sc.file)
                    pc_entry = {
                        "file": sc.file,
                        "start_line": sc.start_line,
                        "end_line": sc.end_line,
                        "text": cited_texts[j] if j < len(cited_texts) else "",
                        "volume_id": vol_id or "",
                    }
                    parsed_citations_with_text.append(pc_entry)
                result_item = {
                    "id": qid,
                    "question": q,
                    "response": response,
                    "parsed_citations": parsed_citations_with_text,
                    "cited_texts": cited_texts,
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
                            "value_system": qt.get("value_system") or qt.get("value") or "",
                            "volume_id": (chunks[i].get("volume_id") or "").strip() if i < len(chunks) else "",
                        }
                        for i, qt in enumerate(data["quotes"])
                    ]
                    if gen_errors:
                        result_item["generation_errors"] = gen_errors
                    per_question_results.append(result_item)

        else:
            retriever = Retriever(books_dir=books_dir)
            retriever._ensure_loaded()

            print("Stage 1: Retrieval...")
            retrieval_results = []
            for i, q_item in enumerate(questions):
                q = q_item.get("question", q_item) if isinstance(q_item, dict) else str(q_item)
                qid = q_item.get("id", i + 1) if isinstance(q_item, dict) else i + 1
                print(f"  [{i+1}/{len(questions)}] {q[:60]}...")
                print("  Retrieving passages...", flush=True)
                if rerank:
                    chunks, _ = retriever.search_with_rerank(q, bi_top_k=50, final_top_k=top_k)
                else:
                    chunks = retriever.search(q, top_k=top_k)
                print(f"  Found {len(chunks)} passages.", flush=True)
                all_retrieved.append(chunks)
                retrieval_results.append({
                    "id": qid,
                    "question": q,
                    "retrieved_chunk_keys": [chunk_index.chunk_key(c) for c in chunks],
                })

            out_retrieval.write_text(
                json.dumps(retrieval_results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Saved retrieval checkpoint: {out_retrieval}")
            if retrieval_only:
                return

            print("Stage 2: Generation (LLM value per citation)...")
            for i, q_item in enumerate(questions):
                q = q_item.get("question", q_item) if isinstance(q_item, dict) else str(q_item)
                qid = q_item.get("id", i + 1) if isinstance(q_item, dict) else i + 1
                chunks = all_retrieved[i]
                print(f"  [{i+1}/{len(questions)}] {q[:60]}...")
                print(f"  Found {len(chunks)} passages. Analyzing...", flush=True)

                data = None
                gen_errors = []
                stream_fn = generate_values_batch_stream
                for event in stream_fn(q, chunks):
                    if event[0] == "status":
                        _, cit_i, total = event
                        print(f"    Citation {cit_i}/{total} done...", flush=True)
                    elif event[0] == "done":
                        payload = event[1]
                        data = {"quotes": payload.get("quotes", [])}
                        gen_errors = payload.get("errors", [])
                response = render_quotes_to_bullets(data) if data else ""

                strict_parsed = (
                    strict_citations_from_data(data) if data else parse_citations_strict(response)
                )
                parsed = [strict_to_parsed(sc) for sc in strict_parsed]
                all_parsed.append(parsed)
                all_strict_parsed.append(strict_parsed)
                all_questions.append(q)

                cited_texts = []
                for sc in strict_parsed:
                    book_id, vol_id = parse_book_volume(sc.file)
                    chunk = chunk_index.get(book_id or sc.file, sc.start_line, sc.end_line, volume_id=vol_id or None)
                    if chunk:
                        t = (chunk.get("text") or "").strip()
                        if t:
                            cited_texts.append(t)
                all_cited_texts.append(cited_texts)

                validity = compute_citation_validity(parsed, chunk_index)
                _, _, halluc_rate = verify_citations_against_retrieved(strict_parsed, chunks)
                all_hallucination_rates.append(halluc_rate)

                parsed_citations_with_text = []
                for j, sc in enumerate(strict_parsed):
                    _, vol_id = parse_book_volume(sc.file)
                    pc_entry = {
                        "file": sc.file,
                        "start_line": sc.start_line,
                        "end_line": sc.end_line,
                        "text": cited_texts[j] if j < len(cited_texts) else "",
                        "volume_id": vol_id or "",
                    }
                    parsed_citations_with_text.append(pc_entry)
                result_item = {
                    "id": qid,
                    "question": q,
                    "response": response,
                    "parsed_citations": parsed_citations_with_text,
                    "cited_texts": cited_texts,
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
                            "value_system": qt.get("value_system") or qt.get("value") or "",
                            "volume_id": (chunks[i].get("volume_id") or "").strip() if i < len(chunks) else "",
                        }
                        for i, qt in enumerate(data["quotes"])
                    ]
                    if gen_errors:
                        result_item["generation_errors"] = gen_errors
                    per_question_results.append(result_item)

    # Checkpoint: retrieval + generation + validity (A1-A4)
    checkpoint_path = out_checkpoint
    checkpoint_path.write_text(
        json.dumps(per_question_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved results checkpoint: {checkpoint_path}")

    b1 = compute_retrieval_diversity(all_retrieved, chunk_index)
    b2 = compute_citation_diversity(all_parsed, chunk_index)
    if retriever is None:
        retriever = Retriever(books_dir=books_dir)
        retriever._ensure_loaded()
    c_sims = compute_similarity(all_questions, all_cited_texts, retriever)
    c_mean = sum(c_sims) / len(c_sims) if c_sims else 0.0

    all_validity = compute_citation_validity(
        [p for citations in all_parsed for p in citations],
        chunk_index,
    )
    a4_mean = (
        sum(all_hallucination_rates) / len(all_hallucination_rates)
        if all_hallucination_rates
        else 0.0
    )

    questions_with_errors = [r for r in per_question_results if r.get("generation_errors")]
    total_generation_errors = sum(len(r.get("generation_errors", [])) for r in per_question_results)

    # Optional: LLM-as-a-judge relevancy and faithfulness
    if run_judge:
        print("Running LLM-as-a-judge (relevancy & faithfulness)...", flush=True)
        per_question_results, judge_summary = run_judge_on_results(per_question_results)
        summary_extra = judge_summary
    else:
        summary_extra = {}

    summary = {
        "A1_citation_existence_rate": all_validity["A1_existence_rate"],
        "A2_exact_quote_match_rate": all_validity["A2_quote_match_rate"],
        "A3_fabrication_rate": all_validity["A3_fabrication_rate"],
        "A4_hallucination_rate": round(a4_mean, 4) if a4_mean is not None else None,
        **b1,
        **b2,
        "C_q_citation_similarity_mean": round(c_mean, 4),
        "n_questions": len(questions),
        "n_questions_with_generation_errors": len(questions_with_errors),
        "total_generation_errors": total_generation_errors,
        **summary_extra,
    }
    # Avoid null for numeric summary keys (downstream may expect numbers)
    for k, default in (
        ("A1_citation_existence_rate", 0.0), ("A2_exact_quote_match_rate", 0.0), ("A3_fabrication_rate", 0.0),
        ("C_q_citation_similarity_mean", 0.0), ("n_questions", 0), ("n_questions_with_generation_errors", 0), ("total_generation_errors", 0),
    ):
        if summary.get(k) is None:
            summary[k] = default
    for k in list(summary):
        if (k.startswith("B1") or k.startswith("B2")) and summary.get(k) is None:
            summary[k] = 0.0 if "entropy" in k or "rate" in k else 0

    results_path = out_results
    summary_path = out_summary
    full_path = out_full
    report_path = out_report

    eval_full = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_questions": len(questions),
            "top_k": top_k,
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

    def _fmt_rate(v):
        if v is None:
            return "N/A"
        try:
            return f"{float(v):.2%}"
        except (TypeError, ValueError):
            return str(v) if v is not None else "N/A"

    def _fmt_num(v):
        if v is None:
            return "0"
        return str(v)

    a4_str = "N/A (no retrieval)" if summary.get("A4_hallucination_rate") is None else _fmt_rate(summary["A4_hallucination_rate"])
    report_lines = [
        "# RAG Evaluation Report",
        "",
        "## A. Citation Validity",
        f"- **A1 Citation Existence Rate**: {_fmt_rate(summary.get('A1_citation_existence_rate'))}",
        f"- **A2 Exact Quote Match Rate**: {_fmt_rate(summary.get('A2_exact_quote_match_rate'))}",
        f"- **A3 Fabrication Rate**: {_fmt_rate(summary.get('A3_fabrication_rate'))}",
        f"- **A4 Hallucination Rate** (citation range does not overlap retrieved chunks): {a4_str}",
        "",
        "## B. Diversity",
        "### B1. Retrieval Diversity",
        f"- **Unique retrieved chunks**: {_fmt_num(summary.get('B1a_unique_retrieved_chunks'))}",
        f"- **Unique sections**: {_fmt_num(summary.get('B1b_unique_sections'))}",
        f"- **Rank-1 unique count**: {_fmt_num(summary.get('B1c_rank1_unique_count'))}",
        f"- **Rank-1 entropy**: {_fmt_num(summary.get('B1c_rank1_entropy'))}",
        "### B2. Citation Diversity",
        f"- **Unique model citations**: {_fmt_num(summary.get('B2a_unique_citations'))}",
        f"- **Citation entropy**: {_fmt_num(summary.get('B2b_citation_entropy'))}",
        f"- **Citation reuse rate**: {_fmt_num(summary.get('B2c_citation_reuse_rate'))}",
        "",
        "## C. Similarity",
        f"- **Q-Citation similarity (BGE) mean**: {_fmt_num(summary.get('C_q_citation_similarity_mean'))}",
        "",
        "## D. Generation (LLM)",
        f"- **Questions with token/context errors**: {_fmt_num(summary.get('n_questions_with_generation_errors'))}",
        f"- **Total generation errors** (context length exceeded, etc.): {_fmt_num(summary.get('total_generation_errors'))}",
        "",
        f"**N questions**: {_fmt_num(summary.get('n_questions'))}",
    ]
    if run_judge and summary.get("judge_relevancy_mean") is not None:
        report_lines.extend([
            "",
            "## E. LLM-as-a-Judge",
            f"- **Relevancy (1-5) mean**: {summary.get('judge_relevancy_mean', 0):.2f}",
            f"- **Faithfulness (1-5) mean**: {summary.get('judge_faithfulness_mean', 0):.2f}",
            f"- **N scored**: {summary.get('judge_n_scored', 0)}",
        ])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    retrieval_checkpoint = out_retrieval
    if retrieval_checkpoint.exists():
        print(f"Retrieval checkpoint: {retrieval_checkpoint}")
    print(f"Results checkpoint: {checkpoint_path}")
    print(f"Full output: {full_path}")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--questions", "-q", type=Path, default=PROJECT_ROOT / "eval" / "questions_life.json")
    parser.add_argument("--output", "-o", type=Path, default=PROJECT_ROOT / "eval" / "results")
    parser.add_argument("--books-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--limit", "-n", type=int, default=None, help="Run only first N questions (for quick testing)")
    parser.add_argument(
        "--from-results",
        "-f",
        nargs="?",
        const=_DEFAULT_FROM_RESULTS,
        default=None,
        type=Path,
        help="Skip retrieval+generation; load existing results and run metrics only. With no path, uses <output>/eval_results_checkpoint.json",
    )
    parser.add_argument(
        "--from-retrieval",
        type=Path,
        default=None,
        help="Skip Stage 1; load retrieval JSON and continue with generation + metrics",
    )
    parser.add_argument(
        "--rerun-retrieval",
        action="store_true",
        help="Force Stage 1 retrieval even when a questions-specific retrieval cache exists",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run Stage 1 only; write retrieval JSON and exit (no generation or metrics)",
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
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use one LLM call per question for values (always batch; flag kept for backward compatibility)",
    )
    parser.add_argument(
        "--run-judge",
        action="store_true",
        default=True,
        help="Run LLM-as-a-judge (relevancy & faithfulness 1-5) on each result (default: True)",
    )
    parser.add_argument(
        "--no-run-judge",
        dest="run_judge",
        action="store_false",
        help="Skip LLM-as-a-judge",
    )
    args = parser.parse_args()

    from_results = args.from_results
    if from_results is _DEFAULT_FROM_RESULTS:
        questions_stem = args.questions.stem
        from_results = args.output / f"{questions_stem}_results_checkpoint.json"

    run_eval(
        questions_path=args.questions,
        output_dir=args.output,
        books_dir=args.books_dir,
        top_k=args.top_k,
        limit=args.limit,
        from_results=from_results,
        from_retrieval=args.from_retrieval,
        rerank=args.rerank,
        rerun_retrieval=args.rerun_retrieval,
        retrieval_only=args.retrieval_only,
        batch=args.batch,
        run_judge=args.run_judge,
    )


if __name__ == "__main__":
    main()
