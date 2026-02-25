#!/usr/bin/env python3
"""
Contextual metrics only: recall (quote-in-chunk) and relevancy (embedding similarity).
Reads existing results JSON; no retrieval or generation.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.citation_parser import normalize_text_for_match
from eval.chunk_index import ChunkIndex, parse_book_volume
from src.retriever import Retriever, BGE_QUERY_PREFIX


def _chunks_from_keys(chunk_index: ChunkIndex, keys: list) -> list[dict]:
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


def _compute_contextual_recall_one(
    parsed_citations: list[dict],
    quotes: list[dict],
    chunk_index: ChunkIndex,
) -> tuple[float, int, int]:
    """
    For one result: check each citation's model quote is contained in the cited chunk.
    Returns (rate, matches, total) where total = citations with non-empty quote.
    """
    matches = 0
    total = 0
    for i, pc in enumerate(parsed_citations):
        file_or_book = pc.get("file") or pc.get("book_id") or ""
        book_id, volume_id = parse_book_volume(file_or_book)
        if pc.get("volume_id") is not None:
            volume_id = pc.get("volume_id") or ""
        start_line = pc.get("start_line")
        end_line = pc.get("end_line")
        if not book_id or start_line is None or end_line is None:
            continue
        model_quote = ""
        if i < len(quotes) and isinstance(quotes[i], dict):
            model_quote = (quotes[i].get("text") or "").strip()
        if not model_quote:
            continue
        total += 1
        chunk_text = (pc.get("text") or "").strip()
        if not chunk_text:
            chunk = chunk_index.get(book_id, int(start_line), int(end_line), volume_id=volume_id or None)
            if chunk:
                chunk_text = (chunk.get("text") or "").strip()
        if not chunk_text:
            continue
        norm_quoted = normalize_text_for_match(model_quote)
        norm_chunk = normalize_text_for_match(chunk_text)
        if norm_quoted and norm_quoted in norm_chunk:
            matches += 1
    rate = matches / total if total else 0.0
    return (rate, matches, total)


def _compute_contextual_relevancy_one(
    question: str,
    cited_texts: list[str],
    top5_chunks: list[dict] | None,
    retriever: Retriever,
) -> tuple[list[float], float, list[float] | None, float | None]:
    """
    Embed question and cited chunks (and optionally top5); return similarities.
    Returns (cited_sims, cited_mean, top5_sims_or_none, top5_mean_or_none).
    """
    import numpy as np

    cited_sims: list[float] = []
    cited_mean = 0.0
    top5_sims: list[float] | None = None
    top5_mean: float | None = None

    if not cited_texts:
        return (cited_sims, cited_mean, top5_sims, top5_mean)

    retriever._ensure_loaded()
    prefixed = BGE_QUERY_PREFIX + question
    q_enc = retriever._model.encode(
        [prefixed],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    q_vec = np.asarray(q_enc, dtype=np.float32)

    texts_enc = retriever._model.encode(
        cited_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    cited_sims = [float(x) for x in np.dot(texts_enc, q_vec)]
    cited_mean = float(np.mean(cited_sims)) if cited_sims else 0.0

    if top5_chunks:
        top5_texts = [(c.get("text") or "").strip() for c in top5_chunks]
        top5_texts = [t for t in top5_texts if t]
        if top5_texts:
            top5_enc = retriever._model.encode(
                top5_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            top5_sims = [float(x) for x in np.dot(top5_enc, q_vec)]
            top5_mean = float(np.mean(top5_sims))

    return (cited_sims, cited_mean, top5_sims, top5_mean)


def run_contextual_metrics(
    from_results: Path,
    output_dir: Path | None = None,
    books_dir: Path | None = None,
    limit: int | None = None,
) -> None:
    books_dir = books_dir or PROJECT_ROOT / "books"
    output_dir = Path(output_dir or from_results.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(from_results.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError("from_results must be a JSON object with 'results' or a list of result items")
    if limit is not None:
        results = results[:limit]

    chunk_index = ChunkIndex(books_dir)
    retriever = Retriever(books_dir=books_dir)
    retriever._ensure_loaded()

    out_results: list[dict] = []
    total_recall_matches = 0
    total_recall_total = 0
    all_cited_means: list[float] = []
    all_top5_means: list[float] = []

    for r in results:
        question = r.get("question", "")
        parsed_citations = r.get("parsed_citations") or []
        quotes = r.get("quotes") or []
        cited_texts = r.get("cited_texts") or []
        if not cited_texts and parsed_citations:
            for pc in parsed_citations:
                file_or_book = pc.get("file") or pc.get("book_id") or ""
                book_id, volume_id = parse_book_volume(file_or_book)
                if pc.get("volume_id") is not None:
                    volume_id = pc.get("volume_id") or ""
                start_line = pc.get("start_line", 0)
                end_line = pc.get("end_line", 0)
                chunk = chunk_index.get(book_id, int(start_line), int(end_line), volume_id=volume_id or None) if book_id else None
                if chunk:
                    t = (chunk.get("text") or "").strip()
                    if t:
                        cited_texts.append(t)

        recall_rate, recall_matches, recall_total = _compute_contextual_recall_one(
            parsed_citations, quotes, chunk_index
        )
        total_recall_matches += recall_matches
        total_recall_total += recall_total

        top5_chunks = None
        keys = r.get("retrieved_chunk_keys") or []
        if keys:
            top5_chunks = _chunks_from_keys(chunk_index, keys)

        cited_sims, cited_mean, top5_sims, top5_mean = _compute_contextual_relevancy_one(
            question, cited_texts, top5_chunks, retriever
        )
        if cited_texts:
            all_cited_means.append(cited_mean)
        if top5_mean is not None:
            all_top5_means.append(top5_mean)

        item = {
            "id": r.get("id"),
            "question": question,
            "contextual_recall_rate": round(recall_rate, 4),
            "contextual_recall_matches": recall_matches,
            "contextual_recall_total": recall_total,
            "contextual_relevancy_cited_similarities": [round(x, 4) for x in cited_sims],
            "contextual_relevancy_cited_mean": round(cited_mean, 4),
        }
        if top5_sims is not None:
            item["contextual_relevancy_top5_similarities"] = [round(x, 4) for x in top5_sims]
        if top5_mean is not None:
            item["contextual_relevancy_top5_mean"] = round(top5_mean, 4)
        out_results.append(item)

    contextual_recall_rate_mean = (
        total_recall_matches / total_recall_total if total_recall_total else 0.0
    )
    contextual_relevancy_cited_mean = (
        sum(all_cited_means) / len(all_cited_means) if all_cited_means else 0.0
    )
    contextual_relevancy_top5_mean = (
        sum(all_top5_means) / len(all_top5_means) if all_top5_means else None
    )

    summary = {
        "contextual_recall_rate_mean": round(contextual_recall_rate_mean, 4),
        "contextual_recall_total_citations": total_recall_total,
        "contextual_recall_matched": total_recall_matches,
        "contextual_relevancy_cited_mean": round(contextual_relevancy_cited_mean, 4),
        "contextual_relevancy_top5_mean": round(contextual_relevancy_top5_mean, 4)
        if contextual_relevancy_top5_mean is not None
        else None,
        "n_questions": len(results),
    }

    out_json = output_dir / "eval_contextual_results.json"
    out_summary = output_dir / "eval_contextual_summary.json"
    out_report = output_dir / "eval_contextual_report.md"

    out_json.write_text(
        json.dumps(out_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    out_summary.write_text(
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

    report_lines = [
        "# Contextual Metrics Report",
        "",
        "## Contextual recall",
        f"- **Rate (quote in chunk)**: {_fmt_rate(summary.get('contextual_recall_rate_mean'))}",
        f"- **Matched / total citations**: {summary.get('contextual_recall_matched')} / {summary.get('contextual_recall_total_citations')}",
        "",
        "## Contextual relevancy (BGE similarity to question)",
        f"- **Cited chunks mean similarity**: {summary.get('contextual_relevancy_cited_mean')}",
        f"- **Top-5 retrieved mean similarity** (if available): {summary.get('contextual_relevancy_top5_mean', 'N/A')}",
        "",
        f"**N questions**: {summary.get('n_questions')}",
    ]
    out_report.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Results: {out_json}")
    print(f"Summary: {out_summary}")
    print(f"Report: {out_report}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute contextual recall and relevancy from existing eval results"
    )
    parser.add_argument(
        "--from-results",
        "-f",
        type=Path,
        default=PROJECT_ROOT / "eval" / "results" / "eval_results_checkpoint.json",
        help="Path to results JSON (object with 'results' or list of items)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: same as from-results file)",
    )
    parser.add_argument(
        "--books-dir",
        type=Path,
        default=None,
        help="Books directory for ChunkIndex (default: project books/)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Process only first N results",
    )
    args = parser.parse_args()

    run_contextual_metrics(
        from_results=args.from_results,
        output_dir=args.output_dir,
        books_dir=args.books_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
