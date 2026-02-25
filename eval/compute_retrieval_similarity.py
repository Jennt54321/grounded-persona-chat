#!/usr/bin/env python3
"""
Compute question–retrieval similarity from eval_retrieval.json.
Uses BGE to measure how relevant retrieved chunks are to each question.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.chunk_index import ChunkIndex


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


def compute_question_retrieval_similarity(
    questions: list[str],
    all_retrieved_texts: list[list[str]],
    retriever,
) -> list[float]:
    """
    BGE similarity between each question and its retrieved chunk texts.
    Returns per-question mean similarity.
    """
    import numpy as np

    sims: list[float] = []
    prefixed = "Represent this sentence for searching relevant passages: "

    for q, texts in zip(questions, all_retrieved_texts):
        if not texts:
            sims.append(0.0)
            continue
        q_enc = retriever._model.encode(
            [prefixed + q],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        texts_enc = retriever._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q_vec = np.asarray(q_enc, dtype=np.float32)
        scores = np.dot(texts_enc, q_vec)
        sims.append(float(np.mean(scores)))

    return sims


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute question–retrieval similarity from eval_retrieval.json"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=PROJECT_ROOT / "eval_retrieval.json",
        help="Path to eval_retrieval.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to write results JSON (default: same dir as input, retrieval_similarity.json)",
    )
    parser.add_argument(
        "--books-dir",
        type=Path,
        default=PROJECT_ROOT / "books",
        help="Path to books directory",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Process only first N questions",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or input_path.parent / "retrieval_similarity.json"

    print(f"Loading {input_path}...")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("Error: eval_retrieval.json must be a list of {id, question, retrieved_chunk_keys}", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        data = data[: args.limit]

    chunk_index = ChunkIndex(books_dir=args.books_dir)
    questions: list[str] = []
    all_retrieved_texts: list[list[str]] = []

    for item in data:
        q = item.get("question", "")
        keys = item.get("retrieved_chunk_keys", [])
        chunks = _chunks_from_keys(chunk_index, keys)
        texts = [(c.get("text") or "").strip() for c in chunks if (c.get("text") or "").strip()]
        questions.append(q)
        all_retrieved_texts.append(texts)

    print(f"Loaded {len(questions)} questions. Loading BGE retriever...")
    from src.retriever import Retriever

    retriever = Retriever(books_dir=args.books_dir)
    retriever._ensure_loaded()

    print("Computing question–retrieval similarity...")
    sims = compute_question_retrieval_similarity(questions, all_retrieved_texts, retriever)

    mean_sim = sum(sims) / len(sims) if sims else 0.0
    per_question = [
        {"id": data[i].get("id", i + 1), "question": q, "similarity": round(s, 4)}
        for i, (q, s) in enumerate(zip(questions, sims))
    ]

    result = {
        "metric": "question_retrieval_similarity",
        "description": "Mean BGE similarity between each question and its retrieved chunk texts",
        "mean": round(mean_sim, 4),
        "n_questions": len(questions),
        "per_question": per_question,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nQuestion–Retrieval Similarity: {mean_sim:.4f}")
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
