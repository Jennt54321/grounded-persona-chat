#!/usr/bin/env python3
"""
Add embeddings to chunk JSON files using BAAI/bge-base-en-v1.5.
Reads each chunk's "text", computes a vector, and writes it back as "embedding".
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root for books path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOKS_DIR = PROJECT_ROOT / "books"

CHUNK_FILES = [
    "apology_chunks.json",
    "meno_chunks.json",
    "gorgias_chunks.json",
    "republic_chunks.json",
]

MODEL_ID = "BAAI/bge-base-en-v1.5"
DEFAULT_BATCH_SIZE = 64


def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID)


def embed_chunks(chunks, model, batch_size: int):
    """Compute embeddings for all chunks. Passages are encoded without query instruction."""
    texts = [c["text"] for c in chunks]
    # Encode passages (no instruction prefix per BGE docs for passage embedding)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    # Convert to list of lists for JSON (numpy floats -> Python floats)
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    return chunks


def process_file(path: Path, model, batch_size: int, force: bool = False) -> None:
    print(f"Processing {path.name} ...")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        print(f"  No chunks in {path.name}, skipping.")
        return
    if not force and data[0].get("embedding") is not None:
        print(f"  Already has embeddings (dim={len(data[0]['embedding'])}), skipping.")
        return
    data = embed_chunks(data, model, batch_size)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Wrote {len(data)} chunks with embeddings (dim={len(data[0]['embedding'])}) to {path.name}")


def main():
    parser = argparse.ArgumentParser(description="Add BGE embeddings to chunk JSON files.")
    parser.add_argument(
        "files",
        nargs="*",
        default=CHUNK_FILES,
        help=f"Chunk JSON filenames (default: {CHUNK_FILES})",
    )
    parser.add_argument(
        "--books-dir",
        type=Path,
        default=BOOKS_DIR,
        help="Directory containing chunk JSON files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Encode batch size",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite existing embeddings",
    )
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence_transformers not found. Install with the same Python you use to run this script:", file=sys.stderr)
        print(f"  {sys.executable} -m pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    model = load_model()
    books_dir = args.books_dir
    if not books_dir.is_dir():
        print(f"Books directory not found: {books_dir}", file=sys.stderr)
        sys.exit(1)

    for name in args.files:
        path = books_dir / name
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue
        process_file(path, model, args.batch_size, force=args.force)

    print("Done.")


if __name__ == "__main__":
    main()
