"""
Semantic retrieval over Plato chunk embeddings.
Loads all chunk JSONs, encodes queries with BGE, returns top-k by similarity.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOKS_DIR = PROJECT_ROOT / "books"
CHUNK_FILES = [
    "apology_chunks.json",
    "meno_chunks.json",
    "gorgias_chunks.json",
    "republic_chunks.json",
]
MODEL_ID = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


MIN_CHUNK_TEXT_LEN = 50  # Skip chunks with empty or very short text (no useful content)


def load_chunks(books_dir: Path) -> List[Dict[str, Any]]:
    """Load all chunk files into a single list with embeddings as numpy arrays.
    Skips chunks without embeddings or with empty/short text."""
    chunks: List[Dict[str, Any]] = []
    for name in CHUNK_FILES:
        path = books_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for c in data:
            emb = c.get("embedding")
            text = (c.get("text") or "").strip()
            if emb is not None and len(text) >= MIN_CHUNK_TEXT_LEN:
                c_copy = dict(c)
                c_copy["embedding"] = np.array(emb, dtype=np.float32)
                chunks.append(c_copy)
    return chunks


def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID)


class Retriever:
    def __init__(self, books_dir: Path = BOOKS_DIR):
        self.books_dir = books_dir
        self._chunks: List[Dict[str, Any]] | None = None
        self._embeddings: np.ndarray | None = None
        self._model = None

    def _ensure_loaded(self):
        if self._chunks is None:
            self._chunks = load_chunks(self.books_dir)
            if not self._chunks:
                self._embeddings = np.array([]).reshape(0, 768)  # BGE dim
            else:
                embs = [c["embedding"] for c in self._chunks]
                self._embeddings = np.stack(embs)
        if self._model is None:
            self._model = load_model()

    def search(
        self,
        query: str,
        top_k: int = 6,
        books_dir: Path | None = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k chunks most similar to the query."""
        self._ensure_loaded()
        prefixed = BGE_QUERY_PREFIX + query
        q_emb = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        q_vec = np.asarray(q_emb, dtype=np.float32)
        scores = np.dot(self._embeddings, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_indices:
            c = dict(self._chunks[i])
            text = (c.get("text") or "").strip()
            if len(text) < MIN_CHUNK_TEXT_LEN:
                continue  # Skip empty/short chunks in results
            c["embedding"] = c["embedding"].tolist()
            c["score"] = float(scores[i])
            results.append(c)
        return results


def retrieve(
    query: str,
    top_k: int = 6,
    books_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    """Convenience function: retrieve top-k chunks for a query."""
    r = Retriever(books_dir=books_dir or BOOKS_DIR)
    return r.search(query, top_k=top_k)


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is justice?"
    results = retrieve(q, top_k=3)
    print(f"Query: {q}\n")
    for i, r in enumerate(results, 1):
        book = r.get("book_id", "?")
        speaker = r.get("speaker", "?")
        s, e = r.get("start_line"), r.get("end_line")
        score = r.get("score", 0)
        text = (r.get("text", "") or "")[:200] + "..." if len(r.get("text", "") or "") > 200 else (r.get("text", "") or "")
        print(f"[{i}] [{book}, {speaker}, lines {s}-{e}] (score={score:.4f})\n{text}\n")
