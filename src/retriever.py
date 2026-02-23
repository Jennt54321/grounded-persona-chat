"""
Semantic retrieval over Plato chunk embeddings.
Bi-encoder (BGE) + optional Cross-encoder reranking.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

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
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 8
BI_TOP_K = 50  # Bi-encoder candidates for reranking
FINAL_TOP_K = 8

MIN_CHUNK_TEXT_LEN = 5  # Skip chunks with empty or very short text (no useful content)


@dataclass
class RetrievalTrace:
    """Stored trace of Bi-encoder -> Cross-encoder retrieval pipeline."""
    query: str
    timestamp: str
    bi_top_k: int
    final_top_k: int
    bi_candidates: List[Dict[str, Any]] = field(default_factory=list)
    rerank_scores: List[float] = field(default_factory=list)
    final_chunks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def _chunk_summary(c: Dict, bi_rank: Optional[int] = None, rerank_rank: Optional[int] = None) -> Dict:
            d = {
                "book_id": c.get("book_id"),
                "start_line": c.get("start_line"),
                "end_line": c.get("end_line"),
                "speaker": c.get("speaker", c.get("speakers")),
                "bi_score": c.get("bi_score"),
                "rerank_score": c.get("rerank_score"),
                "text_preview": (c.get("text") or "")[:200] + ("..." if len(c.get("text") or "") > 200 else ""),
            }
            if bi_rank is not None:
                d["bi_rank"] = bi_rank
            if rerank_rank is not None:
                d["rerank_rank"] = rerank_rank
            return d

        bi_summaries = [_chunk_summary(c, bi_rank=i + 1) for i, c in enumerate(self.bi_candidates)]
        final_summaries = [_chunk_summary(c, rerank_rank=i + 1) for i, c in enumerate(self.final_chunks)]

        return {
            "query": self.query,
            "timestamp": self.timestamp,
            "bi_top_k": self.bi_top_k,
            "final_top_k": self.final_top_k,
            "bi_candidates": bi_summaries,
            "rerank_scores": self.rerank_scores,
            "final_chunks": final_summaries,
        }

    def save(self, out_dir: Path) -> Path:
        """Save trace to JSON. Returns path to saved file."""
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() or c in " -" else "_" for c in self.query[:50])
        fname = f"retrieval_{ts}_{safe_query[:30]}.json"
        path = out_dir / fname
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


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
                #lists are for general sequences; arrays are for numeric computation.
                c_copy["embedding"] = np.array(emb, dtype=np.float32)
                chunks.append(c_copy)
    return chunks


def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID)


def load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL_ID)


class Retriever:
    def __init__(self, books_dir: Path = BOOKS_DIR):
        self.books_dir = books_dir
        self._chunks: List[Dict[str, Any]] | None = None
        self._embeddings: np.ndarray | None = None
        self._model = None
        self._reranker = None

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

    def _ensure_reranker(self):
        if self._reranker is None:
            self._reranker = load_cross_encoder()

    def search(
        self,
        query: str,
        books_dir: Path | None = None,
        top_k: int = TOP_K,
    ) -> List[Dict[str, Any]]:
        """Return top-k chunks by Bi-encoder similarity (no reranking)."""
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

    def search_with_rerank(
        self,
        query: str,
        bi_top_k: int = BI_TOP_K,
        final_top_k: int = FINAL_TOP_K,
        save_trace_dir: Optional[Path] = None,
    ) -> tuple[List[Dict[str, Any]], RetrievalTrace]:
        """
        Bi-encoder retrieves bi_top_k candidates, Cross-encoder reranks, returns final_top_k.
        Returns (final_chunks, retrieval_trace). If save_trace_dir is set, saves trace to JSON.
        """
        self._ensure_loaded()
        self._ensure_reranker()

        # 1. Bi-encoder: get top bi_top_k
        prefixed = BGE_QUERY_PREFIX + query
        q_emb = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        q_vec = np.asarray(q_emb, dtype=np.float32)
        scores = np.dot(self._embeddings, q_vec)
        top_indices = np.argsort(scores)[::-1][:bi_top_k]

        bi_candidates: List[Dict[str, Any]] = []
        for i in top_indices:
            c = dict(self._chunks[i])
            text = (c.get("text") or "").strip()
            if len(text) < MIN_CHUNK_TEXT_LEN:
                continue
            c["embedding"] = c["embedding"].tolist()
            c["bi_score"] = float(scores[i])
            bi_candidates.append(c)

        if not bi_candidates:
            trace = RetrievalTrace(
                query=query,
                timestamp=datetime.now(timezone.utc).isoformat(),
                bi_top_k=bi_top_k,
                final_top_k=final_top_k,
                bi_candidates=[],
                rerank_scores=[],
                final_chunks=[],
            )
            if save_trace_dir:
                trace.save(save_trace_dir)
            return [], trace

        # 2. Cross-encoder: rerank pairs (query, passage)
        pairs = [(query, (c.get("text") or "").strip()) for c in bi_candidates]
        rerank_scores = self._reranker.predict(pairs, show_progress_bar=False)
        if isinstance(rerank_scores, np.ndarray):
            rerank_scores = rerank_scores.tolist()

        # 3. Sort by rerank score, take final_top_k
        ranked = sorted(
            zip(bi_candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:final_top_k]

        final_chunks = []
        for c, sc in ranked:
            c = dict(c)
            c["rerank_score"] = float(sc)
            c["score"] = float(sc)  # Override for downstream compatibility
            final_chunks.append(c)

        trace = RetrievalTrace(
            query=query,
            timestamp=datetime.now(timezone.utc).isoformat(),
            bi_top_k=bi_top_k,
            final_top_k=final_top_k,
            bi_candidates=bi_candidates,
            rerank_scores=[float(s) for s in rerank_scores],
            final_chunks=final_chunks,
        )
        if save_trace_dir:
            trace.save(save_trace_dir)

        return final_chunks, trace


def retrieve(
    query: str,
    books_dir: Path | None = None,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """Convenience: Bi-encoder only, top-k chunks."""
    r = Retriever(books_dir=books_dir or BOOKS_DIR)
    return r.search(query, top_k=top_k)


def retrieve_with_rerank(
    query: str,
    books_dir: Path | None = None,
    bi_top_k: int = BI_TOP_K,
    final_top_k: int = FINAL_TOP_K,
    save_trace_dir: Optional[Path] = None,
) -> tuple[List[Dict[str, Any]], RetrievalTrace]:
    """Convenience: Bi-encoder -> Cross-encoder rerank. Returns (chunks, trace)."""
    r = Retriever(books_dir=books_dir or BOOKS_DIR)
    return r.search_with_rerank(
        query,
        bi_top_k=bi_top_k,
        final_top_k=final_top_k,
        save_trace_dir=save_trace_dir,
    )


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is justice?"
    trace_dir = PROJECT_ROOT / "retrieval_logs"
    chunks, trace = retrieve_with_rerank(q, save_trace_dir=trace_dir)
    print(f"Query: {q}\n")
    print(f"Bi-encoder selected {len(trace.bi_candidates)} candidates -> Cross-encoder -> {len(chunks)} final\n")
    for i, r in enumerate(chunks, 1):
        book = r.get("book_id", "?")
        speaker = r.get("speaker", "?")
        s, e = r.get("start_line"), r.get("end_line")
        bi_sc = r.get("bi_score", 0)
        rr_sc = r.get("rerank_score", 0)
        text = (r.get("text", "") or "")[:200] + "..." if len(r.get("text", "") or "") > 200 else (r.get("text", "") or "")
        print(f"[{i}] [{book}, {speaker}, lines {s}-{e}] bi={bi_sc:.4f} rerank={rr_sc:.4f}\n{text}\n")
    print(f"Trace saved to {trace_dir}")
