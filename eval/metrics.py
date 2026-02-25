"""
Evaluation metrics: A (Citation Validity), B (Diversity), C (Similarity).
"""

import math
from collections import Counter
from typing import Any

from eval.citation_parser import ParsedCitation, StrictCitation, normalize_text_for_match
from eval.chunk_index import ChunkIndex, _section_key, parse_book_volume


def intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Two intervals [a_start, a_end] and [b_start, b_end] overlap iff a_start <= b_end and b_start <= a_end."""
    return a_start <= b_end and b_start <= a_end


def citation_overlaps_retrieved(citation: StrictCitation, retrieved_chunks: list[dict[str, Any]]) -> bool:
    """Check if citation range overlaps at least one retrieved chunk (same book/volume, interval overlap)."""
    cite_book, cite_vol = parse_book_volume(citation.file)
    if not cite_book:
        cite_book = citation.file.lower().strip()
    for c in retrieved_chunks:
        book = (c.get("book_id") or "").lower().strip()
        if book != cite_book:
            continue
        if cite_vol:
            c_vol = (c.get("volume_id") or "").strip().upper()
            if c_vol and c_vol != cite_vol.upper():
                continue
        try:
            s = int(c.get("start_line", 0))
            e = int(c.get("end_line", 0))
        except (TypeError, ValueError):
            continue
        if intervals_overlap(citation.start_line, citation.end_line, s, e):
            return True
    return False


def verify_citations_against_retrieved(
    parsed_citations: list[StrictCitation],
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[list[StrictCitation], list[StrictCitation], float]:
    """
    Verify each citation overlaps at least one retrieved chunk.
    Returns (passed_list, hallucinated_list, hallucination_rate).
    """
    passed: list[StrictCitation] = []
    hallucinated: list[StrictCitation] = []
    for pc in parsed_citations:
        if citation_overlaps_retrieved(pc, retrieved_chunks):
            passed.append(pc)
        else:
            hallucinated.append(pc)
    total = len(parsed_citations)
    rate = len(hallucinated) / total if total else 0.0
    return passed, hallucinated, rate


def compute_citation_validity(
    parsed_citations: list[ParsedCitation],
    chunk_index: ChunkIndex,
) -> dict[str, float]:
    """
    A1: Citation Existence Rate - % of citations that resolve in chunk index.
    A2: Exact Quote Match Rate - % of cited quotes (when present) that are substring of chunk.
    A3: Fabrication Rate - % of citations that cannot be resolved (1 - A1).
    """
    if not parsed_citations:
        return {"A1_existence_rate": 0.0, "A2_quote_match_rate": 0.0, "A3_fabrication_rate": 1.0}

    exists = 0
    quote_matches = 0
    quote_total = 0

    for pc in parsed_citations:
        book_id, volume_id = parse_book_volume(pc.book)
        chunk = chunk_index.get(book_id or pc.book, pc.start_line, pc.end_line, volume_id=volume_id or None)
        if chunk is not None:
            exists += 1
            if pc.quoted_text and pc.quoted_text.strip():
                quote_total += 1
                chunk_text = (chunk.get("text") or "").strip()
                if chunk_text:
                    norm_quoted = normalize_text_for_match(pc.quoted_text)
                    norm_chunk = normalize_text_for_match(chunk_text)
                    if norm_quoted and norm_quoted in norm_chunk:
                        quote_matches += 1

    total = len(parsed_citations)
    a1 = exists / total if total else 0.0
    a2 = quote_matches / quote_total if quote_total else 0.0
    a3 = 1.0 - a1

    return {
        "A1_existence_rate": round(a1, 4),
        "A2_quote_match_rate": round(a2, 4),
        "A3_fabrication_rate": round(a3, 4),
    }


def compute_retrieval_diversity(
    all_retrieved: list[list[dict[str, Any]]],
    chunk_index: ChunkIndex,
) -> dict[str, Any]:
    """
    B1a: Unique retrieved chunks - distinct (book_id, chunk_id, start_line, end_line).
    B1b: Unique sections - distinct (book_id, thematic_division).
    B1c: Top-k concentration - rank-1 chunk reuse count and entropy.
    """
    unique_chunk_keys: set[tuple[str, int, int] | tuple[str, str, int, int]] = set()
    unique_sections: set[tuple[str, str | int]] = set()
    rank1_chunks: list[tuple[str, int, int] | tuple[str, str, int, int]] = []

    for chunks in all_retrieved:
        for i, c in enumerate(chunks):
            key = chunk_index.chunk_key(c)
            unique_chunk_keys.add(key)
            unique_sections.add(chunk_index.section_key(c))
            if i == 0:
                rank1_chunks.append(key)

    rank1_counts = Counter(rank1_chunks)
    n_rank1 = len(rank1_chunks)
    rank1_entropy = 0.0
    if n_rank1:
        for count in rank1_counts.values():
            p = count / n_rank1
            if p > 0:
                rank1_entropy -= p * math.log2(p)

    return {
        "B1a_unique_retrieved_chunks": len(unique_chunk_keys),
        "B1b_unique_sections": len(unique_sections),
        "B1c_rank1_reuse_total": sum(rank1_counts.values()),
        "B1c_rank1_entropy": round(rank1_entropy, 4),
        "B1c_rank1_unique_count": len(rank1_counts),
    }


def compute_citation_diversity(
    all_parsed_citations: list[list[ParsedCitation]],
    chunk_index: ChunkIndex,
) -> dict[str, Any]:
    """
    B2a: Unique model citations.
    B2b: Citation entropy over sections.
    B2c: Citation reuse rate (total citations / unique citations).
    """
    all_citations: list[ParsedCitation] = []
    for citations in all_parsed_citations:
        all_citations.extend(citations)

    if not all_citations:
        return {
            "B2a_unique_citations": 0,
            "B2b_citation_entropy": 0.0,
            "B2c_citation_reuse_rate": 0.0,
        }

    unique_citation_keys: set[tuple[str, str, int, int]] = set()
    section_counts: Counter[tuple[str, str | int]] = Counter()

    for pc in all_citations:
        book_id, volume_id = parse_book_volume(pc.book)
        key = (book_id or pc.book.lower(), volume_id or "", pc.start_line, pc.end_line)
        unique_citation_keys.add(key)
        chunk = chunk_index.get(book_id or pc.book, pc.start_line, pc.end_line, volume_id=volume_id or "")
        if chunk:
            section_counts[chunk_index.section_key(chunk)] += 1
        else:
            section_counts[(pc.book.lower(), "?")] += 1

    total_citations = len(all_citations)
    reuse_rate = total_citations / len(unique_citation_keys) if unique_citation_keys else 0.0

    entropy = 0.0
    if total_citations:
        for count in section_counts.values():
            p = count / total_citations
            if p > 0:
                entropy -= p * math.log2(p)

    return {
        "B2a_unique_citations": len(unique_citation_keys),
        "B2b_citation_entropy": round(entropy, 4),
        "B2c_citation_reuse_rate": round(reuse_rate, 4),
    }


def compute_similarity(
    questions: list[str],
    all_cited_chunk_texts: list[list[str]],
    retriever: Any,
) -> list[float]:
    """
    C: BGE similarity between question and cited chunk text.
    Returns per-question mean similarity (or empty list if no citations).
    """
    import numpy as np

    sims: list[float] = []
    prefixed = "Represent this sentence for searching relevant passages: "

    for q, cited_texts in zip(questions, all_cited_chunk_texts):
        if not cited_texts:
            continue
        q_enc = retriever._model.encode(
            [prefixed + q],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        texts_enc = retriever._model.encode(
            cited_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q_vec = np.asarray(q_enc, dtype=np.float32)
        scores = np.dot(texts_enc, q_vec)
        sims.append(float(np.mean(scores)))

    return sims
