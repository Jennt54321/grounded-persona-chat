#!/usr/bin/env python3
"""
Chunk the Republic.text file for RAG implementation.
Skips the introduction (lines 1-572), chunks by Book and speaker blocks.
Schema: book_id, volume_id, thematic_division, speakers, start_line, end_line, chunk_id, text
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Introduction ends before line 573 (BOOK I starts at 573)
INTRODUCTION_END_LINE = 572

# Thematic divisions (from the introduction):
# 1: Book I + first half of Book II (introductory)
# 2: Remainder of II, III, IV (first State, first education)
# 3: V, VI, VII (philosophy, second State)
# 4: VIII, IX (perversions, tyranny)
# 5: X (conclusion)
BOOK_TO_DIVISION = {
    "I": 1,
    "II": 2,   # Simplified: full Book II in division 2
    "III": 2,
    "IV": 2,
    "V": 3,
    "VI": 3,
    "VII": 3,
    "VIII": 4,
    "IX": 4,
    "X": 5,
}

# Max words per chunk before splitting by paragraph
MAX_CHUNK_WORDS = 450


def is_speaker_line(line: str) -> bool:
    """Check if line is a speaker marker, e.g. 'Socrates - GLAUCON' or 'Glaucon - CEPHALUS - SOCRATES'."""
    stripped = line.strip()
    # Pattern: Word(s) - WORD(s) or Word - WORD - WORD
    return bool(re.match(r"^[A-Za-z]+\s+-\s+[A-Za-z]+(\s+-\s+[A-Za-z]+)*\s*$", stripped)) and " - " in stripped


def parse_speakers(speaker_line: str) -> List[str]:
    """Extract speaker names from a speaker line."""
    parts = [p.strip() for p in speaker_line.split(" - ") if p.strip()]
    # Normalize: use title case for consistency
    return [p.title() if p.isupper() else p for p in parts]


def is_book_header(line: str) -> Optional[str]:
    """Check if line is a BOOK header, return Roman numeral or None."""
    stripped = line.strip()
    match = re.match(r"^BOOK\s+(I{1,3}|IV|V|VI{1,3}|IX|X|XI)$", stripped, re.IGNORECASE)
    return match.group(1).upper() if match else None


def is_separator(line: str) -> bool:
    """Check if line is the ------ separator."""
    return bool(re.match(r"^-+\s*$", line.strip()))


def get_thematic_division(book: str) -> int:
    """Get thematic division (1-5) for a book."""
    return BOOK_TO_DIVISION.get(book.upper(), 1)


PARA_MARKER = "\x00PARA\x00"
SEP_MARKER = "\x00SEP\x00"
BOOK_MARKER = "\x00BOOK\x00"
SPEAKER_LINE_MARKER = "\x00SPEAKER\x00"


def chunk_republic(text_lines: List[str]) -> List[Dict]:
    """
    Chunk the Republic text (lines after introduction) by Book and speaker blocks.
    Returns list of chunk dicts with the specified schema.
    """
    chunks = []
    chunk_id = 0

    # Use only content after introduction
    content_start = min(INTRODUCTION_END_LINE, len(text_lines))
    content_lines = text_lines[content_start:]
    line_offset = content_start

    i = 0
    current_book = "I"
    current_speakers: List[str] = []
    current_block_lines: List[Tuple[int, str]] = []  # (source_line, text or PARA_MARKER)
    current_block_start = line_offset + 1
    prev_was_blank = False

    while i < len(content_lines):
        line = content_lines[i]
        source_line = line_offset + i + 1
        stripped = line.strip()

        # Check for BOOK header: put BOOK line in next chunk; extend previous chunk to include line before BOOK (separator/blank)
        book_num = is_book_header(stripped)
        if book_num:
            if current_block_lines:
                chunk_id = _flush_block(
                    chunks, chunk_id, current_book, current_speakers,
                    current_block_lines, current_block_start,
                    end_line_include_until=source_line - 1,
                )
                current_block_lines = []

            current_book = book_num
            current_block_lines = [(source_line, BOOK_MARKER)]
            current_block_start = source_line
            prev_was_blank = False
            i += 1
            continue

        # Check for speaker line: put speaker line in next chunk's range
        if is_speaker_line(stripped):
            if current_block_lines:
                chunk_id = _flush_block(
                    chunks, chunk_id, current_book, current_speakers,
                    current_block_lines, current_block_start,
                )
                current_block_lines = []

            current_speakers = parse_speakers(stripped)
            current_block_lines = [(source_line, SPEAKER_LINE_MARKER)]
            current_block_start = source_line
            prev_was_blank = False
            i += 1
            continue

        # Blank line: mark paragraph boundary (include in range)
        if not stripped:
            if not current_block_lines:
                current_block_start = source_line
                current_block_lines.append((source_line, PARA_MARKER))
            elif current_block_lines[-1][1] != PARA_MARKER:
                current_block_lines.append((source_line, PARA_MARKER))
            prev_was_blank = True
            i += 1
            continue

        # Separator: include in current block range so no line is deleted
        if is_separator(stripped):
            current_block_lines.append((source_line, SEP_MARKER))
            i += 1
            continue

        # Regular content
        current_block_lines.append((source_line, stripped))
        prev_was_blank = False
        i += 1

    if current_block_lines:
        _flush_block(
            chunks, chunk_id, current_book, current_speakers,
            current_block_lines, current_block_start,
        )

    return chunks


def _flush_block(
    chunks: List[Dict],
    chunk_id: int,
    book: str,
    speakers: List[str],
    block_lines: List[Tuple[int, str]],
    block_start: int,
    end_line_include_until: Optional[int] = None,
) -> int:
    """
    Flush a speaker block to one or more chunks.
    Splits by paragraph if block exceeds MAX_CHUNK_WORDS.
    end_line_include_until: if set, extend chunk end_line to include this line (BOOK/speaker).
    """
    # Build full text (skip structural markers)
    full_text = " ".join(
        text for _, text in block_lines
        if text not in (PARA_MARKER, SEP_MARKER, BOOK_MARKER, SPEAKER_LINE_MARKER)
    )
    word_count = len(full_text.split())

    end_line = block_lines[-1][0]  # include trailing blank/separator in range
    if end_line_include_until is not None:
        end_line = max(end_line, end_line_include_until)

    if word_count <= MAX_CHUNK_WORDS or not block_lines:
        chunk_id += 1
        chunk = {
            "book_id": "republic",
            "volume_id": book,
            "thematic_division": get_thematic_division(book),
            "speakers": speakers,
            "start_line": block_start,
            "end_line": end_line,
            "chunk_id": chunk_id,
            "text": full_text,
        }
        chunks.append(chunk)
        return chunk_id

    # Split by PARA_MARKER into paragraphs (include blank/sep in range)
    paragraphs: List[Tuple[int, int, str]] = []
    para_start = block_start
    para_lines = []

    for src_line, text in block_lines:
        if text in (PARA_MARKER, SEP_MARKER, BOOK_MARKER, SPEAKER_LINE_MARKER):
            if para_lines:
                paragraphs.append((para_start, src_line, " ".join(t for _, t in para_lines)))
            para_lines = []
            para_start = src_line + 1
        else:
            para_lines.append((src_line, text))

    if para_lines:
        paragraphs.append((para_start, block_lines[-1][0], " ".join(t for _, t in para_lines if t != SEP_MARKER)))

    if not paragraphs:
        return chunk_id

    # Merge paragraphs to stay under MAX_CHUNK_WORDS; first chunk starts at block_start so BOOK/speaker/blank lines are covered
    current_group: List[Tuple[int, int, str]] = []
    current_words = 0
    group_start = block_start

    for start_ln, end_ln, text in paragraphs:
        words = len(text.split())
        if current_words + words > MAX_CHUNK_WORDS and current_group:
            group_text = " ".join(t for _, _, t in current_group)
            chunk_id += 1
            chunk = {
                "book_id": "republic",
                "volume_id": book,
                "thematic_division": get_thematic_division(book),
                "speakers": speakers,
                "start_line": group_start,
                "end_line": current_group[-1][1],
                "chunk_id": chunk_id,
                "text": group_text,
            }
            chunks.append(chunk)
            current_group = []
            current_words = 0
            group_start = start_ln

        current_group.append((start_ln, end_ln, text))
        current_words += words

    if current_group:
        group_text = " ".join(t for _, _, t in current_group)
        chunk_id += 1
        last_end = max(current_group[-1][1], block_lines[-1][0])  # include trailing blank/sep in range
        if end_line_include_until is not None:
            last_end = max(last_end, end_line_include_until)
        chunks.append({
            "book_id": "republic",
            "volume_id": book,
            "thematic_division": get_thematic_division(book),
            "speakers": speakers,
            "start_line": group_start,
            "end_line": last_end,
            "chunk_id": chunk_id,
            "text": group_text,
        })

    return chunk_id


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_file = project_root / "books" / "republic.txt"
    output_file = project_root / "books" / "republic_chunks.json"

    print(f"Reading {input_file}...")
    text_lines = []

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text_lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        try:
            with open(input_file, "rb") as f:
                content = f.read()
            text_lines = content.decode("utf-8", errors="ignore").splitlines(keepends=True)
        except Exception as e2:
            print(f"Error with binary read: {e2}")
            return

    if not text_lines:
        print("Error: No content found in file!")
        return

    if len(text_lines) < INTRODUCTION_END_LINE + 10:
        print(f"Error: File has only {len(text_lines)} lines. Need at least {INTRODUCTION_END_LINE + 10} lines (intro + start of BOOK I).")
        return

    print(f"Total lines read: {len(text_lines)}")
    print(f"Skipping introduction (lines 1-{INTRODUCTION_END_LINE})")
    print("Chunking by Book and speaker blocks...")

    chunks = chunk_republic(text_lines)
    print(f"Created {len(chunks)} chunks")

    # Save as array of chunk objects (matching apology_chunks format)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {output_file}")

    # Summary
    print("\n=== Chunking Summary ===")
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        avg_words = sum(len(c["text"].split()) for c in chunks) / len(chunks)
        print(f"Average chunk size: {avg_words:.1f} words")
        by_volume = {}
        for c in chunks:
            v = c["volume_id"]
            by_volume[v] = by_volume.get(v, 0) + 1
        vol_order = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
        sorted_vols = sorted(by_volume.items(), key=lambda x: (vol_order.get(x[0], 99), x[0]))
        print("\nChunks per book:", dict(sorted_vols))


if __name__ == "__main__":
    main()
