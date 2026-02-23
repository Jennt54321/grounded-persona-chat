#!/usr/bin/env python3
"""
Chunk the Meno dialogue for RAG with high-quality citation.

Strategy (one chunk per speaker turn / utterance):
- Each chunk = one continuous utterance by one speaker.
- Preserve exact start_line/end_line for precise citations.
- Text retains speaker label at start (e.g. "Meno. Can you tell me...").
- Long single speeches are split by paragraph when exceeding MAX_CHUNK_WORDS;
  when split, chunk_id stays the same, sub_chunk_id = 1, 2, 3, ...
- Schema: book_id, volume_id, thematic_division, speaker, start_line, end_line,
  chunk_id, sub_chunk_id, text.
- volume_id = "I" (single dialogue). thematic_division = phase name for context/filtering.
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Speaker tags at start of line (full name or abbreviation; Jowett-style)
SPEAKER_PATTERN = re.compile(
    r"^(Meno\.|Socrates\.|Men\.|Soc\.|Boy\.|Any\.)\s*",
    re.IGNORECASE
)

SPEAKER_TO_NAME = {
    "meno.": "Meno",
    "socrates.": "Socrates",
    "men.": "Meno",
    "soc.": "Socrates",
    "boy.": "Slave",
    "any.": "Anytus",
}

# Line ranges for thematic_division (Meno has no section headers)
PHASE_BOUNDARIES = [
    (8, 160, "definition_of_virtue"),      # What is virtue? definitions & refutations
    (161, 630, "paradox_recollection"),   # Meno's paradox, recollection, immortality
    (631, 1050, "slave_boy"),              # Slave boy geometry demonstration
    (1051, 1293, "teachability_hypothesis"), # Return to teachability, hypothesis
    (1294, 1600, "teachability_anytus"),   # Anytus, no teachers of virtue
    (1601, 99999, "conclusion"),           # Virtue as divine gift, closing
]

MAX_CHUNK_WORDS = 350
PARA_MARKER = "\x00PARA\x00"


def get_thematic_division(line_num: int) -> str:
    """Return thematic_division for a line number (context of that passage)."""
    for start, end, phase in PHASE_BOUNDARIES:
        if start <= line_num <= end:
            return phase
    return "conclusion"


def find_content_start(lines: List[str]) -> int:
    """Return index of first line of dialogue (after header/separator)."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if SPEAKER_PATTERN.match(stripped):
            return i
    return 0


def is_speaker_line(line: str) -> Optional[str]:
    """If line starts with Men./Soc./Boy./Any., return the tag (lowercase); else None."""
    stripped = line.strip()
    m = SPEAKER_PATTERN.match(stripped)
    if m:
        return m.group(1).lower()
    return None


def parse_speaker_tag(tag: str) -> str:
    """Map speaker tag to display name."""
    return SPEAKER_TO_NAME.get(tag.lower(), tag)


def _emit_turn_chunks(
    chunks: List[Dict],
    chunk_id: int,
    speaker: str,
    turn_lines: List[Tuple[int, str]],
) -> None:
    """
    Emit one or more chunks for a single speaker turn.
    If turn exceeds MAX_CHUNK_WORDS, split by paragraph; chunk_id stays same,
    sub_chunk_id = 1, 2, 3, ...
    """
    full_text = " ".join(t for _, t in turn_lines if t != PARA_MARKER)
    word_count = len(full_text.split())
    block_start = turn_lines[0][0]
    # Include trailing empty line in range so no line is deleted (verify_all_chunks)
    end_line = turn_lines[-1][0]
    prefixed = f"{speaker}. {full_text}"

    if word_count <= MAX_CHUNK_WORDS:
        thematic_division = get_thematic_division(block_start)
        chunks.append({
            "book_id": "meno",
            "volume_id": "I",
            "thematic_division": thematic_division,
            "speaker": speaker,
            "start_line": block_start,
            "end_line": end_line,
            "chunk_id": chunk_id,
            "sub_chunk_id": 1,
            "text": prefixed,
        })
        return

    # Split by paragraph; chunk_id stays same, sub_chunk_id = 1, 2, 3, ...
    paragraphs: List[Tuple[int, int, str]] = []
    para_start = block_start
    para_buf: List[Tuple[int, str]] = []

    for src_line, text in turn_lines:
        if text == PARA_MARKER:
            if para_buf:
                # Include empty line src_line in range so no line is deleted
                paragraphs.append((para_start, src_line, " ".join(t for _, t in para_buf)))
            para_buf = []
            para_start = src_line + 1
        else:
            para_buf.append((src_line, text))
    if para_buf:
        paragraphs.append((para_start, para_buf[-1][0], " ".join(t for _, t in para_buf)))

    if not paragraphs:
        return

    group: List[Tuple[int, int, str]] = []
    group_words = 0
    group_start = paragraphs[0][0]
    sub_chunk_idx = 1

    for start_ln, end_ln, text in paragraphs:
        words = len(text.split())
        if group_words + words > MAX_CHUNK_WORDS and group:
            group_text = " ".join(t for _, _, t in group)
            thematic_division = get_thematic_division(group_start)
            chunks.append({
                "book_id": "meno",
                "volume_id": "I",
                "thematic_division": thematic_division,
                "speaker": speaker,
                "start_line": group_start,
                "end_line": group[-1][1],
                "chunk_id": chunk_id,
                "sub_chunk_id": sub_chunk_idx,
                "text": f"{speaker}. {group_text}",
            })
            sub_chunk_idx += 1
            group = []
            group_words = 0
            group_start = start_ln
        group.append((start_ln, end_ln, text))
        group_words += words

    if group:
        group_text = " ".join(t for _, _, t in group)
        thematic_division = get_thematic_division(group_start)
        chunks.append({
            "book_id": "meno",
            "volume_id": "I",
            "thematic_division": thematic_division,
            "speaker": speaker,
            "start_line": group_start,
            "end_line": group[-1][1],
            "chunk_id": chunk_id,
            "sub_chunk_id": sub_chunk_idx,
            "text": f"{speaker}. {group_text}",
        })


def chunk_meno(text_lines: List[str]) -> List[Dict]:
    """
    Chunk Meno by speaker turn. chunk_id = turn index; when split, sub_chunk_id = 1, 2, 3, ...
    """
    chunks: List[Dict] = []
    content_start = find_content_start(text_lines)
    content_lines = text_lines[content_start:]
    line_offset = content_start
    chunk_id = 0

    i = 0

    while i < len(content_lines):
        line = content_lines[i]
        source_line = line_offset + i + 1
        stripped = line.strip()

        speaker = is_speaker_line(line)
        if speaker is not None:
            name = parse_speaker_tag(speaker)
            rest = SPEAKER_PATTERN.sub("", stripped, count=1).strip()
            turn_text = rest
            turn_lines: List[Tuple[int, str]] = [(source_line, rest)] if rest else []
            i += 1
            while i < len(content_lines):
                next_line = content_lines[i]
                next_src = line_offset + i + 1
                next_stripped = next_line.strip()
                if is_speaker_line(next_line) is not None:
                    break
                if not next_stripped:
                    if turn_lines and turn_lines[-1][1] != PARA_MARKER:
                        turn_lines.append((next_src, PARA_MARKER))
                    i += 1
                    continue
                turn_lines.append((next_src, next_stripped))
                turn_text += " " + next_stripped
                i += 1

            if turn_lines:
                chunk_id += 1
                _emit_turn_chunks(chunks, chunk_id, name, turn_lines)
            continue

        if not stripped:
            i += 1
            continue
        i += 1

    return chunks


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_file = project_root / "books" / "meno.txt"
    output_file = project_root / "books" / "meno_chunks.json"

    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return

    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        text_lines = f.readlines()

    content_start = find_content_start(text_lines)
    print(f"Total lines: {len(text_lines)}, content starts at line {content_start + 1}")
    print("Chunking by speaker blocks (Men./Soc./Boy./Any.)...")

    chunks = chunk_meno(text_lines)
    print(f"Created {len(chunks)} chunks")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {output_file}")

    print("\n=== Chunking Summary ===")
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        split_count = sum(1 for c in chunks if c.get("sub_chunk_id", 1) > 1)
        print(f"Chunks with sub_chunk_id > 1: {split_count}")
        avg = sum(len(c["text"].split()) for c in chunks) / len(chunks)
        print(f"Average chunk size: {avg:.1f} words")
        print(f"Min words: {min(len(c['text'].split()) for c in chunks)}")
        print(f"Max words: {max(len(c['text'].split()) for c in chunks)}")
        by_division = {}
        for c in chunks:
            d = c["thematic_division"]
            by_division[d] = by_division.get(d, 0) + 1
        print("\nChunks per thematic_division:", dict(sorted(by_division.items())))


if __name__ == "__main__":
    main()
