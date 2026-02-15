#!/usr/bin/env python3
"""
Chunk the Gorgias dialogue for RAG with high-quality citation.

Strategy (one chunk per speaker turn / utterance):
- Each dialogue_id = one continuous utterance by one speaker.
- If a speaker turn exceeds MAX_CHUNK_WORDS, it is split into multiple chunks
  (chunk_id 1, 2, 3, ... within that dialogue_id).
- Preserve exact start_line/end_line for precise citations.
- Text retains speaker label at start (e.g. "Socrates. ...").
- Schema: book_id, volume_id, thematic_division, speaker, start_line, end_line,
  dialogue_id, chunk_id, text.
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Speaker tags at start of line (full name or abbreviation; Jowett-style)
# Order: longer names first so "Socrates." matches before "Soc."
SPEAKER_PATTERN = re.compile(
    r"^(Callicles\.|Socrates\.|Chaerephon\.|Gorgias\.|Polus\.|Cal\.|Soc\.|Chaer\.|Gor\.|Pol\.)\s*",
    re.IGNORECASE,
)

SPEAKER_TO_NAME = {
    "callicles.": "Callicles",
    "socrates.": "Socrates",
    "chaerephon.": "Chaerephon",
    "gorgias.": "Gorgias",
    "polus.": "Polus",
    "cal.": "Callicles",
    "soc.": "Socrates",
    "chaer.": "Chaerephon",
    "gor.": "Gorgias",
    "pol.": "Polus",
}

# Line ranges for thematic_division (Gorgias structure)
PHASE_BOUNDARIES = [
    (13, 626, "opening"),
    (627, 814, "gorgias_on_rhetoric"),
    (815, 1220, "polus_power_injustice"),
    (1221, 99999, "callicles_nature_convention"),
]

MAX_CHUNK_WORDS = 400
PARA_MARKER = "\x00PARA\x00"


def get_thematic_division(line_num: int) -> str:
    """Return thematic_division for a line number."""
    for start, end, phase in PHASE_BOUNDARIES:
        if start <= line_num <= end:
            return phase
    return "callicles_nature_convention"


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
    """If line starts with a speaker tag, return the tag (lowercase); else None."""
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
    dialogue_id: int,
    speaker: str,
    turn_lines: List[Tuple[int, str]],
) -> None:
    """
    Emit one or more chunks for a single speaker turn (one dialogue_id).
    If turn exceeds MAX_CHUNK_WORDS, split by paragraph into chunk_id 1, 2, ...
    """
    if not turn_lines:
        return
    full_text = " ".join(t for _, t in turn_lines if t != PARA_MARKER)
    word_count = len(full_text.split())
    block_start = turn_lines[0][0]
    # Include trailing empty line in range so no line is deleted (verify_all_chunks)
    end_line = turn_lines[-1][0]
    prefixed = f"{speaker}. {full_text}"

    if word_count <= MAX_CHUNK_WORDS:
        thematic_division = get_thematic_division(block_start)
        chunks.append({
            "book_id": "gorgias",
            "volume_id": "I",
            "thematic_division": thematic_division,
            "speaker": speaker,
            "start_line": block_start,
            "end_line": end_line,
            "dialogue_id": dialogue_id,
            "chunk_id": 1,
            "text": prefixed,
        })
        return

    # Split by paragraph into sub-chunks (chunk_id 1, 2, 3, ...)
    paragraphs: List[Tuple[int, int, str]] = []
    para_start = block_start
    para_buf: List[Tuple[int, str]] = []

    for src_line, text in turn_lines:
        if text == PARA_MARKER:
            if para_buf:
                # Include empty line src_line in range so no line is deleted
                paragraphs.append(
                    (para_start, src_line, " ".join(t for _, t in para_buf))
                )
            para_buf = []
            para_start = src_line + 1
        else:
            para_buf.append((src_line, text))
    if para_buf:
        paragraphs.append(
            (para_start, para_buf[-1][0], " ".join(t for _, t in para_buf))
        )

    if not paragraphs:
        return

    group: List[Tuple[int, int, str]] = []
    group_words = 0
    group_start = paragraphs[0][0]
    chunk_idx = 1

    for start_ln, end_ln, text in paragraphs:
        words = len(text.split())
        if group_words + words > MAX_CHUNK_WORDS and group:
            group_text = " ".join(t for _, _, t in group)
            thematic_division = get_thematic_division(group_start)
            chunks.append({
                "book_id": "gorgias",
                "volume_id": "I",
                "thematic_division": thematic_division,
                "speaker": speaker,
                "start_line": group_start,
                "end_line": group[-1][1],
                "dialogue_id": dialogue_id,
                "chunk_id": chunk_idx,
                "text": f"{speaker}. {group_text}",
            })
            chunk_idx += 1
            group = []
            group_words = 0
            group_start = start_ln
        group.append((start_ln, end_ln, text))
        group_words += words

    if group:
        group_text = " ".join(t for _, _, t in group)
        thematic_division = get_thematic_division(group_start)
        chunks.append({
            "book_id": "gorgias",
            "volume_id": "I",
            "thematic_division": thematic_division,
            "speaker": speaker,
            "start_line": group_start,
            "end_line": group[-1][1],
            "dialogue_id": dialogue_id,
            "chunk_id": chunk_idx,
            "text": f"{speaker}. {group_text}",
        })


def chunk_gorgias(text_lines: List[str]) -> List[Dict]:
    """
    Chunk Gorgias by speaker turn: one dialogue_id per utterance; if an
    utterance exceeds MAX_CHUNK_WORDS, split into multiple chunks (chunk_id 1, 2, ...).
    """
    chunks: List[Dict] = []
    content_start = find_content_start(text_lines)
    content_lines = text_lines[content_start:]
    line_offset = content_start
    dialogue_id = 0
    i = 0

    while i < len(content_lines):
        line = content_lines[i]
        source_line = line_offset + i + 1
        stripped = line.strip()

        speaker = is_speaker_line(line)
        if speaker is not None:
            name = parse_speaker_tag(speaker)
            rest = SPEAKER_PATTERN.sub("", stripped, count=1).strip()
            # Include speaker-only line in range (rest may be empty)
            turn_lines: List[Tuple[int, str]] = [(source_line, rest or "")]
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
                i += 1

            if turn_lines:
                dialogue_id += 1
                _emit_turn_chunks(chunks, dialogue_id, name, turn_lines)
            continue

        if not stripped:
            i += 1
            continue
        i += 1

    return chunks


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_file = project_root / "books" / "gorgias.txt"
    output_file = project_root / "books" / "gorgias_chunks.json"

    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return

    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        text_lines = f.readlines()

    content_start = find_content_start(text_lines)
    print(
        f"Total lines: {len(text_lines)}, content starts at line {content_start + 1}"
    )
    print("Chunking by speaker blocks (Cal./Soc./Chaer./Gor./Pol./full names)...")

    chunks = chunk_gorgias(text_lines)
    print(f"Created {len(chunks)} chunks")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {output_file}")

    print("\n=== Chunking Summary ===")
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        dialogue_ids = {c["dialogue_id"] for c in chunks}
        print(f"Total dialogue_id (speaker turns): {len(dialogue_ids)}")
        split_dialogues = {c["dialogue_id"] for c in chunks if c["chunk_id"] > 1}
        print(f"Dialogues split into multiple chunks: {len(split_dialogues)}")
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
