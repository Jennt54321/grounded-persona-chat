#!/usr/bin/env python3
"""
Verify that *_chunks.json files are identical to their source *.txt:
- Order of every line preserved
- No line added (except overlap where applicable)
- No line deleted

Supports: apology, gorgias, meno, republic.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set


def normalize_whitespace(s: str) -> str:
    return " ".join(s.split())


# Speaker prefix patterns: for file lines (strip tag) and for chunk text (strip "Speaker. ")
GORGIAS_SPEAKER_PATTERN = re.compile(
    r"^(Callicles|Socrates|Chaerephon|Gorgias|Polus|Cal|Soc|Chaer|Gor|Pol)\.\s*",
    re.IGNORECASE,
)
# File has Meno./Socrates./Boy./Any. or Men./Soc.; chunk has Meno/Socrates/Slave/Anytus
MENO_FILE_SPEAKER_PATTERN = re.compile(
    r"^(Meno|Socrates|Men|Soc|Boy|Any)\.\s*",
    re.IGNORECASE,
)
MENO_CHUNK_SPEAKER_PATTERN = re.compile(
    r"^(Meno|Socrates|Slave|Anytus)\.\s*",
    re.IGNORECASE,
)


def verify_apology(project_root: Path) -> List[str]:
    """Verify apology_chunks.json vs apology.txt."""
    txt_path = project_root / "books" / "apology.txt"
    json_path = project_root / "books" / "apology_chunks.json"
    errors = []
    with open(txt_path, "r", encoding="utf-8") as f:
        file_lines = f.readlines()
    total = len(file_lines)
    file_lines_1 = {i: line.rstrip("\n") for i, line in enumerate(file_lines, 1)}
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = sorted(json.load(f), key=lambda c: c["chunk_id"])

    covered = set()
    for c in chunks:
        for ln in range(c["start_line"], c["end_line"] + 1):
            covered.add(ln)
    missing = set(range(1, total + 1)) - covered
    if missing:
        err = f"DELETED LINES: {sorted(missing)[:25]}"
        if len(missing) > 25:
            err += f" ... and {len(missing) - 25} more"
        errors.append(err)

    prev_end = 0
    for c in chunks:
        s, e = c["start_line"], c["end_line"]
        if prev_end > 0 and s < prev_end + 1:
            errors.append(f"ORDER: chunk_id {c['chunk_id']} start={s} prev_end={prev_end}")
        prev_end = max(prev_end, e)

    for c in chunks:
        s, e = c["start_line"], c["end_line"]
        para_parts = []
        for ln in range(s, e + 1):
            if ln in file_lines_1 and file_lines_1[ln].strip():
                para_parts.append(file_lines_1[ln].strip())
        expected = " ".join(para_parts)
        if expected and normalize_whitespace(expected) not in normalize_whitespace(c["text"]):
            errors.append(f"CHUNK {c['chunk_id']} (lines {s}-{e}): file content not in chunk text")

    return errors


def _expected_content_dialogue(
    file_lines_1: Dict[int, str],
    start_line: int,
    end_line: int,
    speaker_prefix_pattern: re.Pattern,
) -> str:
    """Build expected chunk content from file lines; strip speaker tag from lines that have it."""
    parts = []
    for ln in range(start_line, end_line + 1):
        if ln not in file_lines_1:
            continue
        line = file_lines_1[ln].strip()
        if not line:
            continue
        line = speaker_prefix_pattern.sub("", line, count=1).strip()
        parts.append(line)
    return " ".join(parts)


def verify_gorgias(project_root: Path) -> List[str]:
    """Verify gorgias_chunks.json vs gorgias.txt. Content starts at first chunk's start_line."""
    txt_path = project_root / "books" / "gorgias.txt"
    json_path = project_root / "books" / "gorgias_chunks.json"
    errors = []
    with open(txt_path, "r", encoding="utf-8") as f:
        file_lines = f.readlines()
    total = len(file_lines)
    file_lines_1 = {i: line.rstrip("\n") for i, line in enumerate(file_lines, 1)}
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chunks_sorted = sorted(chunks, key=lambda c: (c["dialogue_id"], c["chunk_id"]))

    min_start = min(c["start_line"] for c in chunks)
    required_range = set(range(min_start, total + 1))
    covered = set()
    for c in chunks_sorted:
        for ln in range(c["start_line"], c["end_line"] + 1):
            covered.add(ln)
    missing = required_range - covered
    if missing:
        err = f"DELETED LINES (in content range): {sorted(missing)[:25]}"
        if len(missing) > 25:
            err += f" ... and {len(missing) - 25} more"
        errors.append(err)

    prev_end = 0
    for c in chunks_sorted:
        s, e = c["start_line"], c["end_line"]
        if prev_end > 0 and s < prev_end + 1:
            errors.append(f"ORDER: dialogue_id={c['dialogue_id']} chunk_id={c['chunk_id']} start={s} prev_end={prev_end}")
        prev_end = max(prev_end, e)

    for c in chunks_sorted:
        s, e = c["start_line"], c["end_line"]
        expected = _expected_content_dialogue(file_lines_1, s, e, GORGIAS_SPEAKER_PATTERN)
        chunk_text = c["text"]
        chunk_content = GORGIAS_SPEAKER_PATTERN.sub("", chunk_text, count=1).strip()
        if expected and normalize_whitespace(expected) not in normalize_whitespace(chunk_content):
            errors.append(f"CHUNK dialogue_id={c['dialogue_id']} chunk_id={c['chunk_id']} (lines {s}-{e}): file content not in chunk")

    return errors


def verify_meno(project_root: Path) -> List[str]:
    """Verify meno_chunks.json vs meno.txt."""
    txt_path = project_root / "books" / "meno.txt"
    json_path = project_root / "books" / "meno_chunks.json"
    errors = []
    with open(txt_path, "r", encoding="utf-8") as f:
        file_lines = f.readlines()
    total = len(file_lines)
    file_lines_1 = {i: line.rstrip("\n") for i, line in enumerate(file_lines, 1)}
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = sorted(json.load(f), key=lambda c: c["chunk_id"])

    min_start = min(c["start_line"] for c in chunks)
    required_range = set(range(min_start, total + 1))
    covered = set()
    for c in chunks:
        for ln in range(c["start_line"], c["end_line"] + 1):
            covered.add(ln)
    missing = required_range - covered
    if missing:
        err = f"DELETED LINES (in content range): {sorted(missing)[:25]}"
        if len(missing) > 25:
            err += f" ... and {len(missing) - 25} more"
        errors.append(err)

    prev_end = 0
    for c in chunks:
        s, e = c["start_line"], c["end_line"]
        if prev_end > 0 and s < prev_end + 1:
            errors.append(f"ORDER: chunk_id {c['chunk_id']} start={s} prev_end={prev_end}")
        prev_end = max(prev_end, e)

    for c in chunks:
        s, e = c["start_line"], c["end_line"]
        expected = _expected_content_dialogue(file_lines_1, s, e, MENO_FILE_SPEAKER_PATTERN)
        chunk_text = c["text"]
        chunk_content = MENO_CHUNK_SPEAKER_PATTERN.sub("", chunk_text, count=1).strip()
        if expected and normalize_whitespace(expected) not in normalize_whitespace(chunk_content):
            errors.append(f"CHUNK {c['chunk_id']} (lines {s}-{e}): file content not in chunk")

    return errors


# Republic: introduction lines 1-572 are skipped by design; only 573+ must be covered
REPUBLIC_CONTENT_START = 573
REPUBLIC_BOOK_HEADER = re.compile(r"^BOOK\s+(I{1,3}|IV|V|VI{1,3}|IX|X|XI)\s*$", re.IGNORECASE)
REPUBLIC_SPEAKER_LINE = re.compile(r"^[A-Za-z]+\s+-\s+[A-Za-z]+(\s+-\s+[A-Za-z]+)*\s*$")
REPUBLIC_SEPARATOR = re.compile(r"^-+\s*$")


def _republic_is_structural(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if REPUBLIC_BOOK_HEADER.match(s):
        return True
    if " - " in s and REPUBLIC_SPEAKER_LINE.match(s):
        return True
    if REPUBLIC_SEPARATOR.match(s):
        return True
    return False


def verify_republic(project_root: Path) -> List[str]:
    """Verify republic_chunks.json vs republic.txt. Only lines 573+ are chunked. Chunk text is dialogue only."""
    txt_path = project_root / "books" / "republic.txt"
    json_path = project_root / "books" / "republic_chunks.json"
    errors = []
    with open(txt_path, "r", encoding="utf-8") as f:
        file_lines = f.readlines()
    total = len(file_lines)
    file_lines_1 = {i: line.rstrip("\n") for i, line in enumerate(file_lines, 1)}
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = sorted(json.load(f), key=lambda c: c["chunk_id"])

    required_range = set(range(REPUBLIC_CONTENT_START, total + 1))
    covered = set()
    for c in chunks:
        for ln in range(c["start_line"], c["end_line"] + 1):
            covered.add(ln)
    missing = required_range - covered
    if missing:
        err = f"DELETED LINES (in content range 573+): {sorted(missing)[:25]}"
        if len(missing) > 25:
            err += f" ... and {len(missing) - 25} more"
        errors.append(err)

    prev_end = 0
    for c in chunks:
        s, e = c["start_line"], c["end_line"]
        if prev_end > 0 and s < prev_end + 1:
            errors.append(f"ORDER: chunk_id {c['chunk_id']} start={s} prev_end={prev_end}")
        prev_end = max(prev_end, e)

    # Chunk text is dialogue only (no BOOK/speaker/separator/blank); compare dialogue lines only
    for c in chunks:
        s, e = c["start_line"], c["end_line"]
        para_parts = []
        for ln in range(s, e + 1):
            if ln not in file_lines_1:
                continue
            line = file_lines_1[ln].strip()
            if line and not _republic_is_structural(line):
                para_parts.append(line)
        expected = " ".join(para_parts)
        if expected and normalize_whitespace(expected) not in normalize_whitespace(c["text"]):
            errors.append(f"CHUNK {c['chunk_id']} (lines {s}-{e}): file content not in chunk text")

    return errors


def main():
    project_root = Path(__file__).resolve().parent.parent
    books = [
        ("apology", verify_apology),
        ("gorgias", verify_gorgias),
        ("meno", verify_meno),
        ("republic", verify_republic),
    ]
    failed = 0
    for name, verify_fn in books:
        print(f"\n=== {name.upper()} ===")
        errors = verify_fn(project_root)
        if errors:
            print("VERIFICATION FAILED:")
            for e in errors:
                print("  -", e)
            failed += 1
        else:
            print("OK: All checks passed (order preserved, no line deleted).")
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    exit(main())
