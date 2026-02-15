#!/usr/bin/env python3
"""
Hybrid chunking approach for RAG implementation.
Chunks the Apology.text file by paragraphs with overlap between chunks.
"""

import json
import re
from typing import List, Dict, Tuple
from pathlib import Path


def section_to_thematic_division(section: str) -> str:
    """Map section name to thematic_division (snake_case)."""
    mapping = {
        "Main Defense": "main_defense",
        "Verdict": "verdict",
        "Proposal for Sentence": "proposal_for_sentence",
        "Death Sentence": "death_sentence",
        "Comments on Sentence": "comments_on_sentence",
    }
    return mapping.get(section, "main_defense")


def identify_sections(text_lines: List[str]) -> Dict[int, str]:
    """Identify section markers and their line numbers. Returns line -> section name."""
    sections = {}
    current_section = "Main Defense"
    
    for i, line in enumerate(text_lines, 1):
        line_stripped = line.strip()
        
        # Identify section markers
        if "(The jury finds Socrates guilty.)" in line_stripped:
            sections[i] = "Verdict"
            current_section = "Proposal for Sentence"
        elif "Socrates' Proposal for his Sentence" in line_stripped:
            sections[i] = "Proposal for Sentence"
            current_section = "Proposal for Sentence"
        elif "(The jury condemns Socrates to death.)" in line_stripped:
            sections[i] = "Death Sentence"
            current_section = "Comments on Sentence"
        elif "Socrates' Comments on his Sentence" in line_stripped:
            sections[i] = "Comments on Sentence"
            current_section = "Comments on Sentence"
        else:
            sections[i] = current_section
    
    return sections


def split_into_paragraphs(text_lines: List[str]) -> List[Tuple[int, int, str]]:
    """Split text into paragraphs, returning (start_line, end_line, text) tuples."""
    paragraphs = []
    current_para_start = 1
    current_para_lines = []
    
    for i, line in enumerate(text_lines, 1):
        line_stripped = line.strip()
        
        # Check for section markers (these are their own paragraphs)
        if (line_stripped.startswith("(") and line_stripped.endswith(")")) or \
           "Socrates'" in line_stripped and ("Proposal" in line_stripped or "Comments" in line_stripped):
            # Save current paragraph if exists
            if current_para_lines:
                para_text = " ".join(current_para_lines).strip()
                if para_text:
                    paragraphs.append((current_para_start, i - 1, para_text))
                current_para_lines = []
            
            # Add section marker as its own paragraph
            if line_stripped:
                paragraphs.append((i, i, line_stripped))
            current_para_start = i + 1
        elif line_stripped == "":
            # Empty line indicates paragraph break; include this line in range so no line is deleted
            if current_para_lines:
                para_text = " ".join(current_para_lines).strip()
                if para_text:
                    paragraphs.append((current_para_start, i, para_text))  # end_line includes empty line i
                current_para_lines = []
            else:
                # Empty line right after a section marker: extend previous paragraph range to include it
                if paragraphs and paragraphs[-1][1] == i - 1:
                    prev = paragraphs[-1]
                    paragraphs[-1] = (prev[0], i, prev[2])
            current_para_start = i + 1
        else:
            current_para_lines.append(line_stripped)
    
    # Add the last paragraph if exists
    if current_para_lines:
        para_text = " ".join(current_para_lines).strip()
        if para_text:
            paragraphs.append((current_para_start, len(text_lines), para_text))
    
    return paragraphs


def get_overlap_text(paragraphs: List[Tuple[int, int, str]], 
                     para_idx: int, 
                     overlap_sentences: int = 2) -> Tuple[str, str]:
    """Get overlap text from previous and next paragraphs."""
    prev_overlap = ""
    next_overlap = ""
    
    # Get previous paragraph overlap
    if para_idx > 0:
        prev_text = paragraphs[para_idx - 1][2]
        sentences = re.split(r'[.!?]+', prev_text)
        # Get last N sentences
        prev_sentences = [s.strip() for s in sentences if s.strip()]
        if prev_sentences:
            prev_overlap = ". ".join(prev_sentences[-overlap_sentences:]) + "."
    
    # Get next paragraph overlap
    if para_idx < len(paragraphs) - 1:
        next_text = paragraphs[para_idx + 1][2]
        sentences = re.split(r'[.!?]+', next_text)
        # Get first N sentences
        next_sentences = [s.strip() for s in sentences if s.strip()]
        if next_sentences:
            next_overlap = ". ".join(next_sentences[:overlap_sentences]) + "."
    
    return prev_overlap, next_overlap


def create_chunks_with_overlap(
    paragraphs: List[Tuple[int, int, str]],
    sections: Dict[int, str],
    book_id: str = "apology",
    volume_id: str = "I",
    speaker: str = "Socrates",
    overlap_sentences: int = 2,
) -> List[Dict]:
    """Create chunks with overlap. Schema: book_id, volume_id, thematic_division, speaker, start_line, end_line, chunk_id, text."""
    chunks = []
    
    for idx, (start_line, end_line, para_text) in enumerate(paragraphs):
        # Get overlap text
        prev_overlap, next_overlap = get_overlap_text(paragraphs, idx, overlap_sentences)
        
        # Build chunk text with overlaps
        chunk_text_parts = []
        if prev_overlap:
            chunk_text_parts.append(prev_overlap)
        chunk_text_parts.append(para_text)
        if next_overlap:
            chunk_text_parts.append(next_overlap)
        
        chunk_text = " ".join(chunk_text_parts)
        
        # Thematic division from section at start of this paragraph
        section_name = sections.get(start_line, "Main Defense")
        thematic_division = section_to_thematic_division(section_name)
        
        chunk = {
            "book_id": book_id,
            "volume_id": volume_id,
            "thematic_division": thematic_division,
            "speaker": speaker,
            "start_line": start_line,
            "end_line": end_line,
            "chunk_id": idx + 1,
            "text": chunk_text,
        }
        chunks.append(chunk)
    
    return chunks


def main():
    """Main function to chunk the Apology.text file."""
    # Read the input file
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_file = project_root / "books" / "apology.txt"
    output_file = project_root / "books" / "apology_chunks.json"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        print(f"Current directory: {Path.cwd()}")
        print(f"Script directory: {script_dir}")
        return
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text_lines = f.readlines()
    
    print(f"Total lines: {len(text_lines)}")
    
    # Identify sections
    print("Identifying sections...")
    sections = identify_sections(text_lines)
    
    # Split into paragraphs
    print("Splitting into paragraphs...")
    paragraphs = split_into_paragraphs(text_lines)
    print(f"Found {len(paragraphs)} paragraphs")
    
    # Create chunks with overlap
    print("Creating chunks with overlap...")
    chunks = create_chunks_with_overlap(
        paragraphs,
        sections,
        book_id="apology",
        volume_id="I",
        speaker="Socrates",
        overlap_sentences=2,
    )
    print(f"Created {len(chunks)} chunks")
    
    # Save chunks to JSON - output as array of chunk objects
    print(f"Saving chunks to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Successfully created {len(chunks)} chunks!")
    print(f"✓ Output saved to {output_file}")
    
    # Print summary statistics
    print("\n=== Chunking Summary ===")
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print(f"Average chunk size: {sum(len(c['text'].split()) for c in chunks) / len(chunks):.1f} words")
        print(f"Min chunk size: {min(len(c['text'].split()) for c in chunks)} words")
        print(f"Max chunk size: {max(len(c['text'].split()) for c in chunks)} words")
    else:
        print("No chunks created!")
    
    # Print book_id and volume_id distribution
    book_counts = {}
    volume_counts = {}
    for chunk in chunks:
        book = chunk['book_id']
        volume = chunk['volume_id']
        book_counts[book] = book_counts.get(book, 0) + 1
        volume_counts[volume] = volume_counts.get(volume, 0) + 1
    
    print("\n=== Book Distribution ===")
    for book, count in book_counts.items():
        print(f"{book}: {count} chunks")
    
    print("\n=== Volume Distribution ===")
    for volume, count in volume_counts.items():
        print(f"{volume}: {count} chunks")


if __name__ == "__main__":
    main()
