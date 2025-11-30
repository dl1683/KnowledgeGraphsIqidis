"""
Text chunking with overlapping windows.
"""
from typing import List, Tuple
from dataclasses import dataclass
import re

from ..config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """A chunk of text with position information."""
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    total_chunks: int

    def get_context_window(self, full_text: str, window_size: int = 200) -> str:
        """Get surrounding context for this chunk."""
        start = max(0, self.start_char - window_size)
        end = min(len(full_text), self.end_char + window_size)
        return full_text[start:end]


class Chunker:
    """Split text into overlapping chunks."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[Chunk]:
        """Split text into overlapping chunks.

        Uses sentence-aware splitting to avoid cutting in the middle of sentences.
        """
        if not text or not text.strip():
            return []

        # Clean and normalize text
        text = self._normalize_text(text)

        # Split into sentences first
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk_sentences = []
        current_length = 0
        chunk_start = 0
        current_pos = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk_sentences:
                # Create chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_text),
                    chunk_index=len(chunks),
                    total_chunks=0  # Will update later
                ))

                # Determine overlap: keep sentences that fit in overlap window
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_length + len(s) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s) + 1  # +1 for space
                    else:
                        break

                # Start new chunk with overlap
                current_chunk_sentences = overlap_sentences
                current_length = sum(len(s) + 1 for s in overlap_sentences)
                chunk_start = current_pos - overlap_length if overlap_length > 0 else current_pos

            current_chunk_sentences.append(sentence)
            current_length += sentence_length + 1  # +1 for space
            current_pos += sentence_length + 1

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                start_char=chunk_start,
                end_char=chunk_start + len(chunk_text),
                chunk_index=len(chunks),
                total_chunks=0
            ))

        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    def _normalize_text(self, text: str) -> str:
        """Normalize text for chunking."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove page markers but keep content
        text = re.sub(r'\[Page \d+\]\s*', '\n\n', text)
        return text.strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - handles common legal patterns
        # This regex handles: periods, question marks, exclamation marks
        # But preserves abbreviations like "Inc.", "Corp.", "No.", etc.

        # Common abbreviations to preserve
        abbreviations = r'(?:Inc|Corp|Ltd|LLC|LLP|No|nos|vs|v|Mr|Mrs|Ms|Dr|Prof|Jr|Sr|etc|e\.g|i\.e|cf|al|et|para|paras|sec|secs|art|arts|ch|chs|vol|vols|p|pp|fig|figs|ex|exs|cert|App|Supp|F\.2d|F\.3d|S\.Ct|U\.S|Cal|N\.Y|Tex|Ill)'

        # Split on sentence boundaries
        sentences = []
        current = ""

        # Simple approach: split on period + space + capital letter
        # while preserving abbreviations
        parts = re.split(r'([.!?]+)\s+', text)

        i = 0
        while i < len(parts):
            part = parts[i]

            if i + 1 < len(parts) and re.match(r'^[.!?]+$', parts[i + 1]):
                # This part ends with punctuation
                current += part + parts[i + 1]
                i += 2

                # Check if this is a complete sentence
                # (not ending with an abbreviation)
                if not re.search(abbreviations + r'[.!?]+$', current, re.IGNORECASE):
                    sentences.append(current.strip())
                    current = ""
            else:
                current += part
                i += 1

        if current.strip():
            sentences.append(current.strip())

        # Filter out empty sentences and very short fragments
        sentences = [s for s in sentences if len(s) > 10]

        return sentences

    def chunk_by_sections(self, text: str, section_markers: List[str] = None) -> List[Chunk]:
        """Chunk text by sections/headers, with overlap for context.

        This is useful for legal documents with clear section structure.
        """
        if section_markers is None:
            # Default patterns for legal documents
            section_markers = [
                r'^\s*(?:ARTICLE|Article)\s+[IVX\d]+',
                r'^\s*(?:SECTION|Section)\s+\d+',
                r'^\s*\d+\.\s+[A-Z]',  # Numbered sections
                r'^\s*[A-Z]{2,}(?:\s+[A-Z]+)*\s*$',  # ALL CAPS headers
            ]

        # Combined pattern
        pattern = '|'.join(f'({m})' for m in section_markers)
        combined_pattern = re.compile(pattern, re.MULTILINE)

        # Find section boundaries
        matches = list(combined_pattern.finditer(text))

        if not matches:
            # No sections found, fall back to regular chunking
            return self.chunk_text(text)

        chunks = []
        prev_end = 0

        for i, match in enumerate(matches):
            # Get text before this section
            if match.start() > prev_end:
                section_text = text[prev_end:match.start()].strip()
                if section_text:
                    # Chunk this section
                    section_chunks = self.chunk_text(section_text)
                    for sc in section_chunks:
                        sc.start_char += prev_end
                        sc.end_char += prev_end
                    chunks.extend(section_chunks)

            prev_end = match.start()

        # Handle text after last section
        if prev_end < len(text):
            section_text = text[prev_end:].strip()
            if section_text:
                section_chunks = self.chunk_text(section_text)
                for sc in section_chunks:
                    sc.start_char += prev_end
                    sc.end_char += prev_end
                chunks.extend(section_chunks)

        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(chunks)

        return chunks
