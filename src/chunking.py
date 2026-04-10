from __future__ import annotations

import math
import re
from typing import List

class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split sentences using regex (keep delimiters)
        sentences = re.split(r'(?<=[.!?])\s+|\.\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i : i + self.max_sentences_per_chunk]
            chunk = " ".join(chunk_sentences).strip()
            chunks.append(chunk)

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        return [c.strip() for c in self._split(text, self.separators) if c.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: small enough
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # No separator left → hard split
        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        sep = remaining_separators[0]

        # Last fallback: character-level split
        if sep == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        parts = current_text.split(sep)

        chunks = []
        buffer = ""

        for part in parts:
            if buffer:
                candidate = buffer + sep + part
            else:
                candidate = part

            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.extend(self._split(buffer, remaining_separators[1:]))
                buffer = part

        if buffer:
            chunks.extend(self._split(buffer, remaining_separators[1:]))

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0

    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class MarkdownChunker:
    """
    Advanced Markdown chunker for RAG:

    Improvements:
        - Preserve tables as atomic blocks
        - Preserve bullet groups
        - Split by semantic blocks (paragraph, list, table)
        - Add overlap between chunks
        - Clean heading context (remove ### noise)
    """

    _HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)')
    _TABLE_ROW_RE = re.compile(r'^\|.*\|$')

    def __init__(self, max_chunk_size: int = 500, overlap: int = 100) -> None:
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    # =========================
    # Public API
    # =========================
    def chunk(self, text: str) -> List[str]:
        if not text:
            return []

        sections = self._parse_sections(text)
        chunks: List[str] = []

        for section in sections:
            context = section["heading_context"]
            body = section["body"]

            context_prefix = self._format_context(context)

            blocks = self._split_semantic_blocks(body)

            chunks.extend(self._merge_blocks(blocks, context_prefix))

        return chunks

    # =========================
    # Section parsing
    # =========================
    def _parse_sections(self, text: str):
        lines = text.split("\n")
        sections = []

        stack = []
        current_body = []
        current_heading = None

        def flush():
            if current_heading or current_body:
                sections.append({
                    "heading_context": [h for _, h in stack] + ([current_heading] if current_heading else []),
                    "body": "\n".join(current_body)
                })

        for line in lines:
            m = self._HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                heading_text = m.group(2).strip()

                flush()
                current_body = []

                while stack and stack[-1][0] >= level:
                    stack.pop()

                if current_heading:
                    stack.append((level, current_heading))

                current_heading = heading_text
            else:
                current_body.append(line)

        flush()
        return sections

    # =========================
    # Semantic splitting
    # =========================
    def _split_semantic_blocks(self, text: str) -> List[str]:
        lines = text.split("\n")

        blocks = []
        buffer = []

        def flush():
            if buffer:
                blocks.append("\n".join(buffer).strip())

        i = 0
        while i < len(lines):
            line = lines[i]

            # TABLE block
            if self._TABLE_ROW_RE.match(line):
                flush()
                buffer = [line]
                i += 1
                while i < len(lines) and self._TABLE_ROW_RE.match(lines[i]):
                    buffer.append(lines[i])
                    i += 1
                flush()
                buffer = []
                continue

            # BULLET block
            if re.match(r'^- ', line):
                flush()
                buffer = [line]
                i += 1
                while i < len(lines) and (lines[i].startswith("  ") or lines[i].startswith("- ")):
                    buffer.append(lines[i])
                    i += 1
                flush()
                buffer = []
                continue

            # PARAGRAPH
            if line.strip() == "":
                flush()
                buffer = []
            else:
                buffer.append(line)

            i += 1

        flush()
        return [b for b in blocks if b.strip()]

    # =========================
    # Merge blocks into chunks
    # =========================
    def _merge_blocks(self, blocks: List[str], context_prefix: str) -> List[str]:
        chunks = []
        buffer = ""

        for block in blocks:
            candidate = self._join(context_prefix, buffer, block)

            if len(candidate) <= self.max_chunk_size:
                buffer = (buffer + "\n\n" + block).strip() if buffer else block
            else:
                if buffer:
                    chunks.append(self._join(context_prefix, buffer))
                buffer = block

        if buffer:
            chunks.append(self._join(context_prefix, buffer))

        return self._add_overlap(chunks)

    # =========================
    # Overlap
    # =========================
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        if self.overlap <= 0:
            return chunks

        new_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                new_chunks.append(chunk)
                continue

            prev = chunks[i - 1]
            overlap_text = prev[-self.overlap:]

            new_chunks.append(overlap_text + "\n" + chunk)

        return new_chunks

    # =========================
    # Helpers
    # =========================
    def _format_context(self, headings: List[str]) -> str:
        """Clean heading format for embedding"""
        return " > ".join([h.strip() for h in headings if h.strip()])

    def _join(self, context: str, *parts: str) -> str:
        content = "\n\n".join([p for p in parts if p])
        return f"{context}\n\n{content}" if context else content


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        if not text:
            return {}

        fixed = FixedSizeChunker(chunk_size=chunk_size)
        sentence = SentenceChunker()
        recursive = RecursiveChunker(chunk_size=chunk_size)
        markdown = MarkdownChunker(max_chunk_size=chunk_size)

        fixed_chunks = fixed.chunk(text)
        sentence_chunks = sentence.chunk(text)
        recursive_chunks = recursive.chunk(text)
        markdown_chunks = markdown.chunk(text)

        def stats(chunks: list[str]) -> dict:
            if not chunks:
                return {
                    "chunks": [],
                    "count": 0,
                    "avg_length": 0,
                    "max_length": 0,
                    "min_length": 0,
                }

            lengths = [len(c) for c in chunks]
            return {
                "chunks": chunks,
                "count": len(chunks),
                "avg_length": sum(lengths) / len(lengths),
                "max_length": max(lengths),
                "min_length": min(lengths),
            }

        return {
            "fixed_size": stats(fixed_chunks),
            "by_sentences": stats(sentence_chunks),
            "recursive": stats(recursive_chunks),
            "markdown": stats(markdown_chunks),
        }
