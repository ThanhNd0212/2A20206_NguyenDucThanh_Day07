from __future__ import annotations

import math
import re


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
        sentences = re.split(r'\. |! |\? |\.\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text.strip()] if text.strip() else []
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
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
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        if separator == "":
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        parts = current_text.split(separator)
        result: list[str] = []
        current_chunk = ""

        for part in parts:
            candidate = current_chunk + separator + part if current_chunk else part

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    result.extend(self._split(current_chunk, next_separators))
                if len(part) <= self.chunk_size:
                    current_chunk = part
                else:
                    result.extend(self._split(part, next_separators))
                    current_chunk = ""

        if current_chunk:
            result.extend(self._split(current_chunk, next_separators))

        return [r for r in result if r.strip()]


# Lookahead that matches the start of a markdown or underlined section header.
# Markdown:   ^#{1,6} followed by a space/tab (e.g. "## Overview")
# Underlined: a non-blank line immediately followed by 3+ '=' or '-' chars
_SECTION_SPLIT_RE = re.compile(
    r"(?m)(?=^#{1,6}[ \t]|^(?!\s*$).+\n[ \t]*[=\-]{3,}[ \t]*$)"
)


class SectionChunker:
    """Custom chunking strategy for structured documents (README, policy, SOP, technical docs).

    Design rationale: character-based chunking cuts across section boundaries and forces
    the retriever to reassemble meaning from unrelated fragments. Documents with explicit
    headers already encode the author's intended units of meaning — each section is
    self-contained and directly answers a specific query. Splitting at those boundaries
    preserves coherence, keeps the header as a retrieval signal alongside its body, and
    avoids the token waste of overlapping windows. Short sections are merged into the
    next chunk so that sparse headings never produce orphan one-liners.
    """

    def __init__(self, min_section_length: int = 50) -> None:
        self.min_section_length = max(0, min_section_length)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split at every header boundary; each part keeps its own header.
        parts = _SECTION_SPLIT_RE.split(text)
        sections = [p.strip() for p in parts if p.strip()]

        if len(sections) <= 1:
            return [text.strip()] if text.strip() else []

        # Merge sections that are too short to be useful on their own.
        merged: list[str] = []
        buffer = ""
        for section in sections:
            buffer = (buffer + "\n\n" + section).strip() if buffer else section
            if len(buffer) >= self.min_section_length:
                merged.append(buffer)
                buffer = ""

        if buffer:
            if merged:
                merged[-1] = merged[-1] + "\n\n" + buffer
            else:
                merged.append(buffer)

        return merged if merged else [text.strip()]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed = FixedSizeChunker(chunk_size=chunk_size).chunk(text)
        by_sentence = SentenceChunker().chunk(text)
        recursive = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count else 0.0
            return {"count": count, "avg_length": avg_length, "chunks": chunks}

        return {
            "fixed_size": stats(fixed),
            "by_sentences": stats(by_sentence),
            "recursive": stats(recursive),
        }
