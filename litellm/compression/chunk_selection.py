"""
Chunk-level content selection for oversized messages.

Splits a large message into file sections and fixed-line windows, scores each
window against the query with BM25, and keeps the highest-scoring windows whole
until the token budget is filled.  Dropped regions are collapsed into short
omission markers so the surviving content keeps its structure and file identity.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

from litellm.compression.scoring.bm25 import bm25_score_texts, tokenize

_FILE_START = re.compile(r"^\[start of (.+)\]$")
_FILE_END = re.compile(r"^\[end of (.+)\]$")
_WINDOW_LINES = 40
_WINDOW_MAX_CHARS = 6000
_LINE_SPLIT_CHARS = 2000
_ASSEMBLY_RESERVE_TOKENS = 64
_ECHO_MIN_TERMS = 20
_ECHO_CONTAINMENT_THRESHOLD = 0.85
_ECHO_EXCLUDED_SCORE = -1.0
_APPROX_CHARS_PER_TOKEN = 3
_TARGET_WINDOWS_PER_BUDGET = 6
_MIN_WINDOW_CHARS = 200
_WINDOW_SCORE_WEIGHT = 0.7
_LABEL_SCORE_WEIGHT = 0.3


@dataclass(frozen=True, slots=True)
class _Section:
    label: str
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Chunk:
    section_index: int
    text: str
    line_count: int


def _split_sections(content: str) -> tuple[_Section, ...]:
    sections: list[_Section] = []
    current_label = ""
    current_lines: list[str] = []

    for line in content.split("\n"):
        start_match = _FILE_START.match(line)
        if start_match:
            if current_lines or current_label:
                sections.append(_Section(label=current_label, lines=tuple(current_lines)))
            current_label = start_match.group(1)
            current_lines = []
            continue
        if _FILE_END.match(line):
            sections.append(_Section(label=current_label, lines=tuple(current_lines)))
            current_label = ""
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines or current_label:
        sections.append(_Section(label=current_label, lines=tuple(current_lines)))
    return tuple(sections)


def _split_long_line(line: str, max_chars: int) -> tuple[str, ...]:
    if len(line) <= max_chars:
        return (line,)
    return tuple(line[start : start + max_chars] for start in range(0, len(line), max_chars))


def _windows(
    section_index: int,
    lines: Sequence[str],
    window_lines: int,
    window_max_chars: int,
) -> tuple[_Chunk, ...]:
    expanded = [piece for line in lines for piece in _split_long_line(line, window_max_chars)]

    chunks: list[_Chunk] = []
    window: list[str] = []
    window_chars = 0
    for piece in expanded:
        if window and (len(window) >= window_lines or window_chars + len(piece) > window_max_chars):
            chunks.append(_Chunk(section_index=section_index, text="\n".join(window), line_count=len(window)))
            window = []
            window_chars = 0
        window.append(piece)
        window_chars += len(piece)
    if window:
        chunks.append(_Chunk(section_index=section_index, text="\n".join(window), line_count=len(window)))
    return tuple(chunks)


def _assemble(
    sections: Sequence[_Section],
    chunks: Sequence[_Chunk],
    kept_chunk_indices: frozenset[int],
) -> str:
    chunk_indices_by_section: dict[int, list[int]] = {}
    for chunk_index, chunk in enumerate(chunks):
        chunk_indices_by_section.setdefault(chunk.section_index, []).append(chunk_index)

    parts: list[str] = []
    omitted_labels: list[str] = []

    for section_index, section in enumerate(sections):
        section_chunk_indices = chunk_indices_by_section.get(section_index, [])
        kept_in_section = [i for i in section_chunk_indices if i in kept_chunk_indices]
        if not kept_in_section:
            if section.label:
                omitted_labels.append(section.label)
            continue

        section_parts: list[str] = []
        omitted_run_lines = 0
        for chunk_index in section_chunk_indices:
            if chunk_index in kept_chunk_indices:
                if omitted_run_lines:
                    section_parts.append(f"[... {omitted_run_lines} lines omitted ...]")
                    omitted_run_lines = 0
                section_parts.append(chunks[chunk_index].text)
            else:
                omitted_run_lines += chunks[chunk_index].line_count
        if omitted_run_lines:
            section_parts.append(f"[... {omitted_run_lines} lines omitted ...]")

        body = "\n".join(section_parts)
        if section.label:
            parts.append(f"[start of {section.label}]\n{body}\n[end of {section.label}]")
        else:
            parts.append(body)

    if omitted_labels:
        parts.append(f"[files omitted (not shown): {', '.join(omitted_labels)}]")

    return "\n".join(parts)


def _is_query_echo(chunk_terms: frozenset[str], query_terms: frozenset[str]) -> bool:
    if len(chunk_terms) < _ECHO_MIN_TERMS:
        return False
    return len(chunk_terms & query_terms) / len(chunk_terms) >= _ECHO_CONTAINMENT_THRESHOLD


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    peak = max(values, default=0.0)
    if peak <= 0.0:
        return tuple(0.0 for _ in values)
    return tuple(v / peak for v in values)


def _score_chunks(
    query: str,
    sections: Sequence[_Section],
    chunks: Sequence[_Chunk],
) -> tuple[float, ...]:
    window_scores = bm25_score_texts(query, [chunk.text for chunk in chunks])
    label_scores = bm25_score_texts(query, [re.sub(r"[^a-zA-Z0-9]+", " ", section.label) for section in sections])

    query_terms = frozenset(tokenize(query))
    echo_flags = tuple(_is_query_echo(frozenset(tokenize(chunk.text)), query_terms) for chunk in chunks)

    normalized_windows = _normalize(tuple(0.0 if echo else s for echo, s in zip(echo_flags, window_scores)))
    normalized_labels = _normalize(label_scores)
    return tuple(
        _ECHO_EXCLUDED_SCORE
        if echo
        else _WINDOW_SCORE_WEIGHT * window_score + _LABEL_SCORE_WEIGHT * normalized_labels[chunk.section_index]
        for chunk, window_score, echo in zip(chunks, normalized_windows, echo_flags)
    )


def _select_with_windows(
    sections: Sequence[_Section],
    query: str,
    max_tokens: int,
    count_tokens: Callable[[str], int],
    window_lines: int,
    window_max_chars: int,
) -> Optional[str]:
    chunks = tuple(
        chunk
        for i, section in enumerate(sections)
        for chunk in _windows(i, section.lines, window_lines, window_max_chars)
    )
    if not chunks:
        return None

    scores = _score_chunks(query, sections, chunks)
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))

    budget = max_tokens - _ASSEMBLY_RESERVE_TOKENS

    kept: set[int] = set()
    used_tokens = 0
    for chunk_index in ranked:
        if scores[chunk_index] < 0:
            continue
        chunk_tokens = count_tokens(chunks[chunk_index].text)
        if used_tokens + chunk_tokens > budget:
            continue
        kept.add(chunk_index)
        used_tokens += chunk_tokens

    while kept:
        assembled = _assemble(sections, chunks, frozenset(kept))
        if count_tokens(assembled) <= max_tokens:
            return assembled
        kept.remove(min(kept, key=lambda i: (scores[i], -i)))

    return None


def select_chunks_to_budget(
    content: str,
    query: str,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> Optional[str]:
    if max_tokens <= _ASSEMBLY_RESERVE_TOKENS:
        return None
    sections = _split_sections(content)

    budget_chars = (max_tokens - _ASSEMBLY_RESERVE_TOKENS) * _APPROX_CHARS_PER_TOKEN
    base_window_chars = min(_WINDOW_MAX_CHARS, max(_MIN_WINDOW_CHARS, budget_chars // _TARGET_WINDOWS_PER_BUDGET))
    base_window_lines = max(min(_WINDOW_LINES, base_window_chars // 100), 2)

    for scale in (1, 4, 16):
        selected = _select_with_windows(
            sections=sections,
            query=query,
            max_tokens=max_tokens,
            count_tokens=count_tokens,
            window_lines=max(base_window_lines // scale, 1),
            window_max_chars=max(base_window_chars // scale, 100),
        )
        if selected is not None:
            return selected

    return None
