"""Canonical identifier matching with offsets into the unmodified source text."""

import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import chain
from typing import Protocol


class PatternMatcher(Protocol):
    def _find_raw_pattern_spans(self, text: str, entry: Mapping[str, object]) -> Sequence[tuple[int, int]]: ...


STRUCTURED = frozenset(
    (
        "cn_mobile",
        "cn_id",
        "cn_identity_card",
        "cn_id_card",
        "cn_uscc",
        "cn_bank_labeled",
        "cn_landline_labeled",
        "credit_card",
        "visa",
        "mastercard",
        "amex",
        "discover",
        "iban",
        "eu_iban_enhanced",
    )
)
SEPARATORS = frozenset("-‐‑‒–—−\u200b\u200c\u200d\u2060\ufeff\u00ad")


def structured(name: str) -> bool:
    return name.lower() in STRUCTURED


def view(text: str, compact: bool = False) -> tuple[str, tuple[int, ...]]:
    normalized = tuple(
        (char, index) for index, original in enumerate(text) for char in unicodedata.normalize("NFKC", original)
    )
    removed: frozenset[int] = (
        frozenset(
            index
            for match in re.finditer(
                r"[A-Za-z0-9](?:[\s\-‐‑‒–—−\u200b\u200c\u200d\u2060\ufeff\u00ad]+[A-Za-z0-9])+",
                "".join(char for char, _ in normalized),
            )
            for index in range(match.start(), match.end())
            if normalized[index][0].isspace() or normalized[index][0] in SEPARATORS
        )
        if compact
        else frozenset()
    )
    kept = tuple(pair for index, pair in enumerate(normalized) if index not in removed)
    return "".join(char for char, _ in kept), tuple(offset for _, offset in kept)


def _merged_spans(spans: Iterable[tuple[int, int]]) -> Iterator[tuple[int, int]]:
    pending: tuple[int, int] | None = None
    for start, end in sorted(frozenset(spans)):
        if end <= start:
            continue
        if pending is not None and start < pending[1]:
            pending = (pending[0], max(end, pending[1]))
        else:
            if pending is not None:
                yield pending
            pending = (start, end)
    if pending is not None:
        yield pending


def merge(
    spans: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:  # mutable-ok: Native span callbacks return lists used by pattern-mask callers.
    return list(  # mutable-ok: Native span callback returns a list.
        _merged_spans(spans)
    )


def find_spans(
    guardrail: PatternMatcher, text: str, entry: Mapping[str, object]
) -> list[tuple[int, int]]:  # mutable-ok: Preserve the native content filter span callback list contract.
    original = guardrail._find_raw_pattern_spans(text, entry)  # pyright: ignore[reportPrivateUsage]  # Invoke the native raw matcher without changing its API.
    if not structured(str(entry.get("pattern_name", ""))):
        return list(original)  # mutable-ok: Native span callback result boundary.
    return merge(chain(original, _normalized_spans(guardrail, text, entry)))


def _normalized_spans(guardrail: PatternMatcher, text: str, entry: Mapping[str, object]) -> Iterator[tuple[int, int]]:
    original = guardrail._find_raw_pattern_spans  # pyright: ignore[reportPrivateUsage]  # Calls the native raw matcher without changing its external interface.
    name = str(entry.get("pattern_name", "")).lower()
    for compact in (False, True):
        normalized, offsets = view(text, compact)
        for start, end in original(normalized, entry):
            if end > start:
                yield offsets[start], offsets[end - 1] + 1
        if name == "cn_mobile":
            # Test the configured regex on the domestic number, not the country code.
            for match in re.finditer(r"(?<![0-9])(?:\+86|0086|86)[ -]?(1[3-9][0-9]{9})(?![0-9])", normalized):
                if original(match[1], entry):
                    yield offsets[match.start()], offsets[match.end() - 1] + 1
        # Adjacent formatted identifiers must not become one long identifier.
        # Split only on actual source separators and only if every full segment
        # independently satisfies the configured rule; never split a long integer.
        width = (
            11
            if name == "cn_mobile"
            else 18
            if name in ("cn_id", "cn_identity_card", "cn_id_card", "cn_uscc")
            else None
        )
        if compact and width:
            for run in re.finditer(r"[0-9A-Za-z]+", normalized):
                if len(run[0]) <= width or len(run[0]) % width:
                    continue
                pieces = tuple((start, start + width) for start in range(run.start(), run.end(), width))
                if all(tuple(original(normalized[a:b], entry)) == ((0, width),) for a, b in pieces) and all(
                    offsets[a] > offsets[a - 1] + 1 for a, _ in pieces[1:]
                ):
                    for start, end in pieces:
                        yield offsets[start], offsets[end - 1] + 1


def source_spans(text: str, source: str, name: str) -> tuple[tuple[int, int], ...]:
    name = name.lower()
    if not structured(name):
        return tuple((m.start(), m.end()) for m in re.finditer(re.escape(source), text, re.I))
    canonical, _ = view(source, True)
    if name == "cn_mobile":
        canonical = re.sub(r"^(?:\+86|0086|86)(?=1[3-9][0-9]{9}$)", "", canonical)
    normalized, offsets = view(text, True)
    prefix = r"(?:\+86|0086|86)?" if name == "cn_mobile" else ""
    boundary = (
        "0-9"
        if name
        in (
            "cn_mobile",
            "cn_id",
            "cn_identity_card",
            "cn_id_card",
            "cn_bank_labeled",
            "cn_landline_labeled",
            "credit_card",
            "visa",
            "mastercard",
            "amex",
            "discover",
        )
        else "0-9A-Za-z"
    )
    pattern = f"(?<![{boundary}])" + prefix + re.escape(canonical) + f"(?![{boundary}])"
    return tuple(
        (offsets[m.start()], offsets[m.end() - 1] + 1)
        for m in re.finditer(pattern, normalized, re.I)
        if m.end() > m.start()
    )
