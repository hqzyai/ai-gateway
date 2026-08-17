"""
Single chokepoint for reading prompt-compression token savings out of a parsed
SpendLog ``metadata`` JSON dict. Imported by the daily-spend DB writer and by
cost-savings read endpoints.
"""

from collections.abc import Mapping
from typing import NamedTuple

HEADROOM_GUARDRAIL_PROVIDER = "headroom"


class CompressionTokenSavings(NamedTuple):
    gross_saved: int
    extra_input: int
    net_saved: int


def _token_count_or_zero(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _savings_from_stats(stats: object, *, headroom: bool) -> CompressionTokenSavings:
    if not isinstance(stats, Mapping):
        return CompressionTokenSavings(0, 0, 0)

    net_saved = _token_count_or_zero(stats.get("tokens_saved"))
    if not headroom:
        return CompressionTokenSavings(max(net_saved, 0), max(-net_saved, 0), net_saved)

    tokens_before = _token_count_or_zero(stats.get("tokens_before"))
    compressed_input = _token_count_or_zero(stats.get("ccr_initial_input_tokens"))
    if compressed_input <= 0:
        compressed_input = _token_count_or_zero(stats.get("headroom_compressed_tokens"))
    total_input = _token_count_or_zero(stats.get("tokens_after"))
    if tokens_before <= 0 or compressed_input <= 0 or total_input <= 0:
        return CompressionTokenSavings(max(net_saved, 0), max(-net_saved, 0), net_saved)

    gross_saved = max(tokens_before - compressed_input, 0)
    extra_input = max(compressed_input - tokens_before, 0) + max(total_input - compressed_input, 0)
    return CompressionTokenSavings(gross_saved, extra_input, gross_saved - extra_input)


def _headroom_entry_savings(entry: object) -> CompressionTokenSavings:
    if not isinstance(entry, Mapping):
        return CompressionTokenSavings(0, 0, 0)
    if entry.get("guardrail_provider") != HEADROOM_GUARDRAIL_PROVIDER:
        return CompressionTokenSavings(0, 0, 0)
    return _savings_from_stats(entry.get("guardrail_response"), headroom=True)


def _headroom_savings(guardrail_information: object) -> CompressionTokenSavings:
    entries = [guardrail_information] if isinstance(guardrail_information, Mapping) else guardrail_information
    if not isinstance(entries, list):
        return CompressionTokenSavings(0, 0, 0)
    savings = tuple(_headroom_entry_savings(entry) for entry in entries)
    return CompressionTokenSavings(
        sum(item.gross_saved for item in savings),
        sum(item.extra_input for item in savings),
        sum(item.net_saved for item in savings),
    )


def extract_compression_token_savings(metadata: Mapping[str, object]) -> CompressionTokenSavings:
    native = _savings_from_stats(metadata.get("compression_savings"), headroom=False)
    headroom = _headroom_savings(metadata.get("guardrail_information"))
    return CompressionTokenSavings(
        native.gross_saved + headroom.gross_saved,
        native.extra_input + headroom.extra_input,
        native.net_saved + headroom.net_saved,
    )


def extract_compression_saved_tokens(metadata: Mapping[str, object]) -> int:
    """
    Return signed net prompt tokens saved by compression for one request.

    Sums two disjoint sources:

    - the native ``compression_savings`` key, written only by
      ``CompressionInterceptionLogger`` in its pre-call deployment hook
    - ``guardrail_information`` entries with ``guardrail_provider ==
      "headroom"``, written only by the Headroom guardrail

    Each writer records only its own transform pass and the two run at
    different stages (guardrail pre-call vs deployment pre-call), so when both
    fire on one request their measured savings are independent and additive;
    summing them never double-counts. Malformed or missing values contribute 0.
    A bare dict ``guardrail_information`` is treated as a single entry, matching
    the spend-log redactor's normalization of that legacy shape.
    """
    return extract_compression_token_savings(metadata).net_saved
