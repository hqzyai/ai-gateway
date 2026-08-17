"""
Unit tests for the compression-savings spend-log metadata normalizer.
"""

import pytest

from litellm.proxy.spend_tracking.compression_savings import (
    extract_compression_saved_tokens,
    extract_compression_token_savings,
)

NATIVE_SAVINGS = {
    "tokens_before": 12000,
    "tokens_after": 5000,
    "tokens_saved": 7000,
    "source": "compression_interception",
}

HEADROOM_ENTRY = {
    "guardrail_name": "headroom-compressor",
    "guardrail_provider": "headroom",
    "guardrail_status": "success",
    "guardrail_response": {"tokens_before": 1000, "tokens_after": 400, "tokens_saved": 600},
}


def test_native_key_only():
    assert extract_compression_saved_tokens({"compression_savings": NATIVE_SAVINGS}) == 7000


def test_headroom_only():
    assert extract_compression_saved_tokens({"guardrail_information": [HEADROOM_ENTRY]}) == 600


def test_native_and_headroom_sum():
    metadata = {
        "compression_savings": NATIVE_SAVINGS,
        "guardrail_information": [HEADROOM_ENTRY],
    }
    assert extract_compression_saved_tokens(metadata) == 7600


def test_multiple_headroom_entries_sum():
    metadata = {"guardrail_information": [HEADROOM_ENTRY, HEADROOM_ENTRY]}
    assert extract_compression_saved_tokens(metadata) == 1200


def test_neither_source_present():
    assert extract_compression_saved_tokens({"user_api_key": "abc", "usage_object": {}}) == 0


def test_non_headroom_guardrail_entries_ignored():
    metadata = {
        "guardrail_information": [
            {
                "guardrail_name": "pii-guard",
                "guardrail_provider": "presidio",
                "guardrail_response": {"tokens_saved": 999},
            }
        ]
    }
    assert extract_compression_saved_tokens(metadata) == 0


@pytest.mark.parametrize(
    "compression_savings",
    [
        None,
        "not-a-dict",
        {},
        {"tokens_saved": None},
        {"tokens_saved": "7000"},
        {"tokens_saved": True},
        {"tokens_before": 100, "tokens_after": 50},
    ],
)
def test_malformed_native_key_contributes_zero(compression_savings):
    assert extract_compression_saved_tokens({"compression_savings": compression_savings}) == 0


@pytest.mark.parametrize(
    "guardrail_information",
    [
        None,
        "not-a-list",
        {"guardrail_provider": "headroom"},
        [],
        [None],
        ["not-a-dict"],
        [{"guardrail_provider": "headroom"}],
        [{"guardrail_provider": "headroom", "guardrail_response": "REDACTED"}],
        [{"guardrail_provider": "headroom", "guardrail_response": {"tokens_saved": "600"}}],
        [{"guardrail_response": {"tokens_saved": 600}}],
    ],
)
def test_malformed_headroom_information_contributes_zero(guardrail_information):
    assert extract_compression_saved_tokens({"guardrail_information": guardrail_information}) == 0


def test_valid_headroom_entry_survives_alongside_malformed_ones():
    metadata = {
        "guardrail_information": [
            None,
            {"guardrail_provider": "headroom", "guardrail_response": "REDACTED"},
            HEADROOM_ENTRY,
        ]
    }
    assert extract_compression_saved_tokens(metadata) == 600


def test_negative_native_savings_are_preserved():
    metadata = {"compression_savings": {"tokens_saved": -5}}
    assert extract_compression_token_savings(metadata) == (0, 5, -5)
    assert extract_compression_saved_tokens(metadata) == -5


def test_headroom_breakdown_uses_gross_extra_and_signed_net_tokens():
    metadata = {
        "guardrail_information": [
            {
                "guardrail_provider": "headroom",
                "guardrail_response": {
                    "tokens_before": 1000,
                    "headroom_compressed_tokens": 400,
                    "ccr_initial_input_tokens": 450,
                    "tokens_after": 1100,
                    "tokens_saved": -100,
                },
            }
        ]
    }
    assert extract_compression_token_savings(metadata) == (550, 650, -100)
    assert extract_compression_saved_tokens(metadata) == -100


def test_headroom_compression_expansion_counts_as_extra_input():
    metadata = {
        "guardrail_information": [
            {
                "guardrail_provider": "headroom",
                "guardrail_response": {
                    "tokens_before": 100,
                    "headroom_compressed_tokens": 125,
                    "tokens_after": 125,
                    "tokens_saved": -25,
                },
            }
        ]
    }
    assert extract_compression_token_savings(metadata) == (0, 25, -25)


def test_float_tokens_saved_counts_as_int():
    entry = {"guardrail_provider": "headroom", "guardrail_response": {"tokens_saved": 600.0}}
    assert extract_compression_saved_tokens({"guardrail_information": [entry]}) == 600
    assert extract_compression_saved_tokens({"compression_savings": {"tokens_saved": 7000.0}}) == 7000
    assert extract_compression_saved_tokens({"compression_savings": {"tokens_saved": 12.5}}) == 12


def test_bare_dict_guardrail_information_counts_as_single_entry():
    assert extract_compression_saved_tokens({"guardrail_information": HEADROOM_ENTRY}) == 600


def test_headroom_writer_and_reader_share_provider_slug():
    from litellm.proxy.guardrails.guardrail_hooks.headroom.headroom import (
        HEADROOM_GUARDRAIL_PROVIDER as writer_slug,
    )
    from litellm.proxy.spend_tracking.compression_savings import (
        HEADROOM_GUARDRAIL_PROVIDER as reader_slug,
    )

    assert writer_slug == reader_slug == "headroom"
