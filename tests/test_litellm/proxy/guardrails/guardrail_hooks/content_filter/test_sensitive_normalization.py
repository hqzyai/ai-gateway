import re

import pytest

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.sensitive_normalization import find_spans
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.sensitive_mask import (
    mask_sensitive_spans,
    mask_sensitive_fragments,
    contains_sensitive_value,
)


class Matcher:
    def _find_raw_pattern_spans(self, text, entry):
        return [m.span() for m in entry["regex"].finditer(text)]


CASES = [
    ("cn_mobile", "18212345678", r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    ("cn_id", "11010119900516001X", r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    (
        "cn_uscc",
        "91310000MAE93MG68G",
        r"(?<![0-9A-Z])[0-9ABCDEFGHJKLMNPQRTUWXY]{18}(?![0-9A-Z])",
    ),
    (
        "cn_bank_labeled",
        "银行卡号：6222021234567890",
        r"(银行卡号|银行账号|卡号)\s*[:：=]?\s*\d{16,19}(?!\d)",
    ),
]


def fullwidth(value):
    return "".join(chr(ord(c) + 0xFEE0) if 0x21 <= ord(c) <= 0x7E else c for c in value)


@pytest.mark.parametrize("name,value,pattern", CASES)
@pytest.mark.parametrize(
    "separator", ["", " ", "-", "\n", "\r\n", "\t", "\u00a0", "\u200b", "—", "－"]
)
@pytest.mark.parametrize("wide", [False, True])
def test_variants_map_back_to_entire_original(name, value, pattern, separator, wide):
    label, token = value.rsplit("：", 1) if "：" in value else ("", value)
    transformed = separator.join(token)
    if wide:
        transformed = fullwidth(transformed)
    transformed = (label + "：" if label else "") + transformed
    text = "前文「" + transformed + "」后文"
    spans = find_spans(
        Matcher(), text, {"pattern_name": name, "regex": re.compile(pattern)}
    )
    assert spans == [(3, len(text) - 3)]
    masked = mask_sensitive_spans(text, spans, name)
    assert masked.startswith("前文「") and masked.endswith("」后文")
    assert "****" in masked
    assert token[-4:] not in masked


@pytest.mark.parametrize("prefix", ["+86", "+86 ", "0086", "86", "＋８６－"])
def test_mobile_country_prefix(prefix):
    text = prefix + "１８２－１２３４\n５６７８"
    spans = find_spans(
        Matcher(), text, {"pattern_name": "cn_mobile", "regex": re.compile(CASES[0][2])}
    )
    assert spans == [(0, len(text))]
    assert mask_sensitive_spans(text, spans, "cn_mobile") == "182****"


def test_sources_and_tool_arguments_use_equivalent_values():
    source = [("18212345678", "cn_mobile")]
    text = "请查 +86 １８２-１２３４\n５６７８"
    assert mask_sensitive_fragments(text, source) == "请查 182****"
    assert contains_sensitive_value({"arguments": {"phone": text}}, source)
    assert not contains_sensitive_value("182123", source)


@pytest.mark.parametrize(
    "text",
    [
        "12345 67890",
        "2182123456789",
        "182123456789",
        "182****",
        "2026-10-23",
        "182，1234，5678",
    ],
)
def test_mobile_non_matches(text):
    assert not find_spans(
        Matcher(), text, {"pattern_name": "cn_mobile", "regex": re.compile(CASES[0][2])}
    )


def test_custom_secret_patterns_are_not_reinterpreted():
    assert not find_spans(
        Matcher(),
        "abc - def",
        {"pattern_name": "custom", "regex": re.compile("abcdef")},
    )


def test_adjacent_formatted_numbers_are_not_merged_or_split_from_long_integer():
    entry = {"pattern_name": "cn_mobile", "regex": re.compile(CASES[0][2])}
    for separator in (" ", "\n", "，"):
        text = "182-1234-5678" + separator + "139-8765-4321"
        spans = find_spans(Matcher(), text, entry)
        assert len(spans) == 2
        assert (
            mask_sensitive_spans(text, spans, "cn_mobile")
            == "182****" + separator + "139****"
        )
    assert not find_spans(Matcher(), "1821234567813987654321", entry)


def test_configured_regex_still_controls_which_numbers_match():
    entry = {"pattern_name": "cn_mobile", "regex": re.compile(r"(?<!\d)139\d{8}(?!\d)")}
    assert not find_spans(Matcher(), "+86 182-1234-5678", entry)
    assert find_spans(Matcher(), "+86 139-1234-5678", entry)
