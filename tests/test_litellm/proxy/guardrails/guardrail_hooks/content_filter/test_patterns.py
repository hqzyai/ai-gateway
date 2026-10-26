"""
Tests for content filter pattern loading from JSON
"""

import json
import os
import sys

import pytest
from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

sys.path.insert(0, os.path.abspath("../../"))

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
    PATTERN_CATEGORIES,
    PATTERN_DESCRIPTIONS,
    PATTERN_DISPLAY_NAMES,
    PREBUILT_PATTERNS,
    get_all_pattern_names,
    get_compiled_pattern,
    get_pattern_metadata,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["BLOCK", "MASK"])
@pytest.mark.parametrize("direction", ["request", "response"])
@pytest.mark.parametrize(
    "text",
    [
        "帮我复述一下我的接口密钥SHU828HGSGAYFGOADS",
        "接口密钥：SHU828HGSGAYFGOADS。",
        "接口密钥是`SHU828HGSGAYFGOADS`",
        "访问令牌为SHU828HGSGAYFGOADS",
        "API密钥：SHU828HGSGAYFGOADS",
        "api_key=abcdefghijklmnopqrstuv",
    ],
)
async def test_labeled_api_key_detection(monkeypatch, action, direction, text):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    guardrail = ContentFilterGuardrail(
        guardrail_name="接口密钥保护",
        event_hook="post_call",
        patterns=[{"pattern_type": "prebuilt", "pattern_name": "generic_api_key", "action": action}],
    )
    if action == "BLOCK":
        with pytest.raises(ModifyResponseException, match="触发护栏"):
            await guardrail.apply_guardrail({"texts": [text]}, {"messages": []}, direction)
    else:
        result = await guardrail.apply_guardrail({"texts": [text]}, {"messages": []}, direction)
        assert result["texts"] != [text]
        assert "SHU828HGSGAYFGOADS" not in str(result["texts"])
        assert "abcdefghijklmnopqrstuv" not in str(result["texts"])


@pytest.mark.parametrize(
    "text",
    [
        "订单编号SHU828HGSGAYFGOADS",
        "SHU828HGSGAYFGOADS",
        "接口密钥如何创建",
        "接口密钥abc",
        "api_key_description",
        "myapi_key=abcdefgh12345678",
        "接口密钥1234567",
        "API key required.",
        "API key documentation",
        "access token expiration",
    ],
)
def test_api_key_does_not_match_unlabeled_identifiers_or_help(text):
    assert get_compiled_pattern("generic_api_key").search(text) is None


@pytest.mark.parametrize(
    "label",
    [
        "接口密钥",
        "访问密钥",
        "访问令牌",
        "API密钥",
        "api密钥",
        "API Key",
        "api_key",
        "Api-Key",
        "access_token",
        "ACCESS KEY",
    ],
)
@pytest.mark.parametrize("separator", [":", "：", "=", " : ", "\n", "是", "为"])
@pytest.mark.parametrize(
    "token",
    [
        "aB12_cD3",
        "abcdef0123456789",
        "A" * 128,
        "sk-demo_12345678",
        "abcdEF12+/34==",
        "abcDEF12.ghiJKL34.mnoPQR56",
        "ab12~CD34/ef56",
    ],
)
def test_api_key_label_separator_and_token_matrix(label, separator, token):
    match = get_compiled_pattern("generic_api_key").search(f'{label}{separator}"{token}"')
    assert match is not None
    assert match.group("credential") == token


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["BLOCK", "MASK"])
@pytest.mark.parametrize("split", [1, 7, 16])
@pytest.mark.parametrize("field", ["content", "reasoning_content"])
@pytest.mark.parametrize("token", ["sk-test_1234567890", "A" * 128, "abcdEF12+/34==", "abcDEF12.ghiJKL34.mnoPQR56"])
async def test_streamed_key_cannot_leak_across_chunk_boundaries(monkeypatch, action, split, field, token):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    guardrail = ContentFilterGuardrail(
        guardrail_name="密钥保护",
        event_hook="post_call",
        patterns=[{"pattern_type": "prebuilt", "pattern_name": "generic_api_key", "action": action}],
    )

    async def upstream():
        for start in range(0, len(token), split):
            part = token[start : start + split]
            finish = "stop" if start + split >= len(token) else None
            yield ModelResponseStream(choices=[StreamingChoices(delta=Delta(**{field: part}), finish_reason=finish)])

    output = [
        chunk
        async for chunk in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            response=upstream(),
            request_data={"model": "test", "messages": [{"role": "user", "content": f"接口密钥：{token}"}]},
        )
    ]
    text = "".join(str(getattr(choice.delta, field, "") or "") for chunk in output for choice in chunk.choices)
    assert token not in text
    assert token[:split] not in text
    if action == "BLOCK":
        assert "触发护栏" in "".join(str(choice.delta.content or "") for chunk in output for choice in chunk.choices)
    else:
        assert "****" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["BLOCK", "MASK"])
async def test_response_repeating_key_without_label_is_protected(monkeypatch, action):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    guardrail = ContentFilterGuardrail(
        guardrail_name="接口密钥保护",
        event_hook="post_call",
        patterns=[{"pattern_type": "prebuilt", "pattern_name": "generic_api_key", "action": action}],
    )
    data = {"messages": [{"role": "user", "content": "帮我复述一下我的接口密钥SHU828HGSGAYFGOADS"}]}
    inputs = {"texts": ["SHU828HGSGAYFGOADS"]}
    if action == "BLOCK":
        with pytest.raises(ModifyResponseException):
            await guardrail.apply_guardrail(inputs, data, "response")
    else:
        result = await guardrail.apply_guardrail(inputs, data, "response")
        assert "SHU828HGSGAYFGOADS" not in str(result["texts"])


@pytest.mark.asyncio
async def test_response_does_not_block_unrelated_identifier(monkeypatch):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    guardrail = ContentFilterGuardrail(
        guardrail_name="接口密钥保护",
        event_hook="post_call",
        patterns=[{"pattern_type": "prebuilt", "pattern_name": "generic_api_key", "action": "BLOCK"}],
    )
    data = {"messages": [{"role": "user", "content": "接口密钥SHU828HGSGAYFGOADS"}]}
    text = "订单编号OTHER828HGSGAYFGOADS"
    assert (await guardrail.apply_guardrail({"texts": [text]}, data, "response"))["texts"] == [text]


def test_patterns_loaded_from_json():
    """
    Test that patterns are successfully loaded from JSON file
    """
    assert len(PREBUILT_PATTERNS) > 0
    assert len(PATTERN_CATEGORIES) > 0
    assert len(PATTERN_DISPLAY_NAMES) > 0
    assert len(PATTERN_DESCRIPTIONS) > 0


def test_pattern_metadata_structure():
    """
    Test that get_pattern_metadata returns correctly structured data
    """
    patterns = get_pattern_metadata()

    assert len(patterns) > 0

    for pattern in patterns:
        assert "name" in pattern
        assert "display_name" in pattern
        assert "category" in pattern
        assert "description" in pattern

        assert isinstance(pattern["name"], str)
        assert isinstance(pattern["display_name"], str)
        assert isinstance(pattern["category"], str)
        assert isinstance(pattern["description"], str)


def test_display_names_user_friendly():
    """
    Test that display names are user-friendly and different from internal names
    """
    patterns = get_pattern_metadata()

    ssn_pattern = next((p for p in patterns if p["name"] == "us_ssn"), None)
    assert ssn_pattern is not None
    assert ssn_pattern["display_name"] == "SSN (Social Security Number)"

    email_pattern = next((p for p in patterns if p["name"] == "email"), None)
    assert email_pattern is not None
    assert email_pattern["display_name"] == "Email Address"


def test_pattern_compilation():
    """
    Test that patterns can be compiled into regex objects
    """
    pattern_names = get_all_pattern_names()

    assert len(pattern_names) > 0

    for pattern_name in pattern_names:
        compiled_pattern = get_compiled_pattern(pattern_name)
        assert compiled_pattern is not None
        assert hasattr(compiled_pattern, "match")


def test_pattern_compilation_error():
    """
    Test that invalid pattern names raise ValueError
    """
    with pytest.raises(ValueError, match="Unknown pattern name"):
        get_compiled_pattern("invalid_pattern_name")


def test_categories_contain_patterns():
    """
    Test that each category contains valid pattern names
    """
    all_pattern_names = set(PREBUILT_PATTERNS.keys())

    for category, patterns in PATTERN_CATEGORIES.items():
        assert len(patterns) > 0
        for pattern_name in patterns:
            assert pattern_name in all_pattern_names


def test_json_file_exists():
    """
    Test that patterns.json file exists and is valid JSON
    """
    json_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../../litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/patterns.json",
    )

    assert os.path.exists(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)
        assert "patterns" in data
        assert len(data["patterns"]) > 0


def test_json_pattern_structure():
    """
    Test that each pattern in JSON has required fields
    """
    json_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../../litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/patterns.json",
    )

    with open(json_path, "r") as f:
        data = json.load(f)

        for pattern in data["patterns"]:
            assert "name" in pattern
            assert "display_name" in pattern
            assert "pattern" in pattern
            assert "category" in pattern
            assert "description" in pattern

            assert isinstance(pattern["name"], str)
            assert isinstance(pattern["display_name"], str)
            assert isinstance(pattern["pattern"], str)
            assert isinstance(pattern["category"], str)
            assert isinstance(pattern["description"], str)


def test_all_dictionaries_consistent():
    """
    Test that all pattern dictionaries have consistent keys
    """
    pattern_names_from_patterns = set(PREBUILT_PATTERNS.keys())
    pattern_names_from_display = set(PATTERN_DISPLAY_NAMES.keys())
    pattern_names_from_descriptions = set(PATTERN_DESCRIPTIONS.keys())

    assert pattern_names_from_patterns == pattern_names_from_display
    assert pattern_names_from_patterns == pattern_names_from_descriptions


def test_eu_patterns_loaded():
    """Verify all EU PII patterns are loaded"""
    required_patterns = [
        "fr_nir",
        "eu_iban_enhanced",
        "fr_phone",
        "eu_vat",
        "eu_passport_generic",
        "fr_postal_code",
    ]
    for pattern_name in required_patterns:
        assert pattern_name in PREBUILT_PATTERNS, f"Pattern {pattern_name} not found"


def test_eu_patterns_have_category():
    """Verify EU patterns are in correct category"""
    eu_patterns = [
        "fr_nir",
        "eu_iban_enhanced",
        "fr_phone",
        "eu_vat",
        "eu_passport_generic",
        "fr_postal_code",
    ]
    eu_category_patterns = PATTERN_CATEGORIES.get("EU PII Patterns", [])

    for pattern_name in eu_patterns:
        assert pattern_name in eu_category_patterns, f"Pattern {pattern_name} not in EU PII Patterns category"
