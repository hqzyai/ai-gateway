from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.sensitive_mask import (
    contains_sensitive_value,
    mask_sensitive_fragments,
    mask_sensitive_spans,
    mask_sensitive_value,
)


def test_mobile_mask_keeps_only_prefix_without_internal_rule_name() -> None:
    masked = mask_sensitive_value("18912345678", "cn_mobile")

    assert masked == "189****"
    assert "CN_MOBILE" not in masked
    assert "REDACTED" not in masked


def test_generic_pattern_uses_stars() -> None:
    assert mask_sensitive_value("private-value", "custom_business_secret") == "****"


def test_mask_sensitive_spans_preserves_offsets_for_multiple_matches() -> None:
    text = "主号18912345678，备用号13387654321"

    assert mask_sensitive_spans(
        text,
        [(2, 13), (17, 28)],
        "cn_mobile",
    ) == "主号189****，备用号133****"


def test_derived_attributes_remain_readable_for_multiple_sensitive_types() -> None:
    sources = [
        ("110105199001011234", "cn_identity_card"),
        ("6222021234567890", "bank_card"),
        ("ProjectFalconSecret", "custom_business_secret"),
    ]
    text = "地区码110105，卡片前缀622202，项目代号ProjectFalcon"

    assert mask_sensitive_fragments(text, sources) == text


def test_structured_tool_arguments_detect_exact_sensitive_values() -> None:
    value = {"function": {"arguments": '{"id":"110105199001011234"}'}}

    assert contains_sensitive_value(value, [("110105199001011234", "cn_identity_card")])


def test_identity_card_uses_type_aware_display_mask() -> None:
    assert mask_sensitive_value("360311199905160719", "cn_identity_card") == "360****"


def test_actual_presets_and_labels():
    assert mask_sensitive_value("360311199905160719", "cn_id") == "360****"
    assert mask_sensitive_value("91310000MAE93MG68G", "cn_uscc") == "91****"
    assert mask_sensitive_value("银行卡号：6222021234567890", "cn_bank_labeled") == "银行卡号：6222****"
    assert mask_sensitive_value("微信号：example_user", "cn_account") == "微信号：e****"
    assert mask_sensitive_value("联系电话：010-12345678", "cn_landline_labeled") == "联系电话：010****"
    assert mask_sensitive_value("18212345678", "手机号") == "182****"
    assert mask_sensitive_value("a", "cn_mobile") == "****"
