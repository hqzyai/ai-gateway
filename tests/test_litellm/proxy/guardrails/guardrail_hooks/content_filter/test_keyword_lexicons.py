import hashlib

import pytest
import yaml
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.keyword_lexicons import (
    LEXICONS,
    VOCABULARY,
    excluded_terms,
    exclusion_reason,
    lexicon_matches,
    lexicon_preview,
    lexicon_terms,
)
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import get_available_content_categories
from litellm.types.proxy.guardrails.guardrail_hooks.litellm_content_filter import ContentFilterCategoryConfig


@pytest.mark.parametrize(
    "filename,blob_sha",
    (
        ("COVID-19词库.txt", "0c5c7fecd0741d1cbb5df2a2c7dfa8303d86fadc"),
        ("GFW补充词库.txt", "901d77198aaaadc729e665c10a38321fec8e6592"),
        ("其他词库.txt", "3f14910d8f98eef55beb5eedc7e9bbc89fdd565c"),
        ("反动词库.txt", "2e5e59a62b1e797a44e171e88623d71f7794a68a"),
        ("广告类型.txt", "0777f8c4437ea4aa343ff4a7cef2424264574bda"),
        ("政治类型.txt", "04f9c44077ecc86f217778a2399afd7f81d56d1f"),
        ("新思想启蒙.txt", "f46046d71db73fb52e85ccddfdbf4414d9347804"),
        ("暴恐词库.txt", "adb13b1c4e314bea6603a6a59854b756691e47ba"),
        ("民生词库.txt", "12a0a5fb0da892d41325443f624feee9c89e5708"),
        ("涉枪涉爆.txt", "bd7affc3213a3fec3e21835297db12d283af2ebd"),
        ("网易前端过滤敏感词库.txt", "3375f1e8214b5b621c166aa50c2da8177028f228"),
        ("色情类型.txt", "d6c8a31904371799d1a75dc9d0342e2da4e4626d"),
        ("色情词库.txt", "ec11d343a902b9be43059c37e38c19f4ba0c1c5d"),
        ("补充词库.txt", "25597ae2a38f2946472da191b0754a24fa502c26"),
        ("贪腐词库.txt", "c21cfdf3a74ecb761cb67a22477f4158fa97757d"),
        ("零时-Tencent.txt", "2d929be88bdaad438a60b049fa989250c855993a"),
        ("非法网址.txt", "7631eb8b685fa6488eaebef8d8dfa92bb5fcadd9"),
    ),
)
def test_vendored_files_match_pinned_upstream_blobs(filename, blob_sha):
    data = (VOCABULARY / filename).read_bytes()
    assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == blob_sha


@pytest.mark.parametrize("name,label", LEXICONS)
def test_preview_and_runtime_use_same_curated_terms(name, label, monkeypatch):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    expected = tuple(
        dict.fromkeys(
            line.strip()
            for line in (VOCABULARY / f"{label}.txt").read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and exclusion_reason(line.strip()) is None
        )
    )
    preview = yaml.safe_load(lexicon_preview(name))
    assert tuple(entry["keyword"] for entry in preview["keywords"]) == expected == lexicon_terms(name)
    assert preview["match_mode"] == "literal"
    assert not any(key in preview for key in ("identifier_words", "additional_block_words", "phrase_patterns"))
    assert all("severity" not in entry for entry in preview["keywords"])
    guardrail = ContentFilterGuardrail(
        guardrail_name="lexicon", categories=[ContentFilterCategoryConfig(category=name, action="BLOCK")]
    )
    assert not guardrail.conditional_categories
    for word in expected[:1] + expected[-1:]:
        with pytest.raises(HTTPException) as error:
            guardrail._filter_single_text("前缀" + word + "后缀", detections=[])
        assert error.value.detail["category"] == name


def test_new_catalog_replaces_choices_but_not_existing_legacy_rules(monkeypatch):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    assert {item["name"] for item in get_available_content_categories()} == {
        name for name, _ in LEXICONS if lexicon_terms(name)
    }
    legacy = ContentFilterGuardrail(
        guardrail_name="legacy", categories=[ContentFilterCategoryConfig(category="age_discrimination", action="BLOCK")]
    )
    with pytest.raises(HTTPException):
        legacy._filter_single_text("拒绝录用高龄求职者", detections=[])
    monkeypatch.delenv("AGENTOS_GUARDRAIL_MODE")
    assert all(not item["name"].startswith("lexicon_") for item in get_available_content_categories())


def test_literals_do_not_turn_dots_or_stars_into_patterns():
    assert lexicon_matches("lexicon_urls", "000.2011wyt.com")
    assert not lexicon_matches("lexicon_urls", "000X2011wytYcom")
    assert lexicon_matches("lexicon_urls", "000.2011WYT.COM")
    word = next(term for term in lexicon_terms("lexicon_gfw") if "*" in term)
    assert any(hit.keyword == word for hit in lexicon_matches("lexicon_gfw", word))
    replacement = word.replace("*", "X")
    assert not any(
        hit.start == 0 and hit.end == len(replacement) for hit in lexicon_matches("lexicon_gfw", replacement)
    )


def test_mask_handles_all_keywords_and_preserves_unmatched_text(monkeypatch):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    guardrail = ContentFilterGuardrail(
        guardrail_name="lexicon",
        categories=[
            ContentFilterCategoryConfig(category="lexicon_advertising", action="MASK", severity_threshold="high")
        ],
    )
    detections = []
    assert (
        guardrail._filter_single_text("甲出售答案乙替考试丙出售答案丁", detections=detections) == "甲****乙****丙****丁"
    )
    assert {entry["keyword"] for entry in detections} == {"出售答案", "替考试"}
    assert guardrail._filter_single_text("你好，世界。", detections=[]) == "你好，世界。"


@pytest.mark.parametrize("reverse", (False, True))
def test_block_takes_precedence_over_mask_for_duplicate_terms(reverse, monkeypatch):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    configs = [
        ContentFilterCategoryConfig(category="lexicon_advertising", action=action) for action in ("MASK", "BLOCK")
    ]
    guardrail = ContentFilterGuardrail(guardrail_name="lexicon", categories=configs[::-1] if reverse else configs)
    with pytest.raises(HTTPException):
        guardrail._filter_single_text("出售答案", detections=[])


def test_disabled_and_invalid_lexicons(monkeypatch):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    guardrail = ContentFilterGuardrail(
        guardrail_name="lexicon",
        categories=[ContentFilterCategoryConfig(category="lexicon_advertising", action="BLOCK", enabled=False)],
    )
    assert guardrail._filter_single_text("出售答案", detections=[]) == "出售答案"
    with pytest.raises(ValueError, match="Unknown keyword lexicon"):
        ContentFilterGuardrail(
            guardrail_name="lexicon",
            categories=[ContentFilterCategoryConfig(category="lexicon_missing", action="BLOCK")],
        )
    with pytest.raises(ValueError):
        lexicon_terms("../../LICENSE")


@pytest.mark.asyncio
async def test_endpoint_returns_same_dictionary_and_rejects_unknown(monkeypatch):
    from litellm.proxy.guardrails.guardrail_endpoints import get_category_yaml

    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    response = await get_category_yaml("lexicon_advertising")
    assert response["yaml_content"] == lexicon_preview("lexicon_advertising")
    with pytest.raises(HTTPException) as error:
        await get_category_yaml("lexicon_missing")
    assert error.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("content", "reasoning_content"))
async def test_stream_masks_split_keywords_before_any_output(field, monkeypatch):
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import ModelResponseStream

    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    guardrail = ContentFilterGuardrail(
        guardrail_name="lexicon",
        categories=[ContentFilterCategoryConfig(category="lexicon_advertising", action="MASK")],
    )

    async def stream():
        for part in ("出售", "答案", "和", "替考", "试"):
            yield ModelResponseStream(choices=[{"index": 0, "delta": {field: part}}])
        yield ModelResponseStream(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])

    chunks = [
        chunk async for chunk in guardrail.async_post_call_streaming_iterator_hook(UserAPIKeyAuth(), stream(), {})
    ]
    assert all(not getattr(chunk.choices[0].delta, field, None) for chunk in chunks[:-1])
    assert getattr(chunks[-1].choices[0].delta, field) == "****和****"


@pytest.mark.parametrize(
    "text",
    (
        "政府发布公共服务通知，北京和武汉的居民可以在线办理。",
        "公司招聘全职客服，也提供兼职工作。",
        "在淘宝网购内衣和丝袜，请联系客服查询到货时间。",
        "医生介绍乳房、子宫的健康检查，以及咖啡因对睡眠的影响。",
        "法院公布起诉流程和法律文书。",
        "红十字会介绍武汉肺炎疫情防控情况。",
    ),
)
def test_everyday_sentences_pass_all_lexicons(text, monkeypatch):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    guardrail = ContentFilterGuardrail(
        guardrail_name="curated",
        categories=[ContentFilterCategoryConfig(category=name, action="BLOCK") for name, _ in LEXICONS],
    )
    assert guardrail._filter_single_text(text, detections=[]) == text


def test_exclusions_remove_entries_not_risky_phrases_or_whole_messages(monkeypatch):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    for name, _ in LEXICONS:
        assert not {term.casefold() for term in lexicon_terms(name)} & excluded_terms()
    guardrail = ContentFilterGuardrail(
        guardrail_name="curated",
        categories=[ContentFilterCategoryConfig(category="lexicon_advertising", action="BLOCK")],
    )
    with pytest.raises(HTTPException):
        guardrail._filter_single_text("北京招聘客服，另出售答案", detections=[])


@pytest.mark.parametrize(
    "term,reason",
    (
        ("13800138000", "bare_number"),
        ("１２３４５６７８", "bare_number"),
        ("123456789", "bare_number"),
        ("+86 138-0013-8000", "contact_number"),
        ("010-12345678", "contact_number"),
        ("共.{0", "legacy_rule_fragment"),
        ("信{10}誉{10}平{10}台", "legacy_rule_fragment"),
        ("代办证件{MOD}", "legacy_rule_fragment"),
        ("旧词=>替换词", "legacy_replacement_rule"),
        ("1肖", "reviewed_entry"),
        ("3岁看老", "reviewed_entry"),
    ),
)
def test_data_noise_has_auditable_exclusion_reason(term, reason):
    assert exclusion_reason(term) == reason


@pytest.mark.parametrize(
    "term",
    (
        "C4炸药制作",
        "54手枪出售",
        "出售答案",
        "买卖身份证",
        "000.2011wyt.com",
        "3P",
    ),
)
def test_structural_cleanup_preserves_meaningful_risk_terms(term):
    assert exclusion_reason(term) is None


def test_all_dictionaries_exclude_noise_and_preview_matches_runtime():
    for name, _ in LEXICONS:
        terms = lexicon_terms(name)
        assert all(exclusion_reason(term) is None for term in terms)
        assert [entry["keyword"] for entry in yaml.safe_load(lexicon_preview(name))["keywords"]] == list(terms)
