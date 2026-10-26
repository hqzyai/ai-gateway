import json

import pytest
import yaml

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.chinese_categories import (
    augment_preview,
    category_keywords,
)
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.types.guardrails import ContentFilterAction
from litellm.types.proxy.guardrails.guardrail_hooks.litellm_content_filter import ContentFilterCategoryConfig
from fastapi import HTTPException


def test_preview_preserves_rules_and_does_not_duplicate_keywords():
    content = json.dumps({"category_name": "denied_insults", "default_action": "BLOCK", "exceptions": ["original"]})
    preview = augment_preview("denied_insults", content, "json")
    parsed = json.loads(preview)
    assert parsed["default_action"] == "BLOCK"
    assert parsed["exceptions"] == ["original"]
    assert {"keyword": "蠢货", "severity": "high"} in parsed["keywords"]
    assert json.loads(augment_preview("denied_insults", preview, "json")) == parsed
    assert yaml.safe_load(augment_preview("denied_insults", preview, "yaml")) == parsed
    assert category_keywords("unknown", [{"keyword": "keep", "severity": "high"}]) == [{"keyword": "keep", "severity": "high"}]


def test_chinese_terms_are_opt_in_and_use_existing_filter(monkeypatch):
    config = [ContentFilterCategoryConfig(category="denied_insults", action=ContentFilterAction.BLOCK)]
    monkeypatch.delenv("AGENTOS_GUARDRAIL_MODE", raising=False)
    original = ContentFilterGuardrail(guardrail_name="original", categories=config)
    assert original._filter_single_text("你是个蠢货", detections=[]) == "你是个蠢货"
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")
    managed = ContentFilterGuardrail(guardrail_name="managed", categories=config)
    with pytest.raises(HTTPException):
        managed._filter_single_text("你是个蠢货", detections=[])
