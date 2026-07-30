import os
import sys
import json
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
import pytest
from litellm import get_llm_provider, get_supported_openai_params
from litellm.llms.siliconflow.chat.transformation import SiliconFlowConfig
from litellm.utils import get_optional_params


class TestSiliconFlowConfig:
    def setup_method(self):
        self.config = SiliconFlowConfig()

    def test_custom_llm_provider(self):
        assert self.config.custom_llm_provider == "siliconflow"

    def test_get_api_key(self):
        assert self.config.get_api_key("test-key") == "test-key"

        with patch(
            "litellm.llms.siliconflow.chat.transformation.get_secret_str",
            return_value="env-key",
        ):
            assert self.config.get_api_key() == "env-key"

        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "env-key"}, clear=False):
            assert self.config.get_api_key() == "env-key"

    def test_get_api_base_precedence(self):
        # Explicit argument wins over everything.
        assert self.config.get_api_base("https://custom-base.com/v1") == "https://custom-base.com/v1"

        # SILICONFLOW_API_BASE override (e.g. the China mainland endpoint).
        with patch(
            "litellm.llms.siliconflow.chat.transformation.get_secret_str",
            return_value="https://api.siliconflow.cn/v1",
        ):
            assert self.config.get_api_base() == "https://api.siliconflow.cn/v1"

        with patch(
            "litellm.llms.siliconflow.chat.transformation.get_secret_str",
            return_value=None,
        ):
            assert self.config.get_api_base() == SiliconFlowConfig.API_BASE_URL
            assert SiliconFlowConfig.API_BASE_URL == "https://api.siliconflow.cn/v1"

    def test_get_openai_compatible_provider_info(self):
        def fake_secret(name, *args, **kwargs):
            return "sk-secret" if name == "SILICONFLOW_API_KEY" else None

        with patch(
            "litellm.llms.siliconflow.chat.transformation.get_secret_str",
            side_effect=fake_secret,
        ):
            api_base, api_key = self.config._get_openai_compatible_provider_info(api_base=None, api_key=None)
        assert api_base == "https://api.siliconflow.cn/v1"
        assert api_key == "sk-secret"

    def test_supported_params_include_tools(self):
        params = self.config.get_supported_openai_params(model="deepseek-ai/DeepSeek-V3")
        for expected in ("temperature", "stream", "tools", "tool_choice"):
            assert expected in params

    def test_reasoning_effort_is_supported_for_siliconflow_reasoning_models(self):
        v4_params = self.config.get_supported_openai_params(model="deepseek-ai/DeepSeek-V4-Flash")
        qwen_params = self.config.get_supported_openai_params(model="Qwen/Qwen3.6-27B")
        unsupported_params = self.config.get_supported_openai_params(model="deepseek-ai/DeepSeek-V3")

        assert "reasoning_effort" in v4_params
        assert "reasoning_effort" in qwen_params
        assert "reasoning_effort" not in unsupported_params

    @pytest.mark.parametrize(
        ("reasoning_effort", "expected"),
        (("low", "high"), ("medium", "high"), ("high", "high"), ("xhigh", "max"), ("max", "max")),
    )
    def test_reasoning_effort_is_normalized(self, reasoning_effort, expected):
        optional_params = get_optional_params(
            model="deepseek-ai/DeepSeek-V4-Flash",
            custom_llm_provider="siliconflow",
            reasoning_effort=reasoning_effort,
        )

        assert optional_params["reasoning_effort"] == expected

    def test_default_reasoning_effort_uses_provider_default(self):
        optional_params = get_optional_params(
            model="deepseek-ai/DeepSeek-V4-Flash",
            custom_llm_provider="siliconflow",
            reasoning_effort="default",
        )

        assert "reasoning_effort" not in optional_params

    @pytest.mark.parametrize("reasoning_effort", ("none", "minimal"))
    def test_unsupported_reasoning_effort_is_rejected(self, reasoning_effort):
        with pytest.raises(litellm.BadRequestError, match="supports reasoning_effort values"):
            get_optional_params(
                model="deepseek-ai/DeepSeek-V4-Flash",
                custom_llm_provider="siliconflow",
                reasoning_effort=reasoning_effort,
            )

    @pytest.mark.parametrize(
        ("reasoning_effort", "expected"),
        (
            ("minimal", {"enable_thinking": True, "thinking_budget": 128}),
            ("low", {"enable_thinking": True, "thinking_budget": 1024}),
            ("medium", {"enable_thinking": True, "thinking_budget": 2048}),
            ("high", {"enable_thinking": True, "thinking_budget": 4096}),
            ("xhigh", {"enable_thinking": True, "thinking_budget": 8192}),
            ("max", {"enable_thinking": True, "thinking_budget": 16384}),
            ("none", {"enable_thinking": False}),
        ),
    )
    def test_qwen_reasoning_effort_maps_to_thinking_params(self, reasoning_effort, expected):
        optional_params = get_optional_params(
            model="Qwen/Qwen3.6-27B",
            custom_llm_provider="siliconflow",
            reasoning_effort=reasoning_effort,
        )

        assert optional_params["extra_body"] == expected
        assert "reasoning_effort" not in optional_params

    def test_qwen_default_reasoning_effort_uses_provider_default(self):
        optional_params = get_optional_params(
            model="Qwen/Qwen3.6-27B",
            custom_llm_provider="siliconflow",
            reasoning_effort="default",
        )

        assert optional_params.get("extra_body", {}) == {}
        assert "reasoning_effort" not in optional_params


class TestSiliconFlowProviderResolution:
    def test_get_llm_provider_resolves_prefixed_model(self):
        with patch.dict(os.environ, {"SILICONFLOW_API_KEY": "sk-secret"}, clear=False):
            model, provider, api_key, api_base = get_llm_provider(model="siliconflow/deepseek-ai/DeepSeek-V3")
        assert model == "deepseek-ai/DeepSeek-V3"
        assert provider == "siliconflow"
        assert api_key == "sk-secret"
        assert api_base == "https://api.siliconflow.cn/v1"

    def test_get_llm_provider_detects_provider_from_api_base(self):
        _, provider, _, _ = get_llm_provider(
            model="deepseek-ai/DeepSeek-V3",
            api_base="https://api.siliconflow.com/v1",
            api_key="sk-secret",
        )
        assert provider == "siliconflow"

    def test_get_supported_openai_params_routes_to_config(self):
        params = get_supported_openai_params(model="deepseek-ai/DeepSeek-V3", custom_llm_provider="siliconflow")
        assert params is not None
        assert "tools" in params

    def test_provider_registered_in_enum_and_lists(self):
        from litellm.types.utils import LlmProviders

        assert LlmProviders.SILICONFLOW.value == "siliconflow"
        assert "siliconflow" in litellm.openai_compatible_providers
        assert "api.siliconflow.com/v1" in litellm.openai_compatible_endpoints


def test_siliconflow_qwen3_8b_pricing_metadata() -> None:
    pricing_path = Path(__file__).parents[5] / "model_prices_and_context_window.json"
    pricing = json.loads(pricing_path.read_text())
    model_info = pricing["siliconflow/Qwen/Qwen3-8B"]

    assert model_info["input_cost_per_token"] == 0.0
    assert model_info["output_cost_per_token"] == 0.0
    assert model_info["mode"] == "chat"
