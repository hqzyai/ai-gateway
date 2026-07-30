from typing import Tuple

import litellm
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str


class SiliconFlowConfig(OpenAIGPTConfig):
    """
    Reference: SiliconFlow is OpenAI compatible.

    API Key: SILICONFLOW_API_KEY
    Default API Base: https://api.siliconflow.cn/v1

    Users on the China mainland endpoint can set
    SILICONFLOW_API_BASE=https://api.siliconflow.cn/v1
    """

    API_BASE_URL = "https://api.siliconflow.cn/v1"
    REASONING_EFFORT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"

    @property
    def custom_llm_provider(self) -> str | None:
        return "siliconflow"

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("SILICONFLOW_API_KEY")

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("SILICONFLOW_API_BASE") or SiliconFlowConfig.API_BASE_URL

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> Tuple[str | None, str | None]:
        resolved_api_base = SiliconFlowConfig.get_api_base(api_base)
        resolved_api_key = SiliconFlowConfig.get_api_key(api_key)
        return resolved_api_base, resolved_api_key

    @classmethod
    def _is_qwen_reasoning_model(cls, model: str) -> bool:
        normalized_model = model.removeprefix("siliconflow/").removeprefix("Pro/")
        return normalized_model.startswith("Qwen/Qwen3")

    @classmethod
    def _supports_reasoning_effort(cls, model: str) -> bool:
        normalized_model = model.removeprefix("siliconflow/")
        return normalized_model == cls.REASONING_EFFORT_MODEL or cls._is_qwen_reasoning_model(model)

    def get_supported_openai_params(self, model: str) -> list:
        base_params = (
            "max_tokens",
            "max_completion_tokens",
            "n",
            "temperature",
            "top_p",
            "seed",
            "stream",
            "stream_options",
            "logprobs",
            "top_logprobs",
            "frequency_penalty",
            "presence_penalty",
            "response_format",
            "stop",
            "logit_bias",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "user",
        )
        if self._supports_reasoning_effort(model):
            return [*base_params, "reasoning_effort"]
        return list(base_params)

    @staticmethod
    def _normalize_reasoning_effort(value: object, model: str) -> str | None:
        match value:
            case "default":
                return None
            case "low" | "medium" | "high":
                return "high"
            case "xhigh" | "max":
                return "max"
            case _:
                raise litellm.BadRequestError(
                    message=(
                        "SiliconFlow deepseek-ai/DeepSeek-V4-Flash supports reasoning_effort values "
                        "low, medium, high, xhigh, max, or default."
                    ),
                    model=model,
                    llm_provider="siliconflow",
                )

    @staticmethod
    def _map_reasoning_effort_to_thinking_params(value: object, model: str) -> dict[str, bool | int]:
        match value:
            case "default":
                return {}
            case "none":
                return {"enable_thinking": False}
            case "minimal":
                return {
                    "enable_thinking": True,
                    "thinking_budget": DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
                }
            case "low":
                return {
                    "enable_thinking": True,
                    "thinking_budget": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
                }
            case "medium":
                return {
                    "enable_thinking": True,
                    "thinking_budget": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
                }
            case "high":
                return {
                    "enable_thinking": True,
                    "thinking_budget": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
                }
            case "xhigh":
                return {
                    "enable_thinking": True,
                    "thinking_budget": DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
                }
            case "max":
                return {
                    "enable_thinking": True,
                    "thinking_budget": DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
                }
            case _:
                raise litellm.BadRequestError(
                    message=(
                        "SiliconFlow Qwen reasoning models support reasoning_effort values none, minimal, low, "
                        "medium, high, xhigh, max, or default."
                    ),
                    model=model,
                    llm_provider="siliconflow",
                )

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        if "reasoning_effort" not in non_default_params:
            return super().map_openai_params(non_default_params, optional_params, model, drop_params)

        if self._is_qwen_reasoning_model(model):
            thinking_params = self._map_reasoning_effort_to_thinking_params(
                non_default_params["reasoning_effort"], model
            )
            params_without_effort = {
                key: value for key, value in non_default_params.items() if key != "reasoning_effort"
            }
            mapped_params = super().map_openai_params(params_without_effort, optional_params, model, drop_params)
            if not thinking_params:
                return mapped_params
            existing_extra_body = mapped_params.get("extra_body")
            safe_extra_body = existing_extra_body if isinstance(existing_extra_body, dict) else {}
            return {**mapped_params, "extra_body": {**safe_extra_body, **thinking_params}}

        if model.removeprefix("siliconflow/") != self.REASONING_EFFORT_MODEL:
            return super().map_openai_params(non_default_params, optional_params, model, drop_params)

        normalized_effort = self._normalize_reasoning_effort(non_default_params["reasoning_effort"], model)
        params_without_effort = {key: value for key, value in non_default_params.items() if key != "reasoning_effort"}
        normalized_params = (
            {**params_without_effort, "reasoning_effort": normalized_effort}
            if normalized_effort is not None
            else params_without_effort
        )
        return super().map_openai_params(normalized_params, optional_params, model, drop_params)
