"""
DashScope Image Generation Configuration

Handles transformation between OpenAI-compatible format and DashScope multimodal-generation API.

API endpoint: POST https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation

Request format:
{
    "model": "qwen-image-2.0-pro",
    "input": {
        "messages": [{"role": "user", "content": [{"text": "<prompt>"}]}]
    },
    "parameters": {"size": "1024*1024", ...}
}

Response format:
{
    "output": {
        "choices": [{"message": {"content": [{"image": "<url>"}]}}]
    },
    "usage": {"input_tokens": 0, "output_tokens": 0, "width": 1024, "height": 1024, "image_count": 1}
}
"""

from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse
from litellm.types.utils import ImageUsage, ImageUsageInputTokensDetails

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

DEFAULT_API_BASE = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
DASHSCOPE_IMAGE_GENERATION_PATH = "/services/aigc/multimodal-generation/generation"

# Maps OpenAI size strings (WxH) to DashScope size strings (W*H)
OPENAI_TO_DASHSCOPE_SIZE: dict = {
    "256x256": "256*256",
    "512x512": "512*512",
    "1024x1024": "1024*1024",
    "1792x1024": "1792*1024",
    "1024x1792": "1024*1792",
    "2048x2048": "2048*2048",
}


class DashScopeImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for DashScope image generation (qwen-image-2.0, qwen-image-2.0-pro).
    """

    def get_supported_openai_params(self, model: str) -> List[OpenAIImageGenerationOptionalParams]:
        return [
            "n",
            "size",
            "image_url",
            "image",
            "watermark",
            "negative_prompt",
            "prompt_upsampling",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        mapped: dict = {}
        for k, v in non_default_params.items():
            if k in optional_params:
                continue
            if k not in supported_params:
                continue
            if k == "size":
                # Convert "WxH" → "W*H"
                mapped["size"] = OPENAI_TO_DASHSCOPE_SIZE.get(v, v.replace("x", "*"))
            elif k == "n":
                mapped["n"] = v
            elif k == "prompt_upsampling":
                mapped["prompt_extend"] = v
            else:
                mapped[k] = v
        return mapped

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        configured_api_base = api_base or get_secret_str("DASHSCOPE_API_BASE_IMAGE") or DEFAULT_API_BASE
        normalized_api_base = configured_api_base.rstrip("/")
        if normalized_api_base.endswith(DASHSCOPE_IMAGE_GENERATION_PATH):
            return normalized_api_base
        if normalized_api_base.endswith("/api/v1"):
            return f"{normalized_api_base}{DASHSCOPE_IMAGE_GENERATION_PATH}"
        return normalized_api_base

    @staticmethod
    def _normalize_image_urls(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        raise ValueError("DashScope image input must be a URL string or a list of URL strings.")

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        final_api_key = api_key or get_secret_str("DASHSCOPE_API_KEY")
        if not final_api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")
        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform OpenAI-style image generation request to DashScope multimodal-generation format.
        """
        image_urls = [
            *self._normalize_image_urls(optional_params.get("image_url")),
            *self._normalize_image_urls(optional_params.get("image")),
        ]
        if len(image_urls) > 3:
            raise ValueError("DashScope Qwen image editing supports at most 3 input images.")

        parameters = {k: v for k, v in optional_params.items() if k not in ("image_url", "image")}
        content = [{"image": image_url} for image_url in image_urls] + [{"text": prompt}]

        return {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            },
            "parameters": parameters,
        }

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        """
        Transform DashScope response to litellm ImageResponse.

        DashScope response: output.choices[0].message.content[0].image
        OpenAI response:    data[0].url
        """
        if raw_response.status_code != 200:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Failed to parse DashScope image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        # DashScope can return API-level errors in a 200 response body.
        # Example: {"code": "InvalidParameter", "message": "Size not supported"}
        if "code" in response_data and "output" not in response_data:
            raise self.get_error_class(
                error_message=str(response_data.get("message", response_data)),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []

        choices = response_data.get("output", {}).get("choices", [])
        for choice in choices:
            content_list = choice.get("message", {}).get("content", [])
            for content_item in content_list:
                image_url = content_item.get("image")
                if image_url:
                    model_response.data.append(ImageObject(url=image_url))

        usage_data = response_data.get("usage")
        if isinstance(usage_data, dict):
            input_tokens = usage_data.get("input_tokens", 0)
            output_tokens = usage_data.get("output_tokens", 0)
            normalized_input_tokens = input_tokens if isinstance(input_tokens, int) else 0
            normalized_output_tokens = output_tokens if isinstance(output_tokens, int) else 0
            model_response.usage = ImageUsage(
                input_tokens=normalized_input_tokens,
                input_tokens_details=ImageUsageInputTokensDetails(
                    image_tokens=0,
                    text_tokens=normalized_input_tokens,
                ),
                output_tokens=normalized_output_tokens,
                total_tokens=normalized_input_tokens + normalized_output_tokens,
                input_image_count=usage_data.get("input_image_count"),
                input_image_type=usage_data.get("input_image_type"),
                output_image_count=usage_data.get("output_image_count"),
                output_image_type=usage_data.get("output_image_type"),
                output_width=usage_data.get("output_width"),
                output_height=usage_data.get("output_height"),
            )
            model_response._hidden_params["model"] = response_data.get("model", model)
            model_response._hidden_params["input_image_count"] = usage_data.get("input_image_count")
            model_response._hidden_params["input_image_type"] = usage_data.get("input_image_type")
            model_response._hidden_params["output_image_count"] = usage_data.get("output_image_count")
            model_response._hidden_params["output_image_type"] = usage_data.get("output_image_type")
            model_response._hidden_params["output_width"] = usage_data.get("output_width")
            model_response._hidden_params["output_height"] = usage_data.get("output_height")
            output_width = usage_data.get("output_width")
            output_height = usage_data.get("output_height")
            if isinstance(output_width, int) and isinstance(output_height, int):
                model_response.size = f"{output_width}x{output_height}"

        return model_response
