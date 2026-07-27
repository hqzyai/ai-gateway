from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field, ValidationError

from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
)

from ..common_utils import (
    VolcEngineError,
    get_volcengine_api_base,
    get_volcengine_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class _VolcEngineImageError(BaseModel):
    message: str
    code: str


class _VolcEngineImageData(BaseModel):
    url: str | None = None
    b64_json: str | None = None
    size: str | None = None
    output_format: str | None = None
    error: _VolcEngineImageError | None = None


class _VolcEngineImageToolUsage(BaseModel):
    web_search: int = 0


class _VolcEngineImageUsage(BaseModel):
    generated_images: int | None = None
    input_images: int | None = None
    output_tokens: int = 0
    total_tokens: int = 0
    tool_usage: _VolcEngineImageToolUsage = Field(default_factory=_VolcEngineImageToolUsage)


class _VolcEngineImageResponse(BaseModel):
    model: str | None = None
    data: list[_VolcEngineImageData] = Field(default_factory=list)
    error: _VolcEngineImageError | None = None
    usage: _VolcEngineImageUsage = Field(default_factory=_VolcEngineImageUsage)
    created: int | None = None
    tools: list[dict[str, object]] = Field(default_factory=list)


def _count_request_images(image: object) -> int:
    if isinstance(image, list):
        return len(image)
    if isinstance(image, str) and image:
        return 1
    return 0


def _image_provider_fields(image: _VolcEngineImageData) -> dict[str, object] | None:
    fields: dict[str, object] = {
        key: value
        for key, value in {
            "size": image.size,
            "output_format": image.output_format,
            "error": image.error.model_dump() if image.error is not None else None,
        }.items()
        if value is not None
    }
    return fields or None


class VolcEngineImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        common_params: list[OpenAIImageGenerationOptionalParams] = [
            "image",
            "response_format",
            "size",
            "seed",
            "watermark",
            "optimize_prompt_options",
        ]
        if "seedream-5-0-pro" in model:
            return [*common_params, "output_format"]
        if "seedream-5-0" in model:
            return [
                *common_params,
                "output_format",
                "sequential_image_generation",
                "sequential_image_generation_options",
                "tools",
            ]
        return [
            *common_params,
            "sequential_image_generation",
            "sequential_image_generation_options",
        ]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        supported_params = self.get_supported_openai_params(model)
        mapped_params = {key: value for key, value in non_default_params.items() if key in supported_params}
        unsupported_params = tuple(key for key in non_default_params if key not in supported_params)
        if unsupported_params and not drop_params:
            raise ValueError(
                f"Parameters {unsupported_params} are not supported for Volcengine image generation. "
                f"Supported parameters are {supported_params}."
            )
        return {**optional_params, **mapped_params}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: bool | None = None,
    ) -> str:
        configured_base = api_base or get_secret_str("VOLCENGINE_API_BASE") or get_secret_str("ARK_API_BASE")
        complete_base = (configured_base or "").rstrip("/")
        if complete_base.endswith("/images/generations"):
            return complete_base
        return f"{get_volcengine_api_base(configured_base)}/images/generations"

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        params_api_key = litellm_params.get("api_key")
        final_api_key = (
            api_key
            or (params_api_key if isinstance(params_api_key, str) else None)
            or get_secret_str("VOLCENGINE_API_KEY")
            or get_secret_str("ARK_API_KEY")
        )
        if not final_api_key:
            raise ValueError("Volcengine API key is required. Set VOLCENGINE_API_KEY or ARK_API_KEY.")
        return get_volcengine_headers(api_key=final_api_key, extra_headers=headers)

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        return {"model": model, "prompt": prompt, **optional_params}

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict[str, object],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        if raw_response.status_code >= 400:
            raise self.get_error_class(raw_response.text, raw_response.status_code, raw_response.headers)
        try:
            response = _VolcEngineImageResponse.model_validate(raw_response.json())
        except ValidationError as exc:
            raise self.get_error_class(
                f"Invalid Volcengine image generation response: {exc}",
                raw_response.status_code,
                raw_response.headers,
            ) from exc
        if response.error is not None:
            raise self.get_error_class(
                f"{response.error.code}: {response.error.message}",
                raw_response.status_code,
                raw_response.headers,
            )
        images = [
            ImageObject(
                url=image.url,
                b64_json=image.b64_json,
                provider_specific_fields=_image_provider_fields(image),
            )
            for image in response.data
        ]
        input_images = response.usage.input_images
        if input_images is None:
            input_images = _count_request_images(request_data.get("image"))
        usage = ImageUsage(
            input_tokens=0,
            input_tokens_details=ImageUsageInputTokensDetails(image_tokens=0, text_tokens=0),
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
        )
        return ImageResponse(
            created=response.created if response.created is not None else model_response.created,
            data=images,
            usage=usage,
            hidden_params={
                "model": response.model or model,
                "generated_images": response.usage.generated_images,
                "input_images": input_images,
                "tools": response.tools,
                "tool_usage": response.usage.tool_usage.model_dump(),
                "additional_headers": process_response_headers(raw_response.headers),
            },
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        response_headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers)
        return VolcEngineError(status_code=status_code, message=error_message, headers=response_headers)
