from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import BaseImageGenerationConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

from ..common_utils import SiliconFlowError, get_siliconflow_api_base, get_siliconflow_headers

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class _SiliconFlowGeneratedImage(BaseModel):
    url: str


class _SiliconFlowImageTimings(BaseModel):
    inference: float | None = None


class _SiliconFlowImageResponse(BaseModel):
    images: list[_SiliconFlowGeneratedImage] = Field(default_factory=list)
    timings: _SiliconFlowImageTimings | None = None
    seed: int | None = None


class SiliconFlowImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        common_params: list[OpenAIImageGenerationOptionalParams] = [
            "response_format",
            "size",
            "image_size",
            "seed",
            "image",
            "negative_prompt",
            "num_inference_steps",
        ]
        kolors_params: list[OpenAIImageGenerationOptionalParams] = ["n", "batch_size", "guidance_scale"]
        qwen_image_params: list[OpenAIImageGenerationOptionalParams] = ["cfg"]
        qwen_image_edit_params: list[OpenAIImageGenerationOptionalParams] = ["image2", "image3"]
        return [
            *common_params,
            *(kolors_params if "Kolors" in model else []),
            *(qwen_image_params if "Qwen-Image" in model else []),
            *(qwen_image_edit_params if "Qwen-Image-Edit-2509" in model else []),
        ]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        supported_params = self.get_supported_openai_params(model)
        unsupported_params = tuple(key for key in non_default_params if key not in supported_params)
        if unsupported_params and not drop_params:
            raise ValueError(f"Parameters {unsupported_params} are not supported for SiliconFlow image generation.")
        response_format = non_default_params.get("response_format")
        if response_format is not None and response_format != "url":
            raise ValueError("SiliconFlow image generation only returns URL responses.")
        mapped_params = {
            ("batch_size" if key == "n" else "image_size" if key == "size" else key): value
            for key, value in non_default_params.items()
            if key in supported_params and key != "response_format"
        }
        self._validate_provider_params(mapped_params)
        return {**optional_params, **mapped_params}

    @staticmethod
    def _validate_provider_params(params: dict[str, object]) -> None:
        seed = params.get("seed")
        if isinstance(seed, int) and not 0 <= seed <= 9_999_999_999:
            raise ValueError("SiliconFlow seed must be between 0 and 9999999999.")
        batch_size = params.get("batch_size")
        if isinstance(batch_size, int) and not 1 <= batch_size <= 4:
            raise ValueError("SiliconFlow batch_size must be between 1 and 4.")
        steps = params.get("num_inference_steps")
        if isinstance(steps, int) and not 1 <= steps <= 100:
            raise ValueError("SiliconFlow num_inference_steps must be between 1 and 100.")
        guidance_scale = params.get("guidance_scale")
        if isinstance(guidance_scale, (int, float)) and not 0 <= guidance_scale <= 20:
            raise ValueError("SiliconFlow guidance_scale must be between 0 and 20.")
        cfg = params.get("cfg")
        if isinstance(cfg, (int, float)) and not 0.1 <= cfg <= 20:
            raise ValueError("SiliconFlow cfg must be between 0.1 and 20.")

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: bool | None = None,
    ) -> str:
        base = get_siliconflow_api_base(api_base)
        return base if base.endswith("/images/generations") else f"{base}/images/generations"

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
            or get_secret_str("SILICONFLOW_API_KEY")
        )
        if not final_api_key:
            raise ValueError("SiliconFlow API key is required. Set SILICONFLOW_API_KEY.")
        return get_siliconflow_headers(final_api_key, headers)

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
            response = _SiliconFlowImageResponse.model_validate(raw_response.json())
        except ValidationError as exc:
            raise self.get_error_class(
                f"Invalid SiliconFlow image response: {exc}",
                raw_response.status_code,
                raw_response.headers,
            ) from exc
        return ImageResponse(
            data=[ImageObject(url=image.url) for image in response.images],
            hidden_params={
                "model": model,
                "generated_images": len(response.images),
                "seed": response.seed,
                "timings": response.timings.model_dump(exclude_none=True) if response.timings is not None else None,
            },
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        response_headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers)
        return SiliconFlowError(status_code=status_code, message=error_message, headers=response_headers)
