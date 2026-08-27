"""
Transformation logic from OpenAI /v1/embeddings format to DashScope's /v1/embeddings format.

Supports
- text-embedding-v4
- text-embedding-v3

Endpoint
- https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings

Docs - https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api
"""

from collections.abc import Mapping, Sequence
from typing import List, Optional

import httpx
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, PromptTokensDetailsWrapper, Usage

from ..common_utils import DashScopeError

DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MULTIMODAL_API_BASE = "https://dashscope.aliyuncs.com/api/v1"
MULTIMODAL_EMBEDDING_PATH = "/services/embeddings/multimodal-embedding/multimodal-embedding"
MULTIMODAL_MODELS = frozenset(
    (
        "qwen3-vl-embedding",
        "qwen2.5-vl-embedding",
        "tongyi-embedding-vision-flash",
        "tongyi-embedding-vision-plus",
        "tongyi-embedding-vision-plus-2026-03-06",
        "tongyi-embedding-vision-flash-2026-03-06",
        "multimodal-embedding-v1",
    )
)
VIDEO_MULTIMODAL_MODELS = frozenset(
    (
        "tongyi-embedding-vision-flash",
        "tongyi-embedding-vision-plus",
        "tongyi-embedding-vision-plus-2026-03-06",
        "tongyi-embedding-vision-flash-2026-03-06",
        "multimodal-embedding-v1",
    )
)
JsonObject = dict[str, JsonValue]
JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)


class DashScopeEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api

    DashScope exposes an OpenAI-compatible /v1/embeddings endpoint, so the
    request and response shapes are nearly identical to OpenAI's.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def is_multimodal_embedding(model: str) -> bool:
        return model.lower() in MULTIMODAL_MODELS

    def get_supported_openai_params(self, model: str) -> List[str]:
        # DashScope's compatible-mode embeddings API accepts the same params as OpenAI.
        # `dimensions` / `encoding_format` are only honored by text-embedding-v3 / v4;
        # earlier versions silently ignore them server-side.
        if self.is_multimodal_embedding(model):
            return ["dimensions", "enable_fusion"]
        return ["dimensions", "encoding_format", "user"]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool = False,
    ) -> dict:
        supported = self.get_supported_openai_params(model)
        return {
            **optional_params,
            **{key: value for key, value in non_default_params.items() if value is not None and key in supported},
        }

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            api_key = get_secret_str("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError(
                "DashScope API key is required. Set 'DASHSCOPE_API_KEY' env var or pass api_key explicitly."
            )
        default_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        return {**default_headers, **headers}

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: Optional[bool] = None,
    ) -> str:
        if self.is_multimodal_embedding(model):
            base = (
                api_base
                or get_secret_str("DASHSCOPE_API_BASE_EMBEDDING")
                or get_secret_str("DASHSCOPE_API_BASE")
                or DEFAULT_MULTIMODAL_API_BASE
            )
            normalized_base = base.rstrip("/")
            if normalized_base.endswith(MULTIMODAL_EMBEDDING_PATH):
                return normalized_base
            if normalized_base.endswith("/compatible-mode/v1"):
                normalized_base = f"{normalized_base[: -len('/compatible-mode/v1')]}/api/v1"
            elif not normalized_base.endswith("/api/v1"):
                normalized_base = f"{normalized_base}/api/v1"
            return f"{normalized_base}{MULTIMODAL_EMBEDDING_PATH}"

        base = api_base or get_secret_str("DASHSCOPE_API_BASE") or DEFAULT_API_BASE
        base = base.rstrip("/")
        if base.endswith("/embeddings"):
            return base
        return f"{base}/embeddings"

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict:
        if self.is_multimodal_embedding(model):
            dimension = optional_params.get("dimensions")
            parameters = {
                **({"dimension": dimension} if dimension is not None else {}),
                **(
                    {"enable_fusion": optional_params["enable_fusion"]}
                    if optional_params.get("enable_fusion") is not None
                    else {}
                ),
            }
            return {
                "model": model,
                "input": {"contents": self._normalize_multimodal_input(model, input)},
                "parameters": parameters,
            }
        data: dict = {
            "model": model,
            "input": input,
        }
        for key in ("dimensions", "encoding_format", "user"):
            value = optional_params.get(key)
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def _normalize_multimodal_input(cls, model: str, input_value: object) -> list[JsonObject]:
        input_items = input_value if isinstance(input_value, list) else [input_value]
        if not input_items:
            raise ValueError("DashScope multimodal embeddings require at least one input item.")
        return [cls._normalize_multimodal_item(model, item) for item in input_items]

    @staticmethod
    def _normalize_multimodal_item(model: str, item: object) -> JsonObject:
        if isinstance(item, str):
            return {"text": item}
        try:
            value = JSON_OBJECT_ADAPTER.validate_python(item)
        except ValidationError as exc:
            raise ValueError("DashScope multimodal embedding inputs must be text, image, or video objects.") from exc

        item_type = value.get("type")
        if item_type is None:
            if any(key in value for key in ("text", "image", "video", "multi_images")):
                if "video" in value and model.lower() not in VIDEO_MULTIMODAL_MODELS:
                    raise ValueError(f"DashScope model {model} does not support video embedding input.")
                return value
            raise ValueError("DashScope multimodal embedding input object is missing a supported content field.")
        if item_type == "text" and isinstance(value.get("text"), str):
            return {"text": value["text"]}
        if item_type in ("image_url", "video_url"):
            if item_type == "video_url" and model.lower() not in VIDEO_MULTIMODAL_MODELS:
                raise ValueError(f"DashScope model {model} does not support video embedding input.")
            raw_url = value.get(item_type)
            url = raw_url.get("url") if isinstance(raw_url, dict) else raw_url
            if isinstance(url, str):
                return {"image" if item_type == "image_url" else "video": url}
        raise ValueError(f"Unsupported DashScope multimodal embedding input type: {item_type}")

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: Mapping[str, object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> EmbeddingResponse:
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise DashScopeError(
                status_code=raw_response.status_code,
                message=f"Failed to parse DashScope response as JSON: {str(e)}",
            )

        logging_obj.post_call(
            input=request_data.get("input"),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_json,
        )

        if "error" in response_json:
            error = response_json["error"]
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise DashScopeError(
                status_code=raw_response.status_code,
                message=message,
            )

        if self.is_multimodal_embedding(model):
            output = response_json.get("output") or {}
            embeddings = output.get("embeddings") or []
            model_response.data = [
                {
                    "embedding": item.get("embedding", []),
                    "index": item.get("index", index),
                    "object": "embedding",
                }
                for index, item in enumerate(embeddings)
                if isinstance(item, dict)
            ]
        else:
            model_response.data = response_json.get("data", [])
        model_response.object = "list"
        model_response.model = response_json.get("model", model)

        usage = response_json.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        total_tokens = usage.get("total_tokens", prompt_tokens)
        input_token_details = usage.get("input_tokens_details") or {}
        prompt_token_details = (
            PromptTokensDetailsWrapper(
                text_tokens=input_token_details.get("text_tokens", 0),
                image_tokens=input_token_details.get("image_tokens", 0),
                video_tokens=input_token_details.get("video_tokens", 0),
            )
            if input_token_details
            else None
        )
        setattr(
            model_response,
            "usage",
            Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
                prompt_tokens_details=prompt_token_details,
            ),
        )

        if "id" in response_json:
            setattr(model_response, "id", response_json["id"])
        elif "request_id" in response_json:
            model_response.id = response_json["request_id"]

        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        return DashScopeError(
            status_code=status_code,
            message=error_message,
            headers=httpx.Headers(headers),
        )
