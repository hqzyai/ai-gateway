from typing import List, Literal, Union

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, PromptTokensDetailsWrapper, Usage

from ..common_utils import VolcEngineError, get_volcengine_base_url, get_volcengine_headers

JsonObject = dict[str, JsonValue]
_JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)
_OBJECT_LIST_ADAPTER = TypeAdapter(list[object])


class _VolcEngineEmbeddingData(BaseModel):
    model_config = ConfigDict(extra="allow")

    embedding: list[float] | str
    object: Literal["embedding"] = "embedding"
    index: int = 0
    sparse_embedding: list[JsonObject] | None = None
    multi_embedding: list[list[float]] | str | None = None


class _VolcEnginePromptTokensDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    text_tokens: int = 0
    image_tokens: int = 0


class _VolcEngineEmbeddingUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: _VolcEnginePromptTokensDetails | None = None


class _VolcEngineEmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    data: _VolcEngineEmbeddingData | list[_VolcEngineEmbeddingData]
    model: str | None = None
    object: str = "list"
    id: str | None = None
    usage: _VolcEngineEmbeddingUsage = Field(default_factory=_VolcEngineEmbeddingUsage)


class VolcEngineEmbeddingConfig(BaseEmbeddingConfig):
    encoding_format: str | None = None

    def __init__(self, encoding_format: str | None = None) -> None:
        if encoding_format is not None:
            VolcEngineEmbeddingConfig.encoding_format = encoding_format

    @staticmethod
    def is_multimodal_embedding(model: str) -> bool:
        return "doubao-embedding-vision" in model.lower()

    @staticmethod
    def supports_instructions_and_multi_embedding(model: str) -> bool:
        version = model.lower().rsplit("-", maxsplit=1)[-1]
        return version.isdigit() and int(version) >= 251215

    def get_supported_openai_params(self, model: str) -> List[str]:
        common_params = ["encoding_format", "dimensions", "extra_headers"]
        if self.is_multimodal_embedding(model):
            multimodal_params = ["sparse_embedding"]
            if self.supports_instructions_and_multi_embedding(model):
                return [*common_params, "instructions", *multimodal_params, "multi_embedding"]
            return [*common_params, *multimodal_params]
        return [*common_params, "user"]

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        stream: bool | None = None,
    ) -> str:
        endpoint = "/embeddings/multimodal" if self.is_multimodal_embedding(model) else "/embeddings"
        base_url = get_volcengine_base_url(api_base).rstrip("/")
        if base_url.endswith(endpoint):
            return base_url
        if endpoint.endswith("/multimodal") and base_url.endswith("/embeddings"):
            return f"{base_url}/multimodal"
        if base_url.endswith("/api/v3"):
            return f"{base_url}{endpoint}"
        return f"{base_url}/api/v3{endpoint}"

    def map_openai_params(
        self,
        non_default_params: JsonObject,
        optional_params: JsonObject,
        model: str,
        drop_params: bool,
    ) -> JsonObject:
        supported_params = self.get_supported_openai_params(model)
        unsupported_params = tuple(param for param in non_default_params if param not in supported_params)
        if unsupported_params and not drop_params:
            raise ValueError(f"Unsupported parameters for Volcengine embeddings: {', '.join(unsupported_params)}")

        encoding_format = non_default_params.get("encoding_format")
        if encoding_format is not None and (
            not isinstance(encoding_format, str) or encoding_format not in ("float", "base64")
        ):
            raise ValueError(
                f"Unsupported encoding_format: {encoding_format}. Volcengine supports: float, base64, null"
            )

        dimensions = non_default_params.get("dimensions")
        if self.is_multimodal_embedding(model) and (
            dimensions is not None and (not isinstance(dimensions, int) or dimensions not in (1024, 2048))
        ):
            raise ValueError("Volcengine multimodal embeddings support dimensions of 1024 or 2048")

        return {
            **optional_params,
            **{
                param: value
                for param, value in non_default_params.items()
                if param in supported_params and value is not None
            },
        }

    @staticmethod
    def _normalize_multimodal_item(item: object) -> JsonObject:
        if isinstance(item, str):
            return {"type": "text", "text": item}
        if not isinstance(item, dict):
            raise ValueError("Volcengine multimodal embeddings accept text, image_url, and video_url input blocks")
        normalized_item = _JSON_OBJECT_ADAPTER.validate_python(item)
        item_type = normalized_item.get("type")
        if not isinstance(item_type, str) or item_type not in ("text", "image_url", "video_url"):
            raise ValueError(f"Unsupported Volcengine multimodal embedding input type: {item_type}")
        return normalized_item

    @classmethod
    def _normalize_multimodal_input(cls, input_value: object) -> list[JsonObject]:
        input_items = (
            _OBJECT_LIST_ADAPTER.validate_python(input_value) if isinstance(input_value, list) else [input_value]
        )
        if not input_items:
            raise ValueError("Volcengine multimodal embeddings require at least one input block")
        return [cls._normalize_multimodal_item(item) for item in input_items]

    @classmethod
    def _validate_multimodal_options(
        cls,
        model: str,
        input_items: list[JsonObject],
        optional_params: JsonObject,
    ) -> None:
        if not cls.supports_instructions_and_multi_embedding(model) and (
            optional_params.get("instructions") is not None or optional_params.get("multi_embedding") is not None
        ):
            raise ValueError(
                "Volcengine instructions and multi_embedding require doubao-embedding-vision-251215 or later"
            )
        sparse_embedding = optional_params.get("sparse_embedding")
        sparse_config = (
            _JSON_OBJECT_ADAPTER.validate_python(sparse_embedding) if isinstance(sparse_embedding, dict) else None
        )
        if (
            sparse_config
            and sparse_config.get("type") == "enabled"
            and any(item.get("type") != "text" for item in input_items)
        ):
            raise ValueError("Volcengine sparse_embedding supports text-only input")

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: JsonObject,
        headers: dict[str, str],
    ) -> JsonObject:
        normalized_input: object
        if self.is_multimodal_embedding(model):
            multimodal_input = self._normalize_multimodal_input(input)
            self._validate_multimodal_options(model, multimodal_input, optional_params)
            normalized_input = multimodal_input
        else:
            normalized_input = input if isinstance(input, list) else [input]
        return _JSON_OBJECT_ADAPTER.validate_python({"model": model, "input": normalized_input, **optional_params})

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        request_data: JsonObject,
        optional_params: JsonObject,
        litellm_params: JsonObject,
    ) -> EmbeddingResponse:
        try:
            parsed_response = _VolcEngineEmbeddingResponse.model_validate(raw_response.json())
        except Exception as exc:
            raise ValueError(f"Failed to parse Volcengine embedding response: {exc}") from exc

        response_data = parsed_response.data if isinstance(parsed_response.data, list) else [parsed_response.data]
        prompt_details = parsed_response.usage.prompt_tokens_details
        response_usage = Usage(
            prompt_tokens=parsed_response.usage.prompt_tokens,
            total_tokens=parsed_response.usage.total_tokens,
            prompt_tokens_details=(
                PromptTokensDetailsWrapper(
                    text_tokens=prompt_details.text_tokens,
                    image_tokens=prompt_details.image_tokens,
                )
                if prompt_details is not None
                else None
            ),
        )
        return EmbeddingResponse(
            model=parsed_response.model or model,
            data=[item.model_dump(exclude_none=True) for item in response_data],
            usage=response_usage,
        )

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: List[AllMessageValues],
        optional_params: JsonObject,
        litellm_params: JsonObject,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        if api_key is None:
            raise ValueError("api_key is required for Volcengine authentication")
        return {**headers, **get_volcengine_headers(api_key)}

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict[str, str], httpx.Headers]
    ) -> BaseLLMException:
        normalized_headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers)
        return VolcEngineError(status_code=status_code, message=error_message, headers=normalized_headers)
