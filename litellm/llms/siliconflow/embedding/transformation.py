from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage

from ..common_utils import SiliconFlowError, get_siliconflow_api_base, get_siliconflow_headers

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class _SiliconFlowEmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float] | str
    index: int


class _SiliconFlowEmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class _SiliconFlowEmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[_SiliconFlowEmbeddingData] = Field(default_factory=list)
    model: str | None = None
    usage: _SiliconFlowEmbeddingUsage = Field(default_factory=_SiliconFlowEmbeddingUsage)


class SiliconFlowEmbeddingConfig(BaseEmbeddingConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["encoding_format", "dimensions", "user", "truncate", "extra_headers"]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        unsupported_params = tuple(
            key for key in non_default_params if key not in self.get_supported_openai_params(model)
        )
        if unsupported_params and not drop_params:
            raise ValueError(f"Parameters {unsupported_params} are not supported for SiliconFlow embeddings.")
        encoding_format = non_default_params.get("encoding_format")
        if encoding_format is not None and encoding_format not in {"float", "base64"}:
            raise ValueError("SiliconFlow encoding_format must be 'float' or 'base64'.")
        truncate = non_default_params.get("truncate")
        if truncate is not None and truncate not in {"left", "right"}:
            raise ValueError("SiliconFlow truncate must be 'left' or 'right'.")
        mapped_params = {
            key: value
            for key, value in non_default_params.items()
            if key in self.get_supported_openai_params(model) and key != "extra_headers"
        }
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
        base = get_siliconflow_api_base(api_base)
        return base if base.endswith("/embeddings") else f"{base}/embeddings"

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        request_params = {
            key: value
            for key, value in optional_params.items()
            if key in {"encoding_format", "dimensions", "user", "truncate"}
        }
        return {"model": model, "input": input, **request_params}

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        request_data: dict[str, object],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
    ) -> EmbeddingResponse:
        if raw_response.status_code >= 400:
            raise self.get_error_class(raw_response.text, raw_response.status_code, raw_response.headers)
        try:
            response = _SiliconFlowEmbeddingResponse.model_validate(raw_response.json())
        except ValidationError as exc:
            raise self.get_error_class(
                f"Invalid SiliconFlow embedding response: {exc}",
                raw_response.status_code,
                raw_response.headers,
            ) from exc
        return EmbeddingResponse(
            object=response.object,
            data=[item.model_dump() for item in response.data],
            model=response.model or model,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=0,
                total_tokens=response.usage.total_tokens,
            ),
        )

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

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        response_headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers)
        return SiliconFlowError(status_code=status_code, message=error_message, headers=response_headers)
