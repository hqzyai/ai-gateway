from collections.abc import Mapping, Sequence

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import RerankResponse

from ..common_utils import SiliconFlowError, get_siliconflow_api_base, get_siliconflow_headers


class SiliconFlowRerankParameters(dict[str, object]):
    pass


class SiliconFlowHeaders(dict[str, str]):
    pass


class SiliconFlowRerankConfig(BaseRerankConfig):
    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: Mapping[str, object] | None = None,
    ) -> str:
        base = get_siliconflow_api_base(api_base)
        return base if base.endswith("/rerank") else f"{base}/rerank"

    def get_supported_cohere_rerank_params(self, model: str) -> Sequence[str]:
        return (
            "query",
            "documents",
            "top_n",
            "return_documents",
            "max_chunks_per_doc",
            "instruction",
            "overlap_tokens",
        )

    def map_cohere_rerank_params(
        self,
        non_default_params: Mapping[str, object],
        model: str,
        drop_params: bool,
        query: str | Mapping[str, object],
        documents: Sequence[str | Mapping[str, object]],
        custom_llm_provider: str | None = None,
        top_n: int | None = None,
        rank_fields: Sequence[str] | None = None,
        return_documents: bool | None = True,
        max_chunks_per_doc: int | None = None,
        max_tokens_per_doc: int | None = None,
        instruction: str | None = None,
    ) -> SiliconFlowRerankParameters:
        values = (
            ("query", query),
            ("documents", documents),
            ("top_n", top_n),
            ("return_documents", return_documents),
            ("max_chunks_per_doc", max_chunks_per_doc),
            ("instruction", instruction),
            ("overlap_tokens", non_default_params.get("overlap_tokens")),
        )
        return SiliconFlowRerankParameters((key, value) for key, value in values if value is not None)

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        optional_params: Mapping[str, object] | None = None,
    ) -> SiliconFlowHeaders:
        final_api_key = api_key or get_secret_str("SILICONFLOW_API_KEY")
        if final_api_key is None:
            raise ValueError("SiliconFlow API key is required. Set SILICONFLOW_API_KEY or pass api_key explicitly.")
        return SiliconFlowHeaders(get_siliconflow_headers(final_api_key, headers))

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Mapping[str, object],
        headers: Mapping[str, str],
        litellm_params: Mapping[str, object] | None = None,
    ) -> SiliconFlowRerankParameters:
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for SiliconFlow rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for SiliconFlow rerank")
        allowed_params = self.get_supported_cohere_rerank_params(model)
        values = (
            ("model", model.removeprefix("siliconflow/")),
            *((key, value) for key, value in optional_rerank_params.items() if key in allowed_params),
        )
        return SiliconFlowRerankParameters(values)

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: Mapping[str, object] | None = None,
        optional_params: Mapping[str, object] | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> RerankResponse:
        if raw_response.status_code >= 400:
            raise self.get_error_class(raw_response.text, raw_response.status_code, raw_response.headers)
        try:
            response = RerankResponse.model_validate(raw_response.json())
        except ValueError as exc:
            raise self.get_error_class(
                f"Invalid SiliconFlow rerank response: {exc}",
                raw_response.status_code,
                raw_response.headers,
            ) from exc
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, str] | httpx.Headers,
    ) -> SiliconFlowError:
        response_headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers)
        return SiliconFlowError(status_code=status_code, message=error_message, headers=response_headers)
