"""Transformation logic for DashScope text and multimodal rerank APIs."""

from typing import Any, Dict, List, Union

import httpx

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
    RerankTokens,
)
from litellm.types.utils import ModelInfo

from ..common_utils import DashScopeError

DEFAULT_RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
DEFAULT_NATIVE_RERANK_BASE = "https://dashscope.aliyuncs.com/api/v1"
NATIVE_RERANK_PATH = "/services/rerank/text-rerank/text-rerank"
NATIVE_RERANK_MODELS = frozenset(("gte-rerank-v2", "qwen3-vl-rerank"))


class DashScopeRerankConfig(BaseRerankConfig):
    """
    Reference: https://help.aliyun.com/zh/model-studio/text-rerank-api

    Targets DashScope's qwen3-rerank model. Request fields: model, query,
    documents, top_n, return_documents. Response: results[].index,
    results[].relevance_score, optionally results[].document.text (when
    return_documents=true), plus a top-level usage.total_tokens counter.
    """

    def __init__(self) -> None:
        pass

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: dict | None = None,
    ) -> str:
        native_api = model.lower() in NATIVE_RERANK_MODELS
        if api_base is None:
            api_base = get_secret_str("DASHSCOPE_API_BASE_RERANK") or (
                DEFAULT_NATIVE_RERANK_BASE if native_api else DEFAULT_RERANK_URL
            )

        cleaned = api_base.rstrip("/")
        if native_api:
            if cleaned.endswith(NATIVE_RERANK_PATH):
                return cleaned
            if cleaned.endswith(("/compatible-api/v1", "/compatible-mode/v1")):
                cleaned = f"{cleaned.rsplit('/', 2)[0]}/api/v1"
            elif not cleaned.endswith("/api/v1"):
                cleaned = f"{cleaned}/api/v1"
            return f"{cleaned}{NATIVE_RERANK_PATH}"

        if api_base == DEFAULT_RERANK_URL:
            return DEFAULT_RERANK_URL

        if cleaned.endswith("/reranks") or cleaned.endswith("/rerank"):
            return cleaned

        if cleaned.endswith("/v1"):
            return f"{cleaned}/reranks"

        # Unknown base: append /reranks rather than silently ignoring the caller's api_base.
        return f"{cleaned}/reranks"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        optional_params: dict | None = None,
    ) -> dict:
        if api_key is None:
            api_key = get_secret_str("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError(
                "DashScope API key is required. Set 'DASHSCOPE_API_KEY' env var or pass api_key explicitly."
            )

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
            "content-type": "application/json",
        }
        return {**default_headers, **headers}

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return ["query", "documents", "top_n", "return_documents", "instruction"]

    def map_cohere_rerank_params(
        self,
        non_default_params: dict | None,
        model: str,
        drop_params: bool,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: str | None = None,
        top_n: int | None = None,
        rank_fields: List[str] | None = None,
        return_documents: bool | None = True,
        max_chunks_per_doc: int | None = None,
        max_tokens_per_doc: int | None = None,
        instruction: str | None = None,
    ) -> Dict:
        # qwen3-rerank accepts query/documents/top_n/return_documents. The
        # rest (rank_fields, max_*_per_doc) are silently dropped.
        params: OptionalRerankParams = OptionalRerankParams(
            query=query,
            documents=documents,
        )
        if top_n is not None:
            params["top_n"] = top_n
        if return_documents is not None:
            params["return_documents"] = return_documents
        if instruction is not None:
            params["instruction"] = instruction
        extra_body = non_default_params.get("extra_body") if non_default_params else None
        return {
            **dict(params),
            **(
                {"fps": extra_body["fps"]} if isinstance(extra_body, dict) and extra_body.get("fps") is not None else {}
            ),
        }

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Dict,
        headers: dict,
        litellm_params: dict | None = None,
    ) -> dict:
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for DashScope rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for DashScope rerank")

        if model.lower() in NATIVE_RERANK_MODELS:
            return {
                "model": model,
                "input": {
                    "query": self._normalize_native_content(optional_rerank_params["query"]),
                    "documents": [
                        self._normalize_native_content(document) for document in optional_rerank_params["documents"]
                    ],
                },
                "parameters": self._native_parameters(optional_rerank_params),
            }

        request: Dict[str, Any] = {
            "model": model,
            "query": optional_rerank_params["query"],
            "documents": optional_rerank_params["documents"],
        }
        if optional_rerank_params.get("top_n") is not None:
            request["top_n"] = optional_rerank_params["top_n"]
        if optional_rerank_params.get("return_documents") is not None:
            request["return_documents"] = optional_rerank_params["return_documents"]
        if optional_rerank_params.get("instruction") is not None:
            request["instruct"] = optional_rerank_params["instruction"]
        return request

    @staticmethod
    def _native_parameters(optional_rerank_params: Dict) -> Dict[str, Any]:
        return {
            target: optional_rerank_params[source]
            for source, target in (
                ("top_n", "top_n"),
                ("return_documents", "return_documents"),
                ("instruction", "instruct"),
                ("fps", "fps"),
            )
            if optional_rerank_params.get(source) is not None
        }

    @staticmethod
    def _normalize_native_content(content: object) -> Dict[str, Any]:
        if isinstance(content, str):
            return {"text": content}
        if not isinstance(content, dict):
            raise TypeError("DashScope native rerank inputs must be strings or content objects.")
        if any(key in content for key in ("text", "image", "video")):
            return content
        content_type = content.get("type")
        if content_type == "text" and isinstance(content.get("text"), str):
            return {"text": content["text"]}
        if content_type in ("image_url", "video_url"):
            raw_url = content.get(content_type)
            url = raw_url.get("url") if isinstance(raw_url, dict) else raw_url
            if isinstance(url, str):
                return {"image" if content_type == "image_url" else "video": url}
        raise ValueError("DashScope native rerank content must contain text, image, or video.")

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: dict | None = None,
        optional_params: dict | None = None,
        litellm_params: dict | None = None,
    ) -> RerankResponse:
        request_data = request_data or {}
        optional_params = optional_params or {}
        litellm_params = litellm_params or {}
        try:
            response_json = raw_response.json()
        except Exception:
            raise DashScopeError(
                status_code=raw_response.status_code,
                message=raw_response.text,
            )

        logging_obj.post_call(
            input=request_data.get("query"),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_json,
        )

        # DashScope error envelope: {"code": "...", "message": "...", "request_id": "..."}
        output = response_json.get("output") if isinstance(response_json.get("output"), dict) else {}
        results = output.get("results") if model.lower() in NATIVE_RERANK_MODELS else response_json.get("results")

        if "code" in response_json and results is None:
            raise DashScopeError(
                status_code=raw_response.status_code,
                message=response_json.get("message", str(response_json)),
            )

        if results is None:
            raise DashScopeError(
                status_code=raw_response.status_code,
                message=f"No results in DashScope rerank response: {response_json}",
            )

        # qwen3-rerank returns:
        #   {"index": int, "relevance_score": float}
        # plus, when return_documents=true was sent:
        #   "document": {"text": "..."}
        # which already matches LiteLLM's RerankResponseDocument shape.
        transformed_results: List[dict] = []
        for r in results:
            item: Dict[str, Any] = {
                "index": r["index"],
                "relevance_score": r["relevance_score"],
            }
            doc = r.get("document")
            if isinstance(doc, dict):
                item["document"] = doc
            elif isinstance(doc, str):
                # Defensive: spec says dict, but normalize string-shaped echoes.
                item["document"] = {"text": doc}
            transformed_results.append(item)

        usage = response_json.get("usage") or {}
        total_tokens = usage.get("total_tokens")
        input_tokens = usage.get("input_tokens", total_tokens)
        image_tokens = usage.get("image_tokens")
        billed_units = RerankBilledUnits(
            total_tokens=total_tokens,
            **({"input_tokens": input_tokens} if "input_tokens" in usage else {}),
            **({"image_tokens": image_tokens} if image_tokens is not None else {}),
        )
        tokens = RerankTokens(
            input_tokens=input_tokens,
            **({"image_tokens": image_tokens} if image_tokens is not None else {}),
        )
        meta = RerankResponseMeta(billed_units=billed_units, tokens=tokens)

        return RerankResponse(
            id=response_json.get("id") or response_json.get("request_id") or str(uuid.uuid4()),
            results=transformed_results,  # type: ignore
            meta=meta,
        )

    def calculate_rerank_cost(
        self,
        model: str,
        custom_llm_provider: str | None = None,
        billed_units: RerankBilledUnits | None = None,
        model_info: ModelInfo | None = None,
    ) -> tuple[float, float]:
        if billed_units is None or model_info is None:
            return 0.0, 0.0
        total_tokens = billed_units.get("input_tokens")
        if total_tokens is None:
            total_tokens = billed_units.get("total_tokens") or 0
        image_tokens = billed_units.get("image_tokens") or 0
        text_tokens = max(total_tokens - image_tokens, 0)
        text_rate = float(model_info.get("input_cost_per_token") or 0.0)
        image_rate = float(model_info.get("input_cost_per_image_token") or text_rate)
        return text_tokens * text_rate + image_tokens * image_rate, 0.0

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        if isinstance(headers, dict):
            headers = httpx.Headers(headers)
        return DashScopeError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
