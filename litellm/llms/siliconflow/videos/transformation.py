from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeVar

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, Field, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
    extract_original_video_id,
)

from ..common_utils import SiliconFlowError, get_siliconflow_api_base, get_siliconflow_headers

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class _SiliconFlowVideoSubmitResponse(BaseModel):
    requestId: str


class _SiliconFlowVideoResult(BaseModel):
    url: str


class _SiliconFlowVideoTimings(BaseModel):
    inference: float | None = None


class _SiliconFlowVideoResults(BaseModel):
    videos: list[_SiliconFlowVideoResult] = Field(default_factory=list)


class _SiliconFlowVideoStatusResponse(BaseModel):
    requestId: str | None = None
    status: str
    reason: str | None = None
    results: _SiliconFlowVideoResults | None = None
    timings: _SiliconFlowVideoTimings | None = None
    seed: int | None = None


class SiliconFlowVideoConfig(BaseVideoConfig):
    _PROVIDER_PARAMS = frozenset({"negative_prompt", "image_size", "image", "seed"})

    def __init__(
        self,
        sync_http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__()
        self._sync_http_client = sync_http_client
        self._async_http_client = async_http_client

    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["model", "prompt", "input_reference", "size", "extra_headers"]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        input_reference = video_create_optional_params.get("input_reference")
        if input_reference is not None and not isinstance(input_reference, str):
            raise ValueError("SiliconFlow input_reference must be an image URL or data URL.")
        size = video_create_optional_params.get("size")
        if size is not None and size not in {"1280x720", "720x1280", "960x960"}:
            raise ValueError("SiliconFlow video size must be 1280x720, 720x1280, or 960x960.")
        provider_params = {
            key: value
            for key, value in video_create_optional_params.items()
            if key in self._PROVIDER_PARAMS and value is not None
        }
        accepted_params = self._PROVIDER_PARAMS | {"extra_body", "extra_headers", "input_reference", "size"}
        unsupported_params = tuple(key for key in video_create_optional_params if key not in accepted_params)
        if unsupported_params and not drop_params:
            raise ValueError(f"Parameters {unsupported_params} are not supported for SiliconFlow video generation.")
        seed = provider_params.get("seed")
        if isinstance(seed, int) and not 0 <= seed <= 9_999_999_999:
            raise ValueError("SiliconFlow seed must be between 0 and 9999999999.")
        return {
            **provider_params,
            **({"image": input_reference} if input_reference is not None else {}),
            **({"image_size": size} if size is not None else {}),
        }

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:
        params_api_key = litellm_params.api_key if litellm_params is not None else None
        final_api_key = api_key or params_api_key or get_secret_str("SILICONFLOW_API_KEY")
        if not final_api_key:
            raise ValueError("SiliconFlow API key is required. Set SILICONFLOW_API_KEY.")
        return get_siliconflow_headers(final_api_key, headers)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],
    ) -> str:
        base = get_siliconflow_api_base(api_base)
        for suffix in ("/video/submit", "/video/status"):
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return base

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> tuple[dict[str, object], RequestFiles, str]:
        request_params = {
            key: value
            for key, value in video_create_optional_request_params.items()
            if key not in {"extra_headers", "extra_body"}
        }
        return {"model": model, "prompt": prompt, **request_params}, [], f"{api_base.rstrip('/')}/video/submit"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict[str, object] | None = None,
    ) -> VideoObject:
        response = self._parse_response(raw_response, _SiliconFlowVideoSubmitResponse)
        provider = custom_llm_provider or LlmProviders.SILICONFLOW.value
        request_size = request_data.get("image_size") if request_data is not None else None
        return VideoObject(
            id=encode_video_id_with_provider(response.requestId, provider, model),
            object="video",
            status="queued",
            model=model,
            size=request_size if isinstance(request_size, str) else None,
            usage={"generated_videos": 1},
        )

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> tuple[str, dict[str, object]]:
        return f"{api_base.rstrip('/')}/video/status", {"requestId": extract_original_video_id(video_id)}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        response = self._parse_response(raw_response, _SiliconFlowVideoStatusResponse)
        encoded_video_id = self._video_id_from_logging_obj(logging_obj)
        decoded_video_id = decode_video_id_with_provider(encoded_video_id)
        request_id = response.requestId or decoded_video_id.get("video_id") or encoded_video_id
        model = decoded_video_id.get("model_id")
        status = self._map_status(response.status)
        inference_time = response.timings.inference if response.timings is not None else None
        return VideoObject(
            id=encode_video_id_with_provider(
                request_id,
                custom_llm_provider or LlmProviders.SILICONFLOW.value,
                model,
            ),
            object="video",
            status=status,
            model=model,
            error={"message": response.reason} if response.reason is not None else None,
            usage={
                "generated_videos": 1 if status == "completed" else 0,
                **({"inference_time": inference_time} if inference_time is not None else {}),
                **({"seed": response.seed} if response.seed is not None else {}),
            },
        )

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
        variant: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        if variant is not None:
            raise ValueError("SiliconFlow video content variants are not supported.")
        return self.transform_video_status_retrieve_request(video_id, api_base, litellm_params, headers)

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        video_url = self._extract_video_url(raw_response)
        if self._sync_http_client is not None:
            video_response = self._sync_http_client.get(video_url)
            video_response.raise_for_status()
            return video_response.content
        with httpx.Client(follow_redirects=True) as client:
            video_response = client.get(video_url)
            video_response.raise_for_status()
            return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        video_url = self._extract_video_url(raw_response)
        if self._async_http_client is not None:
            video_response = await self._async_http_client.get(video_url)
            video_response.raise_for_status()
            return video_response.content
        async with httpx.AsyncClient(follow_redirects=True) as client:
            video_response = await client.get(video_url)
            video_response.raise_for_status()
            return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:
        raise NotImplementedError("Video remix is not supported by SiliconFlow.")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by SiliconFlow.")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:
        raise NotImplementedError("Video listing is not supported by SiliconFlow.")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        raise NotImplementedError("Video listing is not supported by SiliconFlow.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> tuple[str, dict[str, object]]:
        raise NotImplementedError("Video deletion is not supported by SiliconFlow.")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video deletion is not supported by SiliconFlow.")

    def _extract_video_url(self, raw_response: httpx.Response) -> str:
        response = self._parse_response(raw_response, _SiliconFlowVideoStatusResponse)
        if self._map_status(response.status) != "completed":
            raise ValueError(f"SiliconFlow video is not ready. Current status: {response.status}.")
        videos = response.results.videos if response.results is not None else []
        if not videos:
            raise ValueError("SiliconFlow video task completed without a video URL.")
        return videos[0].url

    @staticmethod
    def _video_id_from_logging_obj(logging_obj: LiteLLMLoggingObj) -> str:
        raw_litellm_params: object = getattr(logging_obj, "litellm_params", None)
        if not isinstance(raw_litellm_params, Mapping):
            return ""
        video_id: object = raw_litellm_params.get("video_id")
        return video_id if isinstance(video_id, str) else ""

    @staticmethod
    def _map_status(status: str) -> str:
        return {
            "succeed": "completed",
            "inqueue": "queued",
            "inprogress": "in_progress",
            "failed": "failed",
        }.get(status.lower(), status.lower())

    def _parse_response(self, raw_response: httpx.Response, response_type: type[ResponseModel]) -> ResponseModel:
        if raw_response.status_code >= 400:
            raise self.get_error_class(raw_response.text, raw_response.status_code, raw_response.headers)
        try:
            return response_type.model_validate(raw_response.json())
        except ValidationError as exc:
            raise self.get_error_class(
                f"Invalid SiliconFlow video response: {exc}",
                raw_response.status_code,
                raw_response.headers,
            ) from exc

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        response_headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers)
        return SiliconFlowError(status_code=status_code, message=error_message, headers=response_headers)
