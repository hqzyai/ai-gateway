from __future__ import annotations

from collections.abc import Mapping
from math import gcd, isfinite
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from litellm.litellm_core_utils.url_utils import encode_url_path_segment
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

from ..common_utils import DashScopeError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = object


DEFAULT_DASHSCOPE_VIDEO_API_BASE = "https://dashscope.aliyuncs.com/api/v1"
VIDEO_SYNTHESIS_PATH = "/services/aigc/video-generation/video-synthesis"
LEGACY_KEYFRAME_SYNTHESIS_PATH = "/services/aigc/image2video/video-synthesis"
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
OBJECT_DICT_ADAPTER = TypeAdapter(dict[str, object])
JSONObject: TypeAlias = dict[str, object]
Headers: TypeAlias = dict[str, str]
StringList: TypeAlias = list[str]
ObjectList: TypeAlias = list[object]


class _DashScopeVideoOutput(BaseModel):
    task_id: str
    task_status: str = "PENDING"
    video_url: str | None = None
    message: str | None = None
    code: str | None = None
    orig_prompt: str | None = None


class _DashScopeVideoResponse(BaseModel):
    output: _DashScopeVideoOutput
    usage: JSONObject = Field(default_factory=dict)
    request_id: str | None = None


class DashScopeVideoConfig(BaseVideoConfig):
    _INPUT_PARAMS = frozenset(
        (
            "audio_url",
            "first_clip_url",
            "first_frame_url",
            "function",
            "img_url",
            "last_clip_url",
            "last_frame_url",
            "mask_image_url",
            "mask_video_url",
            "media",
            "negative_prompt",
            "ref_images_url",
            "reference_urls",
            "reference_voice",
            "video_url",
        )
    )
    _PARAMETER_PARAMS = frozenset(
        (
            "audio",
            "audio_setting",
            "bottom_scale",
            "control_condition",
            "duration",
            "expand_ratio",
            "left_scale",
            "mask_frame_id",
            "mask_type",
            "obj_or_bg",
            "prompt_extend",
            "ratio",
            "resolution",
            "right_scale",
            "seed",
            "shot_type",
            "size",
            "strength",
            "top_scale",
            "watermark",
        )
    )
    _STANDARD_PARAMS = frozenset(
        (
            "extra_body",
            "extra_headers",
            "input",
            "input_reference",
            "parameters",
            "seconds",
            "size",
        )
    )
    _WAN_27_SIZE_MAP = {
        "1280x720": ("720P", "16:9"),
        "720x1280": ("720P", "9:16"),
        "1920x1080": ("1080P", "16:9"),
        "1080x1920": ("1080P", "9:16"),
    }

    def __init__(
        self,
        sync_http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__()
        self._sync_http_client = sync_http_client
        self._async_http_client = async_http_client

    def get_supported_openai_params(self, model: str) -> StringList:
        return StringList(("model", "prompt", "input_reference", "seconds", "size", "extra_headers"))

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> JSONObject:
        extra_body = self._mapping_param(video_create_optional_params, "extra_body")
        extra_body_keys = frozenset(extra_body)
        accepted_params = self._STANDARD_PARAMS | self._INPUT_PARAMS | self._PARAMETER_PARAMS | extra_body_keys
        unsupported_params = tuple(key for key in video_create_optional_params if key not in accepted_params)
        if unsupported_params and not drop_params:
            raise ValueError(f"Parameters {unsupported_params} are not supported for DashScope video generation.")

        input_reference = video_create_optional_params.get("input_reference")
        if input_reference is not None and not isinstance(input_reference, str):
            raise ValueError("DashScope input_reference must be a publicly accessible image URL.")

        seconds = video_create_optional_params.get("seconds")
        try:
            duration = int(seconds) if seconds is not None else None
        except (TypeError, ValueError) as exc:
            raise ValueError("DashScope video seconds must be an integer.") from exc

        raw_size = video_create_optional_params.get("size")
        mapped_size = self._map_size(model, raw_size) if isinstance(raw_size, str) else JSONObject()
        passthrough = JSONObject(
            (key, value)
            for key, value in video_create_optional_params.items()
            if key in (self._INPUT_PARAMS | self._PARAMETER_PARAMS | {"input", "parameters"})
            and key != "size"
            and value is not None
        )
        return {
            **passthrough,
            **({"input_reference": input_reference} if input_reference is not None else {}),
            **({"duration": duration} if duration is not None else {}),
            **mapped_size,
        }

    def validate_environment(
        self,
        headers: Headers,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> Headers:
        params_api_key = litellm_params.api_key if litellm_params is not None else None
        final_api_key = api_key or params_api_key or get_secret_str("DASHSCOPE_API_KEY")
        if not final_api_key:
            raise ValueError("DashScope API key is required. Set DASHSCOPE_API_KEY.")
        return Headers(
            (*headers.items(), ("Authorization", f"Bearer {final_api_key}"), ("Content-Type", "application/json"))
        )

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: JSONObject,
    ) -> str:
        configured_base = api_base or get_secret_str("DASHSCOPE_API_BASE_VIDEO") or DEFAULT_DASHSCOPE_VIDEO_API_BASE
        return self._normalize_api_base(configured_base)

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: JSONObject,
        litellm_params: GenericLiteLLMParams,
        headers: Headers,
    ) -> tuple[JSONObject, RequestFiles, str]:
        headers["X-DashScope-Async"] = "enable"
        request_data = self._build_generation_request(model, prompt, video_create_optional_request_params)
        synthesis_path = (
            LEGACY_KEYFRAME_SYNTHESIS_PATH
            if "kf2v" in model.lower() and not self._uses_media_input(model)
            else VIDEO_SYNTHESIS_PATH
        )
        return request_data, [], f"{api_base}{synthesis_path}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: JSONObject | None = None,
    ) -> VideoObject:
        response = self._parse_response(raw_response, _DashScopeVideoResponse)
        return self._response_to_video(
            response=response,
            provider=custom_llm_provider or LlmProviders.DASHSCOPE.value,
            model=model,
            request_data=request_data,
        )

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Headers,
    ) -> tuple[str, JSONObject]:
        task_id = encode_url_path_segment(extract_original_video_id(video_id), field_name="video_id")
        return f"{api_base}/tasks/{task_id}", JSONObject()

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        response = self._parse_response(raw_response, _DashScopeVideoResponse)
        encoded_video_id = self._video_id_from_logging_obj(logging_obj)
        decoded_video_id = decode_video_id_with_provider(encoded_video_id)
        model = decoded_video_id.get("model_id")
        return self._response_to_video(
            response=response,
            provider=custom_llm_provider or LlmProviders.DASHSCOPE.value,
            model=model,
        )

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Headers,
        variant: str | None = None,
    ) -> tuple[str, JSONObject]:
        if variant is not None:
            raise ValueError("DashScope video content variants are not supported.")
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

    def get_video_edit_prefetch_params(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Headers,
        extra_body: JSONObject | None = None,
    ) -> tuple[str, JSONObject] | None:
        body = extra_body or JSONObject()
        raw_input = self._mapping_param(body, "input")
        if (
            isinstance(body.get("video_url") or raw_input.get("video_url"), str)
            or body.get("media") is not None
            or raw_input.get("media") is not None
        ):
            return None
        return self.transform_video_status_retrieve_request(video_id, api_base, litellm_params, headers)

    def get_video_edit_prefetch_method(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Headers,
    ) -> Literal["GET", "POST"]:
        return "GET"

    def transform_video_edit_request(
        self,
        prompt: str,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Headers,
        extra_body: JSONObject | None = None,
        prefetched_source_data: JSONObject | None = None,
    ) -> tuple[str, JSONObject]:
        body = extra_body or JSONObject()
        model_value = body.get("model", "wan2.7-videoedit")
        if not isinstance(model_value, str):
            raise TypeError("DashScope video edit model must be a string.")
        raw_input = self._mapping_param(body, "input")
        raw_parameters = self._mapping_param(body, "parameters")
        explicit_media = body.get("media") or raw_input.get("media")
        source_url = (
            self._source_video_url(body, prefetched_source_data)
            if explicit_media is None or not self._uses_media_input(model_value)
            else None
        )
        media = (
            explicit_media if explicit_media is not None else ObjectList((JSONObject(type="video", url=source_url),))
        )
        direct_input = JSONObject(
            (key, value)
            for key, value in body.items()
            if key in self._INPUT_PARAMS and key not in {"media", "video_url"} and value is not None
        )
        parameters = JSONObject(
            (
                *raw_parameters.items(),
                *((key, value) for key, value in body.items() if key in self._PARAMETER_PARAMS and value is not None),
            )
        )
        source_input = (
            JSONObject(media=media) if self._uses_media_input(model_value) else JSONObject(video_url=source_url)
        )
        headers["X-DashScope-Async"] = "enable"
        return (
            f"{api_base}{VIDEO_SYNTHESIS_PATH}",
            JSONObject(
                model=model_value,
                input=JSONObject(
                    (*raw_input.items(), *direct_input.items(), ("prompt", prompt), *source_input.items())
                ),
                parameters=parameters,
            ),
        )

    def transform_video_edit_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: JSONObject | None = None,
    ) -> VideoObject:
        response = self._parse_response(raw_response, _DashScopeVideoResponse)
        request_model = request_data.get("model") if request_data is not None else None
        model = request_model if isinstance(request_model, str) else "wan2.7-videoedit"
        return self._response_to_video(
            response=response,
            provider=custom_llm_provider or LlmProviders.DASHSCOPE.value,
            model=model,
            request_data=request_data,
        )

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Headers,
        extra_body: JSONObject | None = None,
    ) -> tuple[str, JSONObject]:
        raise NotImplementedError("Video remix is not supported by DashScope.")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by DashScope.")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Headers,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: JSONObject | None = None,
    ) -> tuple[str, JSONObject]:
        raise NotImplementedError("Video listing is not supported by DashScope.")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> Headers:
        raise NotImplementedError("Video listing is not supported by DashScope.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Headers,
    ) -> tuple[str, JSONObject]:
        raise NotImplementedError("Video deletion is not supported by DashScope.")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video deletion is not supported by DashScope.")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Headers | httpx.Headers,
    ) -> DashScopeError:
        response_headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers)
        return DashScopeError(status_code=status_code, message=error_message, headers=response_headers)

    def _build_generation_request(
        self,
        model: str,
        prompt: str,
        request_params: JSONObject,
    ) -> JSONObject:
        raw_input = self._mapping_param(request_params, "input")
        raw_parameters = self._mapping_param(request_params, "parameters")
        direct_input = {
            key: value for key, value in request_params.items() if key in self._INPUT_PARAMS and value is not None
        }
        input_reference = request_params.get("input_reference")
        normalized_input = self._normalize_input(model, direct_input, input_reference)
        parameters = {
            **raw_parameters,
            **{
                key: value
                for key, value in request_params.items()
                if key in self._PARAMETER_PARAMS and value is not None
            },
        }
        return {
            "model": model,
            "input": {**raw_input, **normalized_input, "prompt": prompt},
            "parameters": parameters,
        }

    def _normalize_input(
        self,
        model: str,
        direct_input: JSONObject,
        input_reference: object,
    ) -> JSONObject:
        if input_reference is not None and not isinstance(input_reference, str):
            raise ValueError("DashScope input_reference must be a publicly accessible image URL.")
        if self._uses_media_input(model):
            explicit_media = direct_input.get("media")
            first_frame = direct_input.get("first_frame_url") or input_reference
            last_frame = direct_input.get("last_frame_url")
            media = (
                explicit_media
                if explicit_media is not None
                else [
                    *([{"type": "first_frame", "url": first_frame}] if first_frame is not None else []),
                    *([{"type": "last_frame", "url": last_frame}] if last_frame is not None else []),
                ]
            )
            return {
                **{
                    key: value
                    for key, value in direct_input.items()
                    if key not in {"first_frame_url", "img_url", "last_frame_url", "media"}
                },
                **({"media": media} if media else {}),
            }
        if "kf2v" in model.lower():
            return {
                **{key: value for key, value in direct_input.items() if key != "img_url"},
                **({"first_frame_url": input_reference} if input_reference is not None else {}),
            }
        return {
            **direct_input,
            **({"img_url": input_reference} if input_reference is not None else {}),
        }

    def _response_to_video(
        self,
        response: _DashScopeVideoResponse,
        provider: str,
        model: str | None,
        request_data: JSONObject | None = None,
    ) -> VideoObject:
        status = self._map_status(response.output.task_status)
        duration = response.usage.get("duration") or response.usage.get("output_video_duration")
        requested_size = self._request_size(request_data)
        duration_seconds = self._duration_seconds(duration)
        video_resolution = self._video_resolution(response.usage, requested_size)
        error = (
            {
                "code": response.output.code or "video_generation_failed",
                "message": response.output.message or "DashScope video generation failed.",
            }
            if status == "failed"
            else None
        )
        video = VideoObject(
            id=encode_video_id_with_provider(response.output.task_id, provider, model),
            object="video",
            status=status,
            model=model,
            seconds=str(duration) if duration is not None else None,
            size=requested_size,
            error=error,
            usage={
                **response.usage,
                **(
                    JSONObject(duration_seconds=duration_seconds)
                    if duration_seconds is not None and status == "completed"
                    else JSONObject()
                ),
                **(JSONObject(video_resolution=video_resolution) if video_resolution is not None else JSONObject()),
                "generated_videos": 1 if status == "completed" else 0,
            },
        )
        hidden_params = {
            **({"request_id": response.request_id} if response.request_id is not None else {}),
            **({"video_url": response.output.video_url} if response.output.video_url is not None else {}),
        }
        video.set_hidden_params(hidden_params)
        return video

    def _extract_video_url(self, raw_response: httpx.Response) -> str:
        response = self._parse_response(raw_response, _DashScopeVideoResponse)
        if self._map_status(response.output.task_status) != "completed":
            raise ValueError(f"DashScope video is not ready. Current status: {response.output.task_status}.")
        if response.output.video_url is None:
            raise ValueError("DashScope video task completed without a video URL.")
        return response.output.video_url

    def _parse_response(self, raw_response: httpx.Response, response_type: type[ResponseModel]) -> ResponseModel:
        if raw_response.status_code >= 400:
            raise self.get_error_class(raw_response.text, raw_response.status_code, raw_response.headers)
        try:
            response_data = OBJECT_DICT_ADAPTER.validate_json(raw_response.content)
        except ValidationError as exc:
            raise self.get_error_class(
                f"Invalid DashScope video response: {exc}", raw_response.status_code, raw_response.headers
            ) from exc
        if "code" in response_data and "output" not in response_data:
            message = response_data.get("message", response_data.get("code", "DashScope video request failed."))
            raise self.get_error_class(str(message), raw_response.status_code, raw_response.headers)
        try:
            return response_type.model_validate(response_data)
        except ValidationError as exc:
            raise self.get_error_class(
                f"Invalid DashScope video response: {exc}", raw_response.status_code, raw_response.headers
            ) from exc

    def _source_video_url(
        self,
        body: JSONObject,
        prefetched_source_data: JSONObject | None,
    ) -> str:
        explicit_url = body.get("video_url")
        if isinstance(explicit_url, str):
            return explicit_url
        raw_input = self._mapping_param(body, "input")
        nested_url = raw_input.get("video_url")
        if isinstance(nested_url, str):
            return nested_url
        if prefetched_source_data is None:
            raise ValueError("DashScope video edit requires a completed source video or extra_body.video_url.")
        try:
            response = _DashScopeVideoResponse.model_validate(prefetched_source_data)
        except ValidationError as exc:
            raise ValueError(f"Invalid DashScope source video response: {exc}") from exc
        if self._map_status(response.output.task_status) != "completed":
            raise ValueError("Source video generation is not complete yet. Check the video status before editing.")
        if response.output.video_url is None:
            raise ValueError("The completed DashScope source video does not include a video URL.")
        return response.output.video_url

    @classmethod
    def _map_size(cls, model: str, size: str) -> JSONObject:
        normalized_size = size.replace("*", "x")
        upper_size = normalized_size.upper()
        if upper_size in {"480P", "720P", "1080P"}:
            return {"resolution": upper_size}
        if cls._uses_media_input(model):
            mapped_size = cls._WAN_27_SIZE_MAP.get(normalized_size)
            if mapped_size is not None:
                return {"resolution": mapped_size[0], "ratio": mapped_size[1]}
            dimensions = cls._parse_dimensions(normalized_size)
            if dimensions is None:
                raise ValueError(
                    "DashScope Wan 2.7 size must be a resolution such as 720P or dimensions such as 1280x720."
                )
            width, height = dimensions
            divisor = gcd(width, height)
            resolution = "1080P" if max(width, height) > 1280 else "720P"
            return {"resolution": resolution, "ratio": f"{width // divisor}:{height // divisor}"}
        if "i2v" in model.lower() or "kf2v" in model.lower():
            dimensions = cls._parse_dimensions(normalized_size)
            if dimensions is None:
                raise ValueError("DashScope image-to-video size must be a resolution or width x height dimensions.")
            largest_dimension = max(dimensions)
            resolution = "1080P" if largest_dimension > 1280 else "720P" if largest_dimension > 832 else "480P"
            return {"resolution": resolution}
        return {"size": normalized_size.replace("x", "*")}

    @staticmethod
    def _parse_dimensions(size: str) -> tuple[int, int] | None:
        parts = size.lower().split("x")
        if len(parts) != 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None

    @staticmethod
    def _uses_media_input(model: str) -> bool:
        return model.lower().startswith("wan2.7-")

    @staticmethod
    def _normalize_api_base(api_base: str) -> str:
        normalized = api_base.rstrip("/")
        if "/services/aigc/" in normalized:
            return normalized.split("/services/aigc/", 1)[0]
        if "/tasks/" in normalized:
            return normalized.split("/tasks/", 1)[0]
        if normalized.endswith("/compatible-mode/v1"):
            return f"{normalized[: -len('/compatible-mode/v1')]}/api/v1"
        if normalized.endswith("/api/v1"):
            return normalized
        return f"{normalized}/api/v1"

    @staticmethod
    def _mapping_param(params: Mapping[str, object], key: str) -> JSONObject:
        value = params.get(key)
        if value is None:
            return {}
        try:
            return OBJECT_DICT_ADAPTER.validate_python(value)
        except ValidationError as exc:
            raise ValueError(f"DashScope video {key} must be an object.") from exc

    @staticmethod
    def _request_size(request_data: JSONObject | None) -> str | None:
        if request_data is None:
            return None
        try:
            parameters = OBJECT_DICT_ADAPTER.validate_python(request_data.get("parameters"))
        except ValidationError:
            return None
        size = parameters.get("size")
        if isinstance(size, str):
            return size.replace("*", "x")
        resolution = parameters.get("resolution")
        ratio = parameters.get("ratio")
        if isinstance(resolution, str) and isinstance(ratio, str):
            return f"{resolution} {ratio}"
        return resolution if isinstance(resolution, str) else None

    @staticmethod
    def _duration_seconds(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            duration = float(value)
            return duration if isfinite(duration) and duration >= 0 else None
        if not isinstance(value, str):
            return None
        try:
            duration = float(value)
        except ValueError:
            return None
        return duration if isfinite(duration) and duration >= 0 else None

    @staticmethod
    def _video_resolution(usage: JSONObject, requested_size: str | None) -> str | None:
        raw_resolution = usage.get("video_resolution") or usage.get("resolution") or usage.get("SR") or requested_size
        if isinstance(raw_resolution, bool):
            return None
        if isinstance(raw_resolution, (int, float)):
            resolution = float(raw_resolution)
            return f"{int(resolution)}p" if isfinite(resolution) and resolution > 0 else None
        if not isinstance(raw_resolution, str):
            return None
        normalized = raw_resolution.strip().lower()
        for resolution in ("480p", "720p", "1080p", "4k"):
            if resolution in normalized:
                return resolution
        if normalized in {"480", "720", "1080"}:
            return f"{normalized}p"
        return normalized or None

    @staticmethod
    def _map_status(status: str) -> str:
        return {
            "PENDING": "queued",
            "RUNNING": "in_progress",
            "SUCCEEDED": "completed",
            "FAILED": "failed",
            "CANCELED": "cancelled",
            "CANCELLED": "cancelled",
        }.get(status.upper(), status.lower())

    @staticmethod
    def _video_id_from_logging_obj(logging_obj: LiteLLMLoggingObj) -> str:
        raw_litellm_params: object = getattr(logging_obj, "litellm_params", None)
        try:
            parsed_litellm_params = OBJECT_DICT_ADAPTER.validate_python(raw_litellm_params)
        except ValidationError:
            return ""
        video_id = parsed_litellm_params.get("video_id")
        return video_id if isinstance(video_id, str) else ""
