from collections.abc import Mapping
from typing import Tuple, Union

import httpx
from pydantic import JsonValue

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import HttpxBinaryResponseContent

from ..common_utils import DashScopeError

DEFAULT_DASHSCOPE_TTS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
JsonObject = dict[str, JsonValue]


class DashScopeTextToSpeechConfig(BaseTextToSpeechConfig):
    CONTENT_TYPES = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "pcm": "application/octet-stream",
        "opus": "audio/ogg",
    }
    PROVIDER_PARAMS = frozenset(
        (
            "aigc_propagate_id",
            "aigc_propagator",
            "bit_rate",
            "disable_markdown_filter",
            "enable_aigc_tag",
            "enable_ssml",
            "hot_fix",
            "instruction",
            "language_hints",
            "pitch",
            "sample_rate",
            "seed",
            "text_type",
            "volume",
            "word_timestamp_enabled",
        )
    )

    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["voice", "response_format", "speed", "instructions"]

    def map_openai_params(
        self,
        model: str,
        optional_params: JsonObject,
        voice: Union[str, JsonObject] | None = None,
        drop_params: bool = False,
        kwargs: JsonObject | None = None,
    ) -> Tuple[str | None, JsonObject]:
        if voice is not None and (not isinstance(voice, str) or not voice.strip()):
            raise ValueError("DashScope TTS voice must be a non-empty string.")
        instructions = optional_params.get("instructions")
        call_kwargs = kwargs or {}
        extra_body = call_kwargs.get("extra_body")
        provider_params = extra_body if isinstance(extra_body, dict) else {}
        response_format = optional_params.get("response_format", provider_params.get("format", "mp3"))
        if not isinstance(response_format, str) or response_format not in self.CONTENT_TYPES:
            raise ValueError("DashScope TTS response_format must be mp3, wav, pcm, or opus.")
        speed = optional_params.get("speed", provider_params.get("rate", 1.0))
        return (
            voice.strip() if isinstance(voice, str) else None,
            {
                "format": response_format,
                "rate": speed,
                **({"instruction": instructions} if isinstance(instructions, str) and instructions else {}),
                **{
                    key: value
                    for key, value in provider_params.items()
                    if key in self.PROVIDER_PARAMS and value is not None
                },
            },
        )

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        resolved_api_key = api_key or get_secret_str("DASHSCOPE_API_KEY")
        if resolved_api_key is None:
            raise ValueError("DashScope API key is required. Set DASHSCOPE_API_KEY or pass api_key explicitly.")
        return {**headers, "Authorization": f"Bearer {resolved_api_key}"}

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: JsonObject) -> str:
        configured = api_base or get_secret_str("DASHSCOPE_API_BASE_AUDIO")
        if configured and configured.startswith(("ws://", "wss://")):
            normalized = configured.rstrip("/")
            return normalized if normalized.endswith("/api-ws/v1/inference") else f"{normalized}/api-ws/v1/inference"
        if configured:
            host = configured.split("://", 1)[-1].split("/", 1)[0]
            return f"wss://{host}/api-ws/v1/inference"
        return DEFAULT_DASHSCOPE_TTS_URL

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        headers: Mapping[str, str],
    ) -> TextToSpeechRequestData:
        return TextToSpeechRequestData(
            dict_body={
                "model": model,
                "input": input,
                "voice": voice or "longanlingxi",
                **optional_params,
            }
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> HttpxBinaryResponseContent:
        return HttpxBinaryResponseContent(raw_response)

    def get_error_class(self, error_message: str, status_code: int, headers: Mapping[str, str]) -> BaseLLMException:
        return DashScopeError(status_code=status_code, message=error_message, headers=httpx.Headers(headers))
