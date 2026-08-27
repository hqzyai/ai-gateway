import base64
from collections.abc import Mapping, Sequence
from typing import List

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, OpenAIAudioTranscriptionOptionalParams
from litellm.types.utils import FileTypes, TranscriptionResponse

from ..common_utils import DashScopeError

DEFAULT_DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/api/v1"
SYNCHRONOUS_ASR_PATH = "/services/aigc/multimodal-generation/generation"
FILE_TRANSCRIPTION_PATH = "/services/audio/asr/transcription"
STREAMING_ASR_MODEL = "qwen-audio-3.0-asr-flash-streaming"
FILE_TRANSCRIPTION_MODEL = "qwen-audio-3.0-asr-flash-filetrans"
JsonObject = dict[str, JsonValue]
JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)


class _DashScopeASRMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: str | tuple[JsonObject, ...] = ""


class _DashScopeASRChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: _DashScopeASRMessage


class _DashScopeASROutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    choices: tuple[_DashScopeASRChoice, ...] = Field(default_factory=tuple)
    text: str | None = None


class _DashScopeASRResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    output: _DashScopeASROutput
    usage: JsonObject = Field(default_factory=dict)
    request_id: str | None = None


class DashScopeAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(self, model: str) -> List[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "response_format"]

    def map_openai_params(
        self,
        non_default_params: JsonObject,
        optional_params: JsonObject,
        model: str,
        drop_params: bool,
    ) -> JsonObject:
        language = non_default_params.get("language")
        response_format = non_default_params.get("response_format")
        return {
            **optional_params,
            **({"language_hints": [language]} if isinstance(language, str) and language else {}),
            **({"response_format": response_format} if response_format is not None else {}),
        }

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: JsonObject,
        litellm_params: JsonObject,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        resolved_api_key = api_key or get_secret_str("DASHSCOPE_API_KEY")
        if resolved_api_key is None:
            raise ValueError("DashScope API key is required. Set DASHSCOPE_API_KEY or pass api_key explicitly.")
        return {**headers, "Authorization": f"Bearer {resolved_api_key}", "Content-Type": "application/json"}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        stream: bool | None = None,
    ) -> str:
        base = self.normalize_api_base(api_base)
        if model == FILE_TRANSCRIPTION_MODEL:
            return f"{base}{FILE_TRANSCRIPTION_PATH}"
        if model == STREAMING_ASR_MODEL:
            return self.websocket_url(api_base)
        return f"{base}{SYNCHRONOUS_ASR_PATH}"

    @staticmethod
    def normalize_api_base(api_base: str | None) -> str:
        base = (api_base or get_secret_str("DASHSCOPE_API_BASE_AUDIO") or DEFAULT_DASHSCOPE_API_BASE).rstrip("/")
        if "/services/" in base:
            return base.split("/services/", 1)[0]
        if base.endswith("/compatible-mode/v1"):
            return f"{base[: -len('/compatible-mode/v1')]}/api/v1"
        if base.endswith("/api/v1"):
            return base
        return f"{base}/api/v1"

    @staticmethod
    def websocket_url(api_base: str | None) -> str:
        configured = api_base or get_secret_str("DASHSCOPE_API_BASE_AUDIO")
        if configured and configured.startswith(("ws://", "wss://")):
            normalized = configured.rstrip("/")
            return normalized if normalized.endswith("/api-ws/v1/inference") else f"{normalized}/api-ws/v1/inference"
        if configured:
            host = configured.split("://", 1)[-1].split("/", 1)[0]
            return f"wss://{host}/api-ws/v1/inference"
        return "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: JsonObject,
        litellm_params: JsonObject,
    ) -> AudioTranscriptionRequestData:
        if model == FILE_TRANSCRIPTION_MODEL:
            return AudioTranscriptionRequestData(data=self.transform_file_transcription_request(model, optional_params))

        processed_audio = process_audio_file(audio_file)
        encoded_audio = base64.b64encode(processed_audio.file_content).decode("ascii")
        params = self.flatten_optional_params(optional_params)
        parameters = {
            key: value
            for key, value in params.items()
            if key in ("format", "sample_rate", "language_hints") and value is not None
        }
        return AudioTranscriptionRequestData(
            data={
                "model": model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": f"data:{processed_audio.content_type};base64,{encoded_audio}"
                                    },
                                }
                            ],
                        }
                    ]
                },
                "parameters": parameters,
            }
        )

    def transform_file_transcription_request(self, model: str, optional_params: JsonObject) -> JsonObject:
        params = self.flatten_optional_params(optional_params)
        raw_urls = params.get("file_urls")
        file_urls = raw_urls if isinstance(raw_urls, list) else [params.get("file_url")]
        urls = [url for url in file_urls if isinstance(url, str) and url]
        if not urls:
            raise ValueError("DashScope filetrans requires a public file_url or file_urls parameter.")
        if len(urls) != 1:
            raise ValueError("DashScope filetrans accepts exactly one public audio URL per request.")
        parameters = {
            key: value for key, value in params.items() if key in ("channel_id", "language_hints") and value is not None
        }
        return JSON_OBJECT_ADAPTER.validate_python(
            {"model": model, "input": {"file_urls": urls}, "parameters": parameters}
        )

    def transform_audio_transcription_response(self, raw_response: httpx.Response) -> TranscriptionResponse:
        if raw_response.status_code >= 400:
            raise DashScopeError(raw_response.status_code, raw_response.text, raw_response.headers)
        try:
            parsed = _DashScopeASRResponse.model_validate_json(raw_response.content)
        except ValueError as exc:
            raise DashScopeError(raw_response.status_code, raw_response.text, raw_response.headers) from exc
        text = parsed.output.text or self._choice_text(parsed.output.choices)
        response = TranscriptionResponse(text=text)
        object.__setattr__(
            response,
            "_hidden_params",
            {
                "request_id": parsed.request_id,
                "usage": parsed.usage,
                **(
                    {"audio_transcription_duration": parsed.usage["duration"]}
                    if isinstance(parsed.usage.get("duration"), (int, float))
                    else {}
                ),
                "custom_llm_provider": "dashscope",
            },
        )
        return response

    @staticmethod
    def _choice_text(choices: Sequence[_DashScopeASRChoice]) -> str:
        if not choices:
            return ""
        content = choices[0].message.content
        if isinstance(content, str):
            return content
        return "".join(str(item.get("text", "")) for item in content if item.get("text") is not None)

    @staticmethod
    def flatten_optional_params(optional_params: JsonObject) -> JsonObject:
        extra_body = optional_params.get("extra_body")
        return {
            **{key: value for key, value in optional_params.items() if key != "extra_body"},
            **(extra_body if isinstance(extra_body, dict) else {}),
        }

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        normalized_headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers)
        return DashScopeError(status_code=status_code, message=error_message, headers=normalized_headers)
