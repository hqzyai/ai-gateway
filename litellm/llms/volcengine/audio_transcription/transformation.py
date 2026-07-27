import base64
import uuid
from typing import List, Union

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue

import litellm
from litellm.litellm_core_utils.audio_utils.utils import (
    normalize_transcription_language_to_bcp47,
    process_audio_file,
)
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

from ..common_utils import VolcEngineError

JsonObject = dict[str, JsonValue]


class _VolcEngineASRResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str


class _VolcEngineASRAudioInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    duration: float | None = None


class _VolcEngineASRResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    result: _VolcEngineASRResult
    audio_info: _VolcEngineASRAudioInfo | None = None


class VolcEngineAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    DEFAULT_API_BASE = "https://openspeech.bytedance.com"
    ENDPOINT_PATH = "/api/v3/auc/bigmodel/recognize/flash"
    RESOURCE_ID = "volc.bigasr.auc_turbo"

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
        if language is not None and not isinstance(language, str):
            if not drop_params:
                raise ValueError("Volcengine ASR language must be a string")
            language = None
        response_format = non_default_params.get("response_format")
        if response_format is not None and (
            not isinstance(response_format, str) or response_format not in ("json", "text")
        ):
            if not drop_params:
                raise ValueError("Volcengine ASR supports response_format values json and text")
            response_format = None
        return {
            **optional_params,
            **({"language": language} if language else {}),
            **({"response_format": response_format} if response_format else {}),
        }

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
        resolved_api_key = (
            api_key
            or get_secret_str("VOLCENGINE_SPEECH_API_KEY")
            or get_secret_str("VOLCENGINE_API_KEY")
            or litellm.api_key
        )
        if resolved_api_key is None:
            raise ValueError(
                "Volcengine Speech API key is required. Set VOLCENGINE_SPEECH_API_KEY or pass api_key explicitly."
            )
        return {
            **headers,
            "Content-Type": "application/json",
            "X-Api-Key": resolved_api_key,
            "X-Api-Resource-Id": self.RESOURCE_ID,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        stream: bool | None = None,
    ) -> str:
        configured_base = get_secret_str("VOLCENGINE_SPEECH_API_BASE")
        usable_api_base = api_base if api_base and "ark.cn-beijing.volces.com" not in api_base else None
        base_url = (usable_api_base or configured_base or self.DEFAULT_API_BASE).rstrip("/")
        if base_url.endswith(self.ENDPOINT_PATH):
            return base_url
        return f"{base_url}{self.ENDPOINT_PATH}"

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: JsonObject,
        litellm_params: JsonObject,
    ) -> AudioTranscriptionRequestData:
        processed_audio = process_audio_file(audio_file)
        language = optional_params.get("language")
        normalized_language = normalize_transcription_language_to_bcp47(language) if isinstance(language, str) else None
        return AudioTranscriptionRequestData(
            data={
                "user": {"uid": str(uuid.uuid4())},
                "audio": {
                    "data": base64.b64encode(processed_audio.file_content).decode("ascii"),
                    **({"language": normalized_language} if normalized_language else {}),
                },
                "request": {
                    "model_name": "bigmodel",
                    "enable_itn": True,
                    "enable_punc": True,
                    "enable_ddc": True,
                },
            }
        )

    def transform_audio_transcription_response(self, raw_response: httpx.Response) -> TranscriptionResponse:
        response_headers = {str(key): str(value) for key, value in raw_response.headers.items()}
        status_code = response_headers.get("x-api-status-code")
        if status_code != "20000000":
            message = response_headers.get("x-api-message") or raw_response.text
            raise VolcEngineError(
                status_code=raw_response.status_code,
                message=f"Volcengine ASR returned status {status_code or 'unknown'}: {message}",
                headers=raw_response.headers,
            )
        try:
            parsed_response = _VolcEngineASRResponse.model_validate_json(raw_response.content)
        except ValueError as exc:
            raise VolcEngineError(
                status_code=raw_response.status_code,
                message=f"Failed to parse Volcengine ASR response: {exc}",
                headers=raw_response.headers,
            ) from exc
        response = TranscriptionResponse(text=parsed_response.result.text)
        if parsed_response.audio_info and parsed_response.audio_info.duration is not None:
            response["duration"] = parsed_response.audio_info.duration / 1000
        return response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict[str, str], httpx.Headers]
    ) -> BaseLLMException:
        normalized_headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers)
        return VolcEngineError(status_code=status_code, message=error_message, headers=normalized_headers)
