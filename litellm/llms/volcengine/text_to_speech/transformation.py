import base64
import uuid
from typing import Tuple, Union

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import HttpxBinaryResponseContent

from ..common_utils import VolcEngineError

JsonObject = dict[str, JsonValue]


class _VolcEngineTextToSpeechChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: int
    data: str | None = None
    message: str | None = None


class _VolcEngineAudioParams(BaseModel):
    format: str = "mp3"


class _VolcEngineRequestParams(BaseModel):
    audio_params: _VolcEngineAudioParams = Field(default_factory=_VolcEngineAudioParams)


class _VolcEngineTextToSpeechRequest(BaseModel):
    req_params: _VolcEngineRequestParams = Field(default_factory=_VolcEngineRequestParams)


class VolcEngineTextToSpeechConfig(BaseTextToSpeechConfig):
    DEFAULT_BASE_URL = "https://openspeech.bytedance.com"
    ENDPOINT_PATH = "/api/v3/tts/unidirectional"
    RESOURCE_ID = "seed-tts-2.0"
    FORMAT_MAPPINGS: dict[str, str] = {"mp3": "mp3", "opus": "ogg_opus", "wav": "wav", "pcm": "pcm"}
    CONTENT_TYPES: dict[str, str] = {
        "mp3": "audio/mpeg",
        "ogg_opus": "audio/ogg",
        "wav": "audio/wav",
        "pcm": "application/octet-stream",
    }

    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["voice", "response_format"]

    def map_openai_params(
        self,
        model: str,
        optional_params: JsonObject,
        voice: Union[str, JsonObject] | None = None,
        drop_params: bool = False,
        kwargs: JsonObject | None = None,
    ) -> Tuple[str | None, JsonObject]:
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("Volcengine TTS requires `voice` to be a valid speaker ID")
        unsupported_params = tuple(
            param for param in ("speed", "instructions") if optional_params.get(param) is not None
        )
        if unsupported_params and not drop_params:
            raise ValueError(f"Volcengine TTS does not support: {', '.join(unsupported_params)}")
        response_format = optional_params.get("response_format", "mp3")
        if not isinstance(response_format, str) or response_format not in self.FORMAT_MAPPINGS:
            raise ValueError(
                f"Unsupported response_format: {response_format}. Volcengine supports mp3, opus, wav, and pcm"
            )
        return voice.strip(), {"format": self.FORMAT_MAPPINGS[response_format]}

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
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
            "X-Control-Require-Usage-Tokens-Return": "*",
        }

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: JsonObject) -> str:
        configured_base = get_secret_str("VOLCENGINE_SPEECH_API_BASE")
        usable_api_base = api_base if api_base and "ark.cn-beijing.volces.com" not in api_base else None
        base_url = (usable_api_base or configured_base or self.DEFAULT_BASE_URL).rstrip("/")
        if base_url.endswith(self.ENDPOINT_PATH):
            return base_url
        return f"{base_url}{self.ENDPOINT_PATH}"

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        headers: dict[str, str],
    ) -> TextToSpeechRequestData:
        if voice is None:
            raise ValueError("Volcengine TTS requires `voice` to be a valid speaker ID")
        audio_format = optional_params.get("format", "mp3")
        if not isinstance(audio_format, str):
            raise ValueError("Volcengine TTS audio format must be a string")
        return TextToSpeechRequestData(
            dict_body={
                "req_params": {
                    "text": input,
                    "speaker": voice,
                    "audio_params": {"format": audio_format, "sample_rate": 24000},
                }
            }
        )

    @classmethod
    def _response_content_type(cls, raw_response: httpx.Response) -> str:
        try:
            request_payload = _VolcEngineTextToSpeechRequest.model_validate_json(raw_response.request.content)
            audio_format = request_payload.req_params.audio_params.format
        except (AttributeError, TypeError, ValueError, RuntimeError):
            audio_format = "mp3"
        return cls.CONTENT_TYPES.get(audio_format, "application/octet-stream")

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> HttpxBinaryResponseContent:
        try:
            chunks = tuple(
                _VolcEngineTextToSpeechChunk.model_validate_json(line)
                for line in raw_response.text.splitlines()
                if line.strip()
            )
        except Exception as exc:
            raise VolcEngineError(
                status_code=raw_response.status_code,
                message=f"Failed to parse Volcengine TTS response: {exc}",
                headers=raw_response.headers,
            ) from exc
        error_chunk = next((chunk for chunk in chunks if chunk.code not in (0, 20000000)), None)
        if error_chunk is not None:
            raise VolcEngineError(
                status_code=raw_response.status_code,
                message=error_chunk.message or f"Volcengine TTS returned error code {error_chunk.code}",
                headers=raw_response.headers,
            )
        try:
            audio_content = b"".join(
                base64.b64decode(chunk.data, validate=True) for chunk in chunks if chunk.code == 0 and chunk.data
            )
        except ValueError as exc:
            raise VolcEngineError(
                status_code=raw_response.status_code,
                message=f"Failed to decode Volcengine TTS audio: {exc}",
                headers=raw_response.headers,
            ) from exc
        if not audio_content:
            raise VolcEngineError(
                status_code=raw_response.status_code,
                message="Volcengine TTS response did not contain audio data",
                headers=raw_response.headers,
            )
        binary_response = httpx.Response(
            status_code=200,
            headers={
                "Content-Type": self._response_content_type(raw_response),
                "Content-Length": str(len(audio_content)),
            },
            content=audio_content,
        )
        return HttpxBinaryResponseContent(binary_response)

    def get_error_class(self, error_message: str, status_code: int, headers: dict[str, str]) -> BaseLLMException:
        return VolcEngineError(status_code=status_code, message=error_message, headers=httpx.Headers(headers))
