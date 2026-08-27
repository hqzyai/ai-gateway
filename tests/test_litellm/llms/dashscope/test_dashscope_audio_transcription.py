import json
from contextlib import nullcontext
from typing import cast
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.dashscope.audio_transcription.handler import DashScopeAudioTranscriptionHandler
from litellm.llms.dashscope.audio_transcription.transformation import (
    FILE_TRANSCRIPTION_MODEL,
    STREAMING_ASR_MODEL,
    DashScopeAudioTranscriptionConfig,
)
from litellm.types.utils import TranscriptionResponse
from litellm.utils import ProviderConfigManager


def test_dashscope_synchronous_asr_transformation_and_duration() -> None:
    config = DashScopeAudioTranscriptionConfig()
    request = config.transform_audio_transcription_request(
        model="qwen-audio-3.0-asr-flash",
        audio_file=("sample.wav", b"RIFF-audio"),
        optional_params={"language_hints": ["zh"]},
        litellm_params={},
    )
    response = config.transform_audio_transcription_response(
        httpx.Response(
            200,
            json={
                "request_id": "asr-request",
                "output": {"choices": [{"message": {"content": [{"text": "你好"}]}}]},
                "usage": {"duration": 2.5},
            },
        )
    )

    assert request.data["input"]["messages"][0]["content"][0]["input_audio"]["data"].startswith(
        "data:audio/wav;base64,"
    )
    assert response.text == "你好"
    assert response._hidden_params["audio_transcription_duration"] == 2.5


def test_dashscope_filetrans_request_and_polling() -> None:
    requests: list[tuple[str, str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, str(request.url), body))
        if request.method == "POST":
            return httpx.Response(200, json={"output": {"task_id": "file-task"}})
        if request.url.host == "cdn.example.com":
            return httpx.Response(200, json={"transcripts": [{"text": "第一段"}, {"text": "第二段"}]})
        return httpx.Response(
            200,
            json={
                "output": {
                    "task_status": "SUCCEEDED",
                    "results": [{"transcription_url": "https://cdn.example.com/result.json"}],
                },
                "usage": {"duration": 12.5},
            },
        )

    handler = DashScopeAudioTranscriptionHandler(sync_sleep=lambda _: None)
    result = handler.audio_transcriptions(
        model=FILE_TRANSCRIPTION_MODEL,
        audio_file=("unused.wav", b""),
        optional_params={"extra_body": {"file_url": "https://example.com/audio.wav", "polling_interval": 0}},
        litellm_params={},
        model_response=TranscriptionResponse(),
        timeout=10,
        max_retries=0,
        logging_obj=MagicMock(),
        api_key="sk-test",
        api_base="https://dashscope.example.com/api/v1",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
    )

    assert isinstance(result, TranscriptionResponse)
    assert result.text == "第一段\n第二段"
    assert result._hidden_params["audio_transcription_duration"] == 12.5
    assert requests[0] == (
        "POST",
        "https://dashscope.example.com/api/v1/services/audio/asr/transcription",
        {
            "model": FILE_TRANSCRIPTION_MODEL,
            "input": {"file_urls": ["https://example.com/audio.wav"]},
            "parameters": {},
        },
    )
    assert requests[1][1] == "https://dashscope.example.com/api/v1/tasks/file-task"
    assert requests[2][1] == "https://cdn.example.com/result.json"


def test_litellm_transcription_dispatches_dashscope_filetrans() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["input"] == {"file_urls": ["https://example.com/audio.wav"]}
            return httpx.Response(200, json={"output": {"task_id": "public-task"}})
        if request.url.host == "cdn.example.com":
            return httpx.Response(200, json={"transcripts": [{"text": "公开调用成功"}]})
        return httpx.Response(
            200,
            json={
                "output": {
                    "task_status": "SUCCEEDED",
                    "results": [{"transcription_url": "https://cdn.example.com/public.json"}],
                },
                "usage": {"duration": 3.0},
            },
        )

    result = litellm.transcription(
        model=f"dashscope/{FILE_TRANSCRIPTION_MODEL}",
        file=("unused.wav", b""),
        api_key="sk-test",
        api_base="https://dashscope.example.com/api/v1",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
        extra_body={"file_url": "https://example.com/audio.wav", "polling_interval": 0},
    )

    assert isinstance(result, TranscriptionResponse)
    assert result.text == "公开调用成功"
    assert result._hidden_params["audio_transcription_duration"] == 3.0


def test_dashscope_filetrans_rejects_multiple_urls() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        DashScopeAudioTranscriptionConfig().transform_audio_transcription_request(
            model=FILE_TRANSCRIPTION_MODEL,
            audio_file=("unused.wav", b""),
            optional_params={"extra_body": {"file_urls": ["one.wav", "two.wav"]}},
            litellm_params={},
        )


class _SyncWebSocket:
    def __init__(self, messages: list[str | bytes]) -> None:
        self.messages = iter(messages)
        self.sent: list[str | bytes] = []

    def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    def recv(self, timeout: float | None = None) -> str | bytes:
        return next(self.messages)


def test_dashscope_streaming_asr_collects_audio_results_and_duration() -> None:
    websocket = _SyncWebSocket(
        [
            json.dumps({"header": {"event": "task-started"}, "payload": {}}),
            json.dumps(
                {
                    "header": {"event": "result-generated"},
                    "payload": {
                        "output": {"sentence": {"sentence_id": 1, "text": "你"}},
                        "usage": {"duration": 0.5},
                    },
                }
            ),
            json.dumps(
                {
                    "header": {"event": "result-generated"},
                    "payload": {
                        "output": {"sentence": {"sentence_id": 2, "text": "好"}},
                        "usage": {"duration": 1.25},
                    },
                }
            ),
            json.dumps({"header": {"event": "task-finished"}, "payload": {}}),
        ]
    )
    handler = DashScopeAudioTranscriptionHandler(
        sync_connector=lambda *args, **kwargs: nullcontext(websocket),
        sync_sleep=lambda _: None,
    )
    result = handler.audio_transcriptions(
        model=STREAMING_ASR_MODEL,
        audio_file=("sample.wav", b"abcd"),
        optional_params={"extra_body": {"chunk_size": 2, "chunk_interval": 0}},
        litellm_params={},
        model_response=TranscriptionResponse(),
        timeout=10,
        max_retries=0,
        logging_obj=MagicMock(),
        api_key="sk-test",
        api_base=None,
    )

    assert isinstance(result, TranscriptionResponse)
    assert result.text == "你好"
    assert result._hidden_params["audio_transcription_duration"] == 1.25
    assert websocket.sent[1:3] == [b"ab", b"cd"]
    assert json.loads(cast(str, websocket.sent[-1]))["header"]["action"] == "finish-task"


def test_dashscope_audio_transcription_registration() -> None:
    config = ProviderConfigManager.get_provider_audio_transcription_config(
        model="qwen-audio-3.0-asr-flash",
        provider=litellm.LlmProviders.DASHSCOPE,
    )
    assert isinstance(config, DashScopeAudioTranscriptionConfig)
