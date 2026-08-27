import json
from contextlib import nullcontext
from typing import cast
from unittest.mock import MagicMock

import litellm

from litellm.llms.dashscope.text_to_speech.handler import DashScopeTextToSpeechHandler
from litellm.llms.dashscope.text_to_speech.transformation import DashScopeTextToSpeechConfig
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.utils import ProviderConfigManager


class _SyncWebSocket:
    def __init__(self, messages: list[str | bytes]) -> None:
        self.messages = iter(messages)
        self.sent: list[str | bytes] = []

    def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    def recv(self, timeout: float | None = None) -> str | bytes:
        return next(self.messages)


def test_dashscope_tts_websocket_events_and_audio() -> None:
    websocket = _SyncWebSocket(
        [
            json.dumps({"header": {"event": "task-started"}, "payload": {}}),
            b"audio-",
            b"bytes",
            json.dumps({"header": {"event": "task-finished"}, "payload": {"usage": {"characters": 2}}}),
        ]
    )
    handler = DashScopeTextToSpeechHandler(sync_connector=lambda *args, **kwargs: nullcontext(websocket))
    response = handler.text_to_speech(
        model="qwen-audio-3.0-tts-plus",
        input="你好",
        voice="longanyang",
        optional_params={"format": "wav", "rate": 1.2},
        litellm_params={},
        logging_obj=MagicMock(),
        timeout=10,
        api_key="sk-test",
        api_base=None,
    )

    assert isinstance(response, HttpxBinaryResponseContent)
    assert response.content == b"audio-bytes"
    assert response.response.headers["content-type"] == "audio/wav"
    assert response._hidden_params["usage"] == {"characters": 2}
    sent_events = [json.loads(cast(str, message)) for message in websocket.sent]
    assert [event["header"]["action"] for event in sent_events] == [
        "run-task",
        "continue-task",
        "finish-task",
    ]
    assert sent_events[0]["payload"]["parameters"]["voice"] == "longanyang"
    assert sent_events[1]["payload"]["input"]["text"] == "你好"


def test_dashscope_tts_param_mapping_and_registration() -> None:
    config = DashScopeTextToSpeechConfig()
    voice, params = config.map_openai_params(
        model="qwen-audio-3.0-tts-flash",
        optional_params={"response_format": "opus", "speed": 0.8, "instructions": "温柔、轻快"},
        voice="longanlingxi",
        kwargs={
            "extra_body": {
                "sample_rate": 24000,
                "volume": 60,
                "language_hints": ["zh"],
                "enable_aigc_tag": True,
            }
        },
    )

    assert voice == "longanlingxi"
    assert params == {
        "format": "opus",
        "rate": 0.8,
        "instruction": "温柔、轻快",
        "sample_rate": 24000,
        "volume": 60,
        "language_hints": ["zh"],
        "enable_aigc_tag": True,
    }
    registered = ProviderConfigManager.get_provider_text_to_speech_config(
        model="qwen-audio-3.0-tts-flash",
        provider=litellm.LlmProviders.DASHSCOPE,
    )
    assert isinstance(registered, DashScopeTextToSpeechConfig)
