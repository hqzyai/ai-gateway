import base64
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.volcengine.text_to_speech.transformation import (
    VolcEngineTextToSpeechConfig,
)
from litellm.utils import ProviderConfigManager


def test_volcengine_text_to_speech_openai_parameter_mapping():
    voice, optional_params = VolcEngineTextToSpeechConfig().map_openai_params(
        model="seed-tts-2.0",
        optional_params={"response_format": "opus"},
        voice="zh_female_vv_uranus_bigtts",
    )

    assert voice == "zh_female_vv_uranus_bigtts"
    assert optional_params == {"format": "ogg_opus"}


def test_volcengine_text_to_speech_rejects_unsupported_openai_speed():
    with pytest.raises(ValueError, match="speed"):
        VolcEngineTextToSpeechConfig().map_openai_params(
            model="seed-tts-2.0",
            optional_params={"speed": 1.25},
            voice="zh_female_vv_uranus_bigtts",
        )


def test_volcengine_text_to_speech_request_and_environment(monkeypatch):
    monkeypatch.setenv("VOLCENGINE_SPEECH_API_KEY", "speech-key")
    config = VolcEngineTextToSpeechConfig()
    headers = config.validate_environment({}, "seed-tts-2.0")
    request = config.transform_text_to_speech_request(
        model="seed-tts-2.0",
        input="你好",
        voice="zh_female_vv_uranus_bigtts",
        optional_params={"format": "mp3"},
        litellm_params={},
        headers=headers,
    )

    assert headers["X-Api-Key"] == "speech-key"
    assert headers["X-Api-Resource-Id"] == "seed-tts-2.0"
    assert headers["X-Api-Request-Id"]
    assert request["dict_body"] == {
        "req_params": {
            "text": "你好",
            "speaker": "zh_female_vv_uranus_bigtts",
            "audio_params": {"format": "mp3", "sample_rate": 24000},
        }
    }
    assert (
        config.get_complete_url("seed-tts-2.0", "https://ark.cn-beijing.volces.com/api/v3", {})
        == "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
    )


def test_volcengine_text_to_speech_decodes_chunked_audio():
    first_audio = b"first-audio-"
    second_audio = b"second-audio"
    request = httpx.Request(
        "POST",
        "https://openspeech.bytedance.com/api/v3/tts/unidirectional",
        json={
            "req_params": {
                "audio_params": {"format": "wav"},
            }
        },
    )
    response = httpx.Response(
        200,
        request=request,
        content="\n".join(
            (
                json.dumps({"code": 0, "data": base64.b64encode(first_audio).decode()}),
                json.dumps({"code": 0, "data": base64.b64encode(second_audio).decode()}),
                json.dumps({"code": 20000000, "message": "done"}),
            )
        ).encode(),
    )

    transformed = VolcEngineTextToSpeechConfig().transform_text_to_speech_response(
        "seed-tts-2.0", response, MagicMock()
    )

    assert transformed.content == first_audio + second_audio
    assert transformed.response.headers["content-type"] == "audio/wav"


def test_provider_config_manager_routes_volcengine_text_to_speech():
    config = ProviderConfigManager.get_provider_text_to_speech_config(
        model="seed-tts-2.0",
        provider=litellm.LlmProviders.VOLCENGINE,
    )

    assert isinstance(config, VolcEngineTextToSpeechConfig)


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_litellm_speech_dispatches_volcengine_request(mock_post):
    audio = b"generated-audio"
    mock_post.return_value = httpx.Response(
        200,
        request=httpx.Request(
            "POST",
            "https://openspeech.bytedance.com/api/v3/tts/unidirectional",
            json={"req_params": {"audio_params": {"format": "mp3"}}},
        ),
        content=(
            json.dumps({"code": 0, "data": base64.b64encode(audio).decode()}) + "\n" + json.dumps({"code": 20000000})
        ).encode(),
    )

    response = litellm.speech(
        model="volcengine/seed-tts-2.0",
        input="你好",
        voice="zh_female_vv_uranus_bigtts",
        response_format="mp3",
        api_key="speech-key",
    )

    assert response.content == audio
    assert mock_post.call_args.kwargs["url"] == "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
    assert mock_post.call_args.kwargs["headers"]["X-Api-Key"] == "speech-key"
    assert mock_post.call_args.kwargs["json"] == {
        "req_params": {
            "text": "你好",
            "speaker": "zh_female_vv_uranus_bigtts",
            "audio_params": {"format": "mp3", "sample_rate": 24000},
        }
    }


def test_volcengine_text_to_speech_model_price():
    assert litellm.model_cost["volcengine/seed-tts-2.0"]["input_cost_per_character"] == 0.0003
    assert (
        litellm.completion_cost(
            model="volcengine/seed-tts-2.0",
            prompt="你好",
            call_type="speech",
            custom_llm_provider="volcengine",
        )
        == 0.0006
    )
