import base64
from io import BytesIO
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm
from litellm.llms.volcengine.audio_transcription.transformation import (
    VolcEngineAudioTranscriptionConfig,
)
from litellm.llms.volcengine.common_utils import VolcEngineError
from litellm.utils import ProviderConfigManager


def _audio_file() -> tuple[str, BytesIO, str]:
    return ("message.ogg", BytesIO(b"ogg-opus-audio"), "audio/ogg")


def _provider_response(text: str = "语音消息", duration: int = 2499) -> httpx.Response:
    return httpx.Response(
        200,
        headers={
            "X-Api-Status-Code": "20000000",
            "X-Api-Message": "OK",
            "X-Tt-Logid": "test-log-id",
        },
        json={"audio_info": {"duration": duration}, "result": {"text": text}},
    )


def test_volcengine_audio_transcription_maps_openai_params():
    config = VolcEngineAudioTranscriptionConfig()

    assert config.map_openai_params(
        {"language": "zh", "response_format": "text"},
        {},
        "volc.bigasr.auc_turbo",
        False,
    ) == {"language": "zh", "response_format": "text"}
    assert (
        config.get_complete_url(
            "https://ark.cn-beijing.volces.com/api/v3",
            "key",
            "volc.bigasr.auc_turbo",
            {},
            {},
        )
        == "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
    )


def test_volcengine_audio_transcription_builds_synchronous_binary_request():
    config = VolcEngineAudioTranscriptionConfig()
    headers = config.validate_environment(
        {},
        "volc.bigasr.auc_turbo",
        [],
        {},
        {},
        api_key="speech-key",
    )
    request = config.transform_audio_transcription_request(
        model="volc.bigasr.auc_turbo",
        audio_file=_audio_file(),
        optional_params={"language": "zh"},
        litellm_params={},
    )

    assert headers["X-Api-Key"] == "speech-key"
    assert headers["X-Api-Resource-Id"] == "volc.bigasr.auc_turbo"
    assert headers["X-Api-Sequence"] == "-1"
    assert headers["X-Api-Request-Id"]
    assert request.data["audio"] == {
        "data": base64.b64encode(b"ogg-opus-audio").decode("ascii"),
        "language": "zh-CN",
    }
    assert request.data["request"] == {
        "model_name": "bigmodel",
        "enable_itn": True,
        "enable_punc": True,
        "enable_ddc": True,
    }


def test_volcengine_audio_transcription_transforms_response_and_duration():
    response = VolcEngineAudioTranscriptionConfig().transform_audio_transcription_response(_provider_response())

    assert response.text == "语音消息"
    assert response.duration == 2.499


def test_volcengine_audio_transcription_rejects_provider_error():
    raw_response = httpx.Response(
        200,
        headers={"X-Api-Status-Code": "45000151", "X-Api-Message": "audio format is invalid"},
        json={},
    )

    with pytest.raises(VolcEngineError, match="45000151.*audio format is invalid"):
        VolcEngineAudioTranscriptionConfig().transform_audio_transcription_response(raw_response)


def test_provider_config_manager_routes_volcengine_audio_transcription():
    config = ProviderConfigManager.get_provider_audio_transcription_config(
        model="volc.bigasr.auc_turbo",
        provider=litellm.LlmProviders.VOLCENGINE,
    )

    assert isinstance(config, VolcEngineAudioTranscriptionConfig)


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_litellm_transcription_dispatches_volcengine_flash_request(mock_post):
    mock_post.return_value = _provider_response()

    response = litellm.transcription(
        model="volcengine/volc.bigasr.auc_turbo",
        file=_audio_file(),
        language="zh",
        response_format="json",
        api_key="speech-key",
    )

    assert response.text == "语音消息"
    assert response.duration == 2.499
    assert mock_post.call_args.kwargs["url"] == ("https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash")
    assert mock_post.call_args.kwargs["headers"]["X-Api-Key"] == "speech-key"
    assert mock_post.call_args.kwargs["headers"]["X-Api-Resource-Id"] == "volc.bigasr.auc_turbo"
    assert mock_post.call_args.kwargs["json"]["audio"] == {
        "data": base64.b64encode(b"ogg-opus-audio").decode("ascii"),
        "language": "zh-CN",
    }


@pytest.mark.asyncio
@patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock)
async def test_litellm_atranscription_dispatches_volcengine_flash_request(mock_post):
    mock_post.return_value = _provider_response(text="异步语音消息", duration=3200)

    response = await litellm.atranscription(
        model="volcengine/volc.bigasr.auc_turbo",
        file=_audio_file(),
        api_key="speech-key",
    )

    assert response.text == "异步语音消息"
    assert response.duration == 3.2
    assert mock_post.call_args.kwargs["url"] == ("https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash")


def test_volcengine_audio_model_prices():
    assert litellm.model_cost["volcengine/volc.bigasr.auc_turbo"]["input_cost_per_second"] == 0.00125
    response = VolcEngineAudioTranscriptionConfig().transform_audio_transcription_response(
        _provider_response(duration=2499)
    )

    assert litellm.completion_cost(
        completion_response=response,
        model="volcengine/volc.bigasr.auc_turbo",
        call_type="transcription",
        custom_llm_provider="volcengine",
    ) == pytest.approx(2.499 * 0.00125)
