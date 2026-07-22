import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.volcengine.videos.transformation import VolcEngineVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.utils import decode_video_id_with_provider
from litellm.utils import ProviderConfigManager


SEEDANCE_VIDEO_MODELS = (
    "volcengine/doubao-seedance-2-0-260128",
    "volcengine/doubao-seedance-2-0-fast-260128",
    "volcengine/doubao-seedance-2-0-mini-260615",
)

REMOVED_SEEDANCE_VIDEO_MODELS = (
    "volcengine/doubao-seedance-1-0-pro-250528",
    "volcengine/doubao-seedance-1-0-pro-fast-251015",
    "volcengine/doubao-seedance-1-5-pro-251215",
)


@pytest.mark.parametrize("model", SEEDANCE_VIDEO_MODELS)
def test_seedance_model_is_registered_for_video_generation(model: str) -> None:
    repo_root = Path(__file__).resolve().parents[5]
    model_maps = (
        repo_root / "model_prices_and_context_window.json",
        repo_root / "litellm/model_prices_and_context_window_backup.json",
    )

    for model_map in model_maps:
        model_info = json.loads(model_map.read_text())[model]
        assert model_info["litellm_provider"] == "volcengine"
        assert model_info["mode"] == "video_generation"
        assert model_info["supported_endpoints"] == ["/v1/videos"]
        assert model_info["video_token_pricing"]["no_video_input"] > 0
        assert model_info["video_token_pricing"]["video_input"] > 0

        for removed_model in REMOVED_SEEDANCE_VIDEO_MODELS:
            assert removed_model not in json.loads(model_map.read_text())


def test_provider_config_manager_returns_volcengine_video_config() -> None:
    config = ProviderConfigManager.get_provider_video_config(
        model="doubao-seedance-2-0-260128",
        provider=LlmProviders.VOLCENGINE,
    )

    assert isinstance(config, VolcEngineVideoConfig)


def test_text_to_video_creates_ark_task_with_mapped_params() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
        assert request.headers["authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "model": "doubao-seedance-2-0-260128",
            "content": [{"type": "text", "text": "Clouds moving over a mountain lake"}],
            "duration": 5,
            "ratio": "16:9",
            "resolution": "720p",
            "generate_audio": True,
            "watermark": False,
        }
        return httpx.Response(200, json={"id": "task-123"})

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    response = litellm.video_generation(
        model="volcengine/doubao-seedance-2-0-260128",
        prompt="Clouds moving over a mountain lake",
        seconds="5",
        size="1280x720",
        api_key="test-key",
        client=client,
        extra_body={"generate_audio": True, "watermark": False},
    )

    assert response.status == "queued"
    assert response.seconds == "5"
    assert response.usage == {
        "duration_seconds": 5.0,
        "video_resolution": "720p",
        "has_video_input": False,
    }
    assert decode_video_id_with_provider(response.id) == {
        "custom_llm_provider": "volcengine",
        "model_id": "doubao-seedance-2-0-260128",
        "video_id": "task-123",
        "has_video_input": False,
    }


def test_video_status_maps_ark_task_to_openai_video() -> None:
    config = VolcEngineVideoConfig()
    response = httpx.Response(
        200,
        json={
            "id": "task-123",
            "model": "doubao-seedance-2-0-260128",
            "status": "succeeded",
            "created_at": 1750000000,
            "updated_at": 1750000030,
            "duration": 5,
            "ratio": "16:9",
            "resolution": "720p",
            "content": {"video_url": "https://example.com/video.mp4"},
            "usage": {"completion_tokens": 100, "total_tokens": 100},
        },
    )

    video = config.transform_video_status_retrieve_response(response, MagicMock())

    assert video.status == "completed"
    assert video.created_at == 1750000000
    assert video.completed_at == 1750000030
    assert video.seconds == "5"
    assert video.usage == {
        "completion_tokens": 100,
        "total_tokens": 100,
        "duration_seconds": 5.0,
        "video_resolution": "720p",
        "has_video_input": False,
    }


def test_video_input_billing_context_survives_status_retrieval() -> None:
    input_content = [
        {"type": "text", "text": "Extend this clip"},
        {
            "type": "video_url",
            "video_url": {"url": "https://example.com/input.mp4"},
            "role": "reference_video",
        },
    ]

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            assert json.loads(request.content)["content"] == input_content
            return httpx.Response(200, json={"id": "task-video-input"})
        return httpx.Response(
            200,
            json={
                "id": "task-video-input",
                "model": "doubao-seedance-2-0-260128",
                "status": "succeeded",
                "resolution": "1080p",
                "usage": {"completion_tokens": 100000, "total_tokens": 100000},
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    created = litellm.video_generation(
        model="volcengine/doubao-seedance-2-0-260128",
        prompt="Extend this clip",
        api_key="test-key",
        client=client,
        extra_body={"content": input_content, "resolution": "1080p"},
    )
    status = litellm.video_status(
        video_id=created.id,
        api_key="test-key",
        client=client,
    )

    assert decode_video_id_with_provider(created.id)["has_video_input"] is True
    assert status.usage == {
        "completion_tokens": 100000,
        "total_tokens": 100000,
        "video_resolution": "1080p",
        "has_video_input": True,
    }


@pytest.mark.parametrize(
    ("model", "resolution", "has_video_input", "yuan_per_million_tokens"),
    (
        ("doubao-seedance-2-0-260128", "720p", False, 46.0),
        ("doubao-seedance-2-0-260128", "720p", True, 28.0),
        ("doubao-seedance-2-0-260128", "1080p", False, 51.0),
        ("doubao-seedance-2-0-260128", "1080p", True, 31.0),
        ("doubao-seedance-2-0-260128", "4k", False, 26.0),
        ("doubao-seedance-2-0-260128", "4k", True, 16.0),
        ("doubao-seedance-2-0-fast-260128", "720p", False, 37.0),
        ("doubao-seedance-2-0-fast-260128", "720p", True, 22.0),
        ("doubao-seedance-2-0-mini-260615", "720p", False, 23.0),
        ("doubao-seedance-2-0-mini-260615", "720p", True, 14.0),
    ),
)
def test_seedance_video_cost_uses_token_scenario_pricing(
    model: str,
    resolution: str,
    has_video_input: bool,
    yuan_per_million_tokens: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[5]
    monkeypatch.setattr(
        litellm,
        "model_cost",
        json.loads((repo_root / "model_prices_and_context_window.json").read_text()),
    )
    response = litellm.VideoObject(
        id="video-cost-test",
        object="video",
        status="completed",
        model=model,
        usage={
            "completion_tokens": 100000,
            "total_tokens": 100000,
            "video_resolution": resolution,
            "has_video_input": has_video_input,
        },
    )

    cost = litellm.response_cost_calculator(
        response_object=response,
        model=f"volcengine/{model}",
        call_type="video_retrieve",
        custom_llm_provider="volcengine",
        optional_params={},
    )

    assert cost == pytest.approx(yuan_per_million_tokens / 10)


def test_video_status_url_encodes_task_id() -> None:
    config = VolcEngineVideoConfig()
    url, params = config.transform_video_status_retrieve_request(
        video_id="../../tasks/other?x=1#frag",
        api_base="https://ark.example.com/api/v3",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert url == ("https://ark.example.com/api/v3/contents/generations/tasks/..%2F..%2Ftasks%2Fother%3Fx%3D1%23frag")
    assert params == {}


def test_complete_url_accepts_full_task_endpoint() -> None:
    config = VolcEngineVideoConfig()

    assert (
        config.get_complete_url(
            model="doubao-seedance-2-0-260128",
            api_base="https://ark.example.com/api/v3/contents/generations/tasks/",
            litellm_params={},
        )
        == "https://ark.example.com/api/v3"
    )


def test_video_content_rejects_unfinished_task() -> None:
    config = VolcEngineVideoConfig()
    response = httpx.Response(200, json={"id": "task-123", "status": "running"})

    with pytest.raises(ValueError, match="not ready"):
        config.transform_video_content_response(response, MagicMock())


def test_video_content_downloads_completed_task_output() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/video.mp4"
        return httpx.Response(200, content=b"video-bytes")

    download_client = httpx.Client(transport=httpx.MockTransport(respond))
    config = VolcEngineVideoConfig(sync_http_client=download_client)
    task_response = httpx.Response(
        200,
        json={
            "id": "task-123",
            "status": "succeeded",
            "content": {"video_url": "https://example.com/video.mp4"},
        },
    )

    assert config.transform_video_content_response(task_response, MagicMock()) == b"video-bytes"
