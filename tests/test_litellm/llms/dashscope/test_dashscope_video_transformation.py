import json
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.dashscope.videos.transformation import (
    LEGACY_KEYFRAME_SYNTHESIS_PATH,
    THREE_D_GENERATION_PATH,
    VIDEO_SYNTHESIS_PATH,
    DashScopeVideoConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider, extract_original_video_id
from litellm.utils import ProviderConfigManager


def test_dashscope_video_generation_status_and_edit() -> None:
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body))
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "request_id": "request-status",
                    "output": {
                        "task_id": "task-123",
                        "task_status": "SUCCEEDED",
                        "video_url": "https://cdn.example.com/source.mp4",
                    },
                    "usage": {"duration": 10, "video_count": 1, "SR": 1080},
                },
            )
        assert body is not None
        task_id = "task-edit" if body["model"] == "wan2.7-videoedit" else "task-123"
        return httpx.Response(
            200,
            json={
                "request_id": "request-create",
                "output": {"task_id": task_id, "task_status": "PENDING"},
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    created = litellm.video_generation(
        model="dashscope/wan2.7-i2v",
        prompt="A cat walks toward the camera",
        input_reference="https://cdn.example.com/first.png",
        seconds="10",
        size="1920x1080",
        extra_body={
            "last_frame_url": "https://cdn.example.com/last.png",
            "prompt_extend": False,
            "watermark": True,
        },
        api_key="test-key",
        api_base="https://dashscope.example.com/api/v1",
        client=client,
    )
    status = litellm.video_status(
        video_id=created.id,
        api_key="test-key",
        api_base="https://dashscope.example.com/api/v1",
        client=client,
    )
    video_edit_call = cast(Callable[..., object], litellm.video_edit)
    edited_result = video_edit_call(
        video_id=created.id,
        prompt="Convert the scene to claymation",
        resolution="720P",
        watermark=True,
        api_key="test-key",
        api_base="https://dashscope.example.com/api/v1",
        client=client,
    )
    assert isinstance(edited_result, VideoObject)
    edited = edited_result
    legacy_edit_result = video_edit_call(
        video_id="external-source",
        prompt="Replace the subject",
        custom_llm_provider="dashscope",
        extra_body={"model": "wan2.1-vace-plus"},
        video_url="https://cdn.example.com/external.mp4",
        function="video_repainting",
        control_condition="depth",
        prompt_extend=False,
        api_key="test-key",
        api_base="https://dashscope.example.com/api/v1",
        client=client,
    )
    assert isinstance(legacy_edit_result, VideoObject)

    assert extract_original_video_id(created.id) == "task-123"
    assert created.status == "queued"
    assert created.model == "wan2.7-i2v"
    assert status.status == "completed"
    assert status.seconds == "10"
    assert status.usage is not None
    assert status.usage["duration_seconds"] == 10.0
    assert status.usage["video_resolution"] == "1080p"
    cost_logging = MagicMock()
    cost_logging.litellm_params = {
        "metadata": {
            "model_info": {
                "output_cost_per_second": 0.1,
                "output_cost_per_second_720p": 0.1,
                "output_cost_per_second_1080p": 0.15,
            }
        }
    }
    assert (
        litellm.completion_cost(
            completion_response=status,
            model="wan2.7-i2v",
            call_type="video_retrieve",
            custom_llm_provider="dashscope",
            custom_pricing=True,
            litellm_logging_obj=cost_logging,
        )
        == 1.5
    )
    assert extract_original_video_id(edited.id) == "task-edit"
    assert edited.model == "wan2.7-videoedit"
    assert legacy_edit_result.model == "wan2.1-vace-plus"
    assert requests == [
        (
            "POST",
            f"/api/v1{VIDEO_SYNTHESIS_PATH}",
            {
                "model": "wan2.7-i2v",
                "input": {
                    "media": [
                        {"type": "first_frame", "url": "https://cdn.example.com/first.png"},
                        {"type": "last_frame", "url": "https://cdn.example.com/last.png"},
                    ],
                    "prompt": "A cat walks toward the camera",
                },
                "parameters": {
                    "duration": 10,
                    "prompt_extend": False,
                    "ratio": "16:9",
                    "resolution": "1080P",
                    "watermark": True,
                },
            },
        ),
        ("GET", "/api/v1/tasks/task-123", None),
        ("GET", "/api/v1/tasks/task-123", None),
        (
            "POST",
            f"/api/v1{VIDEO_SYNTHESIS_PATH}",
            {
                "model": "wan2.7-videoedit",
                "input": {
                    "prompt": "Convert the scene to claymation",
                    "media": [{"type": "video", "url": "https://cdn.example.com/source.mp4"}],
                },
                "parameters": {"resolution": "720P", "watermark": True},
            },
        ),
        (
            "POST",
            f"/api/v1{VIDEO_SYNTHESIS_PATH}",
            {
                "model": "wan2.1-vace-plus",
                "input": {
                    "function": "video_repainting",
                    "prompt": "Replace the subject",
                    "video_url": "https://cdn.example.com/external.mp4",
                },
                "parameters": {"control_condition": "depth", "prompt_extend": False},
            },
        ),
    ]


@pytest.mark.parametrize(
    ("model", "input_reference", "extra_body", "expected_input"),
    [
        (
            "wan2.7-t2v-2026-06-12",
            None,
            {"audio_url": "https://cdn.example.com/audio.mp3"},
            {"audio_url": "https://cdn.example.com/audio.mp3", "prompt": "A cinematic sunrise"},
        ),
        (
            "wan2.6-i2v-flash",
            "https://cdn.example.com/frame.png",
            {"audio_url": "https://cdn.example.com/audio.mp3"},
            {
                "audio_url": "https://cdn.example.com/audio.mp3",
                "img_url": "https://cdn.example.com/frame.png",
                "prompt": "A cinematic sunrise",
            },
        ),
        (
            "wan2.7-r2v-2026-06-12",
            None,
            {
                "media": [
                    {"type": "reference_video", "url": "https://cdn.example.com/person.mp4"},
                    {"type": "reference_image", "url": "https://cdn.example.com/guitar.png"},
                ]
            },
            {
                "media": [
                    {"type": "reference_video", "url": "https://cdn.example.com/person.mp4"},
                    {"type": "reference_image", "url": "https://cdn.example.com/guitar.png"},
                ],
                "prompt": "A cinematic sunrise",
            },
        ),
    ],
)
def test_dashscope_video_generation_input_modes(
    model: str,
    input_reference: str | None,
    extra_body: dict[str, object],
    expected_input: dict[str, object],
) -> None:
    config = DashScopeVideoConfig()
    optional_params = config.map_openai_params(
        video_create_optional_params=cast(
            VideoCreateOptionalRequestParams,
            {"input_reference": input_reference, "seconds": "5", **extra_body},
        ),
        model=model,
        drop_params=False,
    )
    headers: dict[str, str] = {}
    request, files, url = config.transform_video_create_request(
        model=model,
        prompt="A cinematic sunrise",
        api_base="https://dashscope.example.com/api/v1",
        video_create_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers=headers,
    )

    assert request["input"] == expected_input
    assert request["parameters"] == {"duration": 5}
    assert files == []
    assert url.endswith(VIDEO_SYNTHESIS_PATH)
    assert headers["X-DashScope-Async"] == "enable"


def test_dashscope_legacy_keyframe_generation() -> None:
    config = DashScopeVideoConfig()
    optional_params = config.map_openai_params(
        video_create_optional_params=cast(
            VideoCreateOptionalRequestParams,
            {
                "input_reference": "https://cdn.example.com/first.png",
                "size": "1280x720",
                "last_frame_url": "https://cdn.example.com/last.png",
                "prompt_extend": True,
            },
        ),
        model="wan2.2-kf2v-flash",
        drop_params=False,
    )
    request, _, url = config.transform_video_create_request(
        model="wan2.2-kf2v-flash",
        prompt="Smoothly transition between both frames",
        api_base="https://dashscope.example.com/api/v1",
        video_create_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert url.endswith(LEGACY_KEYFRAME_SYNTHESIS_PATH)
    assert request == {
        "model": "wan2.2-kf2v-flash",
        "input": {
            "first_frame_url": "https://cdn.example.com/first.png",
            "last_frame_url": "https://cdn.example.com/last.png",
            "prompt": "Smoothly transition between both frames",
        },
        "parameters": {"resolution": "720P", "prompt_extend": True},
    }


def test_dashscope_video_content_download() -> None:
    def download(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://cdn.example.com/generated.mp4"
        return httpx.Response(200, content=b"video-bytes")

    config = DashScopeVideoConfig(sync_http_client=httpx.Client(transport=httpx.MockTransport(download)))
    raw_response = httpx.Response(
        200,
        json={
            "output": {
                "task_id": "task-123",
                "task_status": "SUCCEEDED",
                "video_url": "https://cdn.example.com/generated.mp4",
            }
        },
    )

    assert config.transform_video_content_response(raw_response, MagicMock()) == b"video-bytes"


def test_dashscope_failed_video_usage_is_not_billable() -> None:
    config = DashScopeVideoConfig()
    video = config.transform_video_create_response(
        model="wan2.7-i2v",
        raw_response=httpx.Response(
            200,
            json={
                "output": {
                    "task_id": "task-failed",
                    "task_status": "FAILED",
                    "code": "InvalidVideo",
                    "message": "Video generation failed",
                },
                "usage": {"duration": 10, "video_count": 0, "SR": 1080},
            },
        ),
        logging_obj=MagicMock(),
    )

    assert video.status == "failed"
    assert video.usage is not None
    assert video.usage["generated_videos"] == 0
    assert "duration_seconds" not in video.usage


def test_dashscope_video_registration_and_environment() -> None:
    config = ProviderConfigManager.get_provider_video_config(
        model="wan2.7-t2v-2026-06-12",
        provider=LlmProviders.DASHSCOPE,
    )

    assert isinstance(config, DashScopeVideoConfig)
    pricing_path = Path(__file__).parents[4] / "model_prices_and_context_window.json"
    pricing = cast(dict[str, dict[str, object]], json.loads(pricing_path.read_text()))
    model_info = pricing["dashscope/wan2.7-t2v-2026-06-12"]
    assert model_info["mode"] == "video_generation"
    assert model_info["supported_output_modalities"] == ["video"]
    expected_prices = {
        "dashscope/wan2.1-vace-plus": (0.1, None, 0.1, None),
        "dashscope/wan2.2-i2v-plus": (0.02, 0.02, None, 0.1),
        "dashscope/wan2.2-kf2v-flash": (0.015, 0.015, 0.036, 0.07),
        "dashscope/wan2.2-t2v-plus": (0.02, 0.02, None, 0.1),
        "dashscope/wan2.5-i2v-preview": (0.05, 0.05, 0.1, 0.15),
        "dashscope/wan2.6-i2v": (0.1, None, 0.1, 0.15),
        "dashscope/wan2.6-i2v-flash": (0.05, None, 0.05, 0.075),
        "dashscope/wan2.6-r2v": (0.1, None, 0.1, 0.15),
        "dashscope/wan2.6-r2v-flash": (0.05, None, 0.05, 0.075),
        "dashscope/wan2.6-t2v": (0.1, None, 0.1, 0.15),
        "dashscope/wan2.7-i2v": (0.1, None, 0.1, 0.15),
        "dashscope/wan2.7-r2v-2026-06-12": (0.1, None, 0.1, 0.15),
        "dashscope/wan2.7-t2v-2026-06-12": (0.1, None, 0.1, 0.15),
        "dashscope/wan2.7-videoedit": (0.1, None, 0.1, 0.15),
    }
    assert {
        model: (
            pricing[model].get("output_cost_per_second"),
            pricing[model].get("output_cost_per_second_480p"),
            pricing[model].get("output_cost_per_second_720p"),
            pricing[model].get("output_cost_per_second_1080p"),
        )
        for model in expected_prices
    } == expected_prices
    assert (
        config.get_complete_url(
            model="wan2.7-t2v-2026-06-12",
            api_base="https://dashscope.example.com/compatible-mode/v1",
            litellm_params={},
        )
        == "https://dashscope.example.com/api/v1"
    )
    assert config.validate_environment({}, "wan2.7-t2v-2026-06-12", api_key="test-key") == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }


@pytest.mark.parametrize(
    ("input_reference", "extra_body", "expected_input"),
    [
        (None, {}, {"prompt": "一只可爱的猫"}),
        ("https://example.com/cat.png", {}, {"image": "https://example.com/cat.png"}),
        (
            None,
            {
                "images": [
                    {"type": "jpeg", "file_token": "https://example.com/front.jpg"},
                    {},
                    {"type": "png", "file_token": "https://example.com/back.png"},
                    {},
                ]
            },
            {
                "images": [
                    {"type": "jpeg", "file_token": "https://example.com/front.jpg"},
                    {},
                    {"type": "png", "file_token": "https://example.com/back.png"},
                    {},
                ]
            },
        ),
    ],
)
def test_dashscope_tripo_input_modes(
    input_reference: str | None,
    extra_body: dict[str, object],
    expected_input: dict[str, object],
) -> None:
    config = DashScopeVideoConfig()
    params = config.map_openai_params(
        cast(VideoCreateOptionalRequestParams, {"input_reference": input_reference, **extra_body}),
        model="Tripo/Tripo-H3.1",
        drop_params=False,
    )
    request, _, url = config.transform_video_create_request(
        model="Tripo/Tripo-H3.1",
        prompt="一只可爱的猫",
        api_base="https://dashscope.example.com/api/v1",
        video_create_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["input"] == expected_input
    assert url.endswith(THREE_D_GENERATION_PATH)


def test_dashscope_tripo_rejects_invalid_multi_image_input() -> None:
    config = DashScopeVideoConfig()
    params = config.map_openai_params(
        cast(VideoCreateOptionalRequestParams, {"images": [{"type": "jpeg", "file_token": "one.jpg"}]}),
        model="Tripo/Tripo-H3.1",
        drop_params=False,
    )

    with pytest.raises(ValueError, match="exactly four"):
        config.transform_video_create_request(
            model="Tripo/Tripo-H3.1",
            prompt="",
            api_base="https://dashscope.example.com/api/v1",
            video_create_optional_request_params=params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )


def test_dashscope_tripo_base_model_response_and_pricing_context() -> None:
    config = DashScopeVideoConfig()
    response = httpx.Response(
        200,
        json={
            "request_id": "tripo-request",
            "output": {
                "task_id": "tripo-task",
                "task_status": "SUCCEEDED",
                "results": [
                    {
                        "base_model_url": "https://example.com/model.glb",
                        "rendered_image_url": "https://example.com/preview.webp",
                    }
                ],
            },
            "usage": {"count": 2, "geometry_quality": "ultra", "texture_quality": "detailed"},
        },
    )
    video = config.transform_video_create_response(
        model="Tripo/Tripo-H3.1",
        raw_response=response,
        logging_obj=MagicMock(),
        request_data={
            "model": "Tripo/Tripo-H3.1",
            "input": {"image": "https://example.com/cat.png"},
            "parameters": {"pbr": False, "texture": False},
        },
    )

    assert video.status == "completed"
    assert video.usage == {
        "count": 2,
        "geometry_quality": "ultra",
        "texture_quality": "detailed",
        "video_resolution": "ultra_no_texture",
        "has_video_input": True,
        "completion_tokens": 2,
        "generated_videos": 2,
    }
    assert video._hidden_params["video_url"] == "https://example.com/model.glb"
    assert video._hidden_params["rendered_image_url"] == "https://example.com/preview.webp"


def test_dashscope_tripo_pbr_forces_texture_pricing() -> None:
    assert (
        DashScopeVideoConfig._three_d_pricing_tier(
            usage={},
            request_data={"parameters": {"geometry_quality": "standard", "texture": False, "pbr": True}},
        )
        == "standard_sd_texture"
    )
    assert (
        DashScopeVideoConfig._three_d_pricing_tier(
            usage={"texture_quality": "detailed"},
            request_data={"parameters": {"geometry_quality": "ultra"}},
        )
        == "ultra_hd_texture"
    )


def test_dashscope_tripo_status_preserves_create_pricing_tier() -> None:
    config = DashScopeVideoConfig()
    created = config.transform_video_create_response(
        model="Tripo/Tripo-H3.1",
        raw_response=httpx.Response(
            200,
            json={"output": {"task_id": "tripo-task", "task_status": "PENDING"}, "usage": {}},
        ),
        logging_obj=MagicMock(),
        request_data={
            "model": "Tripo/Tripo-H3.1",
            "input": {"prompt": "一只白色陶瓷茶杯"},
            "parameters": {"geometry_quality": "standard", "pbr": False, "texture": False},
        },
    )
    decoded_created_id = decode_video_id_with_provider(created.id)
    assert decoded_created_id["video_resolution"] == "standard_no_texture"

    status_logging = MagicMock()
    status_logging.litellm_params = {"video_id": created.id}
    completed = config.transform_video_status_retrieve_response(
        raw_response=httpx.Response(
            200,
            json={
                "output": {"task_id": "tripo-task", "task_status": "SUCCEEDED"},
                "usage": {"count": 1, "geometry_quality": "standard", "3d_task_type": "text-to-3d"},
            },
        ),
        logging_obj=status_logging,
        custom_llm_provider="dashscope",
    )

    assert completed.usage is not None
    assert completed.usage["video_resolution"] == "standard_no_texture"
    assert completed.usage["completion_tokens"] == 1
    assert decode_video_id_with_provider(completed.id)["video_resolution"] == "standard_no_texture"
    assert litellm.completion_cost(
        completion_response=completed,
        model="dashscope/Tripo/Tripo-H3.1",
        call_type="video_retrieve",
    ) == pytest.approx(0.7)
