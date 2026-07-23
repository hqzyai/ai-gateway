import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.siliconflow.videos.transformation import SiliconFlowVideoConfig
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import extract_original_video_id
from litellm.utils import ProviderConfigManager


def test_siliconflow_video_submit_and_status() -> None:
    requests: list[tuple[str, str, dict[str, object]]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.method, str(request.url), body))
        if request.url.path.endswith("/video/submit"):
            return httpx.Response(200, json={"requestId": "request-123"})
        return httpx.Response(
            200,
            json={
                "requestId": "request-123",
                "status": "Succeed",
                "results": {
                    "videos": [{"url": "https://example.com/video.mp4"}],
                },
                "timings": {"inference": 12.5},
                "seed": 8,
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    created = litellm.video_generation(
        model="siliconflow/Wan-AI/Wan2.2-I2V-A14B",
        prompt="waves at sunset",
        input_reference="https://example.com/source.png",
        size="1280x720",
        extra_body={"negative_prompt": "blur", "seed": 8},
        api_key="test-key",
        api_base="https://api.example.com/v1",
        client=client,
    )
    status = litellm.video_status(
        video_id=created.id,
        api_key="test-key",
        api_base="https://api.example.com/v1",
        client=client,
    )

    assert extract_original_video_id(created.id) == "request-123"
    assert created.status == "queued"
    assert created.model == "Wan-AI/Wan2.2-I2V-A14B"
    assert status.status == "completed"
    assert status.model == "Wan-AI/Wan2.2-I2V-A14B"
    assert status.usage == {"generated_videos": 1, "inference_time": 12.5, "seed": 8}
    assert requests == [
        (
            "POST",
            "https://api.example.com/v1/video/submit",
            {
                "model": "Wan-AI/Wan2.2-I2V-A14B",
                "prompt": "waves at sunset",
                "negative_prompt": "blur",
                "seed": 8,
                "image": "https://example.com/source.png",
                "image_size": "1280x720",
            },
        ),
        ("POST", "https://api.example.com/v1/video/status", {"requestId": "request-123"}),
    ]


def test_siliconflow_video_registration_and_per_video_cost() -> None:
    config = ProviderConfigManager.get_provider_video_config(
        model="Wan-AI/Wan2.2-T2V-A14B",
        provider=LlmProviders.SILICONFLOW,
    )
    model_info = {
        "litellm_provider": "siliconflow",
        "mode": "video_generation",
        "output_cost_per_video": 2.0,
    }
    response = VideoObject(
        id="request-123",
        object="video",
        status="queued",
        model="Wan-AI/Wan2.2-T2V-A14B",
        usage={"generated_videos": 1},
    )

    assert isinstance(config, SiliconFlowVideoConfig)
    with patch.dict(litellm.model_cost, {"siliconflow/Wan-AI/Wan2.2-T2V-A14B": model_info}):
        resolved_model_info = litellm.get_model_info(
            model="Wan-AI/Wan2.2-T2V-A14B",
            custom_llm_provider="siliconflow",
        )
        cost = completion_cost(
            completion_response=response,
            model="siliconflow/Wan-AI/Wan2.2-T2V-A14B",
            call_type="create_video",
        )
    assert resolved_model_info["output_cost_per_video"] == pytest.approx(2.0)
    assert cost == pytest.approx(2.0)


def test_siliconflow_video_pricing_metadata() -> None:
    pricing_path = Path(__file__).parents[5] / "model_prices_and_context_window.json"
    pricing = json.loads(pricing_path.read_text())

    assert pricing["siliconflow/Wan-AI/Wan2.2-T2V-A14B"]["output_cost_per_video"] == pytest.approx(2.0)
    assert pricing["siliconflow/Wan-AI/Wan2.2-I2V-A14B"]["output_cost_per_video"] == pytest.approx(2.0)
