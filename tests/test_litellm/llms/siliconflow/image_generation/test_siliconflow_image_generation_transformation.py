import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.siliconflow.image_generation.transformation import SiliconFlowImageGenerationConfig
from litellm.types.utils import ImageObject, ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


def test_siliconflow_image_generation_request_response_and_cost() -> None:
    captured: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "images": [
                    {"url": "https://example.com/first.png"},
                    {"url": "https://example.com/second.png"},
                ],
                "timings": {"inference": 1.5},
                "seed": 42,
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    response = litellm.image_generation(
        model="siliconflow/Kwai-Kolors/Kolors",
        prompt="a mountain lake",
        n=2,
        size="1024x1024",
        negative_prompt="fog",
        num_inference_steps=25,
        guidance_scale=7.5,
        seed=42,
        api_key="test-key",
        api_base="https://api.example.com/v1",
        client=client,
    )

    assert captured == {
        "url": "https://api.example.com/v1/images/generations",
        "body": {
            "model": "Kwai-Kolors/Kolors",
            "prompt": "a mountain lake",
            "batch_size": 2,
            "image_size": "1024x1024",
            "seed": 42,
            "negative_prompt": "fog",
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
        },
    }
    assert [image.url for image in response.data] == [
        "https://example.com/first.png",
        "https://example.com/second.png",
    ]
    assert response._hidden_params["generated_images"] == 2
    assert response._hidden_params["seed"] == 42
    assert response._hidden_params["timings"] == {"inference": 1.5}

    model_info = {
        "litellm_provider": "siliconflow",
        "mode": "image_generation",
        "output_cost_per_image": 0.3,
    }
    with patch.dict(litellm.model_cost, {"siliconflow/Qwen/Qwen-Image": model_info}):
        cost = completion_cost(
            completion_response=ImageResponse(
                data=[ImageObject(url="https://example.com/result.png")],
                hidden_params={"generated_images": 2},
            ),
            model="siliconflow/Qwen/Qwen-Image",
            call_type="image_generation",
        )
    assert cost == pytest.approx(0.6)


def test_siliconflow_image_generation_registration_and_ranges() -> None:
    config = ProviderConfigManager.get_provider_image_generation_config(
        model="Qwen/Qwen-Image",
        provider=LlmProviders.SILICONFLOW,
    )

    assert isinstance(config, SiliconFlowImageGenerationConfig)
    with pytest.raises(ValueError, match="cfg"):
        config.map_openai_params(
            non_default_params={"cfg": 20.1},
            optional_params={},
            model="Qwen/Qwen-Image",
            drop_params=False,
        )


def test_siliconflow_image_pricing_metadata() -> None:
    pricing_path = Path(__file__).parents[5] / "model_prices_and_context_window.json"
    pricing = json.loads(pricing_path.read_text())

    assert pricing["siliconflow/Qwen/Qwen-Image"]["output_cost_per_image"] == pytest.approx(0.3)
    assert pricing["siliconflow/Tongyi-MAI/Z-Image-Turbo"]["output_cost_per_image"] == pytest.approx(0.1)
