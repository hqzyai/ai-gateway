import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.volcengine.common_utils import VolcEngineError
from litellm.llms.volcengine.image_generation.transformation import (
    VolcEngineImageGenerationConfig,
)
from litellm.types.utils import ImageObject, ImageResponse, ImageUsage, LlmProviders
from litellm.utils import ProviderConfigManager


def test_provider_config_manager_returns_volcengine_image_config():
    config = ProviderConfigManager.get_provider_image_generation_config(
        model="doubao-seedream-4-5-251128",
        provider=LlmProviders.VOLCENGINE,
    )
    assert isinstance(config, VolcEngineImageGenerationConfig)


def test_image_generation_uses_ark_endpoint_and_transforms_response():
    captured: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"x-request-id": "image-request-123"},
            json={
                "model": "doubao-seedream-4-5-251128",
                "created": 1750000000,
                "data": [
                    {
                        "url": "https://example.com/image.png",
                        "b64_json": None,
                        "size": "2K",
                        "output_format": "jpeg",
                    },
                    {"url": None, "b64_json": "aW1hZ2U=", "size": "1024x1024"},
                ],
                "usage": {
                    "generated_images": 2,
                    "input_images": 0,
                    "output_tokens": 200,
                    "total_tokens": 200,
                },
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    response = litellm.image_generation(
        model="volcengine/doubao-seedream-4-5-251128",
        prompt="A lighthouse above a stormy sea",
        size="2K",
        response_format="url",
        seed=42,
        api_key="test-key",
        api_base="https://ark.example.com/api/v3",
        client=client,
    )

    assert captured == {
        "url": "https://ark.example.com/api/v3/images/generations",
        "authorization": "Bearer test-key",
        "body": {
            "model": "doubao-seedream-4-5-251128",
            "prompt": "A lighthouse above a stormy sea",
            "response_format": "url",
            "seed": 42,
            "size": "2K",
        },
    }
    assert response.created == 1750000000
    assert response.data[0].url == "https://example.com/image.png"
    assert response.data[0].provider_specific_fields == {"size": "2K", "output_format": "jpeg"}
    assert response.data[1].b64_json == "aW1hZ2U="
    assert response.usage is not None
    assert response.usage.output_tokens == 200
    assert response._hidden_params["model"] == "doubao-seedream-4-5-251128"
    assert response._hidden_params["generated_images"] == 2
    assert response._hidden_params["input_images"] == 0
    assert response._hidden_params["additional_headers"]["llm_provider-x-request-id"] == "image-request-123"


def test_image_generation_accepts_single_and_multiple_reference_images():
    captured_bodies: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_bodies.append(body)
        image = body["image"]
        input_images = len(image) if isinstance(image, list) else 1
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "created": 1750000001,
                "data": [{"url": "https://example.com/result.png", "output_format": "png"}],
                "usage": {
                    "generated_images": 1,
                    "input_images": input_images,
                    "output_tokens": 100,
                    "total_tokens": 100,
                },
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    single_response = litellm.image_generation(
        model="volcengine/doubao-seedream-5-0-pro-260628",
        prompt="Restyle the reference image",
        image="https://example.com/reference.png",
        output_format="png",
        watermark=False,
        api_key="test-key",
        client=client,
    )
    multiple_response = litellm.image_generation(
        model="volcengine/doubao-seedream-5-0-260128",
        prompt="Combine the reference images",
        image=["https://example.com/first.png", "data:image/png;base64,aW1hZ2U="],
        sequential_image_generation="disabled",
        api_key="test-key",
        client=client,
    )

    assert captured_bodies[0]["image"] == "https://example.com/reference.png"
    assert captured_bodies[0]["output_format"] == "png"
    assert captured_bodies[0]["watermark"] is False
    assert captured_bodies[1]["image"] == [
        "https://example.com/first.png",
        "data:image/png;base64,aW1hZ2U=",
    ]
    assert single_response._hidden_params["input_images"] == 1
    assert multiple_response._hidden_params["input_images"] == 2


def test_image_generation_maps_api_errors():
    config = VolcEngineImageGenerationConfig()
    response = httpx.Response(
        400,
        json={"error": {"code": "InvalidParameter", "message": "invalid size"}},
    )

    with pytest.raises(VolcEngineError, match="invalid size"):
        config.transform_image_generation_response(
            model="doubao-seedream-4-5-251128",
            raw_response=response,
            model_response=litellm.ImageResponse(),
            logging_obj=None,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )


def test_image_generation_cost_uses_generated_image_count():
    model_id = "volcengine-image-deployment"
    model_info = {
        "litellm_provider": "volcengine",
        "mode": "image_generation",
        "output_cost_per_image": 0.25,
    }
    response = ImageResponse(
        data=[ImageObject(url="https://example.com/image.png")],
        hidden_params={"generated_images": 2, "model": "doubao-seedream-4-5-251128"},
    )

    with patch.dict(litellm.model_cost, {model_id: model_info}):
        cost = completion_cost(
            completion_response=response,
            model="doubao-seedream-4-5-251128",
            call_type="image_generation",
            custom_llm_provider="volcengine",
            custom_pricing=True,
            router_model_id=model_id,
        )

    assert cost == 0.5


def test_image_generation_cost_falls_back_to_response_image_count():
    model = "doubao-seedream-test"
    model_info = {
        "litellm_provider": "volcengine",
        "mode": "image_generation",
        "output_cost_per_image": 0.1,
    }
    response = ImageResponse(
        data=[
            ImageObject(url="https://example.com/first.png"),
            ImageObject(url="https://example.com/second.png"),
        ],
        hidden_params={"model": model},
    )

    with patch.dict(litellm.model_cost, {f"volcengine/{model}": model_info}):
        cost = completion_cost(
            completion_response=response,
            model=f"volcengine/{model}",
            call_type="image_generation",
        )

    assert cost == 0.2


@pytest.mark.parametrize(
    ("model", "input_cost_per_image", "output_cost_per_image"),
    [
        ("doubao-seedream-5-0-260128", 0.0, 0.22),
        ("doubao-seedream-5-0-pro-260628", 0.02, 0.3),
    ],
)
def test_seedream_model_pricing(
    model: str,
    input_cost_per_image: float,
    output_cost_per_image: float,
):
    pricing_path = Path(__file__).parents[5] / "model_prices_and_context_window.json"
    pricing = json.loads(pricing_path.read_text())
    model_key = f"volcengine/{model}"
    configured_model_info = pricing[model_key]
    response = ImageResponse(
        data=[ImageObject(url="https://example.com/image.png")],
        hidden_params={"generated_images": 2, "input_images": 3, "model": model},
    )

    with patch.dict(litellm.model_cost, {model_key: configured_model_info}):
        model_info = litellm.get_model_info(model=model, custom_llm_provider="volcengine")
        cost = completion_cost(
            completion_response=response,
            model=model_key,
            call_type="image_generation",
        )

    assert model_info["input_cost_per_image"] == pytest.approx(input_cost_per_image)
    assert model_info["output_cost_per_image"] == pytest.approx(output_cost_per_image)
    assert cost == pytest.approx(input_cost_per_image * 3 + output_cost_per_image * 2)


@pytest.mark.parametrize(
    ("output_tokens", "expected_cost"),
    [(16384, 0.32), (16385, 0.62), (18000, 0.62)],
)
def test_seedream_pro_large_image_pricing(output_tokens: int, expected_cost: float) -> None:
    model = "doubao-seedream-5-0-pro-260628"
    response = ImageResponse(
        data=[ImageObject(url="https://example.com/image.png")],
        usage=ImageUsage(
            input_tokens=0,
            input_tokens_details={"image_tokens": 0, "text_tokens": 0},
            output_tokens=output_tokens,
            total_tokens=output_tokens,
        ),
        hidden_params={"generated_images": 1, "input_images": 1, "model": model},
    )

    cost = completion_cost(
        completion_response=response,
        model=f"volcengine/{model}",
        call_type="image_edit",
    )

    assert cost == pytest.approx(expected_cost)


def _image_response(output_tokens: int, generated_images: int, input_images: int, model: str) -> ImageResponse:
    return ImageResponse(
        data=[ImageObject(url="https://example.com/image.png")],
        usage=ImageUsage(
            input_tokens=0,
            input_tokens_details={"image_tokens": 0, "text_tokens": 0},
            output_tokens=output_tokens,
            total_tokens=output_tokens,
        ),
        hidden_params={"generated_images": generated_images, "input_images": input_images, "model": model},
    )


@pytest.mark.parametrize(
    ("output_tokens", "expected_cost"),
    [(4096, 0.32), (4097, 0.47), (8192, 0.47), (8193, 0.62)],
)
def test_image_cost_honors_output_token_thresholds_other_than_16384(output_tokens: int, expected_cost: float) -> None:
    """Large-image tiers come from the price map, not a hard-coded 16384 boundary.

    Before this was generalized the calculator only ever read
    output_cost_per_image_above_16384_tokens, so a model with different tier
    boundaries silently billed every output image at the base rate.
    """
    model = "doubao-seedream-custom-thresholds"
    model_info = {
        "litellm_provider": "volcengine",
        "mode": "image_generation",
        "input_cost_per_image": 0.02,
        "output_cost_per_image": 0.3,
        "output_cost_per_image_above_4096_tokens": 0.45,
        "output_cost_per_image_above_8192_tokens": 0.6,
    }

    with patch.dict(litellm.model_cost, {f"volcengine/{model}": model_info}):
        cost = completion_cost(
            completion_response=_image_response(output_tokens, 1, 1, model),
            model=f"volcengine/{model}",
            call_type="image_generation",
        )

    assert cost == pytest.approx(expected_cost)


@pytest.mark.parametrize(("output_tokens", "expected_cost"), [(16384, 0.45), (16385, 0.95)])
def test_deployment_can_override_large_image_pricing_via_litellm_params(
    output_tokens: int, expected_cost: float
) -> None:
    """output_cost_per_image_above_16384_tokens must survive the litellm_params -> model_info copy.

    Router only forwards fields declared on CustomPricingLiteLLMParams, so an
    undeclared field would leave the deployment billing at the built-in rate.
    """
    from litellm import Router
    from litellm.utils import _invalidate_model_cost_lowercase_map

    model_id = "volcengine-seedream-pro-deployment"
    backend_model = "doubao-seedream-5-0-pro-260628"

    with patch.dict(litellm.model_cost):
        Router(
            model_list=[
                {
                    "model_name": "seedream-pro",
                    "litellm_params": {
                        "model": f"volcengine/{backend_model}",
                        "api_key": "fake-key",
                        "input_cost_per_image": 0.05,
                        "output_cost_per_image": 0.4,
                        "output_cost_per_image_above_16384_tokens": 0.9,
                    },
                    "model_info": {"id": model_id, "mode": "image_generation"},
                }
            ]
        )

        assert litellm.model_cost[model_id]["output_cost_per_image_above_16384_tokens"] == 0.9

        cost = completion_cost(
            completion_response=_image_response(output_tokens, 1, 1, backend_model),
            model=backend_model,
            call_type="image_generation",
            custom_llm_provider="volcengine",
            custom_pricing=True,
            router_model_id=model_id,
        )

    _invalidate_model_cost_lowercase_map()

    assert cost == pytest.approx(expected_cost)
