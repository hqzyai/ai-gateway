from unittest.mock import MagicMock

import httpx
import litellm
import pytest

from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
from litellm.llms.dashscope.image_generation.transformation import DashScopeImageGenerationConfig
from litellm.types.utils import ImageObject, ImageResponse


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    original = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original
        litellm.get_model_info.cache_clear()


def test_transform_preserves_dashscope_image_resolution_usage(local_model_cost_map: None) -> None:
    raw_response = httpx.Response(
        200,
        json={
            "model": "qwen-image-3.0-pro",
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"image": "https://example.com/one.png"},
                                {"image": "https://example.com/two.png"},
                                {"image": "https://example.com/three.png"},
                            ]
                        }
                    }
                ]
            },
            "usage": {
                "output_height": 2048,
                "output_width": 2048,
                "input_image_count": 2,
                "input_image_type": "qima_input_2k",
                "output_image_count": 3,
                "output_image_type": "qima_output_2k",
            },
        },
        request=httpx.Request("POST", "https://dashscope.example/generation"),
    )
    result = DashScopeImageGenerationConfig().transform_image_generation_response(
        model="qwen-image-3.0-pro",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert result.size == "2048x2048"
    assert result.usage is not None
    usage = result.usage.model_dump()
    assert usage["input_image_count"] == 2
    assert usage["input_image_type"] == "qima_input_2k"
    assert usage["output_image_count"] == 3
    assert usage["output_image_type"] == "qima_output_2k"
    assert result.get_hidden_param("output_width") == 2048
    assert litellm.completion_cost(
        completion_response=result,
        model="dashscope/qwen-image-3.0-pro",
        call_type="image_generation",
    ) == pytest.approx(2 * 0.02 + 3 * 0.5)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1500, 1500, 0.25),
        (1501, 1500, 0.5),
    ],
)
def test_resolution_pricing_uses_pixel_boundary_when_response_type_is_missing(
    local_model_cost_map: None,
    width: int,
    height: int,
    expected: float,
) -> None:
    response = ImageResponse(
        data=[ImageObject(url="https://example.com/image.png")],
        hidden_params={
            "output_image_count": 1,
            "output_width": width,
            "output_height": height,
        },
    )

    assert litellm.completion_cost(
        completion_response=response,
        model="dashscope/qwen-image-3.0-pro",
        call_type="image_generation",
    ) == pytest.approx(expected)


def test_resolution_pricing_can_be_configured_for_another_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "openai/resolution-priced-image": {
                "litellm_provider": "openai",
                "mode": "image_generation",
                "image_resolution_pricing": {"output_4k": 1.25},
            }
        },
    )
    litellm.get_model_info.cache_clear()
    response = ImageResponse(
        data=[ImageObject(url="https://example.com/image.png")],
        hidden_params={"output_image_count": 1, "output_image_type": "output_4k"},
    )

    try:
        cost = CostCalculatorUtils.route_image_generation_cost_calculator(
            model="resolution-priced-image",
            completion_response=response,
            custom_llm_provider="openai",
        )
    finally:
        litellm.get_model_info.cache_clear()

    assert cost == pytest.approx(1.25)
