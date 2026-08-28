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


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (
            "https://dashscope.aliyuncs.com/api/v1",
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        ),
        (
            "https://dashscope.aliyuncs.com/api/v1/",
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        ),
        (
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        ),
    ],
)
def test_dashscope_image_generation_completes_api_base(api_base: str, expected: str) -> None:
    result = DashScopeImageGenerationConfig().get_complete_url(
        api_base=api_base,
        api_key=None,
        model="qwen-image-3.0-pro",
        optional_params={},
        litellm_params={},
    )

    assert result == expected


def test_dashscope_image_editing_maps_reference_images_and_parameters() -> None:
    config = DashScopeImageGenerationConfig()
    optional_params = config.map_openai_params(
        non_default_params={
            "image_url": ["https://example.com/input-1.png", "https://example.com/input-2.png"],
            "n": 2,
            "size": "1024x1536",
            "watermark": False,
            "negative_prompt": "blurred",
            "prompt_upsampling": True,
        },
        optional_params={},
        model="qwen-image-3.0-pro",
        drop_params=False,
    )

    request = config.transform_image_generation_request(
        model="qwen-image-3.0-pro",
        prompt="Edit the reference images",
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert request == {
        "model": "qwen-image-3.0-pro",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": "https://example.com/input-1.png"},
                        {"image": "https://example.com/input-2.png"},
                        {"text": "Edit the reference images"},
                    ],
                }
            ]
        },
        "parameters": {
            "n": 2,
            "size": "1024*1536",
            "watermark": False,
            "negative_prompt": "blurred",
            "prompt_extend": True,
        },
    }


def test_dashscope_image_editing_rejects_more_than_three_reference_images() -> None:
    with pytest.raises(ValueError, match="at most 3 input images"):
        DashScopeImageGenerationConfig().transform_image_generation_request(
            model="qwen-image-3.0-pro",
            prompt="Edit the reference images",
            optional_params={
                "image": [
                    "https://example.com/input-1.png",
                    "https://example.com/input-2.png",
                    "https://example.com/input-3.png",
                    "https://example.com/input-4.png",
                ]
            },
            litellm_params={},
            headers={},
        )


def test_dashscope_text_to_image_does_not_add_an_input_image() -> None:
    request = DashScopeImageGenerationConfig().transform_image_generation_request(
        model="qwen-image-3.0-pro",
        prompt="Create a paper-cut landscape",
        optional_params={"size": "1024*1024"},
        litellm_params={},
        headers={},
    )

    assert request["input"]["messages"][0]["content"] == [{"text": "Create a paper-cut landscape"}]


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
