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


import unittest
from unittest.mock import patch
from litellm.litellm_core_utils.litellm_logging import Logging, use_custom_pricing_for_model

@pytest.mark.usefixtures("local_model_cost_map")
class DashScopeImageCount(unittest.TestCase):
    def response(self, usage, count=1, logger=None):
        raw = httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"image": "https://example.invalid/image.png"}
                                    for _ in range(count)
                                ]
                            }
                        }
                    ]
                },
                "usage": usage,
            },
        )
        return DashScopeImageGenerationConfig().transform_image_generation_response(
            "qwen-image-2.0", raw, ImageResponse(), logger, {}, {}, {}, None
        )

    def test_legacy_usage_counts_returned_images(self):
        result = self.response({"image_count": 1})
        self.assertEqual(result.usage.output_image_count, 1)
        self.assertEqual(result._hidden_params["output_image_count"], 1)

    def test_missing_count_multiple_images(self):
        self.assertEqual(self.response({}, 2).usage.output_image_count, 2)

    def test_explicit_count_preserved(self):
        self.assertEqual(
            self.response({"output_image_count": 3}).usage.output_image_count, 3
        )

    def test_deployment_flat_price_does_not_use_shared_zero_price(self):
        from types import SimpleNamespace

        logger = SimpleNamespace(
            litellm_params={"model_info": {"output_cost_per_image": 0.2}}
        )
        result = self.response({"image_count": 1}, logger=logger)
        self.assertEqual(result._hidden_params["response_cost"], 0.2)

    def test_router_registration_preserves_catalog_image_price(self):
        canonical = "dashscope/qwen-image-2.0"
        litellm.model_cost[canonical] = {
            "mode": "image_generation",
            "litellm_provider": "dashscope",
            "output_cost_per_image": 0.2,
        }
        litellm.Router(
            model_list=[
                {
                    "model_name": "catalog-image-test",
                    "litellm_params": {"model": canonical, "api_key": "dummy"},
                    "model_info": {
                        "id": "catalog-image-deployment",
                        "mode": "image_generation",
                        "output_cost_per_image": 0.9,
                    },
                }
            ]
        )
        self.assertEqual(litellm.model_cost[canonical]["output_cost_per_image"], 0.2)

    def test_price_map_replacement_restores_served_deployments(self):
        import litellm.proxy.proxy_server as proxy
        from litellm.proxy.management_endpoints.model_cost_map_endpoints import (
            _apply_cost_map,
        )

        canonical = "dashscope/qwen-image-2.0"
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "price-reload-test",
                    "litellm_params": {"model": canonical, "api_key": "dummy"},
                    "model_info": {
                        "id": "reload-deployment-test",
                        "mode": "image_generation",
                        "output_cost_per_image": 0.2,
                    },
                }
            ]
        )
        with patch.object(proxy, "llm_router", router):
            _apply_cost_map(
                {
                    canonical: {
                        "mode": "image_generation",
                        "litellm_provider": "dashscope",
                        "output_cost_per_image": 0.2,
                    }
                }
            )
        self.assertEqual(
            litellm.model_cost["reload-deployment-test"]["output_cost_per_image"], 0.2
        )
        self.assertEqual(litellm.model_cost[canonical]["output_cost_per_image"], 0.2)

    def test_catalog_reload_replay_retains_new_image_price(self):
        from litellm.proxy.management_endpoints.model_cost_map_endpoints import _apply_cost_map as adopt_model_cost_map
        canonical="dashscope/qwen-image-2.0"
        litellm.model_cost[canonical]={"mode":"image_generation","litellm_provider":"dashscope","output_cost_per_image":0.2}
        litellm.Router(model_list=[{"model_name":"catalog-replay-test","litellm_params":{"model":canonical,"api_key":"dummy"},"model_info":{"id":"catalog-replay-deployment","mode":"image_generation","output_cost_per_image":0.9}}])
        adopt_model_cost_map({canonical:{"mode":"image_generation","litellm_provider":"dashscope","output_cost_per_image":0.3}})
        self.assertEqual(litellm.model_cost[canonical]["output_cost_per_image"],0.3)

    def test_startup_restores_persisted_catalog_before_serving(self):
        import asyncio
        from unittest.mock import AsyncMock
        import litellm.proxy.proxy_server as proxy
        from litellm.proxy.management_endpoints import model_cost_map_endpoints as endpoint
        canonical="dashscope/qwen-image-2.0"
        router=litellm.Router(model_list=[{"model_name":"startup-test","litellm_params":{"model":canonical,"api_key":"dummy"},"model_info":{"id":"startup-image-deployment","mode":"image_generation","output_cost_per_image":0.2}}])
        with patch.object(proxy,"llm_router",router),patch.object(endpoint,"load_model_cost_map_overrides",AsyncMock(return_value={canonical:{"mode":"image_generation","litellm_provider":"dashscope","output_cost_per_image":0.2}})):
            asyncio.run(endpoint.restore_persisted_model_cost_map(object()))
        self.assertEqual(litellm.model_cost[canonical]["output_cost_per_image"],0.2)
        self.assertEqual(litellm.model_cost["startup-image-deployment"]["output_cost_per_image"],0.2)

    def test_per_image_cost_uses_actual_count(self):
        litellm.register_model(
            {
                "dashscope/qwen-image-2.0": {
                    "mode": "image_generation",
                    "litellm_provider": "dashscope",
                    "output_cost_per_image": 0.2,
                }
            }
        )
        result = self.response({"image_count": 1})
        self.assertAlmostEqual(
            litellm.completion_cost(
                completion_response=result,
                model="dashscope/qwen-image-2.0",
                call_type="image_generation",
                custom_llm_provider="dashscope",
            ),
            0.2,
        )

    def test_no_image_means_no_output(self):
        self.assertEqual(self.response({}, 0).usage.output_image_count, 0)


@pytest.mark.usefixtures("local_model_cost_map")
class ImageLogging(unittest.TestCase):
    def test_image_logging_retains_deployment_prices(self):
        calls = []
        original = Logging.update_from_kwargs

        def capture(self, *a, **k):
            calls.append(dict(k.get("litellm_params", {})))
            return original(self, *a, **k)

        with patch.object(Logging, "update_from_kwargs", capture):
            litellm.image_generation(
                model="dashscope/qwen-image-2.0",
                prompt="test",
                mock_response="https://example.invalid/image.png",
                metadata={
                    "model_info": {
                        "id": "deployment-test",
                        "output_cost_per_image": 0.2,
                    }
                },
            )
        self.assertTrue(any(use_custom_pricing_for_model(x) for x in calls), calls)
