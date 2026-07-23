import json

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.siliconflow.image_edit.transformation import SiliconFlowImageEditConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_siliconflow_image_edit_maps_three_images_to_generation_endpoint() -> None:
    captured: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"images": [{"url": "https://example.com/edited.png"}], "seed": 7})

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    response = litellm.image_edit(
        model="siliconflow/Qwen/Qwen-Image-Edit-2509",
        prompt="combine the references",
        image=[b"\x89PNG\r\n\x1a\nfirst", b"\xff\xd8\xffsecond", b"\x89PNG\r\n\x1a\nthird"],
        size="1024x1024",
        response_format="url",
        api_key="test-key",
        api_base="https://api.example.com/v1",
        client=client,
    )

    body = captured["body"]
    assert captured["url"] == "https://api.example.com/v1/images/generations"
    assert isinstance(body, dict)
    assert body["model"] == "Qwen/Qwen-Image-Edit-2509"
    assert body["image"].startswith("data:image/png;base64,")
    assert body["image2"].startswith("data:image/jpeg;base64,")
    assert body["image3"].startswith("data:image/png;base64,")
    assert body["image_size"] == "1024x1024"
    assert response.data[0].url == "https://example.com/edited.png"


def test_siliconflow_image_edit_registration_and_model_limit() -> None:
    config = ProviderConfigManager.get_provider_image_edit_config(
        model="Qwen/Qwen-Image-Edit",
        provider=LlmProviders.SILICONFLOW,
    )

    assert isinstance(config, SiliconFlowImageEditConfig)
    with pytest.raises(ValueError, match="multiple input images"):
        config.transform_image_edit_request(
            model="Qwen/Qwen-Image-Edit",
            prompt="edit",
            image=[b"first", b"second"],
            image_edit_optional_request_params={},
            litellm_params=litellm.GenericLiteLLMParams(),
            headers={},
        )
