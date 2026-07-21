import base64
from io import BytesIO
from unittest.mock import MagicMock

import httpx

from litellm.llms.volcengine.image_edit.transformation import VolcEngineImageEditConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_provider_config_manager_returns_volcengine_image_edit_config():
    config = ProviderConfigManager.get_provider_image_edit_config(
        model="doubao-seedream-5-0-pro-260628",
        provider=LlmProviders.VOLCENGINE,
    )

    assert isinstance(config, VolcEngineImageEditConfig)


def test_image_edit_converts_multipart_files_to_ark_json_data_uris():
    config = VolcEngineImageEditConfig()
    first_image = BytesIO(b"\x89PNG\r\n\x1a\nfirst")
    second_image = BytesIO(b"\x89PNG\r\n\x1a\nsecond")

    request_body, files = config.transform_image_edit_request(
        model="doubao-seedream-5-0-pro-260628",
        prompt="Combine both references",
        image=[first_image, second_image],
        image_edit_optional_request_params={"size": "2K", "response_format": "url"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert files == []
    assert request_body["model"] == "doubao-seedream-5-0-pro-260628"
    assert request_body["prompt"] == "Combine both references"
    assert request_body["size"] == "2K"
    encoded_images = request_body["image"]
    assert isinstance(encoded_images, list)
    assert base64.b64decode(encoded_images[0].split(",", 1)[1]) == b"\x89PNG\r\n\x1a\nfirst"
    assert base64.b64decode(encoded_images[1].split(",", 1)[1]) == b"\x89PNG\r\n\x1a\nsecond"


def test_image_edit_uses_generation_endpoint_and_transforms_usage():
    config = VolcEngineImageEditConfig()
    raw_response = httpx.Response(
        200,
        json={
            "model": "doubao-seedream-5-0-pro-260628",
            "created": 1750000002,
            "data": [{"url": "https://example.com/edited.png", "output_format": "png"}],
            "usage": {
                "generated_images": 1,
                "input_images": 2,
                "output_tokens": 100,
                "total_tokens": 100,
            },
        },
    )

    response = config.transform_image_edit_response(
        model="doubao-seedream-5-0-pro-260628",
        raw_response=raw_response,
        logging_obj=MagicMock(),
    )

    assert (
        config.get_complete_url(
            model="doubao-seedream-5-0-pro-260628",
            api_base="https://ark.example.com/api/v3",
            litellm_params={},
        )
        == "https://ark.example.com/api/v3/images/generations"
    )
    assert config.use_multipart_form_data() is False
    assert response.created == 1750000002
    assert response.data[0].url == "https://example.com/edited.png"
    assert response._hidden_params["generated_images"] == 1
    assert response._hidden_params["input_images"] == 2
