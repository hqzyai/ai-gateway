import json
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.siliconflow.common_utils import get_siliconflow_api_base
from litellm.llms.siliconflow.embedding.transformation import SiliconFlowEmbeddingConfig
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_siliconflow_default_api_base_is_china_endpoint() -> None:
    with patch("litellm.llms.siliconflow.common_utils.get_secret_str", return_value=None):
        assert get_siliconflow_api_base() == "https://api.siliconflow.cn/v1"


def test_siliconflow_embedding_request_and_response() -> None:
    captured: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
                "model": "BAAI/bge-m3",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    response = litellm.embedding(
        model="siliconflow/BAAI/bge-m3",
        input="hello",
        dimensions=256,
        truncate="right",
        api_key="test-key",
        api_base="https://api.example.com/v1",
        client=client,
    )

    assert captured == {
        "url": "https://api.example.com/v1/embeddings",
        "authorization": "Bearer test-key",
        "body": {
            "model": "BAAI/bge-m3",
            "input": "hello",
            "dimensions": 256,
            "truncate": "right",
        },
    }
    assert response.data[0]["embedding"] == [0.1, 0.2]
    assert response.usage.prompt_tokens == 2


def test_siliconflow_embedding_config_registration_and_validation() -> None:
    config = ProviderConfigManager.get_provider_embedding_config(
        model="BAAI/bge-m3",
        provider=LlmProviders.SILICONFLOW,
    )

    assert isinstance(config, SiliconFlowEmbeddingConfig)
    assert config.map_openai_params(
        non_default_params={"encoding_format": "base64", "truncate": "left"},
        optional_params={},
        model="BAAI/bge-m3",
        drop_params=False,
    ) == {"encoding_format": "base64", "truncate": "left"}


def test_siliconflow_vl_embedding_preserves_mixed_input() -> None:
    captured: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [{"object": "embedding", "embedding": [0.3, 0.4], "index": 0}],
                "model": "Qwen/Qwen3-VL-Embedding-8B",
                "usage": {"prompt_tokens": 51, "completion_tokens": 1260, "total_tokens": 1311},
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    response = litellm.embedding(
        model="siliconflow/Qwen/Qwen3-VL-Embedding-8B",
        input=[{"text": "describe the image"}, {"image": "https://example.com/source.png"}],
        api_key="test-key",
        client=client,
    )

    assert captured["body"] == {
        "model": "Qwen/Qwen3-VL-Embedding-8B",
        "input": [{"text": "describe the image"}, {"image": "https://example.com/source.png"}],
    }
    assert response.data[0]["embedding"] == [0.3, 0.4]
    assert response.usage.prompt_tokens == 1311
    assert response.usage.completion_tokens == 0
    assert response.usage.total_tokens == 1311
    assert response.usage.prompt_tokens_details.text_tokens == 51
    assert response.usage.prompt_tokens_details.image_tokens == 1260

    input_cost, output_cost = generic_cost_per_token(
        model="Qwen/Qwen3-VL-Embedding-8B",
        usage=response.usage,
        custom_llm_provider="siliconflow",
    )

    assert input_cost == pytest.approx((51 * 0.7e-6) + (1260 * 1.8e-6))
    assert output_cost == 0
