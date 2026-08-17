import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.siliconflow.common_utils import SiliconFlowError
from litellm.llms.siliconflow.rerank.transformation import SiliconFlowRerankConfig
from litellm.types.rerank import RerankResponse
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_siliconflow_rerank_end_to_end() -> None:
    captured: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "rerank-test-id",
                "results": [
                    {
                        "index": 0,
                        "relevance_score": 0.98,
                        "document": {"text": "apple"},
                    }
                ],
                "meta": {
                    "tokens": {"input_tokens": 9, "output_tokens": 0, "image_tokens": 0},
                    "billed_units": {
                        "input_tokens": 9,
                        "output_tokens": 0,
                        "image_tokens": 0,
                        "search_units": 1,
                        "classifications": 0,
                    },
                },
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    response = litellm.rerank(
        model="siliconflow/BAAI/bge-reranker-v2-m3",
        query="Apple",
        documents=["apple", "banana"],
        top_n=1,
        return_documents=True,
        max_chunks_per_doc=512,
        overlap_tokens=20,
        api_key="test-key",
        api_base="https://api.example.com/v1",
        client=client,
    )

    assert isinstance(response, RerankResponse)
    assert captured == {
        "url": "https://api.example.com/v1/rerank",
        "authorization": "Bearer test-key",
        "body": {
            "model": "BAAI/bge-reranker-v2-m3",
            "query": "Apple",
            "documents": ["apple", "banana"],
            "top_n": 1,
            "return_documents": True,
            "max_chunks_per_doc": 512,
            "overlap_tokens": 20,
        },
    }
    assert response.id == "rerank-test-id"
    assert response.results == [{"index": 0, "relevance_score": 0.98, "document": {"text": "apple"}}]
    assert response.meta == {
        "tokens": {"input_tokens": 9, "output_tokens": 0, "image_tokens": 0},
        "billed_units": {
            "input_tokens": 9,
            "output_tokens": 0,
            "image_tokens": 0,
            "search_units": 1,
            "classifications": 0,
        },
    }


def test_siliconflow_rerank_supports_instruction_and_multimodal_content() -> None:
    config = SiliconFlowRerankConfig()
    mapped = config.map_cohere_rerank_params(
        non_default_params={},
        model="Qwen/Qwen3-Reranker-8B",
        drop_params=False,
        query="find relevant technical documents",
        documents=["document one", "document two"],
        instruction="prefer recent content",
    )
    assert mapped["instruction"] == "prefer recent content"

    request = config.transform_rerank_request(
        model="siliconflow/Qwen/Qwen3-VL-Reranker-8B",
        optional_rerank_params={
            "query": {"image": "https://example.com/query.png"},
            "documents": [{"image": "https://example.com/document.png"}, "text document"],
            "return_documents": True,
        },
        headers={},
    )
    assert request == {
        "model": "Qwen/Qwen3-VL-Reranker-8B",
        "query": {"image": "https://example.com/query.png"},
        "documents": [{"image": "https://example.com/document.png"}, "text document"],
        "return_documents": True,
    }

    raw_response = httpx.Response(
        200,
        json={
            "id": "vl-rerank-id",
            "results": [
                {
                    "index": 0,
                    "relevance_score": 0.91,
                    "document": {"image": "https://example.com/document.png"},
                }
            ],
            "meta": {
                "tokens": {"input_tokens": 10, "output_tokens": 1, "image_tokens": 8},
                "billed_units": {"input_tokens": 10, "image_tokens": 8},
            },
        },
    )
    response = config.transform_rerank_response(
        model="Qwen/Qwen3-VL-Reranker-8B",
        raw_response=raw_response,
        model_response=RerankResponse(),
        logging_obj=MagicMock(),
    )
    assert response.results == [
        {
            "index": 0,
            "relevance_score": 0.91,
            "document": {"image": "https://example.com/document.png"},
        }
    ]
    assert response.meta == {
        "tokens": {"input_tokens": 10, "output_tokens": 1, "image_tokens": 8},
        "billed_units": {"input_tokens": 10, "image_tokens": 8},
    }


def test_siliconflow_rerank_registration_and_environment() -> None:
    config = ProviderConfigManager.get_provider_rerank_config(
        model="BAAI/bge-reranker-v2-m3",
        provider=LlmProviders.SILICONFLOW,
        api_base=None,
        present_version_params=[],
    )
    assert isinstance(config, SiliconFlowRerankConfig)
    assert config.get_complete_url(None, "BAAI/bge-reranker-v2-m3") == "https://api.siliconflow.cn/v1/rerank"

    with patch("litellm.llms.siliconflow.rerank.transformation.get_secret_str", return_value=None):
        with pytest.raises(ValueError, match="SiliconFlow API key is required"):
            config.validate_environment(headers={}, model="BAAI/bge-reranker-v2-m3")


def test_siliconflow_rerank_maps_error_response() -> None:
    config = SiliconFlowRerankConfig()
    with pytest.raises(SiliconFlowError) as exc_info:
        config.transform_rerank_response(
            model="BAAI/bge-reranker-v2-m3",
            raw_response=httpx.Response(429, json={"message": "rate limited", "data": ""}),
            model_response=RerankResponse(),
            logging_obj=MagicMock(),
        )
    assert exc_info.value.status_code == 429
    assert "rate limited" in str(exc_info.value)
