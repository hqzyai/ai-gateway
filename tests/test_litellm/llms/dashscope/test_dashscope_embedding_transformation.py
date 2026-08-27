"""
Unit tests for DashScope embedding transformation.
"""

import json
import os
import sys
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.dashscope.common_utils import DashScopeError
from litellm.llms.dashscope.embed.transformation import (
    DEFAULT_API_BASE,
    MULTIMODAL_EMBEDDING_PATH,
    DashScopeEmbeddingConfig,
)
from litellm.types.utils import EmbeddingResponse


def test_validate_environment_and_url():
    config = DashScopeEmbeddingConfig()
    headers = config.validate_environment(
        headers={},
        model="text-embedding-v4",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-test",
    )
    assert headers["Authorization"] == "Bearer sk-test"

    url = config.get_complete_url(
        api_base=None,
        api_key="sk-test",
        model="text-embedding-v4",
        optional_params={},
        litellm_params={},
    )
    assert url == f"{DEFAULT_API_BASE}/embeddings"


def test_transform_embedding_request():
    config = DashScopeEmbeddingConfig()
    data = config.transform_embedding_request(
        model="text-embedding-v4",
        input=["风急天高猿啸哀"],
        optional_params={"dimensions": 1024, "encoding_format": "float"},
        headers={},
    )
    assert data == {
        "model": "text-embedding-v4",
        "input": ["风急天高猿啸哀"],
        "dimensions": 1024,
        "encoding_format": "float",
    }


def test_transform_embedding_response_success():
    config = DashScopeEmbeddingConfig()
    payload = {
        "data": [
            {"embedding": [0.1, 0.2], "index": 0, "object": "embedding"},
        ],
        "model": "text-embedding-v4",
        "object": "list",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
        "id": "73591b79-xxxx",
    }
    raw = httpx.Response(
        status_code=200,
        content=json.dumps(payload).encode("utf-8"),
        request=httpx.Request("POST", "https://example.com"),
    )
    result = config.transform_embedding_response(
        model="text-embedding-v4",
        raw_response=raw,
        model_response=EmbeddingResponse(),
        logging_obj=MagicMock(),
        api_key="sk-x",
        request_data={"input": ["a"]},
        optional_params={},
        litellm_params={},
    )
    assert result.model == "text-embedding-v4"
    assert len(result.data) == 1
    assert result.usage.prompt_tokens == 5


def test_transform_embedding_request_user_param():
    config = DashScopeEmbeddingConfig()
    data = config.transform_embedding_request(
        model="text-embedding-v4",
        input=["hello"],
        optional_params={"user": "user-123"},
        headers={},
    )
    assert data["user"] == "user-123"


def test_map_openai_params_drops_unsupported_with_drop_params():
    config = DashScopeEmbeddingConfig()
    result = config.map_openai_params(
        non_default_params={"dimensions": 512, "unknown_param": "value"},
        optional_params={},
        model="text-embedding-v4",
        drop_params=True,
    )
    assert result == {"dimensions": 512}
    assert "unknown_param" not in result


def test_transform_embedding_response_error():
    config = DashScopeEmbeddingConfig()
    payload = {
        "error": {
            "message": "Incorrect API key provided.",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }
    }
    raw = httpx.Response(
        status_code=401,
        content=json.dumps(payload).encode("utf-8"),
        request=httpx.Request("POST", "https://example.com"),
    )
    with pytest.raises(DashScopeError) as exc:
        config.transform_embedding_response(
            model="text-embedding-v4",
            raw_response=raw,
            model_response=EmbeddingResponse(),
            logging_obj=MagicMock(),
            api_key="sk-bad",
            request_data={"input": ["a"]},
            optional_params={},
            litellm_params={},
        )
    assert exc.value.status_code == 401
    assert "Incorrect API key" in exc.value.message


def test_multimodal_embedding_url_and_request():
    config = DashScopeEmbeddingConfig()
    url = config.get_complete_url(
        api_base="https://dashscope.example.com/compatible-mode/v1",
        api_key="sk-test",
        model="qwen3-vl-embedding",
        optional_params={},
        litellm_params={},
    )
    request = config.transform_embedding_request(
        model="qwen3-vl-embedding",
        input=[
            "杭州天气",
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ],
        optional_params={"dimensions": 1024, "enable_fusion": True},
        headers={},
    )

    assert url == f"https://dashscope.example.com/api/v1{MULTIMODAL_EMBEDDING_PATH}"
    assert request == {
        "model": "qwen3-vl-embedding",
        "input": {
            "contents": [
                {"text": "杭州天气"},
                {"image": "https://example.com/cat.png"},
            ]
        },
        "parameters": {"dimension": 1024, "enable_fusion": True},
    }


def test_multimodal_embedding_video_capability_is_model_specific():
    config = DashScopeEmbeddingConfig()
    video_input = [{"type": "video_url", "video_url": "https://example.com/cat.mp4"}]

    with pytest.raises(ValueError, match="does not support video"):
        config.transform_embedding_request(
            model="qwen3-vl-embedding",
            input=video_input,
            optional_params={},
            headers={},
        )

    request = config.transform_embedding_request(
        model="multimodal-embedding-v1",
        input=video_input,
        optional_params={},
        headers={},
    )
    assert request["input"] == {"contents": [{"video": "https://example.com/cat.mp4"}]}


def test_multimodal_embedding_response_usage():
    config = DashScopeEmbeddingConfig()
    raw = httpx.Response(
        200,
        json={
            "request_id": "embedding-request",
            "output": {"embeddings": [{"embedding": [0.2, 0.4], "index": 0}]},
            "usage": {
                "input_tokens": 120,
                "total_tokens": 120,
                "input_tokens_details": {"text_tokens": 80, "image_tokens": 40},
            },
        },
    )

    result = config.transform_embedding_response(
        model="qwen3-vl-embedding",
        raw_response=raw,
        model_response=EmbeddingResponse(),
        logging_obj=MagicMock(),
        api_key="sk-test",
        request_data={},
        optional_params={},
        litellm_params={},
    )

    assert result.id == "embedding-request"
    assert result.data == [{"embedding": [0.2, 0.4], "index": 0, "object": "embedding"}]
    assert result.usage.prompt_tokens == 120
    assert result.usage.prompt_tokens_details.image_tokens == 40
