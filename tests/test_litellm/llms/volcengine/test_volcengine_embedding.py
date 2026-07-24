"""
Integration tests for Volcengine embedding following LiteLLM testing patterns
Based on the BaseLLMEmbeddingTest framework
"""

import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath("../../../../.."))

from tests.llm_translation.base_embedding_unit_tests import BaseLLMEmbeddingTest
import litellm
from litellm.types.utils import EmbeddingResponse, Usage


class TestVolcEngineEmbedding(BaseLLMEmbeddingTest):
    """Test Volcengine embedding integration following LiteLLM patterns"""

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.VOLCENGINE

    def get_base_embedding_call_args(self) -> dict:
        return {
            "model": "volcengine/doubao-embedding-text-240715",
        }

    @pytest.mark.asyncio()
    @pytest.mark.parametrize("sync_mode", [True, False])
    async def test_basic_embedding(self, sync_mode):
        """Test basic embedding functionality with realistic response"""
        litellm.set_verbose = True
        embedding_call_args = self.get_base_embedding_call_args()

        # Mock the embedding functions to avoid actual API calls
        with (
            patch("litellm.embedding") as mock_embedding,
            patch("litellm.aembedding") as mock_aembedding,
        ):
            # Create realistic Volcengine response
            mock_response = MagicMock()
            mock_response.model = "doubao-embedding-text-240715"
            mock_response.object = "list"
            mock_response.data = [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3] + [0.01 * i for i in range(1021)],  # 1024-dim embedding
                    "index": 0,
                },
                {
                    "object": "embedding",
                    "embedding": [0.4, 0.5, 0.6] + [0.02 * i for i in range(1021)],  # 1024-dim embedding
                    "index": 1,
                },
            ]
            mock_response.usage.prompt_tokens = 2
            mock_response.usage.total_tokens = 2

            mock_embedding.return_value = mock_response
            mock_aembedding.return_value = mock_response

            # Test sync mode
            if sync_mode is True:
                response = litellm.embedding(
                    **embedding_call_args,
                    input=["hello", "world"],
                )

                # Verify response structure matches Volcengine format
                assert response.model == "doubao-embedding-text-240715"
                assert response.object == "list"
                assert len(response.data) == 2
                assert len(response.data[0]["embedding"]) == 1024
                assert response.usage.total_tokens > 0

            # Test async mode
            else:
                response = await litellm.aembedding(
                    **embedding_call_args,
                    input=["hello", "world"],
                )

                # Verify response structure
                assert response.model == "doubao-embedding-text-240715"
                assert response.object == "list"
                assert len(response.data) == 2
                assert len(response.data[0]["embedding"]) == 1024
                assert response.usage.total_tokens > 0


def test_volcengine_embedding_with_encoding_formats():
    """Test Volcengine embedding with different encoding formats"""

    test_cases = [
        {"encoding_format": "float"},
        {"encoding_format": "base64"},
        {"encoding_format": None},  # Default
    ]

    for params in test_cases:
        with patch("litellm.embedding") as mock_embedding:
            # Create mock response based on encoding format
            mock_response = MagicMock()
            mock_response.model = "doubao-embedding-text-240715"
            mock_response.object = "list"

            if params["encoding_format"] == "base64":
                # Simulate base64 encoded embeddings
                mock_response.data = [
                    {
                        "object": "embedding",
                        "embedding": "c29tZS1iYXNlNjQtZW5jb2RlZC1lbWJlZGRpbmc=",  # base64 encoded
                        "index": 0,
                    }
                ]
            else:
                # Float embeddings (default)
                mock_response.data = [
                    {
                        "object": "embedding",
                        "embedding": [0.1, 0.2, 0.3, -0.1] * 256,  # 1024 dimensions
                        "index": 0,
                    }
                ]

            mock_response.usage.prompt_tokens = 3
            mock_response.usage.total_tokens = 3
            mock_embedding.return_value = mock_response

            # Test the call
            litellm.embedding(
                model="volcengine/doubao-embedding-text-240715",
                input=["test text"],
                **params,
            )

            # Verify the call was made with correct parameters
            mock_embedding.assert_called_once()
            call_args = mock_embedding.call_args
            assert call_args[1]["model"] == "volcengine/doubao-embedding-text-240715"
            assert call_args[1]["input"] == ["test text"]

            if params["encoding_format"] is not None:
                assert call_args[1]["encoding_format"] == params["encoding_format"]


def test_volcengine_embedding_with_user_parameter():
    """Test Volcengine embedding with user parameter for tracking"""

    with patch("litellm.embedding") as mock_embedding:
        mock_response = MagicMock()
        mock_response.model = "doubao-embedding-text-240715"
        mock_response.object = "list"
        mock_response.data = [{"object": "embedding", "embedding": [0.1] * 1024, "index": 0}]
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.total_tokens = 5
        mock_embedding.return_value = mock_response

        # Test with user parameter
        litellm.embedding(
            model="volcengine/doubao-embedding-text-240715",
            input=["user tracking test"],
            user="test-user-12345",
        )

        # Verify user parameter was passed
        mock_embedding.assert_called_once()
        call_args = mock_embedding.call_args
        assert call_args[1]["user"] == "test-user-12345"


def test_volcengine_embedding_error_scenarios():
    """Test Volcengine embedding error handling in integration context"""

    error_scenarios = [
        # Invalid model name
        {"model": "volcengine/invalid-model-name", "expected_error_pattern": "model"},
        # Invalid encoding format
        {
            "model": "volcengine/doubao-embedding-text-240715",
            "encoding_format": "invalid_format",
            "expected_error_pattern": "encoding_format",
        },
    ]

    for scenario in error_scenarios:
        with patch("litellm.embedding") as mock_embedding:
            # Configure mock to raise appropriate errors
            if "invalid-model" in scenario.get("model", ""):
                mock_embedding.side_effect = Exception("Model not found")
            elif scenario.get("encoding_format") == "invalid_format":
                mock_embedding.side_effect = ValueError("Unsupported encoding_format")

            # Test that errors are properly raised
            with pytest.raises(Exception) as exc_info:
                test_params = {k: v for k, v in scenario.items() if k != "expected_error_pattern"}
                litellm.embedding(input=["test"], **test_params)

            # Verify error message contains expected pattern
            assert scenario["expected_error_pattern"].lower() in str(exc_info.value).lower()


def test_volcengine_embedding_with_multiple_inputs():
    """Test Volcengine embedding with various input lengths and types"""

    test_inputs = [
        # Single short text
        ["hello"],
        # Multiple short texts
        ["hello", "world", "test"],
        # Mixed length texts
        [
            "short",
            "This is a much longer text that should be handled properly by the embedding service",
        ],
        # Unicode content
        ["测试中文文本", "Test English text", "混合语言 mixed language"],
        # Many inputs (batch processing)
        [f"Test sentence number {i}" for i in range(10)],
    ]

    for test_input in test_inputs:
        with patch("litellm.embedding") as mock_embedding:
            # Create proportional mock response
            mock_response = MagicMock()
            mock_response.model = "doubao-embedding-text-240715"
            mock_response.object = "list"
            mock_response.data = [
                {
                    "object": "embedding",
                    "embedding": [0.1 * (i + 1)] * 1024,  # Unique embedding per input
                    "index": i,
                }
                for i in range(len(test_input))
            ]
            mock_response.usage.prompt_tokens = len(test_input) * 5  # Realistic token estimate
            mock_response.usage.total_tokens = len(test_input) * 5
            mock_embedding.return_value = mock_response

            # Test the call
            response = litellm.embedding(model="volcengine/doubao-embedding-text-240715", input=test_input)

            # Verify response matches input count
            assert len(response.data) == len(test_input)
            for i, embedding_data in enumerate(response.data):
                assert embedding_data["index"] == i
                assert len(embedding_data["embedding"]) == 1024


def test_volcengine_vision_embedding_request_supports_text_image_and_video():
    from litellm.llms.volcengine.embedding.transformation import (
        VolcEngineEmbeddingConfig,
    )

    config = VolcEngineEmbeddingConfig()
    model = "doubao-embedding-vision-251215"
    input_blocks = [
        {"type": "text", "text": "找出相似内容"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {
            "type": "video_url",
            "video_url": {
                "url": "https://example.com/video.mp4",
                "fps": 1,
                "max_video_tokens": 120000,
                "min_frame_tokens": 32,
                "max_frame_tokens": 640,
                "min_frames": 10,
            },
        },
    ]

    assert config.get_complete_url(None, None, model, {}, {}) == (
        "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
    )
    request = config.transform_embedding_request(
        model,
        input_blocks,
        {
            "encoding_format": "float",
            "dimensions": 1024,
            "instructions": "为跨模态检索生成向量",
            "multi_embedding": {"type": "enabled", "compression": "zstd"},
        },
        {},
    )

    assert request == {
        "model": model,
        "input": input_blocks,
        "encoding_format": "float",
        "dimensions": 1024,
        "instructions": "为跨模态检索生成向量",
        "multi_embedding": {"type": "enabled", "compression": "zstd"},
    }


def test_volcengine_vision_embedding_supports_sparse_text_input_only():
    from litellm.llms.volcengine.embedding.transformation import (
        VolcEngineEmbeddingConfig,
    )

    config = VolcEngineEmbeddingConfig()
    request = config.transform_embedding_request(
        "doubao-embedding-vision-250615",
        [{"type": "text", "text": "天很蓝"}],
        {"sparse_embedding": {"type": "enabled"}},
        {},
    )

    assert request["sparse_embedding"] == {"type": "enabled"}
    with pytest.raises(ValueError, match="text-only"):
        config.transform_embedding_request(
            "doubao-embedding-vision-250615",
            [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}],
            {"sparse_embedding": {"type": "enabled"}},
            {},
        )


def test_volcengine_vision_embedding_enforces_model_version_capabilities():
    from litellm.llms.volcengine.embedding.transformation import (
        VolcEngineEmbeddingConfig,
    )

    config = VolcEngineEmbeddingConfig()

    assert config.get_supported_openai_params("doubao-embedding-vision-250615") == [
        "encoding_format",
        "dimensions",
        "extra_headers",
        "sparse_embedding",
    ]
    with pytest.raises(ValueError, match="251215"):
        config.transform_embedding_request(
            "doubao-embedding-vision-250615",
            "hello",
            {"multi_embedding": {"type": "enabled"}},
            {},
        )


def test_volcengine_vision_embedding_normalizes_text_input():
    from litellm.llms.volcengine.embedding.transformation import (
        VolcEngineEmbeddingConfig,
    )

    request = VolcEngineEmbeddingConfig().transform_embedding_request("doubao-embedding-vision-251215", "hello", {}, {})

    assert request["input"] == [{"type": "text", "text": "hello"}]


def test_volcengine_vision_embedding_response_preserves_multimodal_data_and_usage():
    from litellm.llms.volcengine.embedding.transformation import (
        VolcEngineEmbeddingConfig,
    )

    payload = {
        "created": 1752133360,
        "data": {
            "embedding": [0.1, 0.2],
            "sparse_embedding": [{"index": 1, "value": 0.08}],
            "multi_embedding": [[0.3, 0.4]],
            "object": "embedding",
        },
        "id": "emb-123",
        "model": "doubao-embedding-vision-251215",
        "object": "list",
        "usage": {
            "prompt_tokens": 100,
            "prompt_tokens_details": {"image_tokens": 40, "text_tokens": 60},
            "total_tokens": 100,
        },
    }
    transformed = VolcEngineEmbeddingConfig().transform_embedding_response(
        model="doubao-embedding-vision-251215",
        raw_response=httpx.Response(200, json=payload),
        model_response=EmbeddingResponse(),
        logging_obj=MagicMock(),
        api_key="test-key",
        request_data={},
        optional_params={},
        litellm_params={},
    )

    assert transformed.data == [
        {
            "embedding": [0.1, 0.2],
            "object": "embedding",
            "index": 0,
            "sparse_embedding": [{"index": 1, "value": 0.08}],
            "multi_embedding": [[0.3, 0.4]],
        }
    ]
    assert transformed.usage.prompt_tokens_details.text_tokens == 60
    assert transformed.usage.prompt_tokens_details.image_tokens == 40


def test_volcengine_vision_embedding_uses_separate_text_and_image_token_prices():
    from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token

    prompt_cost, completion_cost = generic_cost_per_token(
        model="doubao-embedding-vision-251215",
        usage=Usage(
            prompt_tokens=100,
            total_tokens=100,
            prompt_tokens_details={"text_tokens": 60, "image_tokens": 40},
        ),
        custom_llm_provider="volcengine",
    )

    assert prompt_cost == pytest.approx((60 * 0.7e-6) + (40 * 1.8e-6))
    assert completion_cost == 0


def test_volcengine_vision_embedding_reports_supported_parameters():
    from litellm.litellm_core_utils.get_supported_openai_params import (
        get_supported_openai_params,
    )

    assert get_supported_openai_params(
        model="doubao-embedding-vision-251215",
        custom_llm_provider="volcengine",
        request_type="embeddings",
    ) == [
        "encoding_format",
        "dimensions",
        "extra_headers",
        "instructions",
        "sparse_embedding",
        "multi_embedding",
    ]


if __name__ == "__main__":
    pytest.main([__file__])
