from unittest.mock import patch

import litellm
from litellm.types.utils import Choices, LlmProviders, Message, ModelResponse, Usage
from litellm.utils import ProviderConfigManager


def test_siliconflow_responses_uses_chat_completions_bridge() -> None:
    chat_response = ModelResponse(
        id="chatcmpl-1",
        created=1,
        model="Qwen/Qwen3-8B",
        object="chat.completion",
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    assert (
        ProviderConfigManager.get_provider_responses_api_config(
            model="Qwen/Qwen3-8B",
            provider=LlmProviders.SILICONFLOW,
        )
        is None
    )
    with patch("litellm.completion", return_value=chat_response) as completion:
        response = litellm.responses(
            model="siliconflow/Qwen/Qwen3-8B",
            input="hi",
            api_key="test-key",
        )

    assert response.output_text == "hello"
    assert completion.call_args.kwargs["custom_llm_provider"] == "siliconflow"
    assert completion.call_args.kwargs["model"] == "Qwen/Qwen3-8B"
    assert completion.call_args.kwargs["messages"] == [{"role": "user", "content": "hi"}]
