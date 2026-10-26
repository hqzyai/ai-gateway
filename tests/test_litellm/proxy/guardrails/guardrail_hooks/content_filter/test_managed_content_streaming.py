from __future__ import annotations

import pytest

@pytest.fixture(autouse=True)
def managed_guardrails(monkeypatch):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")

import unittest
from types import SimpleNamespace

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)


def _guardrail() -> ContentFilterGuardrail:
    return ContentFilterGuardrail(
        guardrail_name="手机号保护",
        patterns=[
            {
                "pattern_type": "regex",
                "pattern": r"1[3-9]\d{9}",
                "name": "cn_mobile",
                "action": "MASK",
            }
        ],
        event_hook="post_call",
    )


def _identity_guardrail() -> ContentFilterGuardrail:
    return ContentFilterGuardrail(
        guardrail_name="身份证保护",
        patterns=[
            {
                "pattern_type": "regex",
                "pattern": r"\d{17}[0-9Xx]",
                "name": "cn_identity_card",
                "action": "MASK",
            }
        ],
        event_hook="post_call",
    )


def _blocking_guardrail() -> ContentFilterGuardrail:
    return ContentFilterGuardrail(
        guardrail_name="关键词响应后护栏",
        blocked_words=[{"keyword": "禁止词", "action": "BLOCK"}],
        event_hook="post_call",
    )


class ContentFilterReasoningMaskTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_post_call_block_is_returned_as_an_assistant_chunk(self) -> None:
        guardrail = _blocking_guardrail()
        records = []
        guardrail.add_standard_logging_guardrail_information_to_request_data = lambda **kw: records.append(kw)
        chunks = [
            ModelResponseStream(
                choices=[StreamingChoices(index=0, delta=Delta(content="前半段"), finish_reason=None)]
            ),
            ModelResponseStream(
                choices=[StreamingChoices(index=0, delta=Delta(content="包含禁止词"), finish_reason="stop")]
            ),
        ]

        async def source():
            for chunk in chunks:
                yield chunk

        emitted = [
            item
            async for item in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=SimpleNamespace(),
                response=source(),
                request_data={"model": "test", "messages": [{"role": "user", "content": "测试"}]},
            )
        ]
        content = "".join(str(choice.delta.content or "") for item in emitted for choice in item.choices)

        self.assertEqual(content, "触发护栏「关键词响应后护栏」，内容已拦截：命中关键词规则")
        self.assertNotIn("前半段", content)
        self.assertEqual(emitted[-1].choices[0].finish_reason, "stop")
        self.assertEqual(emitted[-1].choices[0].delta.role, "assistant")
        self.assertEqual(records[-1]["guardrail_status"], "guardrail_intervened")

    async def test_identity_explanation_keeps_derived_attributes_readable(self) -> None:
        guardrail = _identity_guardrail()
        text = (
            "结构解析：360311199905160719；地区码360311，"
            "出生年份1999，顺序码071，校验码9。"
        )

        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={
                "messages": [
                    {"role": "user", "content": "360311199905160719这个身份证号能说明什么？"}
                ]
            },
            input_type="response",
        )

        self.assertEqual(
            result["texts"],
            ["结构解析：360****；地区码360311，出生年份1999，顺序码071，校验码9。"],
        )

    async def test_post_call_masks_visible_content_and_reasoning(self) -> None:
        guardrail = _guardrail()
        message = SimpleNamespace(
            content="联系电话是18931156546，号段是189311",
            reasoning_content="先识别号码18931156546，归属号段189311",
            reasoning=None,
        )
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        result = await guardrail.apply_guardrail(
            inputs={"texts": [message.content]},
            request_data={
                "response": response,
                "messages": [{"role": "user", "content": "18931156546这是什么号码"}],
            },
            input_type="response",
        )

        self.assertEqual(result["texts"], ["联系电话是189****，号段是189311"])
        self.assertEqual(message.reasoning_content, "先识别号码189****，归属号段189311")
        self.assertNotIn("CN_MOBILE", str(result))

    async def test_non_streaming_sensitive_tool_argument_is_blocked(self) -> None:
        guardrail = _guardrail()
        records = []
        guardrail.add_standard_logging_guardrail_information_to_request_data = lambda **kw: records.append(kw)
        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "lookup_phone",
                    "arguments": '{"number":"18931156546"}',
                },
            }
        ]

        with self.assertRaisesRegex(Exception, "敏感信息即将发送给工具"):
            await guardrail.apply_guardrail(
                inputs={"texts": [], "tool_calls": tool_calls},
                request_data={
                    "messages": [{"role": "user", "content": "18931156546这是什么号码"}],
                },
                input_type="response",
            )

        self.assertEqual(records[-1]["guardrail_status"], "guardrail_intervened")
        self.assertIn("敏感信息即将发送给工具", records[-1]["guardrail_json_response"])

    async def test_streaming_reasoning_is_buffered_and_masked_before_emission(self) -> None:
        guardrail = _guardrail()
        chunks = [
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content="", reasoning_content="推理号码18912"),
                        finish_reason=None,
                    )
                ]
            ),
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content="答复完成", reasoning_content="345678"),
                        finish_reason="stop",
                    )
                ]
            ),
        ]

        async def source():
            for chunk in chunks:
                yield chunk

        emitted = [
            item
            async for item in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=SimpleNamespace(),
                response=source(),
                request_data={
                    "model": "test",
                    "messages": [{"role": "user", "content": "18912345678这是什么号码"}],
                },
            )
        ]
        reasoning = "".join(
            str(getattr(choice.delta, "reasoning_content", "") or "")
            for item in emitted
            for choice in item.choices
        )

        self.assertEqual(reasoning, "推理号码189****")
        self.assertNotIn("18912345678", reasoning)

    async def test_streaming_tool_arguments_are_blocked_and_logged(self) -> None:
        guardrail = _guardrail()
        records = []
        guardrail.add_standard_logging_guardrail_information_to_request_data = lambda **kw: records.append(kw)
        chunks = [
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            content="",
                            tool_calls=[
                                ChatCompletionDeltaToolCall(
                                    index=0,
                                    id="call-1",
                                    type="function",
                                    function=Function(name="lookup_phone", arguments='{"number":"18931'),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ]
            ),
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            content="",
                            tool_calls=[
                                ChatCompletionDeltaToolCall(
                                    index=0,
                                    function=Function(name=None, arguments='156546"}'),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ]
            ),
            ModelResponseStream(
                choices=[StreamingChoices(index=0, delta=Delta(content=""), finish_reason="tool_calls")]
            ),
        ]

        async def source():
            for chunk in chunks:
                yield chunk

        with self.assertRaisesRegex(Exception, "敏感信息即将发送给工具"):
            _emitted = [
                item
                async for item in guardrail.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=SimpleNamespace(),
                    response=source(),
                    request_data={
                        "model": "test",
                        "messages": [{"role": "user", "content": "18931156546这是什么号码"}],
                    },
                )
            ]

        self.assertEqual(records[-1]["guardrail_status"], "guardrail_intervened")
        self.assertIn("敏感信息即将发送给工具", records[-1]["guardrail_json_response"])


if __name__ == "__main__":
    unittest.main()
