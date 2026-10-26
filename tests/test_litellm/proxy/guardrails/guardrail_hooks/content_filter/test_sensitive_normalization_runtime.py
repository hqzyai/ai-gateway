import json
import unittest

import pytest


@pytest.fixture(autouse=True)
def managed_mode(monkeypatch):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")


from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


class NormalizationRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_mask_then_keyword_block_logs_final_intervention(self):
        from litellm.integrations.custom_guardrail import ModifyResponseException

        for direction in ("request", "response"):
            g = ContentFilterGuardrail(guardrail_name="组合过滤", patterns=[{
                "pattern_type": "regex", "name": "cn_mobile", "action": "MASK",
                "pattern": r"1[3-9]\d{9}",
            }], blocked_words=[{"keyword": "禁止词", "action": "BLOCK"}], event_hook="post_call")
            records = []
            g.add_standard_logging_guardrail_information_to_request_data = lambda **kw: records.append(kw)
            with self.assertRaises(ModifyResponseException):
                await g.apply_guardrail({"texts": ["18212345678 禁止词"]}, {"messages": []}, direction)
            self.assertEqual(records[-1]["guardrail_status"], "guardrail_intervened")

    async def test_real_failure_is_not_intervention(self):
        g = self.guardrail("MASK")
        def fail(*args, **kwargs):
            raise RuntimeError("detector unavailable")
        g._filter_single_text = fail
        with self.assertRaisesRegex(RuntimeError, "detector unavailable"):
            await g.apply_guardrail({"texts": ["test"]}, {"messages": []}, "response")
        self.assertEqual(self.records[-1]["guardrail_status"], "guardrail_failed_to_respond")

    def guardrail(self, action):
        g = ContentFilterGuardrail(
            guardrail_name="敏感信息",
            patterns=[
                {
                    "pattern_type": "regex",
                    "name": "cn_mobile",
                    "action": action,
                    "pattern": r"(?<!\d)1[3-9]\d{9}(?!\d)",
                }
            ],
            event_hook="post_call",
        )
        self.records = []
        g.add_standard_logging_guardrail_information_to_request_data = lambda **kw: (
            self.records.append(kw)
        )
        return g

    async def test_block_and_mask_request_response_variants(self):
        for action in ("BLOCK", "MASK"):
            for direction in ("request", "response"):
                for value in (
                    "182 1234 5678",
                    "+86-182-1234-5678",
                    "１８２\n１２３４\n５６７８",
                    "1\u200b8212345678",
                ):
                    with self.subTest(action=action, direction=direction, value=value):
                        g = self.guardrail(action)
                        if action == "BLOCK":
                            with self.assertRaises(Exception):
                                await g.apply_guardrail(
                                    {"texts": [value]}, {"messages": []}, direction
                                )
                            self.assertEqual(
                                self.records[-1]["guardrail_status"],
                                "guardrail_intervened",
                            )
                        else:
                            result = await g.apply_guardrail(
                                {"texts": [value]}, {"messages": []}, direction
                            )
                            self.assertEqual(result["texts"], ["182****"])
                            self.assertEqual(
                                self.records[-1]["masked_entity_count"]["cn_mobile"], 1
                            )

    async def test_history_adapter(self):
        from litellm.llms.openai.chat.guardrail_translation.handler import (
            OpenAIChatCompletionsHandler,
        )

        result = await OpenAIChatCompletionsHandler().process_input_messages(
            {
                "messages": [
                    {"role": "user", "content": "182-1234-5678"},
                    {"role": "assistant", "content": "１８２１２３４５６７８"},
                    {"role": "user", "content": "复述"},
                ]
            },
            self.guardrail("MASK"),
        )
        self.assertEqual(result["messages"][0]["content"], "182****")
        self.assertEqual(result["messages"][1]["content"], "182****")

    async def test_stream_split_inside_number_and_reasoning(self):
        g = self.guardrail("MASK")

        async def source():
            for text, finish in [
                ("＋８６ １８２-", None),
                ("１２３４\n５６７８", "stop"),
            ]:
                yield ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(content=text, reasoning_content=text),
                            finish_reason=finish,
                        )
                    ]
                )

        chunks = [
            c
            async for c in g.async_post_call_streaming_iterator_hook(
                None, source(), {"messages": []}
            )
        ]
        for field in ("content", "reasoning_content"):
            self.assertEqual(
                "".join(
                    getattr(c.delta, field, "") or ""
                    for chunk in chunks
                    for c in chunk.choices
                ),
                "182****",
            )

    async def test_json_escaped_tool_arguments_are_blocked(self):
        g = self.guardrail("MASK")
        with self.assertRaises(Exception):
            await g.apply_guardrail(
                {
                    "texts": [],
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": json.dumps(
                                    {"phone": "１８２\n１２３４\n５６７８"}
                                ),
                            }
                        }
                    ],
                },
                {"messages": [{"role": "user", "content": "18212345678"}]},
                "response",
            )
