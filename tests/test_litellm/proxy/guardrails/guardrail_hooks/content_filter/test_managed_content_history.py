import unittest
from copy import deepcopy
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def managed_guardrails(monkeypatch):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


class SensitiveHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_masks_both_reasoning_fields_and_visible_text(self):
        guardrail = self.make_guardrail()
        number = "91310000MAE93MG68G"
        message = SimpleNamespace(
            content=f"结果 {number}",
            reasoning_content=f"分析 {number}",
            reasoning=f"复核 {number}",
        )
        result = await guardrail.apply_guardrail(
            {"texts": [message.content]},
            {
                "response": SimpleNamespace(choices=[SimpleNamespace(message=message)]),
                "messages": [{"role": "user", "content": number}],
            },
            "response",
        )
        self.assertEqual(result["texts"], ["结果 91****"])
        self.assertEqual(message.reasoning_content, "分析 91****")
        self.assertEqual(message.reasoning, "复核 91****")
        self.assertEqual(self.records[-1]["masked_entity_count"]["cn_uscc"], 3)

    async def test_unknown_pattern_uses_stars(self):
        g = ContentFilterGuardrail(
            guardrail_name="custom",
            patterns=[{"pattern_type": "regex", "name": "custom_secret",
                       "pattern": "secret123", "action": "MASK"}],
            event_hook="pre_call",
        )
        g.add_standard_logging_guardrail_information_to_request_data = lambda **kw: None
        self.assertEqual(g.pattern_redaction_format, "****")
        self.assertEqual(g.keyword_redaction_tag, "****")
        from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.sensitive_mask import mask_history_text

        self.assertEqual(mask_history_text(g, "内容 secret123", []), "内容 ****")

    async def test_actual_request_adapter_and_policy_change(self):
        from litellm.llms.openai.chat.guardrail_translation.handler import (
            OpenAIChatCompletionsHandler,
        )

        number = "91310000MAE93MG68G"
        original = {
            "messages": [
                {"role": "user", "content": number},
                {
                    "role": "assistant",
                    "content": number,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"value":"' + number + '"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": number},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "复述之前的号码"}],
                },
            ]
        }
        handler = OpenAIChatCompletionsHandler()
        for _ in range(2):
            g = self.make_guardrail()
            result = await handler.process_input_messages(deepcopy(original), g)
            self.assertNotIn(number, str(result["messages"]))
            self.assertGreater(self.records[-1]["masked_entity_count"]["cn_uscc"], 0)
        no_mask = ContentFilterGuardrail(
            guardrail_name="new-policy", patterns=[], event_hook="pre_call"
        )
        no_mask.add_standard_logging_guardrail_information_to_request_data = (
            lambda **kw: None
        )
        result = await handler.process_input_messages(deepcopy(original), no_mask)
        self.assertIn(number, str(result["messages"]))
        self.assertIn(number, str(original))

    def make_guardrail(self):
        g = ContentFilterGuardrail(
            guardrail_name="敏感信息拦截",
            patterns=[
                {
                    "pattern_type": "regex",
                    "name": "cn_uscc",
                    "pattern": r"(?<![0-9A-Z])[0-9ABCDEFGHJKLMNPQRTUWXY]{18}(?![0-9A-Z])",
                    "action": "MASK",
                }
            ],
            blocked_words=[{"keyword": "禁止词", "action": "BLOCK"}],
            event_hook="pre_call",
        )
        self.records = []
        g.add_standard_logging_guardrail_information_to_request_data = lambda **kw: (
            self.records.append(kw)
        )
        return g

    async def test_history_and_current_redacted_and_counted(self):
        g = self.make_guardrail()
        number = "91310000MAE93MG68G"
        messages = [
            {"role": "user", "content": "禁止词 " + number},
            {"role": "assistant", "content": number},
            {"role": "user", "content": number + "复述一遍"},
        ]
        result = await g.apply_guardrail(
            {
                "texts": [m["content"] for m in messages],
                "structured_messages": messages,
            },
            {"messages": messages},
            "request",
        )
        self.assertNotIn(number, str(result["texts"]))
        self.assertIn("禁止词", result["texts"][0])
        self.assertEqual(self.records[-1]["masked_entity_count"]["cn_uscc"], 3)
        self.assertTrue(
            all(
                d["action"] == "MASK"
                for d in self.records[-1]["guardrail_json_response"]
            )
        )
        self.assertIn(number, messages[0]["content"])

    async def test_current_block_still_blocks(self):
        g = self.make_guardrail()
        with self.assertRaises(Exception):
            await g.apply_guardrail(
                {"texts": ["禁止词"]},
                {"messages": [{"role": "user", "content": "禁止词"}]},
                "request",
            )

    async def test_clean_turn_has_no_false_mask_records(self):
        g = self.make_guardrail()
        result = await g.apply_guardrail(
            {"texts": ["你好"]}, {"messages": []}, "request"
        )
        self.assertEqual(result["texts"], ["你好"])
        self.assertEqual(self.records[-1]["masked_entity_count"], {})

    async def test_stream_source_mask_is_logged(self):
        g = self.make_guardrail()
        number = "91310000MAE93MG68G"

        async def source():
            yield ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=0, delta=Delta(content=number[:9]), finish_reason=None
                    )
                ]
            )
            yield ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=0, delta=Delta(content=number[9:]), finish_reason="stop"
                    )
                ]
            )

        out = [
            c
            async for c in g.async_post_call_streaming_iterator_hook(
                None, source(), {"messages": [{"role": "user", "content": number}]}
            )
        ]
        text = "".join(c.delta.content or "" for item in out for c in item.choices)
        self.assertEqual(text, "91****")
        self.assertEqual(self.records[-1]["masked_entity_count"]["cn_uscc"], 1)


if __name__ == "__main__":
    unittest.main()
