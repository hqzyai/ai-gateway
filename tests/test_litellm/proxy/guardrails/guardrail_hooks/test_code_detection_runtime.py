import unittest

import pytest


@pytest.fixture(autouse=True)
def managed_mode(monkeypatch):
    monkeypatch.setenv("AGENTOS_GUARDRAIL_MODE", "true")


from litellm.proxy.guardrails.guardrail_hooks.block_code_execution.block_code_execution import (
    BlockCodeExecutionGuardrail,
)
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
    ChatCompletionDeltaToolCall,
    Function,
)


class CodeStreamTests(unittest.IsolatedAsyncioTestCase):
    def test_legacy_threshold_does_not_change_enforcement_or_emit_scores(self):
        for threshold in (0, 0.5, 1):
            g = BlockCodeExecutionGuardrail(confidence_threshold=threshold)
            for text in ('print(1)', '```python\nprint(1)\n```', '```text\nprint(1)\n```'):
                detections = []
                self.assertTrue(g._scan_text(text, detections)[1])
                self.assertTrue(all("confidence" not in item for item in detections))

    async def run_stream(self, chunks, action="block"):
        g = BlockCodeExecutionGuardrail(
            guardrail_name="代码执行",
            blocked_languages=["python"],
            action=action,
            event_hook="post_call",
        )
        g.add_standard_logging_guardrail_information_to_request_data = lambda **kw: None

        async def source():
            for chunk in chunks:
                yield chunk

        return [
            item
            async for item in g.async_post_call_streaming_iterator_hook(
                None, source(), {}
            )
        ]

    def chunk(self, text=None, finish=None, tools=None, reasoning=None):
        return ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=text, tool_calls=tools, reasoning_content=reasoning
                    ),
                    finish_reason=finish,
                )
            ]
        )

    async def test_split_code_never_leaks_prefix(self):
        chunks = [self.chunk("这是代码：cus"), self.chunk("tom_task()", "stop")]
        output = await self.run_stream(chunks)
        text = "".join(c.delta.content or "" for item in output for c in item.choices)
        self.assertIn("触发护栏", text)
        self.assertNotIn("cus", text)

    async def test_split_tool_arguments_block_even_in_mask_mode(self):
        chunks = [
            self.chunk(
                tools=[
                    ChatCompletionDeltaToolCall(
                        index=0,
                        id="call_1",
                        type="function",
                        function=Function(
                            name="terminal", arguments='{"code":"custom_'
                        ),
                    )
                ]
            ),
            self.chunk(
                tools=[
                    ChatCompletionDeltaToolCall(
                        index=0, function=Function(arguments='task()"}')
                    )
                ],
                finish="tool_calls",
            ),
        ]
        output = await self.run_stream(chunks, "mask")
        self.assertTrue(
            all(not c.delta.tool_calls for item in output for c in item.choices)
        )
        self.assertIn("工具调用", output[0].choices[0].delta.content)

    async def test_mask_across_chunk_boundary(self):
        output = await self.run_stream(
            [self.chunk("pri"), self.chunk("nt(1)", "stop")], "mask"
        )
        self.assertEqual(
            "".join(c.delta.content or "" for item in output for c in item.choices),
            "[代码已隐藏]",
        )

    async def test_normal_text_preserved(self):
        output = await self.run_stream([self.chunk("你好"), self.chunk("世界", "stop")])
        self.assertEqual(
            "".join(c.delta.content or "" for item in output for c in item.choices),
            "你好世界",
        )

    async def test_reasoning_checked(self):
        output = await self.run_stream(
            [
                self.chunk(reasoning="custom_"),
                self.chunk("好的", "stop", reasoning="task()"),
            ]
        )
        self.assertIn("触发护栏", output[0].choices[0].delta.content)


if __name__ == "__main__":
    unittest.main()
