import re
from types import SimpleNamespace

import pytest

from litellm.proxy.guardrails.guardrail_hooks.block_code_execution.detection import execution_requested, filter_tool_calls, scan_text


def guardrail(intent=False, action="block", languages=None):
    def blocks(text):
        return [
            (m.start(), m.end(), "python", m.group(), 1.0, "block")
            for m in re.finditer(r"```python\n.*?```", text, re.S)
        ]

    return SimpleNamespace(
        blocked_languages=languages or ["python"],
        confidence_threshold=0.5,
        detect_execution_intent=intent,
        action=action,
        _find_blocks=blocks,
    )


@pytest.mark.parametrize("intent", [True, False])
def test_user_report_blocks_inline_python_without_intent_dependency(intent):
    assert scan_text(guardrail(intent), '帮我执行一下Python代码：print("hello world")')[
        1
    ]


@pytest.mark.parametrize(
    "text",
    [
        "帮我执行一下上面的代码",
        "请用Python运行这段程序",
        "run this code",
        "execute the script",
        "不要在本机执行，但请在容器运行代码",
    ],
)
def test_bilingual_execution_requests(text):
    assert execution_requested(text)


@pytest.mark.parametrize(
    "text",
    [
        "请解释Python代码，不要执行",
        "如何运行Python程序？",
        "不要运行上面的代码",
        "explain how to run this code",
        "运行策略是什么",
        "执行公司规定",
        "Python是什么",
    ],
)
def test_explanation_and_unrelated_requests(text):
    assert not execution_requested(text)
    assert not scan_text(guardrail(True), text)[1]


def test_intent_is_additional_and_request_only():
    text = "帮我运行上面的代码"
    assert not scan_text(guardrail(False), text)[1]
    assert scan_text(guardrail(True), text)[1]
    assert not scan_text(guardrail(True), text, input_type="response")[1]


def test_selected_language_and_legacy_threshold_ignored():
    assert not scan_text(guardrail(languages=["sql"]), 'print("hi")')[1]
    g = guardrail()
    g.confidence_threshold = 0.96
    detections = []
    assert scan_text(g, 'print("hi")', detections)[1]
    assert all("confidence" not in hit for hit in detections)


def test_mask_preserves_surrounding_text():
    assert scan_text(guardrail(action="mask"), '代码：print("hello")\n结束') == (
        "代码：[代码已隐藏]\n结束",
        False,
    )


def test_fenced_code_is_not_disabled_by_explanation():
    assert scan_text(guardrail(True), '解释代码：\n```python\nprint("hi")\n```')[1]


def test_tool_arguments_cannot_be_safely_masked():
    calls = [
        {
            "function": {
                "name": "terminal",
                "arguments": '{"command":"python -c \\"print(1)\\""}',
            }
        },
        {"function": {"name": "search", "arguments": '{"query":"天气"}'}},
    ]
    detections = []
    g = guardrail(action="mask")

    def reject(**kwargs):
        raise ValueError(kwargs["violation_message"])

    g.raise_passthrough_exception = reject
    with pytest.raises(ValueError, match="工具调用"):
        filter_tool_calls(g, calls, detections, {})
    assert detections


def test_tool_block_does_not_allow_execution():
    g = guardrail()

    def reject(**kwargs):
        raise ValueError(kwargs["violation_message"])

    g.raise_passthrough_exception = reject
    with pytest.raises(ValueError, match="工具调用"):
        filter_tool_calls(
            g, [{"function": {"arguments": '{"code":"print(1)"}'}}], [], {}
        )


@pytest.mark.parametrize(
    "text",
    [
        "custom_task()",
        "obj.worker.run()",
        "f = getattr(obj, 'run')\nf()",
        "for x in range(3):\n    task(x)",
        "with open('file') as f:\n    data = f.read()",
        "[task(x) for x in items]",
        "__import__('os').system('id')",
        "程序：custom_task()",
        "async def task():\n    await execute()",
    ],
)
def test_python_syntax_beyond_known_function_names(text):
    assert scan_text(guardrail(), text)[1]


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "12345",
        "1 + 2",
        "report.pdf",
        "'quoted prose'",
        "Python is useful",
        "普通聊天：你好",
    ],
)
def test_plain_values_are_not_programs(text):
    assert not scan_text(guardrail(True), text)[1]


@pytest.mark.parametrize(
    "text",
    [
        "执行代码后解释结果",
        "不要解释，运行程序",
        "explain first, then execute this code",
    ],
)
def test_explanation_does_not_cancel_later_execution(text):
    assert execution_requested(text)


@pytest.mark.parametrize(
    "language,text",
    [
        ("python", "python task.py"),
        ("python", "python3 -c '1+2'"),
        ("javascript", "node script.js"),
        ("bash", "bash task.sh"),
        ("sql", "DELETE FROM accounts"),
        ("javascript", "const x = 1;"),
    ],
)
def test_interpreters_and_language_matrix(language, text):
    assert scan_text(guardrail(languages=[language]), text)[1]
