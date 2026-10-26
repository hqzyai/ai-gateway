"""Text-level code detection; never executes submitted code.

Fenced blocks retain the gateway's language matching. Inline rules
cover recognizable syntax, not every possible program or natural-language intent.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterator
from typing import TYPE_CHECKING, Literal, TypeVar

from pydantic import JsonValue, TypeAdapter

from litellm.types.llms.openai import ChatCompletionToolCallChunk
from litellm.types.proxy.guardrails.guardrail_hooks.block_code_execution import CodeBlockDetection
from litellm.types.utils import ChatCompletionMessageToolCall, Choices, ModelResponseStream

Call = TypeVar("Call", ChatCompletionToolCallChunk, ChatCompletionMessageToolCall)

if TYPE_CHECKING:
    from .block_code_execution import BlockCodeExecutionGuardrail

import ast
import re
import unicodedata

ALIASES = (
    ("py", "python"),
    ("js", "javascript"),
    ("ts", "typescript"),
    ("sh", "bash"),
    ("shell", "bash"),
    ("zsh", "bash"),
)
SYNTAX = (
    (
        "python",
        r"\b(?:print|exec|eval)\s*\([^\n]+\)|\b(?:import\s+[a-zA-Z_]\w*|from\s+[a-zA-Z_][\w.]*\s+import\s+\w+)|\bdef\s+\w+\([^\n]*\)\s*:",
    ),
    ("javascript", r"\bconsole\.\w+\([^\n]+\)|\b(?:const|let|var)\s+\w+\s*=\s*[^\n]+|\bfunction\s+\w+\([^\n]*\)\s*\{"),
    ("typescript", r"\b(?:interface|type)\s+[A-Za-z_]\w*\s*(?:=|\{)|\b(?:const|let)\s+\w+\s*:\s*\w+\s*=\s*[^\n]+"),
    (
        "bash",
        r"\b(?:python[23]?|node|bash|sh|zsh)\s+-(?:c|e)\s+[^\n]+|\b(?:curl|wget)\s+https?://\S+|\brm\s+-[rf]+\s+\S+",
    ),
    (
        "sql",
        r"\b(?:SELECT\s+[^\n]+?\s+FROM\s+\w+|DELETE\s+FROM\s+\w+|DROP\s+TABLE\s+\w+|INSERT\s+INTO\s+\w+|UPDATE\s+\w+\s+SET\s+[^\n]+)",
    ),
    ("powershell", r"\b(?:Invoke-Expression|Invoke-WebRequest|Start-Process|Remove-Item)\s+[^\n]+"),
    ("java", r"\bSystem\.out\.print(?:ln)?\([^\n]+\)"),
    ("csharp", r"\bConsole\.Write(?:Line)?\([^\n]+\)"),
    ("go", r"\bfmt\.Print(?:ln|f)?\([^\n]+\)"),
    ("rust", r"\b(?:println|print)!\([^\n]+\)"),
    ("php", r"<\?php[^\n]*"),
    ("ruby", r"\bputs\s+[\"'][^\n]+"),
    ("c", r"\bprintf\s*\([^\n]+\)|#include\s*[<\"][^>\"]+[>\"]"),
    ("cpp", r"\bstd::cout\s*<<[^\n]+"),
)
PATTERNS = tuple((language, re.compile(pattern, re.I)) for language, pattern in SYNTAX)
INTERPRETERS = (
    ("python", re.compile(r"\bpython(?:[23](?:\.\d+)?)?\s+(?:-[cm]\s+\S+|[\w./-]+\.py\b)", re.I)),
    ("javascript", re.compile(r"\bnode\s+(?:-[ep]\s+\S+|[\w./-]+\.(?:js|mjs|cjs)\b)", re.I)),
    ("bash", re.compile(r"\b(?:bash|sh|zsh)\s+(?:-c\s+\S+|[\w./-]+\.sh\b)", re.I)),
)
EXECUTE = re.compile(r"执行|运行|跑一下|跑一遍|跑起来|启动|\b(?:run|execute|evaluate|launch)\b", re.I)
CONTEXT = re.compile(
    r"代码|脚本|命令|程序|终端|控制台|单元格|这段|上述|上面|刚才|python|javascript|typescript|bash|shell|sql|powershell|\b(?:code|script|command|program|snippet|cell|this|it)\b",
    re.I,
)
NEGATION = re.compile(
    r"不要|别|无需|不必|禁止|不能|不允许|不执行|不运行|不用|\b(?:do\s+not|don't|never|without|no\s+need\s+to)\b",
    re.I,
)
EXPLANATION = re.compile(
    r"解释|说明|讲解|含义|什么意思|如何|怎么|怎样|\b(?:explain|describe|how\s+to|what\s+does)\b",
    re.I,
)


def execution_requested(text: str) -> bool:
    # Evaluate clauses independently: "不要在本机执行，但请在容器运行代码"
    # must not be exempted by a negation in a different clause.
    for clause in re.split(
        r"[\n。！？!?；;,，]|但是|但|然后|再|并且|并|\b(?:but|then|and)\b",
        text,
        flags=re.I,
    ):
        verb = EXECUTE.search(clause)
        if not verb or not CONTEXT.search(clause):
            continue
        prefix = clause[: verb.start()]
        if NEGATION.search(prefix) or EXPLANATION.search(prefix):
            continue
        return True
    return False


def python_spans(text: str) -> Iterator[tuple[int, int]]:
    """Parse candidate snippets without compiling or executing them.

    Limit candidate sizes to keep pasted documents from causing unbounded AST work.
    A bare name, number, string or arithmetic expression is not evidence of code.
    """
    candidates = (
        *(((0, len(text)),) if len(text) <= 65536 else ()),
        *((m.start(), m.end()) for m in re.finditer(r"[^\n：]+", text)),
    )
    for start, end in candidates:
        candidate = text[start:end]
        if len(candidate) > 65536:
            continue
        try:
            tree = ast.parse(candidate.strip())
        except (SyntaxError, ValueError, RecursionError):
            continue
        if any(
            isinstance(
                node,
                (
                    ast.Call,
                    ast.Import,
                    ast.ImportFrom,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.For,
                    ast.While,
                    ast.With,
                ),
            )
            for node in ast.walk(tree)
        ):
            yield start, end


def code_hits(
    guardrail: BlockCodeExecutionGuardrail,
    text: str,
    input_type: Literal["request", "response"],
) -> Iterator[tuple[int, int, str]]:
    normalized = unicodedata.normalize("NFKC", text)
    selected = frozenset(
        next((canonical for alias, canonical in ALIASES if alias == lang.lower()), lang.lower())
        for lang in (guardrail.blocked_languages or ())
    )
    yield from (
        (start, end, language)
        for start, end, language, _, _confidence, action in guardrail._find_blocks(text)  # pyright: ignore[reportPrivateUsage]  # Preserve the native fence detector without changing its public API.
        if action != "allow"  # Ignore legacy threshold-based log_only decisions.
    )
    fenced = tuple((m.start(), m.end()) for m in re.finditer(r"```[^\n]*\n.*?```", text, re.S))
    for language, pattern in INTERPRETERS:
        if not selected or language in selected:
            for match in pattern.finditer(text):
                if not any(start <= match.start() < end for start, end in fenced):
                    yield match.start(), match.end(), language
    if not selected or "python" in selected:
        for start, end in python_spans(text):
            if not any(a < end and start < b for a, b in fenced):
                yield start, end, "python"
    for language, pattern in PATTERNS:
        if selected and language not in selected:
            continue
        for match in pattern.finditer(text):
            if any(start <= match.start() < end for start, end in fenced):
                continue
            yield match.start(), match.end(), language
    intent = input_type == "request" and guardrail.detect_execution_intent and execution_requested(normalized)
    if intent:
        yield 0, len(text), "execution_request"


def merged_code_spans(hits: tuple[tuple[int, int, str], ...]) -> Iterator[tuple[int, int]]:
    pending: tuple[int, int] | None = None
    for start, end, _ in sorted(hits):
        if pending is not None and start <= pending[1]:
            pending = (pending[0], max(end, pending[1]))
        else:
            if pending is not None:
                yield pending
            pending = (start, end)
    if pending is not None:
        yield pending


def scan_text(
    guardrail: BlockCodeExecutionGuardrail,
    text: str,
    detections: list[CodeBlockDetection]  # mutable-ok: Native scanner appends findings to the supplied log.
    | None = None,
    input_type: Literal["request", "response"] = "request",
) -> tuple[str, bool]:
    if not text:
        return text, False
    hits = tuple(code_hits(guardrail, text, input_type))
    if detections is not None:
        detections.extend(
            {  # mutable-ok: Native guardrail log entries use the CodeBlockDetection dictionary contract.
                "type": "code_block",
                "language": language,
                "action_taken": "block" if guardrail.action == "block" else "mask",
            }
            for _, _, language in hits
        )
    if not hits:
        return text, False
    if guardrail.action == "block":
        return text, True
    for start, end in reversed(tuple(merged_code_spans(hits))):
        text = text[:start] + "[代码已隐藏]" + text[end:]
    return text, False


def filter_tool_calls(
    guardrail: BlockCodeExecutionGuardrail,
    calls: list[Call],  # mutable-ok: Native tool-call adapter list contract.
    detections: list[  # mutable-ok: Native scanner appends findings to the supplied log.
        CodeBlockDetection
    ],
    request_data: dict[  # mutable-ok: Native passthrough handler updates request metadata.
        str, object
    ],
) -> list[Call]:  # mutable-ok: Native tool-call adapters require a list result.
    """Check emitted calls, never the list of tools offered to the model."""
    kept: tuple[Call, ...] = ()
    for call in calls:
        raw_arguments = call["function"].get("arguments") if isinstance(call, dict) else call.function.arguments
        try:
            arguments: JsonValue = (
                TypeAdapter[JsonValue](JsonValue).validate_json(raw_arguments)
                if isinstance(raw_arguments, str)
                else TypeAdapter[JsonValue](JsonValue).validate_python(raw_arguments)
            )
        except ValueError:
            arguments = TypeAdapter[JsonValue](JsonValue).validate_python(raw_arguments)

        def strings(value: JsonValue) -> Iterator[str]:
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from strings(item)
            elif isinstance(value, list):
                for item in value:
                    yield from strings(item)

        before = len(detections)
        for text in strings(arguments):
            scan_text(guardrail, text, detections, "response")
        if len(detections) == before:
            kept = (*kept, call)
        else:
            guardrail.raise_passthrough_exception(
                violation_message="检测到工具调用中的代码执行内容",
                request_data=request_data,
            )
        # Never shorten the list: gateway adapters can restore original calls
        # when counts differ. Executable arguments cannot safely be text-masked.
    return list(kept)  # mutable-ok: Native tool-call adapters consume a list; only inspected calls are included.


async def checked_stream(
    guardrail: BlockCodeExecutionGuardrail,
    response: AsyncIterable[ModelResponseStream],
    request_data: dict[str, object],  # mutable-ok: Native guardrail hook updates request logging metadata in place.
) -> AsyncIterator[ModelResponseStream]:
    """Hold Chat Completions chunks until code and tool arguments are checked."""
    from litellm import (
        stream_chunk_builder,  # pyright: ignore[reportUnknownVariableType]  # Native stream assembler has incomplete parameter annotations.
    )
    from litellm.integrations.custom_guardrail import ModifyResponseException
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    chunks: list[
        ModelResponseStream
    ] = []  # mutable-ok: Native stream_chunk_builder consumes this size-bounded list of model chunks.
    size = 0
    try:
        async for chunk in response:
            size += len(chunk.model_dump_json().encode("utf-8"))
            if size > 8 * 1024 * 1024:
                guardrail.raise_passthrough_exception(
                    violation_message="响应超过代码检查容量，已停止输出",
                    request_data=request_data,
                )
            chunks.append(chunk)
        if not chunks:
            return
        assembled = stream_chunk_builder(chunks=chunks)
        if assembled is None:
            raise RuntimeError("Unable to assemble response for code inspection")
        choices = tuple(choice for choice in assembled.choices if isinstance(choice, Choices))
        fields = tuple(
            ((choice.index, field), value)
            for choice in choices
            for field in ("content", "reasoning_content", "reasoning")
            for value in (getattr(choice.message, field, None),)
            if isinstance(value, str) and value
        )
        texts = [  # mutable-ok: Native GenericGuardrailAPIInputs requires a text list.
            value for _, value in fields
        ]
        calls = [  # mutable-ok: Native GenericGuardrailAPIInputs requires a tool-call list.
            call for choice in choices for call in (choice.message.tool_calls or ())
        ]
        result = await guardrail.apply_guardrail(  # pyright: ignore[reportUnknownMemberType]  # Native guardrail request_data boundary has incomplete annotations.
            inputs={  # mutable-ok: Native guardrail updates this input dictionary.
                "texts": texts,
                "tool_calls": calls,
            },
            request_data=request_data,
            input_type="response",
        )
        replacements = tuple(
            (key, value) for (key, before), value in zip(fields, result.get("texts", texts)) if before != value
        )
        emitted: frozenset[tuple[int, str]] = frozenset()
        for chunk in chunks:
            for choice in chunk.choices:
                for field in ("content", "reasoning_content", "reasoning"):
                    key = (choice.index, field)
                    replacement = next((value for candidate, value in replacements if candidate == key), None)
                    if replacement is not None and getattr(choice.delta, field, None) is not None:
                        setattr(
                            choice.delta,
                            field,
                            replacement if key not in emitted else "",
                        )
                        emitted = emitted.union((key,))
            yield chunk
    except ModifyResponseException as exc:
        yield ModelResponseStream(
            choices=[  # mutable-ok: Native ModelResponseStream choices field requires a list.
                StreamingChoices(
                    index=0,
                    delta=Delta(role="assistant", content=str(exc)),
                    finish_reason="stop",
                )
            ]
        )
