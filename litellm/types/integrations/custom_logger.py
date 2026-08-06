from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

CHAT_COMPLETION_AGENTIC_SURFACE = "chat_completions"
RESPONSES_AGENTIC_SURFACE = "responses"
CODE_INTERPRETER_INTERCEPTION_PREFIX = "_code_interpreter_interception"
NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES = frozenset(
    ("_websearch_interception", "_compression_interception")
)
INTERCEPTION_INTERNAL_PREFIXES = frozenset(
    (
        *NON_CODE_INTERPRETER_INTERCEPTION_INTERNAL_PREFIXES,
        CODE_INTERPRETER_INTERCEPTION_PREFIX,
    )
)


def is_interception_internal_key(
    key: str,
    prefixes: frozenset[str] = INTERCEPTION_INTERNAL_PREFIXES,
) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


# Metadata an interception measures once for the whole client request. An agentic
# loop answers one request with several billed turns, so carrying these onto a
# follow-up turn would report the same measurement on more than one spend row and
# the daily aggregates would sum it twice.
REQUEST_SCOPED_METADATA_KEYS = frozenset(("compression_savings", "compression_retrieval"))


def without_request_scoped_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Copy metadata for a follow-up turn, dropping what the first turn already reported."""
    return {key: value for key, value in metadata.items() if key not in REQUEST_SCOPED_METADATA_KEYS}


class StandardCustomLoggerInitParams(BaseModel):
    """
    Params for initializing a CustomLogger.
    """

    turn_off_message_logging: Optional[bool] = False


class AgenticLoopRequestPatch(BaseModel):
    """
    Patch returned by callbacks to request a follow-up LLM call.
    """

    model: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    max_tokens: Optional[int] = None
    optional_params: Dict[str, Any] = Field(default_factory=dict)
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class AgenticLoopPlan(BaseModel):
    """
    Typed callback response for agentic-loop reruns.
    """

    run_agentic_loop: bool = False
    request_patch: Optional[AgenticLoopRequestPatch] = None
    response_override: Optional[Any] = None
    terminate: bool = False
    stop_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
