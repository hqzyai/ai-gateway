"""
Type definitions for the Headroom server-side compression interception callback.
"""

from typing import Any, Dict, Literal, Optional, TypedDict


class HeadroomInterceptionConfig(TypedDict, total=False):
    """
    Configuration parameters for HeadroomInterceptionLogger.

    Used in proxy_config.yaml under litellm_settings:
        litellm_settings:
          callbacks: ["headroom_interception"]
          headroom_interception_params:
            api_base: os.environ/HEADROOM_API_BASE
            api_key: os.environ/HEADROOM_API_KEY
            model: null
            unreachable_fallback: fail_closed
            tool_call_format: hermes
    """

    api_base: Optional[str]
    api_key: Optional[str]
    model: Optional[str]
    unreachable_fallback: Literal["fail_closed", "fail_open"]
    tool_call_format: Literal["openai", "hermes"]


class HeadroomInterceptionSavingsMetadata(TypedDict):
    """
    Per-request prompt-compression savings recorded into the spend-log metadata
    JSON so daily spend aggregates can track tokens saved by Headroom.
    """

    tokens_before: int
    tokens_after: int
    tokens_saved: int
    source: Literal["headroom_interception"]


class HeadroomCompressResult(TypedDict):
    """Result of a single compress_and_inject_retrieve_tool() call."""

    messages: list[Dict[str, Any]]
    tools: Optional[list[Any]]
    compression_applied: bool
    retrieve_tool_injected: bool
    call_id: Optional[str]
    stats: Dict[str, Any]
