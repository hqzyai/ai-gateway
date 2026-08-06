"""
Internal request context for LiteLLM.

Provides a ContextVar-based mechanism for internal signals that must not
be settable from user input. Context variables are scoped to the current
asyncio task and cannot be injected via HTTP request bodies.
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

# When True, suppresses async logging and billing for internal sub-calls
# (e.g., emulated file-search steps that make nested LLM calls).
is_internal_call: ContextVar[bool] = ContextVar("is_internal_call", default=False)


@contextmanager
def suppressed_sub_call_billing() -> Generator[None]:
    """
    Suppress a sub-call's own billing event so the parent event bills it.

    Every suppressed sub-call's tokens must be folded into the parent's
    response, otherwise they are billed by nobody.
    """
    previous = is_internal_call.get()
    is_internal_call.set(True)
    try:
        yield
    finally:
        is_internal_call.set(previous)
