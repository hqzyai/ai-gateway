"""Mask blocked-word hits for LiteLLM content filter (Bug #158).

Upstream MASK only replaces the keyword label, leaving assigned values such as
「身份证号是2745…」in plaintext for the model to echo. Also redact the value
after common Chinese/English assignment connectors.
"""

from __future__ import annotations

import re

_ASSIGNMENT_CONNECTOR = r"(?:是|为|如下|：|:|=)"


def _keyword_pattern(keyword: str) -> str:
    return re.escape(keyword).replace(r"\*", ".?")


def mask_blocked_keyword(text: str, keyword: str, tag: str) -> str:
    """Redact keyword-assigned values, then any remaining bare keyword hits."""
    if not text or not keyword:
        return text
    escaped = _keyword_pattern(keyword)
    assigned = re.compile(
        rf"({escaped})(\s*{_ASSIGNMENT_CONNECTOR}\s*)(\S+)",
        flags=re.IGNORECASE,
    )
    text = assigned.sub(lambda match: f"{match.group(1)}{match.group(2)}{tag}", text)
    return re.compile(escaped, flags=re.IGNORECASE).sub(tag, text)
