#!/usr/bin/env python3
"""A/B lean-ctx compression for Hermes-style LiteLLM traffic.

Compares tokens before/after ``POST {LEAN_CTX_PROXY_URL}/v1/compress`` on fixtures
that mirror Hermes tool loops (logs, JSON tool results, short chat, multi-turn).

Optionally compares LiteLLM billed ``prompt_tokens`` with vs without
``x-headroom-bypass: true`` when ``--litellm`` is set.

Usage:
    python scripts/compare_lean_ctx_compression.py
    python scripts/compare_lean_ctx_compression.py --litellm --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping, Sequence


def _default_proxy_url() -> str:
    return os.environ.get("LEAN_CTX_PROXY_URL", "http://127.0.0.1:4444")


def _default_proxy_token() -> str:
    env_token = os.environ.get("LEAN_CTX_PROXY_TOKEN")
    if env_token:
        return env_token
    try:
        return subprocess.check_output(["lean-ctx", "proxy", "token"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"LEAN_CTX_PROXY_TOKEN unset and `lean-ctx proxy token` failed: {exc}") from exc


def _log_lines(n: int = 150) -> str:
    return "\n".join(
        f"2024-01-01T12:00:{i % 60:02d}Z INFO worker={i % 8} ok path=/var/log/app request_id=req-{i}"
        for i in range(n)
    )


def _array_json(n_files: int = 40, n_per: int = 15) -> str:
    payload = {
        "files": [
            {"path": f"src/m{i}/f{j}.py", "size": 1000 + i * j, "hash": f"h{i}{j}"}
            for i in range(n_files)
            for j in range(n_per)
        ]
    }
    return json.dumps(payload, separators=(",", ":"))


def _hermes_multi_turn() -> list[dict[str, object]]:
    tool_json = _array_json(n_files=20, n_per=10)
    return [
        {"role": "system", "content": "You are Hermes. Use tools when needed."},
        {"role": "user", "content": "Inspect the workspace and summarize the Python modules."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_list",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "find src -name '*.py' | head"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_list", "content": tool_json},
        {"role": "user", "content": "Focus on the modules that look related to networking."},
    ]


def _fixtures() -> list[tuple[str, list[dict[str, object]]]]:
    return [
        (
            "short_chat",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "list files and summarize structure"},
            ],
        ),
        (
            "medium_json_tool",
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_medium_json",
                    "content": _array_json(n_files=8, n_per=8),
                }
            ],
        ),
        (
            "repetitive_logs",
            [{"role": "tool", "tool_call_id": "call_logs", "content": _log_lines()}],
        ),
        (
            "array_json_tool",
            [{"role": "tool", "tool_call_id": "call_json", "content": _array_json()}],
        ),
        ("hermes_multi_turn", _hermes_multi_turn()),
    ]


@dataclass(frozen=True)
class CompressResult:
    name: str
    tokens_before: int | None
    tokens_after: int | None
    compression_ratio: float | None
    has_ccr_marker: bool
    error: str | None = None

    @property
    def saved_pct(self) -> float | None:
        if self.tokens_before is None or self.tokens_after is None or self.tokens_before <= 0:
            return None
        return round(100.0 * (1.0 - (self.tokens_after / self.tokens_before)), 1)


def _post_json(url: str, headers: Mapping[str, str], body: Mapping[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object response from {url}")
    return payload


def compress_messages(
    *,
    proxy_url: str,
    token: str,
    messages: Sequence[Mapping[str, object]],
    model: str,
) -> dict[str, object]:
    return _post_json(
        f"{proxy_url.rstrip('/')}/v1/compress",
        {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        {"messages": list(messages), "model": model},
    )


def measure_fixture(
    name: str,
    messages: Sequence[Mapping[str, object]],
    *,
    proxy_url: str,
    token: str,
    model: str,
) -> CompressResult:
    try:
        body = compress_messages(
            proxy_url=proxy_url,
            token=token,
            messages=messages,
            model=model,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        return CompressResult(name, None, None, None, False, error=str(exc))

    compressed = body.get("messages")
    text_blob = json.dumps(compressed) if compressed is not None else ""
    before = body.get("tokens_before")
    after = body.get("tokens_after")
    ratio = body.get("compression_ratio")
    return CompressResult(
        name=name,
        tokens_before=before if isinstance(before, int) else None,
        tokens_after=after if isinstance(after, int) else None,
        compression_ratio=ratio if isinstance(ratio, (int, float)) else None,
        has_ccr_marker=("hash=" in text_blob) or ("lean-ctx CCR" in text_blob),
    )


def _print_sidecar_table(results: Sequence[CompressResult]) -> None:
    print("lean-ctx /v1/compress")
    print(f"{'fixture':<22} {'before':>8} {'after':>8} {'saved%':>8} {'ratio':>7} {'ccr':>5}")
    print("-" * 64)
    heavy_ccr: list[str] = []
    for row in results:
        if row.error:
            print(f"{row.name:<22} ERROR {row.error}")
            continue
        saved = "-" if row.saved_pct is None else f"{row.saved_pct}"
        ratio = "-" if row.compression_ratio is None else f"{row.compression_ratio:.2f}"
        print(
            f"{row.name:<22} {row.tokens_before or 0:>8} {row.tokens_after or 0:>8} "
            f"{saved:>8} {ratio:>7} {str(row.has_ccr_marker):>5}"
        )
        if row.has_ccr_marker and row.saved_pct is not None and row.saved_pct >= 90:
            heavy_ccr.append(row.name)
    if heavy_ccr:
        print(
            "note: "
            + ", ".join(heavy_ccr)
            + " are mostly CCR stubs; quality depends on headroom_retrieve round-trips"
        )


def measure_litellm_prompt_tokens(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[Mapping[str, object]],
    bypass: bool,
) -> int:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if bypass:
        headers["x-headroom-bypass"] = "true"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": list(messages),
                "max_tokens": 1,
                "temperature": 0,
            }
        ).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ValueError(f"HTTP {exc.code}: {detail}") from exc
    if not isinstance(body, dict):
        raise ValueError("LiteLLM response was not a JSON object")
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("LiteLLM response missing usage")
    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(prompt_tokens, int):
        raise ValueError("LiteLLM usage.prompt_tokens missing")
    return prompt_tokens


def _print_litellm_table(
    rows: Sequence[tuple[str, int, int]],
) -> None:
    print()
    print("LiteLLM prompt_tokens (bypass vs compressed)")
    print(f"{'fixture':<22} {'bypass':>8} {'compressed':>10} {'saved%':>8}")
    print("-" * 52)
    for name, bypass_tokens, compressed_tokens in rows:
        saved = round(100.0 * (1.0 - (compressed_tokens / bypass_tokens)), 1) if bypass_tokens else 0.0
        print(f"{name:<22} {bypass_tokens:>8} {compressed_tokens:>10} {saved:>8}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-url", default=_default_proxy_url())
    parser.add_argument("--proxy-token", default=None)
    parser.add_argument("--compress-model", default="gpt-4o-mini")
    parser.add_argument(
        "--litellm",
        action="store_true",
        help="Also A/B LiteLLM /v1/chat/completions with x-headroom-bypass",
    )
    parser.add_argument("--litellm-url", default=os.environ.get("LITELLM_PROXY_BASE_URL", "http://127.0.0.1:4000"))
    parser.add_argument("--litellm-key", default=os.environ.get("LITELLM_API_KEY", "sk-1234"))
    parser.add_argument("--model", default=os.environ.get("LITELLM_TEST_MODEL", "gpt-4o-mini"))
    args = parser.parse_args(argv)

    token = args.proxy_token or _default_proxy_token()
    fixtures = _fixtures()
    results = [
        measure_fixture(
            name,
            messages,
            proxy_url=args.proxy_url,
            token=token,
            model=args.compress_model,
        )
        for name, messages in fixtures
    ]
    _print_sidecar_table(results)

    failed = [row for row in results if row.error]
    if failed:
        print(f"\n{len(failed)} fixture(s) failed talking to {args.proxy_url}", file=sys.stderr)
        return 1

    if not args.litellm:
        return 0

    litellm_rows: list[tuple[str, int, int]] = []
    for name, messages in fixtures:
        try:
            bypass_tokens = measure_litellm_prompt_tokens(
                base_url=args.litellm_url,
                api_key=args.litellm_key,
                model=args.model,
                messages=messages,
                bypass=True,
            )
            compressed_tokens = measure_litellm_prompt_tokens(
                base_url=args.litellm_url,
                api_key=args.litellm_key,
                model=args.model,
                messages=messages,
                bypass=False,
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            print(f"\nLiteLLM A/B failed on {name}: {exc}", file=sys.stderr)
            return 1
        litellm_rows.append((name, bypass_tokens, compressed_tokens))

    _print_litellm_table(litellm_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
