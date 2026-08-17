from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request


def payload(index: int) -> str:
    return "".join(hashlib.sha256(f"{index}:{part}".encode()).hexdigest() for part in range(8))


def main() -> None:
    base_url = os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4000").rstrip("/")
    master_key = os.environ["LITELLM_MASTER_KEY"]
    model = os.environ["LITELLM_MODEL"]
    rows = [
        {
            "id": index,
            "payload": payload(index),
            "status": "retry" if index % 17 == 0 else "ok",
        }
        for index in range(400)
    ]
    messages = [
        {
            "role": "system",
            "content": "Answer accurately from tool data. If omitted data is needed, retrieve it before answering.",
        },
        {"role": "user", "content": "Keep this database result available for a follow-up question."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_database_query",
                    "type": "function",
                    "function": {"name": "database_query", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_database_query",
            "content": json.dumps(rows, separators=(",", ":")),
        },
        {
            "role": "user",
            "content": "Return the exact payload for item 377 and nothing else.",
        },
    ]
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {master_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
        response_status = response.status
    answer = result.get("choices", [{}])[0].get("message", {}).get("content") or ""
    expected = payload(377)
    output = json.dumps(
        {
            "status": response_status,
            "model": result.get("model"),
            "usage": result.get("usage"),
            "finish_reason": result.get("choices", [{}])[0].get("finish_reason"),
            "answer_matches": answer.strip() == expected,
            "answer_chars": len(answer.strip()),
            "expected_chars": len(expected),
        },
        sort_keys=True,
    )
    sys.stdout.write(f"{output}\n")


if __name__ == "__main__":
    main()
