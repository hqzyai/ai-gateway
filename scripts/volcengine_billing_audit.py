from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4


CHAT_MODELS = (
    "volcengine/doubao-seed-2-0-pro-260215",
)
IMAGE_MODELS = (
    "volcengine/doubao-seedream-4-0-250828",
    "volcengine/doubao-seedream-4-5-251128",
    "volcengine/doubao-seedream-5-0-260128",
    "volcengine/doubao-seedream-5-0-pro-260628",
)
VIDEO_MODELS = (
    "volcengine/doubao-seedance-2-0-260128",
)
VISION_EMBEDDING_MODELS = (
    "volcengine/doubao-embedding-vision-250615",
    "volcengine/doubao-embedding-vision-251215",
)
TEXT_EMBEDDING_MODELS = ("doubao-embedding-text-240715",)
ALL_MODELS = CHAT_MODELS + IMAGE_MODELS + VIDEO_MODELS + VISION_EMBEDDING_MODELS + TEXT_EMBEDDING_MODELS
CHAT_RATES = {
    CHAT_MODELS[0]: ((3.2e-6, 16e-6, 0.64e-6), (4.8e-6, 24e-6, 0.96e-6), (9.6e-6, 48e-6, 1.92e-6)),
}
IMAGE_RATES = {
    IMAGE_MODELS[0]: (0.0, 0.20, None),
    IMAGE_MODELS[1]: (0.0, 0.25, None),
    IMAGE_MODELS[2]: (0.0, 0.22, None),
    IMAGE_MODELS[3]: (0.02, 0.30, 0.60),
}
VIDEO_RATES = {
    (False, "480p"): 46e-6,
    (False, "720p"): 46e-6,
    (False, "1080p"): 51e-6,
    (False, "4k"): 26e-6,
    (True, "480p"): 28e-6,
    (True, "720p"): 28e-6,
    (True, "1080p"): 31e-6,
    (True, "4k"): 16e-6,
}
VISION_EMBEDDING_TEXT_RATE = 0.7e-6
VISION_EMBEDDING_IMAGE_RATE = 1.8e-6
TEXT_EMBEDDING_RATES = {TEXT_EMBEDDING_MODELS[0]: 0.5e-6}
TERMINAL_VIDEO_STATES = frozenset({"completed", "failed", "cancelled", "expired"})
CACHE_MAX_ATTEMPTS = 20
CACHE_TEST_MODELS = CHAT_MODELS


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: object
    error: str


@dataclass(frozen=True)
class VideoScenario:
    resolution: str
    size: str
    has_video_input: bool


@dataclass(frozen=True)
class ChatScenario:
    name: str
    prefix_repetitions: int
    minimum_tokens: int
    maximum_tokens: int


CHAT_SCENARIOS = (
    ChatScenario("输入小于 32K", 360, 1, 32767),
    ChatScenario("输入大于 32K", 2400, 32769, 131072),
)


VIDEO_SCENARIOS = tuple(
    VideoScenario(resolution, size, has_video_input)
    for has_video_input in (False, True)
    for resolution, size in (
        ("480p", "1280x720"),
        ("720p", "1280x720"),
        ("1080p", "1920x1080"),
        ("4k", "3840x2160"),
    )
)


@dataclass(frozen=True)
class AuditRecord:
    model: str
    capability: str
    phase: str
    result: str
    http_status: int
    call_id: str
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    total_tokens: int
    text_tokens: int
    image_tokens: int
    generated_images: int
    input_images: int
    output_image_tokens: int
    resolution: str
    litellm_cost: float | None
    expected_cost: float | None
    delta: float | None
    detail: str
    provider_request_id: str = ""
    provider_task_id: str = ""
    scenario: str = ""
    unit_rate: float | None = None
    has_video_input: bool = False
    duration_seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用火山引擎模型并生成 LiteLLM 中文计费审计报告")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LITELLM_BASE_URL", "http://localhost:4000"),
        help="LiteLLM Proxy 地址",
    )
    parser.add_argument("--output", type=Path, help="Markdown 输出路径；不填则输出到当前目录")
    parser.add_argument("--video-timeout", type=int, default=900, help="每个视频任务的超时秒数")
    parser.add_argument("--poll-interval", type=int, default=8, help="视频任务轮询间隔秒数")
    parser.add_argument("--video-seconds", type=int, default=4, help="视频输出时长，默认 4 秒")
    parser.add_argument("--video-workers", type=int, default=4, help="同时等待的视频任务数，默认 4")
    parser.add_argument(
        "--video-input-url",
        default=os.environ.get("VOLCENGINE_AUDIT_VIDEO_INPUT_URL"),
        help="火山引擎可公开访问的测试视频 URL，也可设置 VOLCENGINE_AUDIT_VIDEO_INPUT_URL",
    )
    return parser.parse_args()


def object_map(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}


def object_list(value: object) -> Sequence[object]:
    return value if isinstance(value, list) else ()


def integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def text(value: object) -> str:
    return value if isinstance(value, str) else ""


def number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def compact_error(body: object, fallback: str) -> str:
    body_map = object_map(body)
    error = body_map.get("error")
    error_map = object_map(error)
    message = text(error_map.get("message")) or text(error) or text(body_map.get("message")) or fallback
    return " ".join(message.split())[:300]


def request_json(
    base_url: str,
    api_key: str,
    method: str,
    path: str,
    payload: Mapping[str, object] | None = None,
    timeout: int = 180,
) -> HttpResponse:
    body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw_body = response.read()
            parsed_body = json.loads(raw_body) if raw_body else {}
            headers = {key.lower(): value for key, value in response.headers.items()}
            return HttpResponse(response.status, headers, parsed_body, "")
    except HTTPError as exc:
        raw_body = exc.read()
        try:
            parsed_body = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            parsed_body = raw_body.decode(errors="replace")
        headers = {key.lower(): value for key, value in exc.headers.items()}
        return HttpResponse(exc.code, headers, parsed_body, compact_error(parsed_body, str(exc)))
    except (URLError, TimeoutError) as exc:
        return HttpResponse(0, {}, {}, " ".join(str(exc).split())[:300])


def response_cost(response: HttpResponse) -> float | None:
    return number(response.headers.get("x-litellm-response-cost"))


def call_id(response: HttpResponse) -> str:
    return response.headers.get("x-litellm-call-id", "")


def provider_request_id(response: HttpResponse) -> str:
    header_id = next(
        (
            text(response.headers.get(key))
            for key in ("llm_provider-x-request-id", "llm_provider-x-tt-logid", "x-request-id", "x-tt-logid")
            if text(response.headers.get(key))
        ),
        "",
    )
    if header_id:
        return header_id
    match = re.search(r"request id:\s*([a-zA-Z0-9_-]+)", response.error, flags=re.IGNORECASE)
    return match.group(1) if match is not None else ""


def provider_video_task_id(video_id: str) -> str:
    if not video_id.startswith("video_"):
        return video_id
    encoded = video_id.removeprefix("video_")
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded).decode()
    except (ValueError, UnicodeDecodeError):
        return ""
    fields = dict(field.split(":", 1) for field in decoded.split(";") if ":" in field)
    return fields.get("video_id", "")


def usage_fields(response: HttpResponse) -> tuple[int, int, int, int, int, int]:
    usage = object_map(object_map(response.body).get("usage"))
    prompt_details = object_map(usage.get("prompt_tokens_details"))
    input_details = object_map(usage.get("input_tokens_details"))
    details = prompt_details or input_details
    prompt_tokens = integer(usage.get("prompt_tokens")) or integer(usage.get("input_tokens"))
    completion_tokens = integer(usage.get("completion_tokens")) or integer(usage.get("output_tokens"))
    total_tokens = integer(usage.get("total_tokens")) or prompt_tokens + completion_tokens
    cached_tokens = integer(details.get("cached_tokens")) or integer(details.get("cache_read_tokens"))
    text_tokens = integer(details.get("text_tokens"))
    image_tokens = integer(details.get("image_tokens"))
    return prompt_tokens, cached_tokens, completion_tokens, total_tokens, text_tokens, image_tokens


def cost_delta(actual: float | None, expected: float | None) -> float | None:
    return actual - expected if actual is not None and expected is not None else None


def chat_expected_cost(model: str, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
    tier = 0 if prompt_tokens <= 32768 else 1 if prompt_tokens <= 131072 else 2
    input_rate, output_rate, cache_rate = CHAT_RATES[model][tier]
    return (prompt_tokens - cached_tokens) * input_rate + cached_tokens * cache_rate + completion_tokens * output_rate


def chat_record(model: str, phase: str, scenario: ChatScenario, response: HttpResponse) -> AuditRecord:
    prompt_tokens, cached_tokens, completion_tokens, total_tokens, text_tokens, image_tokens = usage_fields(response)
    actual = response_cost(response)
    expected = (
        chat_expected_cost(model, prompt_tokens, cached_tokens, completion_tokens) if response.status == 200 else None
    )
    tokens_in_scenario = scenario.minimum_tokens <= prompt_tokens <= scenario.maximum_tokens
    result = "ok" if response.status == 200 and tokens_in_scenario else "error"
    detail = response.error or (
        f"输入 tokens={prompt_tokens} 未落入目标区间 {scenario.minimum_tokens}-{scenario.maximum_tokens}"
        if response.status == 200 and not tokens_in_scenario
        else ""
    )
    return AuditRecord(
        model=model,
        capability="chat",
        phase=phase,
        result=result,
        http_status=response.status,
        call_id=call_id(response),
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        text_tokens=text_tokens,
        image_tokens=image_tokens,
        generated_images=0,
        input_images=0,
        output_image_tokens=0,
        resolution="",
        litellm_cost=actual,
        expected_cost=expected,
        delta=cost_delta(actual, expected),
        detail=detail,
        provider_request_id=provider_request_id(response),
        scenario=scenario.name,
    )


def collect_chat_responses(
    base_url: str,
    api_key: str,
    payload: Mapping[str, object],
    remaining_attempts: int,
    accumulated: tuple[HttpResponse, ...] = (),
) -> tuple[HttpResponse, ...]:
    response = request_json(base_url, api_key, "POST", "/v1/chat/completions", payload)
    collected = accumulated + (response,)
    if usage_fields(response)[1] > 0 or remaining_attempts <= 1 or response.status != 200:
        return collected
    return collect_chat_responses(base_url, api_key, payload, remaining_attempts - 1, collected)


def audit_chat_scenario(
    base_url: str,
    api_key: str,
    model: str,
    scenario: ChatScenario,
) -> tuple[AuditRecord, ...]:
    prefix_sentence = (
        f"Audit run {uuid4().hex}. Cache billing validation uses an identical stable prefix with numbered facts and "
        "deterministic wording. "
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prefix_sentence * scenario.prefix_repetitions},
            {"role": "user", "content": "Reply with exactly OK."},
        ],
        "max_tokens": 16,
        "temperature": 0,
        "thinking": {"type": "disabled"},
    }
    responses = collect_chat_responses(base_url, api_key, payload, CACHE_MAX_ATTEMPTS)
    return tuple(
        chat_record(model, "warm" if index == 0 else f"repeat_{index}", scenario, response)
        for index, response in enumerate(responses)
    )


def audit_chat(base_url: str, api_key: str, model: str) -> tuple[AuditRecord, ...]:
    return tuple(
        record
        for scenario in CHAT_SCENARIOS
        for record in audit_chat_scenario(base_url, api_key, model, scenario)
    )


def image_record(model: str, response: HttpResponse, resolution: str) -> AuditRecord:
    body = object_map(response.body)
    usage = object_map(body.get("usage"))
    images = object_list(body.get("data"))
    generated_images = integer(usage.get("generated_images")) or len(images)
    input_images = integer(usage.get("input_images"))
    output_tokens = integer(usage.get("output_tokens"))
    total_tokens = integer(usage.get("total_tokens"))
    input_rate, output_rate, large_output_rate = IMAGE_RATES[model]
    selected_output_rate = large_output_rate if large_output_rate is not None and output_tokens > 16384 else output_rate
    expected = input_images * input_rate + generated_images * selected_output_rate if response.status == 200 else None
    actual = response_cost(response)
    return AuditRecord(
        model=model,
        capability="image",
        phase="generate",
        result="ok" if response.status == 200 else "error",
        http_status=response.status,
        call_id=call_id(response),
        prompt_tokens=0,
        cached_tokens=0,
        completion_tokens=0,
        total_tokens=total_tokens,
        text_tokens=0,
        image_tokens=0,
        generated_images=generated_images,
        input_images=input_images,
        output_image_tokens=output_tokens,
        resolution=resolution,
        litellm_cost=actual,
        expected_cost=expected,
        delta=cost_delta(actual, expected),
        detail=response.error,
        provider_request_id=provider_request_id(response),
    )


def audit_image(base_url: str, api_key: str, model: str) -> AuditRecord:
    resolution = "2048x2048" if model in (IMAGE_MODELS[1], IMAGE_MODELS[2]) else "1024x1024"
    payload = {
        "model": model,
        "prompt": "A simple blue circle centered on a plain white background, minimal flat design",
        "size": resolution,
        "response_format": "url",
        "watermark": False,
    }
    return image_record(
        model, request_json(base_url, api_key, "POST", "/v1/images/generations", payload, 300), resolution
    )


def video_scenario_name(scenario: VideoScenario) -> str:
    input_name = "包含视频输入" if scenario.has_video_input else "不含视频输入"
    return f"{input_name} / {scenario.resolution}"


def video_rate(resolution: str, has_video_input: bool) -> float | None:
    return VIDEO_RATES.get((has_video_input, resolution.lower()))


def video_payload(
    model: str,
    scenario: VideoScenario,
    video_input_url: str,
    seconds: int,
) -> Mapping[str, object]:
    prompt = "A blue circle gently moves from left to right on a plain white background"
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "video_url",
            "video_url": {"url": video_input_url},
            "role": "reference_video",
        },
    ]
    return {
        "model": model,
        "prompt": prompt,
        "seconds": str(seconds),
        "size": scenario.size,
        "extra_body": {
            "generate_audio": False,
            "resolution": scenario.resolution,
            "watermark": False,
            **({"content": content} if scenario.has_video_input else {}),
        },
    }


def video_record(
    model: str,
    response: HttpResponse,
    scenario: VideoScenario,
    creation_request_id: str,
    task_id: str,
) -> AuditRecord:
    body = object_map(response.body)
    usage = object_map(body.get("usage"))
    completion_tokens = integer(usage.get("completion_tokens"))
    total_tokens = integer(usage.get("total_tokens"))
    actual = response_cost(response)
    reported_resolution = text(usage.get("video_resolution")) or text(body.get("resolution")) or scenario.resolution
    reported_video_input = (
        usage.get("has_video_input") if isinstance(usage.get("has_video_input"), bool) else scenario.has_video_input
    )
    rate = video_rate(reported_resolution, reported_video_input)
    expected = completion_tokens * rate if response.status == 200 and rate is not None else None
    state = text(body.get("status"))
    result = "ok" if response.status == 200 and state == "completed" else "error"
    detail = response.error or (f"terminal video status: {state}" if result == "error" else "")
    return AuditRecord(
        model=model,
        capability="video",
        phase="retrieve",
        result=result,
        http_status=response.status,
        call_id=call_id(response),
        prompt_tokens=0,
        cached_tokens=0,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        text_tokens=0,
        image_tokens=0,
        generated_images=0,
        input_images=0,
        output_image_tokens=0,
        resolution=reported_resolution,
        litellm_cost=actual,
        expected_cost=expected,
        delta=cost_delta(actual, expected),
        detail=detail,
        provider_request_id=creation_request_id or provider_request_id(response),
        provider_task_id=task_id,
        scenario=video_scenario_name(VideoScenario(reported_resolution, scenario.size, reported_video_input)),
        unit_rate=rate,
        has_video_input=reported_video_input,
        duration_seconds=number(usage.get("duration_seconds")) or 0.0,
    )


def video_create_error(model: str, response: HttpResponse, scenario: VideoScenario) -> AuditRecord:
    return AuditRecord(
        model=model,
        capability="video",
        phase="create",
        result="error",
        http_status=response.status,
        call_id=call_id(response),
        prompt_tokens=0,
        cached_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        text_tokens=0,
        image_tokens=0,
        generated_images=0,
        input_images=0,
        output_image_tokens=0,
        resolution=scenario.resolution,
        litellm_cost=response_cost(response),
        expected_cost=None,
        delta=None,
        detail=response.error or "video task id missing",
        provider_request_id=provider_request_id(response),
        scenario=video_scenario_name(scenario),
        unit_rate=video_rate(scenario.resolution, scenario.has_video_input),
        has_video_input=scenario.has_video_input,
    )


def poll_video(
    base_url: str,
    api_key: str,
    model: str,
    video_id: str,
    scenario: VideoScenario,
    creation_request_id: str,
    task_id: str,
    deadline: float,
    poll_interval: int,
) -> AuditRecord:
    while time.monotonic() < deadline:
        response = request_json(base_url, api_key, "GET", f"/v1/videos/{quote(video_id, safe='')}", timeout=60)
        status = text(object_map(response.body).get("status"))
        if response.status != 200 or status in TERMINAL_VIDEO_STATES:
            return video_record(model, response, scenario, creation_request_id, task_id)
        time.sleep(poll_interval)
    timeout_response = HttpResponse(0, {}, {}, "video polling timed out")
    return video_create_error(model, timeout_response, scenario)


def audit_video(
    base_url: str,
    api_key: str,
    model: str,
    scenario: VideoScenario,
    video_input_url: str,
    seconds: int,
    video_timeout: int,
    poll_interval: int,
) -> AuditRecord:
    payload = video_payload(model, scenario, video_input_url, seconds)
    created = request_json(base_url, api_key, "POST", "/v1/videos", payload, 180)
    video_id = text(object_map(created.body).get("id"))
    if created.status != 200 or not video_id:
        return video_create_error(model, created, scenario)
    return poll_video(
        base_url,
        api_key,
        model,
        video_id,
        scenario,
        provider_request_id(created),
        provider_video_task_id(video_id),
        time.monotonic() + video_timeout,
        poll_interval,
    )


def embedding_record(model: str, capability: str, response: HttpResponse) -> AuditRecord:
    prompt_tokens, cached_tokens, completion_tokens, total_tokens, text_tokens, image_tokens = usage_fields(response)
    actual = response_cost(response)
    expected = None
    if response.status == 200 and capability == "vision_embedding":
        normalized_text_tokens = text_tokens or prompt_tokens - image_tokens
        expected = normalized_text_tokens * VISION_EMBEDDING_TEXT_RATE + image_tokens * VISION_EMBEDDING_IMAGE_RATE
    if response.status == 200 and capability == "text_embedding":
        expected = prompt_tokens * TEXT_EMBEDDING_RATES[model]
    return AuditRecord(
        model=model,
        capability=capability,
        phase="embed",
        result="ok" if response.status == 200 else "error",
        http_status=response.status,
        call_id=call_id(response),
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        text_tokens=text_tokens,
        image_tokens=image_tokens,
        generated_images=0,
        input_images=0,
        output_image_tokens=0,
        resolution="",
        litellm_cost=actual,
        expected_cost=expected,
        delta=cost_delta(actual, expected),
        detail=response.error,
        provider_request_id=provider_request_id(response),
    )


def audit_embedding(base_url: str, api_key: str, model: str, vision: bool) -> AuditRecord:
    input_value: object = (
        [{"type": "text", "text": "LiteLLM Volcengine billing audit embedding sample"}]
        if vision
        else "LiteLLM Volcengine billing audit embedding sample"
    )
    payload = {"model": model, "input": input_value, "encoding_format": "float"}
    response = request_json(base_url, api_key, "POST", "/v1/embeddings", payload)
    capability = "vision_embedding" if vision else "text_embedding"
    return embedding_record(model, capability, response)


def audit_all(
    base_url: str,
    api_key: str,
    video_input_url: str,
    video_seconds: int,
    video_timeout: int,
    poll_interval: int,
    video_workers: int,
) -> tuple[AuditRecord, ...]:
    chat_records = tuple(record for model in CHAT_MODELS for record in audit_chat(base_url, api_key, model))
    image_records = tuple(audit_image(base_url, api_key, model) for model in IMAGE_MODELS)
    audit_video_scenario = partial(
        audit_video,
        base_url,
        api_key,
        VIDEO_MODELS[0],
        video_input_url=video_input_url,
        seconds=video_seconds,
        video_timeout=video_timeout,
        poll_interval=poll_interval,
    )
    with ThreadPoolExecutor(max_workers=min(video_workers, len(VIDEO_SCENARIOS))) as executor:
        video_records = tuple(executor.map(audit_video_scenario, VIDEO_SCENARIOS))
    vision_embedding_records = tuple(
        audit_embedding(base_url, api_key, model, True) for model in VISION_EMBEDDING_MODELS
    )
    text_embedding_records = tuple(audit_embedding(base_url, api_key, model, False) for model in TEXT_EMBEDDING_MODELS)
    return chat_records + image_records + video_records + vision_embedding_records + text_embedding_records


def money(value: float | None) -> str:
    return "-" if value is None else f"{value:.9f}"


def usage_summary(record: AuditRecord) -> str:
    if record.capability == "chat":
        return f"输入={record.prompt_tokens}; 缓存命中={record.cached_tokens}; 输出={record.completion_tokens}"
    if record.capability == "image":
        return f"输出图片={record.generated_images}; 输入图片={record.input_images}; 输出 tokens={record.output_image_tokens}"
    if record.capability == "video":
        input_name = "是" if record.has_video_input else "否"
        duration = f"; 输出时长={record.duration_seconds:g}s" if record.duration_seconds > 0 else ""
        return f"视频输入={input_name}; 分辨率={record.resolution}{duration}; 输出 tokens={record.completion_tokens}"
    return f"输入={record.prompt_tokens}; 文本={record.text_tokens}; 图片={record.image_tokens}"


def unit_rate_summary(record: AuditRecord) -> str:
    if record.capability == "chat":
        tier = 0 if record.prompt_tokens <= 32768 else 1 if record.prompt_tokens <= 131072 else 2
        input_rate, output_rate, cache_rate = CHAT_RATES[record.model][tier]
        return (
            f"输入={input_rate * 1_000_000:g}; "
            f"缓存={cache_rate * 1_000_000:g}; "
            f"输出={output_rate * 1_000_000:g}/百万 tokens"
        )
    if record.capability == "video" and record.unit_rate is not None:
        return f"{record.unit_rate * 1_000_000:g}/百万 tokens"
    return "-"


def capability_name(capability: str) -> str:
    return {
        "chat": "对话",
        "image": "图片生成",
        "video": "视频生成",
        "vision_embedding": "多模态向量",
        "text_embedding": "文本向量",
    }.get(capability, capability)


def phase_name(phase: str) -> str:
    if phase == "warm":
        return "首次请求"
    if phase.startswith("repeat_"):
        return f"重复请求 {phase.removeprefix('repeat_')}"
    return {"generate": "生成", "retrieve": "查询完成结果", "embed": "向量化", "create": "创建任务"}.get(phase, phase)


def markdown_report(base_url: str, records: Sequence[AuditRecord], started_at: datetime, finished_at: datetime) -> str:
    rows = "\n".join(
        "| "
        + " | ".join(
            (
                record.model,
                capability_name(record.capability),
                phase_name(record.phase),
                record.scenario or "-",
                f"{'成功' if record.result == 'ok' else '失败'} ({record.http_status})",
                usage_summary(record),
                unit_rate_summary(record),
                money(record.litellm_cost),
                money(record.expected_cost),
                money(record.delta),
                record.provider_request_id or "-",
                record.provider_task_id or "-",
                record.call_id or "-",
                record.detail.replace("|", "\\|") or "-",
            )
        )
        + " |"
        for record in records
    )
    success_count = sum(record.result == "ok" for record in records)
    failed_count = len(records) - success_count
    chat_scenarios_with_cache_hits = {
        record.scenario
        for record in records
        if record.model in CACHE_TEST_MODELS and record.capability == "chat" and record.cached_tokens > 0
    }
    total_cost = sum(record.litellm_cost or 0.0 for record in records)
    return f"""# 火山引擎 LiteLLM 计费实测报告

- 开始时间：{started_at.astimezone().isoformat(timespec="seconds")}
- 结束时间：{finished_at.astimezone().isoformat(timespec="seconds")}
- LiteLLM 地址：`{base_url}`
- 配置模型数：{len(ALL_MODELS)}
- 实测记录数：{len(records)}
- 成功记录数：{success_count}
- 失败记录数：{failed_count}
- 对话模型：仅测试 `{CHAT_MODELS[0]}`
- 已测缓存且命中的对话计费区间：{len(chat_scenarios_with_cache_hits)}/{len(CHAT_SCENARIOS)}
- 对话 Lite 和 Mini：按要求不测试
- 视频计费场景：{len(VIDEO_SCENARIOS)} 个，仅测试 `{VIDEO_MODELS[0]}`
- 本表对比记录的 LiteLLM 费用合计：`{total_cost:.9f}`

“LiteLLM 费用”取自响应头 `x-litellm-response-cost`。“公式复算”使用火山返回的原始 usage 和当前火山单价独立计算。按照本仓库现有约定，人民币单价数字以 1:1 数值写入美元计费字段，因此表内数字用于数值对账，不表示实际汇率换算。

表内费用合计只统计当前展示的样本。如果在同一时间段执行过调试或补采请求，这些请求也会出现在实际账单中，因此应按下表中的火山 ID 逐条核对，不要按整个调试时间段的费用合计核对。

| 模型 | 类型 | 阶段 | 计费场景 | 结果 | 原始用量 | 场景单价 | LiteLLM 费用 | 公式复算 | 差额 | 火山 Request ID | 火山任务 ID | LiteLLM call ID | 详情 |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|
{rows}

## 计费公式

- 对话：`(输入 tokens - 缓存命中 tokens) × 输入单价 + 缓存命中 tokens × 缓存单价 + 输出 tokens × 输出单价`。仅测试 Pro，并分别覆盖输入小于 32K、输入大于 32K 且不超过 131K 两个价格区间；每个区间在首次缓存命中后停止，最多发送 {CACHE_MAX_ATTEMPTS} 次相同请求
- 图片：`输入图片数 × 输入图片单价 + 输出图片数 × 输出图片单价`。Seedream 5.0 Pro 的输出超过 16,384 tokens 时使用大图价格
- 视频：`completion_tokens × 场景单价`。仅测试 Seedance 2.0，覆盖不含/包含视频输入两类场景，以及 480p、720p、1080p、4K 四种分辨率，共 8 个场景
- 多模态向量：`文本 tokens × 文本单价 + 图片 tokens × 图片单价`
- 文本向量：官方公开价独立换算为 `0.5/百万输入 tokens`，当前 LiteLLM 价格表配置为零

## 对账方法

- 对话、图片和向量模型：在火山侧使用“火山 Request ID”定位请求，在 LiteLLM 管理日志中使用同一行的“LiteLLM call ID”定位请求
- 视频模型：在火山侧使用“火山任务 ID”定位异步任务，在 LiteLLM 管理日志中使用同一行的“LiteLLM call ID”定位最终计费记录
- LiteLLM call ID 与火山 Request ID 不是同一个值，必须使用同一表行建立映射，不能拿 LiteLLM call ID 直接搜索火山账单

## 文本向量模型异常

方舟模型元数据将 `doubao-embedding-text-240715` 标记为 `Retiring`，并且 `online_inference=false`，与本次上游返回 404 一致。方舟仍列出输入价 `0.0005/千 tokens`，但当前 LiteLLM 价格表记录为零。如果已有 endpoint 仍能调用该模型，修正价格配置前 LiteLLM 会少计输入费用。

官方资料：[模型与缓存能力](https://www.volcengine.com/docs/82379/1330310)、[隐式缓存原理](https://www.volcengine.com/docs/82379/1398933)、[多模态模型计费](https://www.volcengine.com/docs/82379/1544106)。

报告未保存 API key、图片 URL 或视频 URL。火山 Request ID / 任务 ID 用于与火山账单逐条核对，LiteLLM call ID 用于查询管理员 spend logs。
"""


def output_path(requested: Path | None, started_at: datetime) -> Path:
    if requested is not None:
        return requested
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / f"volcengine_billing_audit_{timestamp}.md"


def write_reports(
    path: Path, base_url: str, records: Sequence[AuditRecord], started_at: datetime, finished_at: datetime
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(base_url, records, started_at, finished_at))
    json_path = path.with_suffix(".json")
    payload = {
        "started_at": started_at.astimezone().isoformat(timespec="seconds"),
        "finished_at": finished_at.astimezone().isoformat(timespec="seconds"),
        "base_url": base_url,
        "records": [asdict(record) for record in records],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("LITELLM_API_KEY")
    if not api_key:
        sys.stderr.write("LITELLM_API_KEY is required\n")
        return 2
    video_input_url = args.video_input_url
    if not isinstance(video_input_url, str) or not valid_http_url(video_input_url):
        sys.stderr.write("--video-input-url or VOLCENGINE_AUDIT_VIDEO_INPUT_URL must be a public HTTP(S) URL\n")
        return 2
    if args.video_seconds <= 0 or args.video_workers <= 0 or args.video_timeout <= 0 or args.poll_interval <= 0:
        sys.stderr.write("video seconds, workers, timeout, and poll interval must be greater than zero\n")
        return 2
    sys.stderr.write(
        f"Running {len(VIDEO_SCENARIOS)} Seedance billing scenarios; video generation may incur substantial charges.\n"
    )
    started_at = datetime.now().astimezone()
    records = audit_all(
        args.base_url,
        api_key,
        video_input_url,
        args.video_seconds,
        args.video_timeout,
        args.poll_interval,
        args.video_workers,
    )
    finished_at = datetime.now().astimezone()
    path = output_path(args.output, started_at)
    write_reports(path, args.base_url, records, started_at, finished_at)
    sys.stdout.write(markdown_report(args.base_url, records, started_at, finished_at))
    sys.stdout.write(f"\nMarkdown: {path}\n")
    sys.stdout.write(f"JSON: {path.with_suffix('.json')}\n")
    return 0 if all(record.result == "ok" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
