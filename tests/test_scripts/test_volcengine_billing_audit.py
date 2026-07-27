import pytest

from scripts.volcengine_billing_audit import (
    CHAT_MODELS,
    CHAT_SCENARIOS,
    ChatScenario,
    HttpResponse,
    VIDEO_MODELS,
    VIDEO_SCENARIOS,
    VideoScenario,
    chat_expected_cost,
    chat_record,
    image_record,
    provider_request_id,
    provider_video_task_id,
    usage_fields,
    video_payload,
    video_rate,
    video_record,
)


@pytest.mark.parametrize(
    ("usage", "expected"),
    (
        (
            {
                "prompt_tokens": 2000,
                "completion_tokens": 10,
                "total_tokens": 2010,
                "prompt_tokens_details": {"cached_tokens": 1500},
            },
            (2000, 1500, 10, 2010, 0, 0),
        ),
        (
            {
                "input_tokens": 2000,
                "output_tokens": 10,
                "total_tokens": 2010,
                "input_tokens_details": {"cached_tokens": 1500},
            },
            (2000, 1500, 10, 2010, 0, 0),
        ),
    ),
)
def test_usage_fields_reads_chat_and_responses_cache_shapes(
    usage: dict[str, object], expected: tuple[int, int, int, int, int, int]
) -> None:
    response = HttpResponse(status=200, headers={}, body={"usage": usage}, error="")

    assert usage_fields(response) == expected


def test_chat_expected_cost_separates_cached_and_uncached_tokens() -> None:
    cost = chat_expected_cost(CHAT_MODELS[0], prompt_tokens=2000, cached_tokens=1500, completion_tokens=10)

    assert cost == pytest.approx(500 * 3.2e-6 + 1500 * 0.64e-6 + 10 * 16e-6)


def test_chat_audit_only_targets_pro_and_covers_both_input_price_ranges() -> None:
    assert CHAT_MODELS == ("volcengine/doubao-seed-2-0-pro-260215",)
    assert tuple((scenario.minimum_tokens, scenario.maximum_tokens) for scenario in CHAT_SCENARIOS) == (
        (1, 32767),
        (32769, 131072),
    )
    assert CHAT_SCENARIOS[1].prefix_repetitions > CHAT_SCENARIOS[0].prefix_repetitions


def test_chat_expected_cost_uses_above_32k_price_tier_with_cache() -> None:
    cost = chat_expected_cost(CHAT_MODELS[0], prompt_tokens=40000, cached_tokens=30000, completion_tokens=2)

    assert cost == pytest.approx(10000 * 4.8e-6 + 30000 * 0.96e-6 + 2 * 24e-6)


def test_chat_record_rejects_response_outside_requested_token_range() -> None:
    response = HttpResponse(
        status=200,
        headers={"x-litellm-response-cost": "0.2"},
        body={"usage": {"prompt_tokens": 40000, "completion_tokens": 1, "total_tokens": 40001}},
        error="",
    )

    record = chat_record(
        CHAT_MODELS[0],
        "warm",
        ChatScenario("输入小于 32K", 360, 1, 32767),
        response,
    )

    assert record.result == "error"
    assert record.scenario == "输入小于 32K"
    assert "未落入目标区间" in record.detail


def test_seedream_pro_large_output_uses_large_image_rate() -> None:
    response = HttpResponse(
        status=200,
        headers={"x-litellm-response-cost": "0.6", "x-litellm-call-id": "call-1"},
        body={"data": [{"url": "redacted"}], "usage": {"output_tokens": 16385, "total_tokens": 16385}},
        error="",
    )

    record = image_record(
        "volcengine/doubao-seedream-5-0-pro-260628",
        response,
        "2048x2048",
    )

    assert record.generated_images == 1
    assert record.expected_cost == pytest.approx(0.6)
    assert record.delta == pytest.approx(0.0)


def test_provider_request_id_prefers_upstream_header_and_falls_back_to_error() -> None:
    header_response = HttpResponse(
        status=200,
        headers={"llm_provider-x-request-id": "request-from-header"},
        body={},
        error="",
    )
    error_response = HttpResponse(
        status=404,
        headers={},
        body={},
        error="model unavailable. Request id: 02170000000000000000",
    )

    assert provider_request_id(header_response) == "request-from-header"
    assert provider_request_id(error_response) == "02170000000000000000"


def test_provider_video_task_id_decodes_litellm_video_id() -> None:
    video_id = (
        "video_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOnZvbGNlbmdpbmU7bW9kZWxfaWQ6ZG91YmFvLXNlZWRhbmNl"
        "LTItdGVzdDt2aWRlb19pZDpjZ3QtMjAyNjA3MjQxNTEwMjgtbXN6cDc7aGFzX3ZpZGVvX2lucHV0OjA="
    )

    assert provider_video_task_id(video_id) == "cgt-20260724151028-mszp7"


def test_video_audit_only_targets_seedance_2_and_covers_eight_billing_scenarios() -> None:
    assert "volcengine/doubao-seed-2-0-lite-260215" not in CHAT_MODELS
    assert VIDEO_MODELS == ("volcengine/doubao-seedance-2-0-260128",)
    assert {
        (scenario.resolution, scenario.has_video_input) for scenario in VIDEO_SCENARIOS
    } == {
        (resolution, has_video_input)
        for resolution in ("480p", "720p", "1080p", "4k")
        for has_video_input in (False, True)
    }
    assert next(
        scenario.size
        for scenario in VIDEO_SCENARIOS
        if scenario.resolution == "480p" and scenario.has_video_input is False
    ) == "1280x720"


@pytest.mark.parametrize(
    ("resolution", "has_video_input", "yuan_per_million_tokens"),
    (
        ("480p", False, 46.0),
        ("720p", False, 46.0),
        ("1080p", False, 51.0),
        ("4K", False, 26.0),
        ("480p", True, 28.0),
        ("720p", True, 28.0),
        ("1080p", True, 31.0),
        ("4K", True, 16.0),
    ),
)
def test_video_rate_covers_resolution_and_input_dimensions(
    resolution: str, has_video_input: bool, yuan_per_million_tokens: float
) -> None:
    assert video_rate(resolution, has_video_input) == pytest.approx(yuan_per_million_tokens / 1_000_000)


def test_video_payload_includes_reference_video_only_for_video_input() -> None:
    video_url = "https://example.com/audit-input.mp4"
    with_input = video_payload(
        VIDEO_MODELS[0],
        VideoScenario("1080p", "1920x1080", True),
        video_url,
        4,
    )
    without_input = video_payload(
        VIDEO_MODELS[0],
        VideoScenario("480p", "1280x720", False),
        video_url,
        4,
    )

    assert with_input["size"] == "1920x1080"
    assert with_input["seconds"] == "4"
    assert with_input["extra_body"] == {
        "generate_audio": False,
        "resolution": "1080p",
        "watermark": False,
        "content": [
            {
                "type": "text",
                "text": "A blue circle gently moves from left to right on a plain white background",
            },
            {
                "type": "video_url",
                "video_url": {"url": video_url},
                "role": "reference_video",
            },
        ],
    }
    assert without_input["extra_body"] == {
        "generate_audio": False,
        "resolution": "480p",
        "watermark": False,
    }


def test_video_record_uses_reported_billing_scenario() -> None:
    response = HttpResponse(
        status=200,
        headers={"x-litellm-response-cost": "3.1", "x-litellm-call-id": "call-1"},
        body={
            "status": "completed",
            "usage": {
                "completion_tokens": 100000,
                "total_tokens": 100000,
                "duration_seconds": 4.0,
                "video_resolution": "1080p",
                "has_video_input": True,
            },
        },
        error="",
    )

    record = video_record(
        VIDEO_MODELS[0],
        response,
        VideoScenario("720p", "1280x720", False),
        "request-1",
        "task-1",
    )

    assert record.scenario == "包含视频输入 / 1080p"
    assert record.resolution == "1080p"
    assert record.has_video_input is True
    assert record.unit_rate == pytest.approx(31e-6)
    assert record.expected_cost == pytest.approx(3.1)
    assert record.delta == pytest.approx(0.0)
