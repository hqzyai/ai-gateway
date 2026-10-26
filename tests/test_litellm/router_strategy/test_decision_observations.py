from unittest.mock import Mock

import pytest

from litellm.router_strategy import decision_observations as observations


def deployment(identifier, **limits):
    return {"model_info": {"id": identifier, "agentos_config_version": 7}, "litellm_params": limits}


@pytest.fixture
def recorder(monkeypatch):
    value = Mock()
    monkeypatch.setattr(observations, "recorder", value)
    monkeypatch.setenv("AGENTOS_ROUTING_OBSERVATIONS_ENABLED", "true")
    return value


def test_latency_observation_matches_cold_start_and_exclusion(recorder):
    candidates = [deployment("cold"), deployment("fast", rpm=1), deployment("slow")]
    observations.capture_latency_decision(
        "group", candidates,
        {"cold": {"latency": [0]}, "fast": {"latency": [0.1], "minute": {"rpm": 1}}, "slow": {"latency": [0.3, 0.5]}},
        candidates[0], stream=False, lowest_latency_buffer=0.1, input_tokens=0, precise_minute="minute",
    )
    event = recorder.enqueue.call_args.args[0]
    assert event.config_version == 7
    cold, fast, slow = event.candidates
    assert cold.selected and cold.preferred and cold.metric_value == 0
    assert not fast.eligible and fast.excluded_reason == "rpm_limit" and not fast.preferred
    assert slow.metric_value == 0.4 and not slow.preferred


def test_latency_ttft_mode_uses_same_float_samples(recorder):
    candidates = [deployment("a"), deployment("b")]
    observations.capture_latency_decision(
        "group", candidates,
        {"a": {"latency": [9.0], "time_to_first_token": [0, 0.1, 0.3]}, "b": {"latency": [0.22]}},
        candidates[0], stream=True, lowest_latency_buffer=0.2, input_tokens=0, precise_minute="minute",
    )
    first, second = recorder.enqueue.call_args.args[0].candidates
    assert first.metric_value == pytest.approx(0.2) and first.sample_count == 2
    assert first.metric_mode == "ttft_per_output_token"
    assert second.metric_mode == "latency_per_output_token"
    assert first.preferred and second.preferred


def test_usage_observation_excludes_limits_and_preserves_zero(recorder):
    candidates = [deployment("zero", tpm=0), deployment("a", rpm=3), deployment("b")]
    observations.capture_usage_decision("group", candidates, [0, 1, 10], [0, 2, 0], 1, candidates[2])
    zero, first, second = recorder.enqueue.call_args.args[0].candidates
    assert zero.excluded_reason == "tpm_limit" and zero.tpm_limit == 0
    assert first.excluded_reason == "rpm_limit"
    assert second.selected and second.preferred and second.metric_value == 10


def test_least_busy_defaults_missing_to_zero_and_filters_stale_lines(recorder):
    candidates = [deployment("a"), deployment("b")]
    observations.capture_least_busy_decision("group", candidates, {"a": 4, "removed": -1}, candidates[1])
    first, second = recorder.enqueue.call_args.args[0].candidates
    assert first.metric_value == 4 and not first.preferred
    assert second.metric_value == 0 and second.preferred and second.selected


def test_disabled_and_malformed_observations_never_interrupt_routing(recorder, monkeypatch):
    monkeypatch.setenv("AGENTOS_ROUTING_OBSERVATIONS_ENABLED", "false")
    observations.capture_least_busy_decision("group", [deployment("a")], {}, deployment("a"))
    recorder.enqueue.assert_not_called()
    monkeypatch.setenv("AGENTOS_ROUTING_OBSERVATIONS_ENABLED", "true")
    observations.capture_least_busy_decision("group", [{"model_info": {}}], {}, {})
    recorder.enqueue.assert_not_called()
    recorder.enqueue.side_effect = RuntimeError("queue unavailable")
    observations.capture_least_busy_decision("group", [deployment("a")], {}, deployment("a"))
    recorder.enqueue.assert_called_once()
