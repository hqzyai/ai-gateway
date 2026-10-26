import time

import fakeredis
import pytest

from litellm import RateLimitError
from litellm.integrations.agentos.live_counters import (
    AgentOSLiveCounters, CALL_DEP_KEY, INFLIGHT_KEY, RPM_KEY, TPM_KEY,
)
from litellm.types.utils import ModelResponse, Usage


def deployment(tpm=None, rpm=None):
    return {
        "model_name": "group",
        "model_info": {"id": "line"},
        "litellm_params": {"model": "openai/model", "tpm": tpm, "rpm": rpm},
    }


def request(line="line"):
    return {"litellm_call_id": "call", "litellm_params": {"model_info": {"id": line}}}


@pytest.mark.parametrize("kind", ["rpm", "tpm"])
def test_limits_use_current_minute_and_preserve_unlimited_lines(kind):
    storage = fakeredis.FakeRedis()
    counter = AgentOSLiveCounters(storage)
    minute = time.strftime("%H-%M", time.gmtime())
    key = (RPM_KEY if kind == "rpm" else TPM_KEY).format(dep="line", minute=minute)
    storage.set(key, 10)
    unlimited = deployment()
    assert counter.pre_call_check(unlimited) is unlimited
    capped = deployment(**{kind: 10})
    with pytest.raises(RateLimitError) as failure:
        counter.pre_call_check(capped)
    assert failure.value.status_code == 429
    assert failure.value.response.headers["retry-after"] == "60"
    storage.set(key, 9)
    assert counter.pre_call_check(capped) is capped


@pytest.mark.asyncio
async def test_success_counts_total_tokens_once_and_clears_inflight():
    storage = fakeredis.FakeRedis()
    counter = AgentOSLiveCounters(storage)
    data = request()
    counter.log_pre_api_call("model", (), data)
    await counter.async_log_pre_api_call("model", (), data)
    assert storage.zcard(INFLIGHT_KEY.format(dep="line")) == 1
    response = ModelResponse(usage=Usage(prompt_tokens=12, completion_tokens=3, total_tokens=15))
    await counter.async_log_success_event(data, response, None, None)
    await counter.async_post_call_success_hook(data, None, response)
    minute = time.strftime("%H-%M", time.gmtime())
    assert storage.get(RPM_KEY.format(dep="line", minute=minute)) == b"1"
    assert storage.get(TPM_KEY.format(dep="line", minute=minute)) == b"15"
    assert storage.zcard(INFLIGHT_KEY.format(dep="line")) == 0
    assert storage.get(CALL_DEP_KEY.format(call_id="call")) is None


@pytest.mark.asyncio
async def test_retry_moving_line_preserves_new_mapping_and_settles_final_failure_once():
    storage = fakeredis.FakeRedis()
    counter = AgentOSLiveCounters(storage)
    counter.log_pre_api_call("model", (), request("first"))
    counter.log_pre_api_call("model", (), request("second"))
    assert storage.zcard(INFLIGHT_KEY.format(dep="first")) == 0
    assert storage.zcard(INFLIGHT_KEY.format(dep="second")) == 1
    await counter.async_log_failure_event(request("first"), None, None, None)
    assert storage.get(CALL_DEP_KEY.format(call_id="call")) == b"second"
    await counter.async_post_call_failure_hook(request("second"), RuntimeError("failed"), None)
    await counter.async_post_call_failure_hook(request("second"), RuntimeError("failed"), None)
    minute = time.strftime("%H-%M", time.gmtime())
    for line in ("first", "second"):
        assert storage.get(RPM_KEY.format(dep=line, minute=minute)) == b"1"
        assert storage.get(TPM_KEY.format(dep=line, minute=minute)) is None
        assert storage.zcard(INFLIGHT_KEY.format(dep=line)) == 0


def test_native_configuration_loader_resolves_callback():
    from litellm.proxy.types_utils.utils import get_instance_fn

    instance = get_instance_fn("litellm.integrations.agentos.live_counters.agentos_live_counters_callback")
    assert isinstance(instance, AgentOSLiveCounters)


@pytest.mark.parametrize("limit", [0, -1, "", "bad", None])
def test_unset_or_invalid_limits_do_not_query_redis(limit):
    from unittest.mock import Mock
    from redis import Redis

    storage = Mock(spec=Redis)
    counter = AgentOSLiveCounters(storage)
    uncapped = deployment(tpm=limit, rpm=limit)
    assert counter.pre_call_check(uncapped) is uncapped
    storage.mget.assert_not_called()


def test_redis_outage_keeps_original_fail_open_behavior():
    from unittest.mock import Mock
    from redis import Redis

    storage = Mock(spec=Redis)
    storage.mget.side_effect = ConnectionError("unavailable")
    counter = AgentOSLiveCounters(storage)
    capped = deployment(tpm=100, rpm=10)
    assert counter.pre_call_check(capped) is capped
