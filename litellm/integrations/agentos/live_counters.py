"""AgentOS 线路实时计数（在途请求数、当前分钟 RPM/TPM），写入 Redis 供 BFF live-metrics 读取。

为什么不用 LiteLLM 自己的计数：
- global_router:{id}:...:rpm|tpm 只在线路设了 rpm/tpm 限额时才记成功请求；
- {group}_request_count 只在有模型启用最少忙碌策略时才维护。
这里对所有线路、所有策略都计数，且不依赖限额。

限额执行：pre_call_check / async_pre_call_check 读取本模块写入的 RPM/TPM，
若已达 litellm_params 限额则抛 RateLimitError，路由器将该线路冷却并不再转发。

键：
  agentos:gw:inflight:{deployment_id}      zset，member=litellm_call_id，score=开始时间戳
  agentos:gw:call:{call_id}                call_id → 当前所在线路（LiteLLM 重试换线路时 call_id 不变）
  agentos:gw:rpm:{deployment_id}:{HH-MM}   当前 UTC 分钟到达该线路并结束的请求数（成功+失败），ttl 120s
  agentos:gw:tpm:{deployment_id}:{HH-MM}   当前 UTC 分钟 total_tokens（仅成功），ttl 120s

实测（LiteLLM 1.92/1.95）：重试链里中间失败的尝试会触发 failure 回调，但最终失败
（抛给客户端的那一次）不触发，因此用代理级 post_call_failure_hook 兜底收尾。
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping

from pydantic import TypeAdapter
from redis import Redis
from redis.client import Pipeline

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("agentos.live_counters")

INFLIGHT_KEY = "agentos:gw:inflight:{dep}"
CALL_DEP_KEY = "agentos:gw:call:{call_id}"
RPM_KEY = "agentos:gw:rpm:{dep}:{minute}"
TPM_KEY = "agentos:gw:tpm:{dep}:{minute}"
STALE_SECONDS = 600  # 进程异常退出没来得及清理的成员，超过 10 分钟视为不在途
MINUTE_TTL = 120


def _mapping(value: object) -> Mapping[str, object]:
    return (
        TypeAdapter(dict[str, object]).validate_python(value)
        if isinstance(value, Mapping)
        else TypeAdapter(dict[str, object]).validate_python({})  # mutable-ok: Native optional callback field map.
    )


def _dep_and_call(kwargs: Mapping[str, object]) -> tuple[str | None, str | None]:
    params = _mapping(kwargs.get("litellm_params"))
    info = _mapping(params.get("model_info"))
    dep = info.get("id")
    call_id = kwargs.get("litellm_call_id")
    return (str(dep) if dep else None, str(call_id) if call_id else None)


def _total_tokens(response_obj: object) -> int:
    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = TypeAdapter(dict[str, object]).validate_python(response_obj).get("usage")
    if usage is None:
        return 0
    total = (
        TypeAdapter(dict[str, object]).validate_python(usage).get("total_tokens")
        if isinstance(usage, dict)
        else getattr(usage, "total_tokens", None)
    )
    try:
        return int(total) if isinstance(total, (str, bytes, int, float)) else 0
    except (TypeError, ValueError):
        return 0


def _text(value: object) -> str | None:
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def _optional_positive_int(value: object) -> int | None:
    try:
        n = int(value) if isinstance(value, (str, bytes, int, float)) else None
    except (TypeError, ValueError):
        return None
    return n if n is not None and n > 0 else None


def _deployment_limits(deployment: Mapping[str, object]) -> tuple[int | None, int | None]:
    params = _mapping(deployment.get("litellm_params"))
    info = _mapping(deployment.get("model_info"))
    tpm = _optional_positive_int(deployment.get("tpm") or params.get("tpm") or info.get("tpm"))
    rpm = _optional_positive_int(deployment.get("rpm") or params.get("rpm") or info.get("rpm"))
    return tpm, rpm


def _current_usage(value: object) -> int:
    try:
        return int(TypeAdapter(float).validate_python(value)) if value is not None else 0
    except (TypeError, ValueError):
        return 0


class AgentOSLiveCounters(CustomLogger):
    def __init__(self, client: Redis[bytes] | None = None) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # Native CustomLogger constructor leaves kwargs untyped.
        self._redis = client

    def _client(self) -> Redis[bytes]:
        if self._redis is None:
            import redis

            self._redis = redis.Redis(
                host=os.environ.get("REDIS_HOST", "agentos-redis"),
                port=int(os.environ.get("REDIS_PORT", "6379")),
                password=os.environ.get("REDIS_PASSWORD") or None,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
            )
        return self._redis

    def _count_finished(self, pipe: Pipeline[bytes], dep: str, tokens: int) -> None:
        minute = time.strftime("%H-%M", time.gmtime())
        rpm_key = RPM_KEY.format(dep=dep, minute=minute)
        pipe.incr(rpm_key)
        pipe.expire(rpm_key, MINUTE_TTL)
        if tokens:
            tpm_key = TPM_KEY.format(dep=dep, minute=minute)
            pipe.incrby(tpm_key, tokens)
            pipe.expire(tpm_key, MINUTE_TTL)

    def _raise_over_limit(self, *, kind: str, limit: int, current: int, deployment: Mapping[str, object]) -> None:
        import httpx

        import litellm

        params = _mapping(deployment.get("litellm_params"))
        info = _mapping(deployment.get("model_info"))
        model_name = str(params.get("model") or "")
        raise litellm.RateLimitError(
            message=f"Deployment over defined {kind} limit={limit}. current usage={current}",
            llm_provider="",
            model=model_name,
            response=httpx.Response(
                status_code=429,
                content=(
                    f"AgentOS {kind} limit={limit}. current usage={current}. "
                    f"id={info.get('id')}, model_group={deployment.get('model_name', '')}"
                ),
                headers=(("retry-after", "60"),),
                request=httpx.Request(method="agentos_live_counters", url="https://github.com/hqzyai/agentos-mvp"),
            ),
        )

    def _enforce_limits(self, deployment: Mapping[str, object]) -> None:
        """超 TPM/RPM 时拒绝继续转发；与控制台 live-metrics 共用 Redis 计数。"""
        tpm_limit, rpm_limit = _deployment_limits(deployment)
        if tpm_limit is None and rpm_limit is None:
            return
        info = _mapping(deployment.get("model_info"))
        dep = info.get("id")
        if not dep:
            return
        try:
            minute = time.strftime("%H-%M", time.gmtime())
            client = self._client()
            rpm_raw, tpm_raw = client.mget(
                RPM_KEY.format(dep=dep, minute=minute),
                TPM_KEY.format(dep=dep, minute=minute),
            )
        except Exception:
            logger.debug("live counters limit check failed", exc_info=True)
            return
        if rpm_limit is not None and _current_usage(rpm_raw) >= rpm_limit:
            self._raise_over_limit(kind="rpm", limit=rpm_limit, current=_current_usage(rpm_raw), deployment=deployment)
        if tpm_limit is not None and _current_usage(tpm_raw) >= tpm_limit:
            self._raise_over_limit(kind="tpm", limit=tpm_limit, current=_current_usage(tpm_raw), deployment=deployment)

    def pre_call_check(
        self,
        deployment: dict[str, object],  # mutable-ok: Native router passes its deployment dictionary.
    ) -> dict[str, object] | None:  # mutable-ok: Native router callback returns the original mutable deployment object.
        self._enforce_limits(deployment)
        return deployment

    async def async_pre_call_check(
        self,
        deployment: dict[str, object],  # mutable-ok: Native router passes its deployment dictionary.
        parent_otel_span: object = None,
    ) -> (
        dict[str, object] | None  # mutable-ok: Native callback returns the original deployment.
    ):
        del parent_otel_span
        self._enforce_limits(deployment)
        return deployment

    # 进入：LiteLLM 选定线路、真正调上游之前（sync/async 都可能触发，zadd 幂等）
    def log_pre_api_call(self, model: object, messages: object, kwargs: Mapping[str, object]) -> None:
        dep, call_id = _dep_and_call(kwargs)
        if not dep or not call_id:
            return
        try:
            now = time.time()
            client = self._client()
            previous = _text(client.set(CALL_DEP_KEY.format(call_id=call_id), dep, ex=STALE_SECONDS, get=True))
            pipe = client.pipeline()
            if previous and previous != dep:  # 重试换了线路：上一条线路的在途要摘掉
                pipe.zrem(INFLIGHT_KEY.format(dep=previous), call_id)
            key = INFLIGHT_KEY.format(dep=dep)
            pipe.zadd(
                key,
                {call_id: now},  # mutable-ok: Redis requires a member-score dictionary.
            )
            pipe.zremrangebyscore(key, 0, now - STALE_SECONDS)
            pipe.expire(key, STALE_SECONDS)
            pipe.execute()
        except Exception:
            logger.debug("live counters pre-call failed", exc_info=True)

    async def async_log_pre_api_call(self, model: object, messages: object, kwargs: Mapping[str, object]) -> None:
        self.log_pre_api_call(model, messages, kwargs)

    # 单次尝试结束（只实现 async 版本：代理全是异步调用，sync 版本会重复计数）
    def _finish(self, kwargs: Mapping[str, object], response_obj: object, *, success: bool) -> None:
        dep, call_id = _dep_and_call(kwargs)
        if not dep:
            return
        try:
            client = self._client()
            pipe = client.pipeline()
            if call_id:
                pipe.zrem(INFLIGHT_KEY.format(dep=dep), call_id)
                # 只清掉指向本线路的映射；若重试已把映射改到下一条线路，留给它自己收尾
                if _text(client.get(CALL_DEP_KEY.format(call_id=call_id))) == dep:
                    pipe.delete(CALL_DEP_KEY.format(call_id=call_id))
            self._count_finished(pipe, dep, _total_tokens(response_obj) if success else 0)
            pipe.execute()
        except Exception:
            logger.debug("live counters finish failed", exc_info=True)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._finish(kwargs, response_obj, success=True)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._finish(kwargs, response_obj, success=False)

    # 代理级兜底：请求整体结束时，映射还在说明最后一次尝试没触发回调
    def _settle(self, request_data: Mapping[str, object], *, count_failure: bool) -> None:
        call_id = _text(request_data.get("litellm_call_id"))
        if not call_id:
            return
        try:
            client = self._client()
            dep = _text(client.getdel(CALL_DEP_KEY.format(call_id=call_id)))
            if not dep:
                return
            pipe = client.pipeline()
            pipe.zrem(INFLIGHT_KEY.format(dep=dep), call_id)
            if count_failure:
                self._count_finished(pipe, dep, 0)
            pipe.execute()
        except Exception:
            logger.debug("live counters settle failed", exc_info=True)

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, object],
        original_exception: Exception,
        user_api_key_dict: object,
        traceback_str: object = None,
    ) -> None:
        self._settle(request_data, count_failure=True)
        return

    async def async_post_call_success_hook(
        self, data: Mapping[str, object], user_api_key_dict: object, response: object
    ) -> None:
        self._settle(data, count_failure=False)
        return


agentos_live_counters_callback = AgentOSLiveCounters()
