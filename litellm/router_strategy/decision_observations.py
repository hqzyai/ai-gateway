from __future__ import annotations

import logging
import os
import queue
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from functools import cached_property, wraps
from types import MappingProxyType
from typing import Literal, ParamSpec, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from redis import Redis

logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")
LIMIT = 100
KEY = "agentos:routing-decisions:{model_group}"


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    deployment_id: str
    eligible: bool
    excluded_reason: str | None = None
    metric_value: float
    selected: bool
    preferred: bool = False
    sample_count: int | None = None
    metric_mode: str | None = None
    rpm_value: float | None = None
    tpm_limit: float | None = None
    rpm_limit: float | None = None


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True)

    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_group: str
    strategy: Literal["latency-based-routing", "least-busy", "usage-based-routing-v2"]
    config_version: int | None
    selected_deployment_id: str
    candidates: tuple[Candidate, ...]


class Limits(BaseModel):
    model_config = ConfigDict(frozen=True)
    tpm: float | None = None
    rpm: float | None = None


class ModelInfo(Limits):
    id: str
    agentos_config_version: int | None = None


class Deployment(Limits):
    model_info: ModelInfo
    litellm_params: Limits = Field(default_factory=Limits)

    def limit(self, kind: Literal["tpm", "rpm"], *, truthy: bool = False) -> float | None:
        values = (
            (self.tpm, self.litellm_params.tpm, self.model_info.tpm)
            if kind == "tpm"
            else (self.rpm, self.litellm_params.rpm, self.model_info.rpm)
        )
        return next((value for value in values if (bool(value) if truthy else value is not None)), None)


class Recorder:
    def __init__(self) -> None:
        self.events: queue.Queue[Decision] = queue.Queue(maxsize=2000)
        self.lock = threading.Lock()

    @cached_property
    def client(self) -> Redis[bytes]:
        return Redis(
            host=os.environ.get("REDIS_HOST", "agentos-redis"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            password=os.environ.get("REDIS_PASSWORD") or None,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )

    @cached_property
    def worker(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, name="routing-observations", daemon=True)
        thread.start()
        return thread

    def enqueue(self, decision: Decision) -> None:
        with self.lock:
            self.worker
            if self.events.full():
                try:
                    self.events.get_nowait()
                    self.events.task_done()
                except queue.Empty:
                    pass
            self.events.put_nowait(decision)

    def batch(self) -> Iterator[Decision]:
        yield self.events.get()
        for _ in range(99):
            try:
                yield self.events.get_nowait()
            except queue.Empty:
                return

    def run(self) -> None:
        while True:
            events = tuple(self.batch())
            try:
                with self.client.pipeline(transaction=False) as pipe:
                    for event in events:
                        pipe.lpush(KEY.format(model_group=event.model_group), event.model_dump_json())
                    for group in frozenset(event.model_group for event in events):
                        key = KEY.format(model_group=group)
                        pipe.ltrim(key, 0, LIMIT - 1)
                        pipe.expire(key, 30 * 24 * 60 * 60)
                    pipe.execute()
            except Exception:
                logger.debug("Routing observation persistence failed", exc_info=True)
            finally:
                for _ in events:
                    self.events.task_done()


recorder = Recorder()


def fail_open(function: Callable[P, R]) -> Callable[P, R | None]:
    @wraps(function)
    def capture(
        *args: P.args,
        **kwargs: P.kwargs,  # kwargs-ok: ParamSpec retains native strategy signatures.
    ) -> R | None:
        if os.getenv("AGENTOS_ROUTING_OBSERVATIONS_ENABLED", "false").lower() != "true":
            return None
        try:
            return function(*args, **kwargs)
        except Exception:
            logger.debug("Routing observation capture failed", exc_info=True)
            return None

    return capture


def record(
    group: str,
    strategy: Literal["latency-based-routing", "least-busy", "usage-based-routing-v2"],
    deployments: Sequence[Deployment],
    selected: Deployment,
    candidates: tuple[Candidate, ...],
) -> None:
    versions = frozenset(
        d.model_info.agentos_config_version for d in deployments if d.model_info.agentos_config_version is not None
    )
    if group and selected.model_info.id:
        recorder.enqueue(
            Decision(
                model_group=group,
                strategy=strategy,
                config_version=next(iter(versions)) if len(versions) == 1 else None,
                selected_deployment_id=selected.model_info.id,
                candidates=candidates,
            )
        )


def preferred(candidates: tuple[Candidate, ...], buffer: float = 0) -> tuple[Candidate, ...]:
    minimum = min((row.metric_value for row in candidates if row.eligible), default=float("inf"))
    return tuple(
        row.model_copy(
            update=MappingProxyType(
                {
                    "preferred": row.eligible and row.metric_value <= minimum * (1 + buffer)
                }  # mutable-ok: Frozen update map.
            )
        )
        for row in candidates
    )


def mapping(value: JsonValue) -> Mapping[str, JsonValue]:
    return (
        value if isinstance(value, dict) else MappingProxyType({})  # mutable-ok: Immediately frozen empty cache map.
    )


def number(value: JsonValue) -> float:
    return float(value) if isinstance(value, (str, float, int)) else 0.0


def latency_candidate(
    deployment: Deployment,
    item: Mapping[str, JsonValue],
    selected_id: str,
    stream: bool,
    input_tokens: int,
    minute: str,
) -> Candidate:
    ttft = item.get("time_to_first_token")
    use_ttft = stream and isinstance(ttft, list) and bool(ttft)
    raw = ttft if use_ttft else item.get("latency")
    samples = tuple(value for value in raw if isinstance(value, float)) if isinstance(raw, list) else ()
    usage = mapping(item.get(minute))
    tpm_limit = deployment.limit("tpm", truthy=True)
    rpm_limit = deployment.limit("rpm", truthy=True)
    reason = (
        "tpm_limit"
        if tpm_limit is not None and number(usage.get("tpm")) + input_tokens > tpm_limit
        else "rpm_limit"
        if rpm_limit is not None and number(usage.get("rpm")) + 1 > rpm_limit
        else None
    )
    return Candidate(
        deployment_id=deployment.model_info.id,
        eligible=reason is None,
        excluded_reason=reason,
        metric_value=sum(samples) / len(samples) if samples else 0.0,
        sample_count=len(samples),
        metric_mode="ttft_per_output_token" if use_ttft else "latency_per_output_token",
        selected=deployment.model_info.id == selected_id,
    )


@fail_open
def capture_latency_decision(
    model_group: str,
    healthy_deployments: object,
    request_count_dict: object,
    selected: object,
    *,
    stream: bool,
    lowest_latency_buffer: float,
    input_tokens: int,
    precise_minute: str,
) -> None:
    deployments = TypeAdapter(tuple[Deployment, ...]).validate_python(healthy_deployments)
    choice = Deployment.model_validate(selected)
    counts = TypeAdapter(dict[str, JsonValue]).validate_python(request_count_dict)
    rows = tuple(
        latency_candidate(
            d, mapping(counts.get(d.model_info.id)), choice.model_info.id, stream, input_tokens, precise_minute
        )
        for d in deployments
    )
    record(model_group, "latency-based-routing", deployments, choice, preferred(rows, lowest_latency_buffer))


@fail_open
def capture_least_busy_decision(
    model_group: str, healthy_deployments: object, counts: object, selected: object
) -> None:
    deployments = TypeAdapter(tuple[Deployment, ...]).validate_python(healthy_deployments)
    choice = Deployment.model_validate(selected)
    values = TypeAdapter(dict[str, JsonValue]).validate_python(counts)
    rows = tuple(
        Candidate(
            deployment_id=d.model_info.id,
            eligible=True,
            metric_value=number(values.get(d.model_info.id)),
            selected=d.model_info.id == choice.model_info.id,
        )
        for d in deployments
    )
    record(model_group, "least-busy", deployments, choice, preferred(rows))


def usage_candidate(deployment: Deployment, tpm: float, rpm: float, input_tokens: int, selected_id: str) -> Candidate:
    tpm_limit = deployment.limit("tpm")
    rpm_limit = deployment.limit("rpm")
    reason = (
        "tpm_limit"
        if tpm_limit is not None and tpm + input_tokens > tpm_limit
        else "rpm_limit"
        if rpm_limit is not None and rpm + 1 >= rpm_limit
        else None
    )
    return Candidate(
        deployment_id=deployment.model_info.id,
        eligible=reason is None,
        excluded_reason=reason,
        metric_value=tpm,
        rpm_value=rpm,
        tpm_limit=tpm_limit,
        rpm_limit=rpm_limit,
        selected=deployment.model_info.id == selected_id,
    )


@fail_open
def capture_usage_decision(
    model_group: str,
    healthy_deployments: object,
    tpm_values: object,
    rpm_values: object,
    input_tokens: int,
    selected: object,
) -> None:
    deployments = TypeAdapter(tuple[Deployment, ...]).validate_python(healthy_deployments)
    choice = Deployment.model_validate(selected)
    tpm = TypeAdapter(tuple[JsonValue, ...]).validate_python(tpm_values)
    rpm = TypeAdapter(tuple[JsonValue, ...]).validate_python(rpm_values)
    rows = tuple(
        usage_candidate(d, number(tpm[index]), number(rpm[index]), input_tokens, choice.model_info.id)
        for index, d in enumerate(deployments)
    )
    record(model_group, "usage-based-routing-v2", deployments, choice, preferred(rows))
