"""
Per-request cost-savings computation for the Cost Optimization dashboard.

Turns the token-level savings recorded on a request into dollar amounts using
the model's own pricing. Daily rollup rows are keyed by date and entity, not by
model, so the dollars have to be computed here (where the model and its prices
are known) and summed into the daily tables; tokens cannot be priced after they
have been aggregated across models.
"""

import math
from collections.abc import Mapping
from typing import NamedTuple

from typing_extensions import TypedDict

import litellm
from litellm._logging import verbose_proxy_logger

from .compression_savings import extract_compression_token_savings


class RecordedSavings(TypedDict, total=False):
    compression_savings_spend: float | None
    compression_gross_savings_spend: float | None
    compression_extra_input_spend: float | None
    prompt_caching_savings_spend: float | None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else RecordedSavings()


def _amount(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _price(info: Mapping[str, object], name: str, tokens: float) -> float | None:
    names = (*(f"{name}_above_{limit}k_tokens" for limit in (512, 272, 200, 128) if tokens > limit * 1000), name)
    return next((value for key in names if (value := _amount(info.get(key))) is not None and value >= 0), None)


def _token_amount(count: int, price: float | None) -> float | None:
    return 0.0 if count == 0 else count * price if price is not None else None


def _recorded_amount(existing: Mapping[str, object], name: str, calculated: float | None) -> float | None:
    return _amount(existing[name]) if name in existing else calculated


def recorded_savings(metadata: Mapping[str, object]) -> RecordedSavings:
    usage = _mapping(metadata.get("usage_object"))
    details = _mapping(usage.get("prompt_tokens_details"))
    info = _mapping(_mapping(metadata.get("model_map_information")).get("model_map_value"))
    costs = _mapping(metadata.get("cost_breakdown"))
    existing = _mapping(metadata.get("optimization_savings"))
    tokens = _amount(usage.get("prompt_tokens")) or _amount(usage.get("input_tokens")) or 0
    input_price = _price(info, "input_cost_per_token", tokens)
    cache_price = _price(info, "cache_read_input_token_cost", tokens)
    cached = max(_amount(usage.get("cache_read_input_tokens")) or _amount(details.get("cached_tokens")) or 0, 0)
    cache_cost = _amount(costs.get("cache_read_cost"))
    compression = extract_compression_token_savings(metadata)
    cache_amount = (
        0.0
        if cached == 0
        else max(0.0, cached * input_price - cache_cost)
        if input_price is not None and cache_cost is not None and cache_cost >= 0
        else cached * max(0.0, input_price - cache_price)
        if input_price is not None and cache_price is not None
        else None
    )
    return RecordedSavings(
        compression_savings_spend=_recorded_amount(
            existing, "compression_savings_spend", _token_amount(compression.net_saved, input_price)
        ),
        compression_gross_savings_spend=_recorded_amount(
            existing, "compression_gross_savings_spend", _token_amount(compression.gross_saved, input_price)
        ),
        compression_extra_input_spend=_recorded_amount(
            existing, "compression_extra_input_spend", _token_amount(compression.extra_input, input_price)
        ),
        prompt_caching_savings_spend=_recorded_amount(existing, "prompt_caching_savings_spend", cache_amount),
    )


class SavingsSpend(NamedTuple):
    compression: float
    compression_gross: float
    compression_extra_input: float
    prompt_caching: float


def _input_and_cache_read_cost(model: str | None, custom_llm_provider: str | None) -> tuple[float, float]:
    """
    Return ``(input_cost_per_token, cache_read_cost_per_token)`` for a model.

    Falls open to ``(0.0, 0.0)`` when the model is unknown so savings degrade to
    zero rather than raising inside the spend writer. When a model has no
    separate cache-read price the cache-read cost mirrors the input cost, which
    yields zero caching savings.
    """
    if not model:
        return 0.0, 0.0
    try:
        info = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception as e:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models; degrade to zero savings
        verbose_proxy_logger.debug(
            "savings: no model info for provider=%s model=%s (%s)", custom_llm_provider, model, e
        )
        return 0.0, 0.0
    input_cost = float(info.get("input_cost_per_token") or 0.0)
    cache_read_cost = info.get("cache_read_input_token_cost")
    if cache_read_cost is None:
        return input_cost, input_cost
    return input_cost, float(cache_read_cost)


def compute_savings_spend(
    model: str | None,
    custom_llm_provider: str | None,
    compression_saved_tokens: int,
    cache_read_input_tokens: int,
    compression_gross_saved_tokens: int = 0,
    compression_extra_input_tokens: int = 0,
) -> SavingsSpend:
    """
    Dollar savings for one request, split by optimization driver.

    Compression savings price the tokens compression removed at the model's
    input rate. Prompt-caching savings price the cache-read tokens at the
    difference between the input rate and the discounted cache-read rate.
    """
    input_cost, cache_read_cost = _input_and_cache_read_cost(model, custom_llm_provider)
    compression = compression_saved_tokens * input_cost
    compression_gross = max(compression_gross_saved_tokens, 0) * input_cost
    compression_extra_input = max(compression_extra_input_tokens, 0) * input_cost
    prompt_caching = max(cache_read_input_tokens, 0) * max(input_cost - cache_read_cost, 0.0)
    return SavingsSpend(
        compression=compression,
        compression_gross=compression_gross,
        compression_extra_input=compression_extra_input,
        prompt_caching=prompt_caching,
    )
