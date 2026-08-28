import re
from collections.abc import Mapping
from typing import Optional

import litellm
from litellm.types.utils import ImageResponse, ModelInfoBase

_IMAGE_TYPE_PREFIX = re.compile(r"^(?:qima_)?(?:input|output)_")
_SAFE_RESOLUTION = re.compile(r"[^a-z0-9_]")
_QWEN_1K_MAX_PIXELS = 2_250_000


def _non_negative_int(value: object, fallback: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return fallback


def _normalized_resolution(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = _SAFE_RESOLUTION.sub("_", value.strip().lower().replace("-", "_")).strip("_")
    resolution = _IMAGE_TYPE_PREFIX.sub("", normalized)
    return resolution or None


def _resolution_from_dimensions(width: object, height: object) -> Optional[str]:
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        return None
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        return None
    return "1k" if width * height <= _QWEN_1K_MAX_PIXELS else "2k"


def _price(
    model_info: ModelInfoBase,
    pricing: Mapping[str, object],
    direction: str,
    resolution: Optional[str],
) -> float:
    value = pricing.get(f"{direction}_{resolution}") if resolution is not None else None
    fallback = model_info.get(f"{direction}_cost_per_image")
    selected = value if value is not None else fallback
    if isinstance(selected, bool) or not isinstance(selected, (int, float, str)):
        return 0.0
    try:
        return float(selected)
    except ValueError:
        return 0.0


def cost_calculator(
    model: str,
    image_response: ImageResponse,
    custom_llm_provider: Optional[str],
) -> Optional[float]:
    model_info = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    pricing = model_info.get("image_resolution_pricing")
    if not isinstance(pricing, Mapping):
        return None

    input_count = _non_negative_int(image_response.get_hidden_param("input_image_count"))
    output_count = _non_negative_int(
        image_response.get_hidden_param("output_image_count"),
        len(image_response.data or ()),
    )
    input_resolution = _normalized_resolution(image_response.get_hidden_param("input_image_type"))
    output_resolution = _normalized_resolution(image_response.get_hidden_param("output_image_type"))
    if output_resolution is None:
        output_resolution = _resolution_from_dimensions(
            image_response.get_hidden_param("output_width"),
            image_response.get_hidden_param("output_height"),
        )

    return (
        _price(model_info, pricing, "input", input_resolution) * input_count
        + _price(
            model_info,
            pricing,
            "output",
            output_resolution,
        )
        * output_count
    )
