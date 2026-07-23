from collections.abc import Mapping

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str


class SiliconFlowError(BaseLLMException):
    def __init__(self, status_code: int, message: str, headers: httpx.Headers | None = None) -> None:
        super().__init__(status_code=status_code, message=message, headers=dict(headers or httpx.Headers()))


def get_siliconflow_api_base(api_base: str | None = None) -> str:
    return (api_base or get_secret_str("SILICONFLOW_API_BASE") or "https://api.siliconflow.cn/v1").rstrip("/")


def get_siliconflow_headers(api_key: str, extra_headers: Mapping[str, str] | None = None) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        **dict(extra_headers or {}),
    }
