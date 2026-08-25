from litellm.llms.base_llm.videos.transformation import BaseVideoConfig

from .transformation import DashScopeVideoConfig

__all__ = ["DashScopeVideoConfig"]


def get_dashscope_video_config(model: str) -> BaseVideoConfig:
    return DashScopeVideoConfig()
