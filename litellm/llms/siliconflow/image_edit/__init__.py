from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .transformation import SiliconFlowImageEditConfig

__all__ = ["SiliconFlowImageEditConfig"]


def get_siliconflow_image_edit_config(model: str) -> BaseImageEditConfig:
    return SiliconFlowImageEditConfig()
