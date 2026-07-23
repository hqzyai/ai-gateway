from __future__ import annotations

import base64
from os import PathLike
from pathlib import Path
from typing import IO, TYPE_CHECKING, Mapping, TypeAlias

import httpx
from httpx._types import RequestFiles

from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse

from ..common_utils import SiliconFlowError, get_siliconflow_api_base, get_siliconflow_headers
from ..image_generation.transformation import SiliconFlowImageGenerationConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


SiliconFlowFileContent: TypeAlias = IO[bytes] | bytes | PathLike[str]
SiliconFlowFile: TypeAlias = (
    SiliconFlowFileContent
    | tuple[str | None, SiliconFlowFileContent]
    | tuple[str | None, SiliconFlowFileContent, str | None]
    | tuple[str | None, SiliconFlowFileContent, str | None, Mapping[str, str]]
)
SiliconFlowImageInput: TypeAlias = SiliconFlowFile | list[SiliconFlowFile]


def _unwrap_image(image: SiliconFlowFile) -> tuple[SiliconFlowFileContent, str | None]:
    if isinstance(image, tuple):
        return image[1], image[2] if len(image) >= 3 else None
    return image, None


def _read_image_bytes(image: SiliconFlowFileContent) -> bytes:
    if isinstance(image, bytes):
        return image
    if isinstance(image, PathLike):
        return Path(image).read_bytes()
    position = image.tell()
    image.seek(0)
    content = image.read()
    image.seek(position)
    return content


def _image_to_data_uri(image: SiliconFlowFile) -> str:
    content, explicit_content_type = _unwrap_image(image)
    content_type = explicit_content_type or ImageEditRequestUtils.get_image_content_type(content)
    encoded = base64.b64encode(_read_image_bytes(content)).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


class SiliconFlowImageEditConfig(BaseImageEditConfig):
    def __init__(self) -> None:
        self._image_generation_config = SiliconFlowImageGenerationConfig()

    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["size", "response_format"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        response_format = image_edit_optional_params.get("response_format")
        if response_format is not None and response_format != "url":
            raise ValueError("SiliconFlow image editing only returns URL responses.")
        size = image_edit_optional_params.get("size")
        return {"image_size": size} if size is not None else {}

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: dict[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        params_api_key = litellm_params.get("api_key") if litellm_params is not None else None
        final_api_key = (
            api_key
            or (params_api_key if isinstance(params_api_key, str) else None)
            or get_secret_str("SILICONFLOW_API_KEY")
        )
        if not final_api_key:
            raise ValueError("SiliconFlow API key is required. Set SILICONFLOW_API_KEY.")
        return get_siliconflow_headers(final_api_key, headers)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],
    ) -> str:
        base = get_siliconflow_api_base(api_base)
        return base if base.endswith("/images/generations") else f"{base}/images/generations"

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: SiliconFlowImageInput | None,
        image_edit_optional_request_params: dict[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> tuple[dict[str, object], RequestFiles]:
        if image is None:
            raise ValueError("SiliconFlow image editing requires at least one image.")
        images = image if isinstance(image, list) else [image]
        if len(images) > 3:
            raise ValueError("SiliconFlow image editing supports at most three input images.")
        if len(images) > 1 and "Qwen-Image-Edit-2509" not in model:
            raise ValueError("Only Qwen/Qwen-Image-Edit-2509 supports multiple input images on SiliconFlow.")
        encoded_images = [_image_to_data_uri(item) for item in images]
        image_fields = {
            "image" if index == 0 else f"image{index + 1}": value for index, value in enumerate(encoded_images)
        }
        return (
            {
                "model": model,
                "prompt": prompt or "",
                **image_fields,
                **image_edit_optional_request_params,
            },
            [],
        )

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        return self._image_generation_config.transform_image_generation_response(
            model=model,
            raw_response=raw_response,
            model_response=ImageResponse(),
            logging_obj=logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    def use_multipart_form_data(self) -> bool:
        return False

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        response_headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers)
        return SiliconFlowError(status_code=status_code, message=error_message, headers=response_headers)
