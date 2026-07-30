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

from ..common_utils import VolcEngineError, get_volcengine_api_base, get_volcengine_headers
from ..image_generation.transformation import VolcEngineImageGenerationConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


VolcEngineFileContent: TypeAlias = IO[bytes] | bytes | PathLike[str]
VolcEngineFile: TypeAlias = (
    VolcEngineFileContent
    | tuple[str | None, VolcEngineFileContent]
    | tuple[str | None, VolcEngineFileContent, str | None]
    | tuple[str | None, VolcEngineFileContent, str | None, Mapping[str, str]]
)
VolcEngineImageInput: TypeAlias = VolcEngineFile | list[VolcEngineFile]


def _unwrap_image(image: VolcEngineFile) -> tuple[VolcEngineFileContent, str | None]:
    if isinstance(image, tuple):
        content_type = image[2] if len(image) >= 3 else None
        return image[1], content_type
    return image, None


def _read_image_bytes(image: VolcEngineFileContent) -> bytes:
    if isinstance(image, bytes):
        return image
    if isinstance(image, PathLike):
        return Path(image).read_bytes()
    position = image.tell()
    image.seek(0)
    content = image.read()
    image.seek(position)
    return content


def _image_to_data_uri(image: VolcEngineFile) -> str:
    content, explicit_content_type = _unwrap_image(image)
    content_type = explicit_content_type or ImageEditRequestUtils.get_image_content_type(content)
    encoded = base64.b64encode(_read_image_bytes(content)).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


class VolcEngineImageEditConfig(BaseImageEditConfig):
    def __init__(self) -> None:
        self._image_generation_config = VolcEngineImageGenerationConfig()

    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["response_format", "size"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        return dict(image_edit_optional_params)

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
            or get_secret_str("VOLCENGINE_API_KEY")
            or get_secret_str("ARK_API_KEY")
        )
        if not final_api_key:
            raise ValueError("Volcengine API key is required. Set VOLCENGINE_API_KEY or ARK_API_KEY.")
        return get_volcengine_headers(api_key=final_api_key, extra_headers=headers)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],
    ) -> str:
        configured_base = api_base or get_secret_str("VOLCENGINE_API_BASE") or get_secret_str("ARK_API_BASE")
        complete_base = (configured_base or "").rstrip("/")
        if complete_base.endswith("/images/generations"):
            return complete_base
        return f"{get_volcengine_api_base(configured_base)}/images/generations"

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: VolcEngineImageInput | None,
        image_edit_optional_request_params: dict[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> tuple[dict[str, object], RequestFiles]:
        if image is None:
            raise ValueError("Volcengine image edit requires at least one image.")
        images = image if isinstance(image, list) else [image]
        encoded_images = [_image_to_data_uri(item) for item in images]
        request_image: str | list[str] = encoded_images[0] if len(encoded_images) == 1 else encoded_images
        request_body: dict[str, object] = {
            "model": model,
            "prompt": prompt or "",
            "image": request_image,
            **image_edit_optional_request_params,
        }
        return request_body, []

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
        return VolcEngineError(status_code=status_code, message=error_message, headers=response_headers)
