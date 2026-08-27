from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Coroutine, Protocol

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._uuid import uuid
from litellm.types.llms.openai import HttpxBinaryResponseContent

from ..common_utils import DashScopeError
from .transformation import DashScopeTextToSpeechConfig, JsonObject

JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)


class SyncWebSocket(Protocol):
    def send(self, message: str | bytes) -> None: ...

    def recv(self, timeout: float | None = None) -> str | bytes: ...


class AsyncWebSocket(Protocol):
    async def send(self, message: str | bytes) -> None: ...

    async def recv(self, decode: bool | None = None) -> str | bytes: ...


class SyncConnector(Protocol):
    def __call__(
        self,
        uri: str,
        *,
        additional_headers: Mapping[str, str],
        open_timeout: float | None,
    ) -> AbstractContextManager[SyncWebSocket]: ...


class AsyncConnector(Protocol):
    def __call__(
        self,
        uri: str,
        *,
        additional_headers: Mapping[str, str],
        open_timeout: float | None,
    ) -> AbstractAsyncContextManager[AsyncWebSocket]: ...


class LoggingObject(Protocol):
    def pre_call(
        self,
        *,
        input: object,
        api_key: str | None,
        additional_args: object,
    ) -> object: ...

    def post_call(
        self,
        *,
        input: object,
        api_key: str | None,
        original_response: object,
    ) -> object: ...


class DashScopeTextToSpeechHandler:
    def __init__(
        self,
        sync_connector: SyncConnector | None = None,
        async_connector: AsyncConnector | None = None,
    ) -> None:
        self._sync_connector = sync_connector
        self._async_connector = async_connector

    def text_to_speech(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        logging_obj: LoggingObject,
        timeout: float | httpx.Timeout,
        api_key: str | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None = None,
        aspeech: bool = False,
        config: DashScopeTextToSpeechConfig | None = None,
    ) -> HttpxBinaryResponseContent | Coroutine[object, object, HttpxBinaryResponseContent]:
        resolved_config = config or DashScopeTextToSpeechConfig()
        if aspeech:
            return self._async_text_to_speech(
                model,
                input,
                voice,
                optional_params,
                litellm_params,
                logging_obj,
                timeout,
                api_key,
                api_base,
                extra_headers or {},
                resolved_config,
            )
        return self._sync_text_to_speech(
            model,
            input,
            voice,
            optional_params,
            litellm_params,
            logging_obj,
            timeout,
            api_key,
            api_base,
            extra_headers or {},
            resolved_config,
        )

    def _sync_text_to_speech(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        logging_obj: LoggingObject,
        timeout: float | httpx.Timeout,
        api_key: str | None,
        api_base: str | None,
        extra_headers: Mapping[str, str],
        config: DashScopeTextToSpeechConfig,
    ) -> HttpxBinaryResponseContent:
        from websockets.sync.client import connect

        connector = self._sync_connector or connect
        headers = config.validate_environment(extra_headers, model, api_key, api_base)
        url = config.get_complete_url(model, api_base, self._json_params(litellm_params))
        task_id, run_task, continue_task, finish_task = self._events(model, input, voice, optional_params)
        logging_obj.pre_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": run_task, "api_base": url, "headers": headers},
        )
        with connector(url, additional_headers=headers, open_timeout=self._timeout_seconds(timeout)) as websocket:
            websocket.send(json.dumps(run_task))
            self._expect_sync_event(websocket, "task-started", timeout)
            websocket.send(json.dumps(continue_task))
            websocket.send(json.dumps(finish_task))
            audio, usage = self._collect_sync_audio(websocket, timeout)
        logging_obj.post_call(input=input, api_key=api_key, original_response={"usage": usage, "task_id": task_id})
        return self._binary_response(audio, optional_params, usage)

    async def _async_text_to_speech(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        logging_obj: LoggingObject,
        timeout: float | httpx.Timeout,
        api_key: str | None,
        api_base: str | None,
        extra_headers: Mapping[str, str],
        config: DashScopeTextToSpeechConfig,
    ) -> HttpxBinaryResponseContent:
        from websockets.asyncio.client import connect

        connector = self._async_connector or connect
        headers = config.validate_environment(extra_headers, model, api_key, api_base)
        url = config.get_complete_url(model, api_base, self._json_params(litellm_params))
        task_id, run_task, continue_task, finish_task = self._events(model, input, voice, optional_params)
        logging_obj.pre_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": run_task, "api_base": url, "headers": headers},
        )
        async with connector(url, additional_headers=headers, open_timeout=self._timeout_seconds(timeout)) as websocket:
            await websocket.send(json.dumps(run_task))
            await self._expect_async_event(websocket, "task-started", timeout)
            await websocket.send(json.dumps(continue_task))
            await websocket.send(json.dumps(finish_task))
            audio, usage = await self._collect_async_audio(websocket, timeout)
        logging_obj.post_call(input=input, api_key=api_key, original_response={"usage": usage, "task_id": task_id})
        return self._binary_response(audio, optional_params, usage)

    @staticmethod
    def _events(
        model: str,
        input: str,
        voice: str | None,
        optional_params: JsonObject,
    ) -> tuple[str, JsonObject, JsonObject, JsonObject]:
        task_id = str(uuid.uuid4()).replace("-", "")
        run_task: JsonObject = {
            "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
            "payload": {
                "task_group": "audio",
                "task": "tts",
                "function": "SpeechSynthesizer",
                "model": model,
                "parameters": {
                    "text_type": optional_params.get("text_type", "PlainText"),
                    "voice": voice or "longanlingxi",
                    "format": optional_params.get("format", "mp3"),
                    "sample_rate": optional_params.get("sample_rate", 22050),
                    "volume": optional_params.get("volume", 50),
                    "rate": optional_params.get("rate", 1.0),
                    "pitch": optional_params.get("pitch", 1.0),
                    "enable_ssml": optional_params.get("enable_ssml", False),
                    **{
                        key: value
                        for key, value in optional_params.items()
                        if key in DashScopeTextToSpeechConfig.PROVIDER_PARAMS and value is not None
                    },
                },
                "input": {},
            },
        }
        continue_task: JsonObject = {
            "header": {"action": "continue-task", "task_id": task_id, "streaming": "duplex"},
            "payload": {"input": {"text": input}},
        }
        finish_task: JsonObject = {
            "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
            "payload": {"input": {}},
        }
        return task_id, run_task, continue_task, finish_task

    def _expect_sync_event(
        self, websocket: SyncWebSocket, expected_event: str, timeout: float | httpx.Timeout
    ) -> JsonObject:
        while True:
            message = websocket.recv(self._timeout_seconds(timeout))
            if isinstance(message, bytes):
                continue
            event = self._event(message)
            self._raise_event_error(event)
            if self._event_name(event) == expected_event:
                return event

    async def _expect_async_event(
        self, websocket: AsyncWebSocket, expected_event: str, timeout: float | httpx.Timeout
    ) -> JsonObject:
        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=self._timeout_seconds(timeout))
            if isinstance(message, bytes):
                continue
            event = self._event(message)
            self._raise_event_error(event)
            if self._event_name(event) == expected_event:
                return event

    def _collect_sync_audio(self, websocket: SyncWebSocket, timeout: float | httpx.Timeout) -> tuple[bytes, JsonObject]:
        chunks: list[bytes] = []
        while True:
            message = websocket.recv(self._timeout_seconds(timeout))
            if isinstance(message, bytes):
                chunks.append(message)
                continue
            event = self._event(message)
            self._raise_event_error(event)
            if self._event_name(event) == "task-finished":
                audio = b"".join(chunks)
                if not audio:
                    raise DashScopeError(502, "DashScope TTS completed without audio data.")
                return audio, self._usage(event)

    async def _collect_async_audio(
        self, websocket: AsyncWebSocket, timeout: float | httpx.Timeout
    ) -> tuple[bytes, JsonObject]:
        chunks: list[bytes] = []
        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=self._timeout_seconds(timeout))
            if isinstance(message, bytes):
                chunks.append(message)
                continue
            event = self._event(message)
            self._raise_event_error(event)
            if self._event_name(event) == "task-finished":
                audio = b"".join(chunks)
                if not audio:
                    raise DashScopeError(502, "DashScope TTS completed without audio data.")
                return audio, self._usage(event)

    @staticmethod
    def _binary_response(
        audio: bytes,
        optional_params: JsonObject,
        usage: JsonObject,
    ) -> HttpxBinaryResponseContent:
        audio_format = optional_params.get("format", "mp3")
        content_type = DashScopeTextToSpeechConfig.CONTENT_TYPES.get(str(audio_format), "application/octet-stream")
        result = HttpxBinaryResponseContent(
            httpx.Response(
                status_code=200,
                headers={"Content-Type": content_type, "Content-Length": str(len(audio))},
                content=audio,
            )
        )
        object.__setattr__(result, "_hidden_params", {"usage": usage, "custom_llm_provider": "dashscope"})
        return result

    @staticmethod
    def _event(message: str) -> JsonObject:
        try:
            return JSON_OBJECT_ADAPTER.validate_json(message)
        except ValidationError as exc:
            raise DashScopeError(502, f"Invalid DashScope TTS event: {exc}") from exc

    @staticmethod
    def _event_name(event: JsonObject) -> str:
        header = event.get("header")
        return str(header.get("event", "")) if isinstance(header, dict) else ""

    @staticmethod
    def _raise_event_error(event: JsonObject) -> None:
        header = event.get("header")
        if not isinstance(header, dict) or header.get("event") != "task-failed":
            return
        raise DashScopeError(502, str(header.get("error_message") or header.get("error_code") or "TTS failed"))

    @staticmethod
    def _usage(event: JsonObject) -> JsonObject:
        payload = event.get("payload")
        usage = payload.get("usage") if isinstance(payload, dict) else None
        try:
            return JSON_OBJECT_ADAPTER.validate_python(usage)
        except ValidationError:
            return {}

    @staticmethod
    def _json_params(params: JsonObject) -> JsonObject:
        return {
            key: value for key, value in params.items() if isinstance(value, (str, int, float, bool)) or value is None
        }

    @staticmethod
    def _timeout_seconds(timeout: float | httpx.Timeout) -> float | None:
        if isinstance(timeout, (int, float)):
            return float(timeout)
        return timeout.read
