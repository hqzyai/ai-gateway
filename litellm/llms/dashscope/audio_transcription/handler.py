from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Coroutine, Protocol

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._uuid import uuid
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

from ..common_utils import DashScopeError
from .transformation import (
    FILE_TRANSCRIPTION_MODEL,
    STREAMING_ASR_MODEL,
    DashScopeAudioTranscriptionConfig,
    JsonObject,
)

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


class DashScopeAudioTranscriptionHandler:
    def __init__(
        self,
        sync_connector: SyncConnector | None = None,
        async_connector: AsyncConnector | None = None,
        sync_sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._sync_connector = sync_connector
        self._async_connector = async_connector
        self._sync_sleep = sync_sleep
        self._async_sleep = async_sleep

    def audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        model_response: TranscriptionResponse,
        timeout: float | httpx.Timeout,
        max_retries: int,
        logging_obj: LoggingObject,
        api_key: str | None,
        api_base: str | None,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        atranscription: bool = False,
        headers: Mapping[str, str] | None = None,
        provider_config: DashScopeAudioTranscriptionConfig | None = None,
    ) -> TranscriptionResponse | Coroutine[object, object, TranscriptionResponse]:
        config = provider_config or DashScopeAudioTranscriptionConfig()
        if atranscription:
            return self._async_audio_transcriptions(
                model=model,
                audio_file=audio_file,
                optional_params=optional_params,
                litellm_params=litellm_params,
                timeout=timeout,
                logging_obj=logging_obj,
                api_key=api_key,
                api_base=api_base,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
                headers=headers or {},
                config=config,
            )
        return self._sync_audio_transcriptions(
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
            timeout=timeout,
            logging_obj=logging_obj,
            api_key=api_key,
            api_base=api_base,
            client=client if isinstance(client, HTTPHandler) else None,
            headers=headers or {},
            config=config,
        )

    def _sync_audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        timeout: float | httpx.Timeout,
        logging_obj: LoggingObject,
        api_key: str | None,
        api_base: str | None,
        client: HTTPHandler | None,
        headers: Mapping[str, str],
        config: DashScopeAudioTranscriptionConfig,
    ) -> TranscriptionResponse:
        if model == FILE_TRANSCRIPTION_MODEL:
            return self._sync_file_transcription(
                model,
                audio_file,
                optional_params,
                litellm_params,
                timeout,
                logging_obj,
                api_key,
                api_base,
                client,
                headers,
                config,
            )
        if model == STREAMING_ASR_MODEL:
            return self._sync_streaming_transcription(
                model, audio_file, optional_params, litellm_params, timeout, api_key, api_base, headers, config
            )
        raise ValueError(f"DashScope audio handler does not support model {model}.")

    async def _async_audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        timeout: float | httpx.Timeout,
        logging_obj: LoggingObject,
        api_key: str | None,
        api_base: str | None,
        client: AsyncHTTPHandler | None,
        headers: Mapping[str, str],
        config: DashScopeAudioTranscriptionConfig,
    ) -> TranscriptionResponse:
        if model == FILE_TRANSCRIPTION_MODEL:
            return await self._async_file_transcription(
                model,
                audio_file,
                optional_params,
                litellm_params,
                timeout,
                logging_obj,
                api_key,
                api_base,
                client,
                headers,
                config,
            )
        if model == STREAMING_ASR_MODEL:
            return await self._async_streaming_transcription(
                model, audio_file, optional_params, litellm_params, timeout, api_key, api_base, headers, config
            )
        raise ValueError(f"DashScope audio handler does not support model {model}.")

    def _sync_file_transcription(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        timeout: float | httpx.Timeout,
        logging_obj: LoggingObject,
        api_key: str | None,
        api_base: str | None,
        client: HTTPHandler | None,
        headers: Mapping[str, str],
        config: DashScopeAudioTranscriptionConfig,
    ) -> TranscriptionResponse:
        auth_headers, url, request_data, polling_interval, max_attempts = self._prepare_file_request(
            model, optional_params, litellm_params, api_key, api_base, headers, config
        )
        http_client = client.client if client is not None else HTTPHandler().client
        logging_obj.pre_call(
            input="",
            api_key=api_key,
            additional_args={"complete_input_dict": request_data, "api_base": url, "headers": auth_headers},
        )
        create_response = http_client.post(url=url, headers=auth_headers, json=request_data, timeout=timeout)
        task_id = self._task_id(create_response)
        result_payload = self._sync_poll_file_task(
            http_client,
            config.normalize_api_base(api_base),
            auth_headers,
            task_id,
            polling_interval,
            max_attempts,
            timeout,
        )
        response = self._file_result_response(http_client, result_payload, timeout)
        logging_obj.post_call(input="", api_key=api_key, original_response=result_payload)
        return response

    async def _async_file_transcription(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        timeout: float | httpx.Timeout,
        logging_obj: LoggingObject,
        api_key: str | None,
        api_base: str | None,
        client: AsyncHTTPHandler | None,
        headers: Mapping[str, str],
        config: DashScopeAudioTranscriptionConfig,
    ) -> TranscriptionResponse:
        auth_headers, url, request_data, polling_interval, max_attempts = self._prepare_file_request(
            model, optional_params, litellm_params, api_key, api_base, headers, config
        )
        http_client = client.client if client is not None else AsyncHTTPHandler().client
        logging_obj.pre_call(
            input="",
            api_key=api_key,
            additional_args={"complete_input_dict": request_data, "api_base": url, "headers": auth_headers},
        )
        create_response = await http_client.post(url=url, headers=auth_headers, json=request_data, timeout=timeout)
        task_id = self._task_id(create_response)
        result_payload = await self._async_poll_file_task(
            http_client,
            config.normalize_api_base(api_base),
            auth_headers,
            task_id,
            polling_interval,
            max_attempts,
            timeout,
        )
        response = await self._async_file_result_response(http_client, result_payload, timeout)
        logging_obj.post_call(input="", api_key=api_key, original_response=result_payload)
        return response

    @staticmethod
    def _prepare_file_request(
        model: str,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        api_key: str | None,
        api_base: str | None,
        headers: Mapping[str, str],
        config: DashScopeAudioTranscriptionConfig,
    ) -> tuple[Mapping[str, str], str, JsonObject, float, int]:
        auth_headers = {
            **config.validate_environment(headers, model, [], optional_params, litellm_params, api_key, api_base),
            "X-DashScope-Async": "enable",
        }
        url = config.get_complete_url(api_base, api_key, model, optional_params, litellm_params)
        request_data = config.transform_file_transcription_request(model, optional_params)
        params = config.flatten_optional_params(optional_params)
        polling_interval = DashScopeAudioTranscriptionHandler._non_negative_float(params.get("polling_interval"), 1.0)
        max_attempts = DashScopeAudioTranscriptionHandler._positive_int(params.get("max_polling_attempts"), 600)
        return (
            auth_headers,
            url,
            request_data,
            max(0.0, min(polling_interval, 10.0)),
            max(1, min(max_attempts, 3600)),
        )

    def _sync_poll_file_task(
        self,
        client: httpx.Client,
        api_base: str,
        headers: Mapping[str, str],
        task_id: str,
        polling_interval: float,
        max_attempts: int,
        timeout: float | httpx.Timeout,
    ) -> JsonObject:
        for _ in range(max_attempts):
            response = client.get(url=f"{api_base}/tasks/{task_id}", headers=headers, timeout=timeout)
            payload = self._response_object(response)
            status = self._task_status(payload)
            if status == "SUCCEEDED":
                return payload
            if status in ("FAILED", "CANCELED", "CANCELLED"):
                raise DashScopeError(response.status_code, self._task_error(payload), response.headers)
            self._sync_sleep(polling_interval)
        raise DashScopeError(504, f"DashScope filetrans task {task_id} did not complete in time.")

    async def _async_poll_file_task(
        self,
        client: httpx.AsyncClient,
        api_base: str,
        headers: Mapping[str, str],
        task_id: str,
        polling_interval: float,
        max_attempts: int,
        timeout: float | httpx.Timeout,
    ) -> JsonObject:
        for _ in range(max_attempts):
            response = await client.get(url=f"{api_base}/tasks/{task_id}", headers=headers, timeout=timeout)
            payload = self._response_object(response)
            status = self._task_status(payload)
            if status == "SUCCEEDED":
                return payload
            if status in ("FAILED", "CANCELED", "CANCELLED"):
                raise DashScopeError(response.status_code, self._task_error(payload), response.headers)
            await self._async_sleep(polling_interval)
        raise DashScopeError(504, f"DashScope filetrans task {task_id} did not complete in time.")

    def _file_result_response(
        self,
        client: httpx.Client,
        task_payload: JsonObject,
        timeout: float | httpx.Timeout,
    ) -> TranscriptionResponse:
        result_payloads = tuple(
            self._response_object(client.get(url=url, timeout=timeout))
            for url in self._transcription_urls(task_payload)
        )
        return self._transcripts_to_response(result_payloads, task_payload)

    async def _async_file_result_response(
        self,
        client: httpx.AsyncClient,
        task_payload: JsonObject,
        timeout: float | httpx.Timeout,
    ) -> TranscriptionResponse:
        result_payloads = tuple(
            [
                self._response_object(await client.get(url=url, timeout=timeout))
                for url in self._transcription_urls(task_payload)
            ]
        )
        return self._transcripts_to_response(result_payloads, task_payload)

    def _sync_streaming_transcription(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        timeout: float | httpx.Timeout,
        api_key: str | None,
        api_base: str | None,
        headers: Mapping[str, str],
        config: DashScopeAudioTranscriptionConfig,
    ) -> TranscriptionResponse:
        from websockets.sync.client import connect

        connector = self._sync_connector or connect
        auth_headers = config.validate_environment(
            headers, model, [], optional_params, litellm_params, api_key, api_base
        )
        url = config.websocket_url(api_base)
        params = config.flatten_optional_params(optional_params)
        processed = process_audio_file(audio_file)
        task_id = str(uuid.uuid4()).replace("-", "")
        run_task = self._streaming_run_task(model, task_id, params)
        with connector(url, additional_headers=auth_headers, open_timeout=self._timeout_seconds(timeout)) as websocket:
            websocket.send(json.dumps(run_task))
            self._expect_sync_event(websocket, "task-started", timeout)
            chunk_size = self._positive_int(params.get("chunk_size"), 3200)
            interval = self._non_negative_float(params.get("chunk_interval"), 0.1)
            for offset in range(0, len(processed.file_content), chunk_size):
                websocket.send(processed.file_content[offset : offset + chunk_size])
                if interval:
                    self._sync_sleep(interval)
            websocket.send(json.dumps(self._finish_task(task_id)))
            return self._collect_sync_transcription(websocket, timeout)

    async def _async_streaming_transcription(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: JsonObject,
        litellm_params: JsonObject,
        timeout: float | httpx.Timeout,
        api_key: str | None,
        api_base: str | None,
        headers: Mapping[str, str],
        config: DashScopeAudioTranscriptionConfig,
    ) -> TranscriptionResponse:
        from websockets.asyncio.client import connect

        connector = self._async_connector or connect
        auth_headers = config.validate_environment(
            headers, model, [], optional_params, litellm_params, api_key, api_base
        )
        url = config.websocket_url(api_base)
        params = config.flatten_optional_params(optional_params)
        processed = process_audio_file(audio_file)
        task_id = str(uuid.uuid4()).replace("-", "")
        run_task = self._streaming_run_task(model, task_id, params)
        async with connector(
            url, additional_headers=auth_headers, open_timeout=self._timeout_seconds(timeout)
        ) as websocket:
            await websocket.send(json.dumps(run_task))
            await self._expect_async_event(websocket, "task-started", timeout)
            chunk_size = self._positive_int(params.get("chunk_size"), 3200)
            interval = self._non_negative_float(params.get("chunk_interval"), 0.1)
            for offset in range(0, len(processed.file_content), chunk_size):
                await websocket.send(processed.file_content[offset : offset + chunk_size])
                if interval:
                    await self._async_sleep(interval)
            await websocket.send(json.dumps(self._finish_task(task_id)))
            return await self._collect_async_transcription(websocket, timeout)

    @staticmethod
    def _streaming_run_task(model: str, task_id: str, params: JsonObject) -> JsonObject:
        return {
            "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
            "payload": {
                "task_group": "audio",
                "task": "asr",
                "function": "recognition",
                "model": model,
                "parameters": {
                    "format": params.get("format", "wav"),
                    "sample_rate": params.get("sample_rate", 16000),
                    **({"language_hints": params["language_hints"]} if params.get("language_hints") else {}),
                },
                "input": {},
            },
        }

    @staticmethod
    def _finish_task(task_id: str) -> JsonObject:
        return {
            "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
            "payload": {"input": {}},
        }

    def _expect_sync_event(
        self, websocket: SyncWebSocket, expected_event: str, timeout: float | httpx.Timeout
    ) -> JsonObject:
        while True:
            event = self._event(websocket.recv(self._timeout_seconds(timeout)))
            self._raise_event_error(event)
            if self._event_name(event) == expected_event:
                return event

    async def _expect_async_event(
        self, websocket: AsyncWebSocket, expected_event: str, timeout: float | httpx.Timeout
    ) -> JsonObject:
        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=self._timeout_seconds(timeout))
            event = self._event(message)
            self._raise_event_error(event)
            if self._event_name(event) == expected_event:
                return event

    def _collect_sync_transcription(
        self, websocket: SyncWebSocket, timeout: float | httpx.Timeout
    ) -> TranscriptionResponse:
        sentences = dict[str, str]()
        usage: JsonObject = {}
        while True:
            event = self._event(websocket.recv(self._timeout_seconds(timeout)))
            self._raise_event_error(event)
            event_name = self._event_name(event)
            if event_name == "result-generated":
                key, text = self._sentence(event, len(sentences))
                if text:
                    sentences[key] = text
                usage.update(self._payload_usage(event))
            if event_name == "task-finished":
                usage.update(self._payload_usage(event))
                return self._streaming_response(sentences, usage)

    async def _collect_async_transcription(
        self, websocket: AsyncWebSocket, timeout: float | httpx.Timeout
    ) -> TranscriptionResponse:
        sentences = dict[str, str]()
        usage: JsonObject = {}
        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=self._timeout_seconds(timeout))
            event = self._event(message)
            self._raise_event_error(event)
            event_name = self._event_name(event)
            if event_name == "result-generated":
                key, text = self._sentence(event, len(sentences))
                if text:
                    sentences[key] = text
                usage.update(self._payload_usage(event))
            if event_name == "task-finished":
                usage.update(self._payload_usage(event))
                return self._streaming_response(sentences, usage)

    @staticmethod
    def _streaming_response(sentences: Mapping[str, str], usage: JsonObject) -> TranscriptionResponse:
        response = TranscriptionResponse(text="".join(sentences.values()))
        object.__setattr__(
            response,
            "_hidden_params",
            {
                "usage": usage,
                **(
                    {"audio_transcription_duration": usage["duration"]}
                    if isinstance(usage.get("duration"), (int, float))
                    else {}
                ),
                "custom_llm_provider": "dashscope",
            },
        )
        return response

    @staticmethod
    def _event(message: str | bytes) -> JsonObject:
        if isinstance(message, bytes):
            raise DashScopeError(502, "DashScope ASR returned unexpected binary data while an event was expected.")
        try:
            return JSON_OBJECT_ADAPTER.validate_json(message)
        except ValidationError as exc:
            raise DashScopeError(502, f"Invalid DashScope ASR event: {exc}") from exc

    @staticmethod
    def _event_name(event: JsonObject) -> str:
        header = event.get("header")
        return str(header.get("event", "")) if isinstance(header, dict) else ""

    @staticmethod
    def _raise_event_error(event: JsonObject) -> None:
        header = event.get("header")
        if not isinstance(header, dict) or header.get("event") != "task-failed":
            return
        raise DashScopeError(502, str(header.get("error_message") or header.get("error_code") or "ASR failed"))

    @staticmethod
    def _sentence(event: JsonObject, fallback_index: int) -> tuple[str, str]:
        payload = event.get("payload")
        output = payload.get("output") if isinstance(payload, dict) else None
        sentence = output.get("sentence") if isinstance(output, dict) else None
        if not isinstance(sentence, dict):
            return str(fallback_index), ""
        key = str(sentence.get("sentence_id", sentence.get("begin_time", fallback_index)))
        return key, str(sentence.get("text", ""))

    @staticmethod
    def _payload_usage(event: JsonObject) -> JsonObject:
        payload = event.get("payload")
        usage = payload.get("usage") if isinstance(payload, dict) else None
        try:
            return JSON_OBJECT_ADAPTER.validate_python(usage)
        except ValidationError:
            return {}

    @staticmethod
    def _task_id(response: httpx.Response) -> str:
        payload = DashScopeAudioTranscriptionHandler._response_object(response)
        output = payload.get("output")
        task_id = output.get("task_id") if isinstance(output, dict) else None
        if not isinstance(task_id, str) or not task_id:
            raise DashScopeError(
                response.status_code, f"DashScope filetrans response did not contain task_id: {payload}"
            )
        return task_id

    @staticmethod
    def _task_status(payload: JsonObject) -> str:
        output = payload.get("output")
        return str(output.get("task_status", "")) if isinstance(output, dict) else ""

    @staticmethod
    def _task_error(payload: JsonObject) -> str:
        output = payload.get("output")
        if not isinstance(output, dict):
            return "DashScope filetrans failed."
        return str(output.get("message") or output.get("code") or "DashScope filetrans failed.")

    @staticmethod
    def _transcription_urls(payload: JsonObject) -> tuple[str, ...]:
        output = payload.get("output")
        results = output.get("results") if isinstance(output, dict) else None
        if not isinstance(results, list):
            raise DashScopeError(502, "DashScope filetrans result did not contain output.results.")
        urls = tuple(
            url
            for result in results
            if isinstance(result, dict)
            for url in (result.get("transcription_url"),)
            if isinstance(url, str)
        )
        if not urls:
            raise DashScopeError(502, "DashScope filetrans result did not contain a transcription URL.")
        return urls

    @staticmethod
    def _transcripts_to_response(
        result_payloads: tuple[JsonObject, ...], task_payload: JsonObject
    ) -> TranscriptionResponse:
        texts = tuple(
            str(transcript.get("text", ""))
            for payload in result_payloads
            for raw_transcripts in (payload.get("transcripts"),)
            if isinstance(raw_transcripts, list)
            for transcript in raw_transcripts
            if isinstance(transcript, dict)
        )
        response = TranscriptionResponse(text="\n".join(text for text in texts if text))
        usage = task_payload.get("usage")
        usage_object = usage if isinstance(usage, dict) else {}
        object.__setattr__(
            response,
            "_hidden_params",
            {
                "task": task_payload,
                "usage": usage_object,
                **(
                    {"audio_transcription_duration": usage_object["duration"]}
                    if isinstance(usage_object.get("duration"), (int, float))
                    else {}
                ),
                "custom_llm_provider": "dashscope",
            },
        )
        return response

    @staticmethod
    def _response_object(response: httpx.Response) -> JsonObject:
        if response.status_code >= 400:
            raise DashScopeError(response.status_code, response.text, response.headers)
        try:
            payload = JSON_OBJECT_ADAPTER.validate_json(response.content)
        except ValidationError as exc:
            raise DashScopeError(
                response.status_code, f"Invalid DashScope JSON response: {exc}", response.headers
            ) from exc
        if "code" in payload and "output" not in payload:
            raise DashScopeError(response.status_code, str(payload.get("message") or payload["code"]), response.headers)
        return payload

    @staticmethod
    def _positive_int(value: object, default: int) -> int:
        if not isinstance(value, (str, int, float)):
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    @staticmethod
    def _non_negative_float(value: object, default: float) -> float:
        if not isinstance(value, (str, int, float)):
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= 0 else default

    @staticmethod
    def _timeout_seconds(timeout: float | httpx.Timeout) -> float | None:
        if isinstance(timeout, (int, float)):
            return float(timeout)
        return timeout.read
