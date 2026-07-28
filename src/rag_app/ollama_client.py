from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Iterator
from typing import Any

from .config import AppConfig


class OllamaConfigurationError(RuntimeError):
    pass


class GoogleConfigurationError(RuntimeError):
    pass


def _message_content_raw(response: Any) -> str:
    if isinstance(response, dict):
        return str((response.get("message") or {}).get("content") or "")
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _message_content(response: Any) -> str:
    return _message_content_raw(response).strip()


def _build_chat_request(
    config: AppConfig,
    messages: list[dict[str, str]],
    *,
    response_format: str | None = None,
    options_override: dict[str, Any] | None = None,
    stream: bool,
) -> tuple[Any, dict[str, Any]]:
    from ollama import Client

    headers = {}
    if config.uses_direct_ollama_cloud:
        if not config.ollama_api_key:
            raise OllamaConfigurationError(
                "OLLAMA_API_KEY is required when OLLAMA_HOST is https://ollama.com."
            )
        headers["Authorization"] = f"Bearer {config.ollama_api_key}"

    client = Client(host=config.ollama_host, headers=headers)
    options = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "num_ctx": config.num_ctx,
    }
    if options_override:
        options.update(options_override)

    request: dict[str, Any] = {
        "model": config.ollama_model,
        "messages": messages,
        "stream": stream,
        "keep_alive": config.ollama_keep_alive,
        "options": options,
    }
    if response_format:
        request["format"] = response_format

    return client, request


def _google_prompt_parts(messages: list[dict[str, str]]) -> tuple[str | None, str]:
    system_parts: list[str] = []
    input_parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().lower()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            input_parts.append(f"Assistant: {content}")
        else:
            input_parts.append(f"User: {content}")
    return ("\n\n".join(system_parts) or None), "\n\n".join(input_parts)


def _build_google_request(
    config: AppConfig,
    messages: list[dict[str, str]],
) -> tuple[Any, dict[str, Any]]:
    if not config.google_api_key:
        raise GoogleConfigurationError("GOOGLE_API_KEY is required when RAG_ANSWER_PROVIDER=google.")

    from google import genai

    system_instruction, prompt = _google_prompt_parts(messages)
    client = genai.Client(api_key=config.google_api_key)
    generation_config: dict[str, Any] = {
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    if system_instruction:
        generation_config["system_instruction"] = system_instruction
    request: dict[str, Any] = {
        "model": config.google_model,
        "contents": prompt,
        "config": generation_config,
    }
    return client, request


def _google_output_text(response: Any, *, strip: bool = True) -> str:
    for attr in ("text", "output_text"):
        try:
            text = getattr(response, attr, None)
        except Exception:
            text = None
        if text:
            content = str(text)
            return content.strip() if strip else content

    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(str(text))
    content = "".join(parts)
    return content.strip() if strip else content


def _google_stream_text(event: Any) -> str:
    delta = getattr(event, "delta", None)
    if delta is not None:
        delta_type = getattr(delta, "type", None)
        text = getattr(delta, "text", None)
        if delta_type == "text" and text:
            return str(text)
    text = getattr(event, "text", None)
    if text:
        return str(text)
    return ""


def chat(
    config: AppConfig,
    messages: list[dict[str, str]],
    *,
    response_format: str | None = None,
    options_override: dict[str, Any] | None = None,
    provider: str = "ollama",
) -> str:
    provider = (provider or "ollama").strip().lower()
    if provider == "google":
        client, request = _build_google_request(config, messages)
        try:
            response = client.models.generate_content(**request)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        content = _google_output_text(response)
        if not content:
            raise RuntimeError("Google returned an empty response.")
        return content
    if provider != "ollama":
        raise ValueError(f"Unsupported answer provider '{provider}'. Use 'ollama' or 'google'.")

    client, request = _build_chat_request(
        config,
        messages,
        response_format=response_format,
        options_override=options_override,
        stream=False,
    )
    response = client.chat(**request)
    content = _message_content(response)
    if not content:
        raise RuntimeError("Ollama returned an empty response.")
    return content


async def _close_async_resource(resource: Any) -> None:
    for method_name in ("aclose", "close"):
        close = getattr(resource, method_name, None)
        if not callable(close):
            continue
        result = close()
        if inspect.isawaitable(result):
            await result
        return


async def chat_stream(
    config: AppConfig,
    messages: list[dict[str, str]],
    *,
    response_format: str | None = None,
    options_override: dict[str, Any] | None = None,
    provider: str = "ollama",
) -> AsyncIterator[str]:
    provider = (provider or "ollama").strip().lower()
    if provider == "google":
        client, request = _build_google_request(config, messages)
        async_client = client.aio
        saw_content = False
        try:
            stream = await async_client.models.generate_content_stream(**request)
            async for event in stream:
                content = _google_output_text(event, strip=False) or _google_stream_text(event)
                if not content:
                    continue
                saw_content = True
                yield content
        finally:
            await _close_async_resource(async_client)
        if not saw_content:
            raise RuntimeError("Google returned an empty response.")
        return
    if provider != "ollama":
        raise ValueError(f"Unsupported answer provider '{provider}'. Use 'ollama' or 'google'.")

    from ollama import AsyncClient

    headers = {}
    if config.uses_direct_ollama_cloud:
        if not config.ollama_api_key:
            raise OllamaConfigurationError(
                "OLLAMA_API_KEY is required when OLLAMA_HOST is https://ollama.com."
            )
        headers["Authorization"] = f"Bearer {config.ollama_api_key}"

    client = AsyncClient(host=config.ollama_host, headers=headers)
    options = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "num_ctx": config.num_ctx,
    }
    if options_override:
        options.update(options_override)
    request: dict[str, Any] = {
        "model": config.ollama_model,
        "messages": messages,
        "stream": True,
        "keep_alive": config.ollama_keep_alive,
        "options": options,
    }
    if response_format:
        request["format"] = response_format

    saw_content = False
    try:
        stream = await client.chat(**request)
        async for chunk in stream:
            content = _message_content_raw(chunk)
            if not content:
                continue
            saw_content = True
            yield content
        if not saw_content:
            raise RuntimeError("Ollama returned an empty response.")
    finally:
        await _close_async_resource(client)
