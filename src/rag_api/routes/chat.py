from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from rag_api.dependencies.services import get_chat_memory, get_rag_service
from rag_api.schemas.chat import ChatRequest, ChatResponse, ChatStopRequest, SourceResponse
from rag_api.schemas.common import BotId
from rag_app.memory import RedisChatMemory
from rag_app.models import RetrievedDocument
from rag_app.rag import parse_focused_answer
from rag_app.services import RagService


router = APIRouter(prefix="/v1", tags=["chat"])
logger = logging.getLogger("rag_api")


def _source_to_response(source: RetrievedDocument) -> SourceResponse:
    metadata = dict(source.metadata or {})
    title = metadata.get("title")
    document_id = metadata.get("document_id") or metadata.get("doc_id") or metadata.get("source_doc_id")
    source_type = metadata.get("source_type") or metadata.get("source")
    preview = source.text[:500]
    if len(source.text) > 500:
        preview = f"{preview}..."

    return SourceResponse(
        id=source.id,
        title=str(title) if title not in (None, "") else None,
        document_id=str(document_id) if document_id not in (None, "") else None,
        source_type=str(source_type) if source_type not in (None, "") else None,
        score=source.score,
        text_preview=preview,
        metadata=metadata,
    )


def _dump_model(model: SourceResponse) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


async def _stream_chunks(chunks: Iterable[str] | AsyncIterable[str]) -> AsyncIterator[str]:
    if isinstance(chunks, AsyncIterable):
        async for chunk in chunks:
            yield chunk
        return
    for chunk in chunks:
        yield chunk


async def _close_stream(chunks: object) -> None:
    close = getattr(chunks, "aclose", None) or getattr(chunks, "close", None)
    if not callable(close):
        return
    result = close()
    if hasattr(result, "__await__"):
        await result


def _load_history(
    memory: RedisChatMemory,
    bot_id: str,
    conversation_id: str | None,
) -> list[dict[str, Any]]:
    if not conversation_id:
        return []
    try:
        return memory.get_messages(_memory_conversation_id(bot_id, conversation_id))
    except Exception:
        return []


def _remember_turn(
    memory: RedisChatMemory,
    bot_id: str,
    conversation_id: str | None,
    question: str,
    answer: str,
) -> None:
    if not conversation_id:
        return
    try:
        memory.append_turn(_memory_conversation_id(bot_id, conversation_id), question, answer)
    except Exception:
        pass


def _memory_conversation_id(bot_id: str, conversation_id: str) -> str:
    """Namespace Redis history without changing the public conversation ID."""

    return f"{len(bot_id)}:{bot_id}:{conversation_id}"


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: RagService = Depends(get_rag_service),
    memory: RedisChatMemory = Depends(get_chat_memory),
) -> ChatResponse:
    question = request.question.strip()
    history = _load_history(memory, request.bot_id, request.conversation_id)
    response = service.answer(
        bot_id=request.bot_id,
        question=question,
        mode=request.mode,
        top_k=request.top_k,
        fetch_k=max(request.fetch_k, request.top_k),
        mmr_lambda=request.mmr_lambda,
        history=history,
    )
    _remember_turn(memory, request.bot_id, request.conversation_id, question, response.answer)
    return ChatResponse(
        bot_id=request.bot_id,
        answer=response.answer,
        mode=response.mode,
        sources=[_source_to_response(source) for source in response.sources],
        diagnostics=response.diagnostics,
        conversation_id=request.conversation_id,
    )


@router.post("/chat/stream")
async def chat_stream(
    http_request: Request,
    request: ChatRequest,
    service: RagService = Depends(get_rag_service),
    memory: RedisChatMemory = Depends(get_chat_memory),
) -> StreamingResponse:
    question = request.question.strip()
    history = _load_history(memory, request.bot_id, request.conversation_id)
    response = service.answer_stream(
        bot_id=request.bot_id,
        question=question,
        mode=request.mode,
        top_k=request.top_k,
        fetch_k=max(request.fetch_k, request.top_k),
        mmr_lambda=request.mmr_lambda,
        history=history,
    )
    sources = [_source_to_response(source) for source in response.sources]

    async def events():
        chunks = response.chunks
        try:
            if await http_request.is_disconnected():
                logger.info("chat stream cancelled before provider stream started")
                return
            yield _sse_event(
                "metadata",
                {
                    "bot_id": request.bot_id,
                    "mode": response.mode,
                    "sources": [_dump_model(source) for source in sources],
                    "diagnostics": response.diagnostics,
                },
            )

            answer_parts: list[str] = []
            async for chunk in _stream_chunks(chunks):
                if await http_request.is_disconnected():
                    logger.info("chat stream cancelled by client")
                    return
                if not chunk:
                    continue
                answer_parts.append(chunk)
                yield _sse_event("token", {"token": chunk})
            answer = parse_focused_answer("".join(answer_parts))
            _remember_turn(
                memory,
                request.bot_id,
                request.conversation_id,
                question,
                answer,
            )
            yield _sse_event(
                "done",
                {
                    "answer": answer,
                    "bot_id": request.bot_id,
                    "conversation_id": request.conversation_id,
                },
            )
        except asyncio.CancelledError:
            # Starlette cancels this task when the SSE client disconnects.
            # Re-raise so cancellation reaches the async provider iterator.
            logger.info("chat stream task cancelled by client disconnect")
            raise
        except Exception as exc:
            yield _sse_event(
                "error",
                {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
        finally:
            # This runs for normal completion and cancellation, closing the
            # Ollama/Google async HTTP stream in its own finally block.
            await _close_stream(chunks)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.delete("/chat/{conversation_id}/memory")
def clear_chat_memory(
    conversation_id: str,
    bot_id: BotId,
    memory: RedisChatMemory = Depends(get_chat_memory),
) -> dict[str, str]:
    memory.clear(_memory_conversation_id(bot_id, conversation_id))
    return {"status": "ok", "bot_id": bot_id}


@router.post("/chat/stop")
def stop_chat(
    request: ChatStopRequest,
    memory: RedisChatMemory = Depends(get_chat_memory),
) -> dict[str, str]:
    memory.clear(_memory_conversation_id(request.bot_id, request.conversation_id))
    return {"status": "ok", "bot_id": request.bot_id}
