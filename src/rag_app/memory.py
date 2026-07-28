from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import redis


class RedisChatMemory:
    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 3600,
        max_messages: int = 20,
    ) -> None:
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages

    def _key(self, conversation_id: str) -> str:
        return f"rag:chat:{conversation_id}:messages"

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        items = self.client.lrange(self._key(conversation_id), 0, -1)
        return [json.loads(item) for item in items]

    def append_message(self, conversation_id: str, role: str, content: str) -> None:
        key = self._key(conversation_id)
        payload = {
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.client.rpush(key, json.dumps(payload, ensure_ascii=False))
        self.client.ltrim(key, -self.max_messages, -1)
        self.client.expire(key, self.ttl_seconds)

    def append_turn(self, conversation_id: str, question: str, answer: str) -> None:
        self.append_message(conversation_id, "user", question)
        self.append_message(conversation_id, "assistant", answer)

    def clear(self, conversation_id: str) -> None:
        self.client.delete(self._key(conversation_id))
