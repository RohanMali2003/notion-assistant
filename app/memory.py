import threading
import time
from typing import Any, Dict, List, Optional


class ConversationMemory:
    """Thread-safe in-memory rolling conversational history and state tracker per sender."""

    def __init__(self, max_history: int = 6, ttl_seconds: int = 1800):
        self._lock = threading.Lock()
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._query_state: Dict[str, Dict[str, Any]] = {}
        self._last_active: Dict[str, float] = {}
        self.max_history = max_history
        self.ttl_seconds = ttl_seconds

    def _cleanup_expired(self, now: float) -> None:
        """Remove sessions inactive for longer than ttl_seconds."""
        expired_senders = [
            s for s, last_t in self._last_active.items()
            if (now - last_t) > self.ttl_seconds
        ]
        for s in expired_senders:
            self._history.pop(s, None)
            self._query_state.pop(s, None)
            self._last_active.pop(s, None)

    def add_user_message(self, sender_id: str, text: str) -> None:
        if not sender_id or not text:
            return
        now = time.time()
        with self._lock:
            self._cleanup_expired(now)
            self._last_active[sender_id] = now
            if sender_id not in self._history:
                self._history[sender_id] = []
            self._history[sender_id].append({
                "role": "user",
                "content": text.strip(),
                "timestamp": now,
            })
            if len(self._history[sender_id]) > self.max_history:
                self._history[sender_id] = self._history[sender_id][-self.max_history:]

    def add_assistant_message(
        self,
        sender_id: str,
        text: str,
        module: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not sender_id or not text:
            return
        now = time.time()
        with self._lock:
            self._cleanup_expired(now)
            self._last_active[sender_id] = now
            if sender_id not in self._history:
                self._history[sender_id] = []
            self._history[sender_id].append({
                "role": "assistant",
                "content": text.strip(),
                "module": module,
                "metadata": metadata or {},
                "timestamp": now,
            })
            if len(self._history[sender_id]) > self.max_history:
                self._history[sender_id] = self._history[sender_id][-self.max_history:]

    def get_history(self, sender_id: str, max_turns: int = 4) -> List[Dict[str, Any]]:
        with self._lock:
            if sender_id not in self._history:
                return []
            return list(self._history[sender_id][-max_turns:])

    def format_context_prompt(self, sender_id: str, max_turns: int = 4) -> str:
        """Formats the last N turns as conversational context for Gemini prompts."""
        turns = self.get_history(sender_id, max_turns=max_turns)
        if not turns:
            return ""

        formatted_lines = []
        for turn in turns:
            role_label = "User" if turn["role"] == "user" else "Assistant"
            content = turn.get("content", "")
            # Truncate very long messages for prompt compactness
            if len(content) > 300:
                content = content[:300] + "..."
            formatted_lines.append(f"{role_label}: {content}")

        return "\n".join(formatted_lines)

    def get_last_query_state(self, sender_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._query_state.get(sender_id, {}))

    def update_query_state(self, sender_id: str, **kwargs) -> None:
        if not sender_id:
            return
        with self._lock:
            if sender_id not in self._query_state:
                self._query_state[sender_id] = {}
            self._query_state[sender_id].update(kwargs)

    def clear(self, sender_id: Optional[str] = None) -> None:
        with self._lock:
            if sender_id:
                self._history.pop(sender_id, None)
                self._query_state.pop(sender_id, None)
                self._last_active.pop(sender_id, None)
            else:
                self._history.clear()
                self._query_state.clear()
                self._last_active.clear()


# Global singleton
conversation_memory = ConversationMemory()
