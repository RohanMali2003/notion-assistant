import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

CACHE_FILE_PATH = os.path.join("data", "session_cache.json")


class ConversationMemory:
    """Thread-safe persistent rolling conversational history, mutation audit stack, and state tracker per sender."""

    def __init__(self, max_history: int = 6, ttl_seconds: int = 86400, cache_file: str = CACHE_FILE_PATH):
        self._lock = threading.Lock()
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._query_state: Dict[str, Dict[str, Any]] = {}
        self._mutations: Dict[str, List[Dict[str, Any]]] = {}
        self._last_active: Dict[str, float] = {}
        self.max_history = max_history
        self.max_mutations = 10
        self.ttl_seconds = ttl_seconds
        self.cache_file = cache_file

        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self._load_disk_cache()

    def _load_disk_cache(self) -> None:
        """Load session history, mutations, and state from disk file on startup."""
        if not os.path.exists(self.cache_file):
            return
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._history = data.get("history", {})
                self._query_state = data.get("query_state", {})
                self._mutations = data.get("mutations", {})
                self._last_active = data.get("last_active", {})
        except Exception:
            pass

    def _save_disk_cache(self) -> None:
        """Save session history, mutations, and state to disk file."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "history": self._history,
                    "query_state": self._query_state,
                    "mutations": self._mutations,
                    "last_active": self._last_active,
                }, f, ensure_ascii=False)
        except Exception:
            pass

    def _cleanup_expired(self, now: float) -> None:
        """Remove sessions inactive for longer than ttl_seconds."""
        expired_senders = [
            s for s, last_t in self._last_active.items()
            if (now - last_t) > self.ttl_seconds
        ]
        for s in expired_senders:
            self._history.pop(s, None)
            self._query_state.pop(s, None)
            self._mutations.pop(s, None)
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
            self._save_disk_cache()

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
            self._save_disk_cache()

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

    def set_last_query_results(self, sender_id: str, results: List[Dict[str, Any]]) -> None:
        """Store the list of task dicts returned by the latest query for ordinal referencing."""
        if not sender_id:
            return
        with self._lock:
            if sender_id not in self._query_state:
                self._query_state[sender_id] = {}
            self._query_state[sender_id]["last_results"] = results
            self._save_disk_cache()

    def get_last_query_results(self, sender_id: str) -> List[Dict[str, Any]]:
        """Retrieve stored task dicts from latest task query for sender."""
        if not sender_id:
            return []
        with self._lock:
            return list(self._query_state.get(sender_id, {}).get("last_results", []))

    def set_pending_menu(self, sender_id: str, menu_payload: Dict[str, Any]) -> None:
        """Store pending interactive selection menu options for ambiguous matching resolution."""
        if not sender_id:
            return
        with self._lock:
            if sender_id not in self._query_state:
                self._query_state[sender_id] = {}
            self._query_state[sender_id]["pending_menu"] = menu_payload
            self._save_disk_cache()

    def get_pending_menu(self, sender_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve active pending selection menu payload for sender if present."""
        if not sender_id:
            return None
        with self._lock:
            return self._query_state.get(sender_id, {}).get("pending_menu")

    def clear_pending_menu(self, sender_id: str) -> None:
        """Clear active pending menu payload for sender."""
        if not sender_id:
            return
        with self._lock:
            if sender_id in self._query_state:
                self._query_state[sender_id].pop("pending_menu", None)
                self._save_disk_cache()

    def record_mutation(
        self,
        sender_id: str,
        action_type: str,
        target_title: str = "",
        affected_items: Optional[List[Dict[str, Any]]] = None,
        rollback_data: Optional[Dict[str, Any]] = None,
        summary: str = "",
    ) -> Dict[str, Any]:
        """Record an atomic mutating side-effect (task created, row added, property modified) for rollback."""
        if not sender_id:
            return {}
        now = time.time()
        mutation_id = uuid.uuid4().hex[:8]
        record = {
            "mutation_id": mutation_id,
            "timestamp": now,
            "action_type": action_type,
            "target_title": target_title,
            "affected_items": affected_items or [],
            "rollback_data": rollback_data or {},
            "summary": summary or f"{action_type} on {target_title}",
        }
        with self._lock:
            if sender_id not in self._mutations:
                self._mutations[sender_id] = []
            self._mutations[sender_id].append(record)
            if len(self._mutations[sender_id]) > self.max_mutations:
                self._mutations[sender_id] = self._mutations[sender_id][-self.max_mutations:]
            self._save_disk_cache()
        return record

    def get_last_mutation(self, sender_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent mutation record for sender without popping."""
        if not sender_id:
            return None
        with self._lock:
            records = self._mutations.get(sender_id, [])
            return dict(records[-1]) if records else None

    def pop_last_mutation(self, sender_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve and remove the most recent mutation record for sender after rolling back."""
        if not sender_id:
            return None
        with self._lock:
            records = self._mutations.get(sender_id, [])
            if records:
                popped = records.pop()
                self._save_disk_cache()
                return popped
            return None

    def list_recent_mutations(self, sender_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve list of recent mutation records for sender."""
        if not sender_id:
            return []
        with self._lock:
            records = self._mutations.get(sender_id, [])
            return list(records[-limit:])

    def clear(self, sender_id: Optional[str] = None) -> None:
        with self._lock:
            if sender_id:
                self._history.pop(sender_id, None)
                self._query_state.pop(sender_id, None)
                self._mutations.pop(sender_id, None)
                self._last_active.pop(sender_id, None)
            else:
                self._history.clear()
                self._query_state.clear()
                self._mutations.clear()
                self._last_active.clear()


# Global singleton
conversation_memory = ConversationMemory()
