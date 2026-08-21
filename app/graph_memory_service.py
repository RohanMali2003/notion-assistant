"""Ocean v3.1: Persistent Knowledge Graph & Memory Governance Engine.

Provides:
1. SQLite-backed local entity-relation graph DB (`data/ocean_graph.db`).
2. Node lifecycle management (`ACTIVE`, `SUPERSEDED`, `COMPLETED`, `DEPRECATED`).
3. Automatic entity superseding & conversational memory commands ('forget', 'update memory', 'inspect memory').
4. Fast graph context retrieval for Gemini prompts.
"""

import json
import logging
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from contextlib import contextmanager
from app.ai import DEFAULT_GEMINI_MODEL, get_gemini_client, get_gemini_model, get_genai_types

logger = logging.getLogger("notion-assistant.graph_memory")
DEFAULT_DB_PATH = os.path.join("data", "ocean_graph.db")



class GraphMemoryService:
    """SQLite-backed Knowledge Graph Service with temporal recency & memory governance."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize SQLite database tables for nodes and edges."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    entity_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    summary TEXT,
                    metadata_json TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    created_at REAL,
                    FOREIGN KEY(source_id) REFERENCES nodes(id),
                    FOREIGN KEY(target_id) REFERENCES nodes(id)
                )
            """)
            conn.commit()

    def add_node(
        self,
        name: str,
        entity_type: str = "TOPIC",
        summary: str = "",
        status: str = "ACTIVE",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Insert or update an entity node in the Knowledge Graph."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Node name cannot be empty")

        node_id = re.sub(r"[^\w]+", "_", clean_name.lower()).strip("_")
        now = time.time()
        metadata_str = json.dumps(metadata or {})

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, created_at FROM nodes WHERE name = ? OR id = ?", (clean_name, node_id))
            row = cursor.fetchone()

            if row:
                created_at = row["created_at"]
                cursor.execute(
                    """
                    UPDATE nodes
                    SET entity_type = ?, status = ?, summary = ?, metadata_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (entity_type, status, summary, metadata_str, now, row["id"]),
                )
                actual_id = row["id"]
            else:
                created_at = now
                cursor.execute(
                    """
                    INSERT INTO nodes (id, name, entity_type, status, summary, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (node_id, clean_name, entity_type, status, summary, metadata_str, created_at, now),
                )
                actual_id = node_id

            conn.commit()

        return {
            "id": actual_id,
            "name": clean_name,
            "entity_type": entity_type,
            "status": status,
            "summary": summary,
            "created_at": created_at,
            "updated_at": now,
        }

    def add_edge(
        self,
        source_name_or_id: str,
        target_name_or_id: str,
        relation_type: str = "RELATED_TO",
        weight: float = 1.0,
    ) -> Dict[str, Any]:
        """Create a directed relationship edge between two nodes."""
        source_node = self.get_node(source_name_or_id) or self.add_node(source_name_or_id)
        target_node = self.get_node(target_name_or_id) or self.add_node(target_name_or_id)

        edge_id = f"{source_node['id']}_{relation_type}_{target_node['id']}".lower()
        now = time.time()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO edges (id, source_id, target_id, relation_type, weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (edge_id, source_node["id"], target_node["id"], relation_type, weight, now),
            )
            conn.commit()

        return {
            "id": edge_id,
            "source_id": source_node["id"],
            "target_id": target_node["id"],
            "relation_type": relation_type,
            "weight": weight,
        }

    def get_node(self, name_or_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve node by exact name or ID."""
        clean = name_or_id.strip()
        clean_id = re.sub(r"[^\w]+", "_", clean.lower()).strip("_")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM nodes WHERE id = ? OR LOWER(name) = ?",
                (clean_id, clean.lower()),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def supersede_nodes(self, pattern: str, new_entity_name: str, reason: str = "") -> int:
        """Mark matching older nodes as SUPERSEDED when a new lifecycle node is created."""
        clean_pattern = pattern.strip().lower()
        new_node = self.get_node(new_entity_name) or self.add_node(new_entity_name)
        now = time.time()
        count = 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name FROM nodes WHERE LOWER(name) LIKE ? AND id != ? AND status = 'ACTIVE'",
                (f"%{clean_pattern}%", new_node["id"]),
            )
            rows = cursor.fetchall()
            for r in rows:
                cursor.execute(
                    "UPDATE nodes SET status = 'SUPERSEDED', updated_at = ? WHERE id = ?",
                    (now, r["id"]),
                )
                # Link old node --SUPERSEDED_BY--> new node
                edge_id = f"{r['id']}_superseded_by_{new_node['id']}"
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO edges (id, source_id, target_id, relation_type, weight, created_at)
                    VALUES (?, ?, ?, 'SUPERSEDED_BY', 1.0, ?)
                    """,
                    (edge_id, r["id"], new_node["id"], now),
                )
                count += 1
            conn.commit()

        logger.info("Superseded %d nodes matching '%s' with '%s'", count, pattern, new_entity_name)
        return count

    def forget_entity(self, pattern: str) -> Tuple[int, List[str]]:
        """Conversational Governance Command: Mark matching nodes as DEPRECATED (soft delete)."""
        clean_pattern = pattern.strip().lower()
        now = time.time()
        forgotten_names = []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name FROM nodes WHERE (LOWER(name) LIKE ? OR LOWER(summary) LIKE ?) AND status != 'DEPRECATED'",
                (f"%{clean_pattern}%", f"%{clean_pattern}%"),
            )
            rows = cursor.fetchall()
            for r in rows:
                forgotten_names.append(r["name"])
                cursor.execute(
                    "UPDATE nodes SET status = 'DEPRECATED', updated_at = ? WHERE id = ?",
                    (now, r["id"]),
                )
            conn.commit()

        return len(forgotten_names), forgotten_names

    def query_active_knowledge(
        self,
        query: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve active entity nodes, ordered by recency and relevance."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            sql = "SELECT * FROM nodes WHERE status = 'ACTIVE'"
            params: List[Any] = []

            if entity_type:
                sql += " AND entity_type = ?"
                params.append(entity_type)

            if query:
                sql += " AND (LOWER(name) LIKE ? OR LOWER(summary) LIKE ?)"
                clean_q = f"%{query.strip().lower()}%"
                params.extend([clean_q, clean_q])

            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)
            return [dict(r) for r in cursor.fetchall()]

    def format_graph_context_for_prompt(self, query_text: Optional[str] = None, limit: int = 6) -> str:
        """Format active Knowledge Graph nodes as structured grounding text for Gemini system prompts."""
        active_nodes = self.query_active_knowledge(query=query_text, limit=limit)
        if not active_nodes:
            return ""

        lines = ["--- Knowledge Graph Active Context ---"]
        for node in active_nodes:
            summary_text = f": {node['summary']}" if node.get("summary") else ""
            lines.append(f"• [{node['entity_type']}] {node['name']}{summary_text}")
        return "\n".join(lines)

    def extract_and_index_entities_from_text(
        self,
        text: str,
        source_module: str = "MIND",
    ) -> Dict[str, Any]:
        """Background LLM Task: Automatically extract entities, relationships, and superseded topics from text."""
        if not text:
            return {"extracted_count": 0}

        try:
            client = get_gemini_client()
            if client is None:
                return {"extracted_count": 0}

            prompt = (
                f"Analyze the following text logged via '{source_module}' and extract key Knowledge Graph entities, relationships, and obsolete topic patterns:\n\n"
                f"Text: '{text}'\n\n"
                "Return JSON with fields:\n"
                "- entities: list of objects with name, entity_type (PROJECT/ORGANIZATION/TOPIC/PREFERENCE/EVENT), summary\n"
                "- relations: list of objects with source_name, target_name, relation_type (BELONGS_TO/DEPENDS_ON/MENTIONED_IN)\n"
                "- superseded_patterns: list of strings (older topic patterns replaced by this news)"
            )

            gen_types = get_genai_types()
            cfg = None
            if gen_types is not None:
                cfg = gen_types.GenerateContentConfig(response_mime_type="application/json")
            response = client.models.generate_content(
                model=DEFAULT_GEMINI_MODEL,
                contents=prompt,
                config=cfg,
            )

            data = json.loads(response.text or "{}")

            extracted_nodes = []

            for ent in data.get("entities", []):
                n_name = ent.get("name")
                if n_name:
                    node = self.add_node(
                        name=n_name,
                        entity_type=ent.get("entity_type", "TOPIC"),
                        summary=ent.get("summary", ""),
                        status="ACTIVE",
                    )
                    extracted_nodes.append(node["name"])

            for rel in data.get("relations", []):
                s_name = rel.get("source_name")
                t_name = rel.get("target_name")
                r_type = rel.get("relation_type", "RELATED_TO")
                if s_name and t_name:
                    self.add_edge(s_name, t_name, relation_type=r_type)

            for pattern in data.get("superseded_patterns", []):
                if pattern and extracted_nodes:
                    self.supersede_nodes(pattern, extracted_nodes[0])

            logger.info("Background entity extraction indexed %d entities from %s text", len(extracted_nodes), source_module)
            return {"extracted_count": len(extracted_nodes), "nodes": extracted_nodes}
        except Exception as exc:
            logger.warning("Background entity extraction failed for text '%s': %s", text[:50], exc)
            return {"extracted_count": 0, "error": str(exc)}

    def sync_graph_to_notion(self, notion_client: Any) -> Dict[str, Any]:
        """Sync active Knowledge Graph nodes & edges to Notion 'Home > Knowledge Graph' page."""
        if notion_client is None or notion_client.client is None:
            return {"status": "error", "message": "Notion client not connected."}

        try:
            from app.notion_utils import build_notion_block
            from app.workspace_service import find_page_node_in_workspace

            # 1. Locate or create Knowledge Graph container page
            kg_node = find_page_node_in_workspace("Knowledge Graph", notion_client=notion_client)
            page_id = kg_node.page_id if kg_node else None

            if not page_id:
                home_node = find_page_node_in_workspace("Home", notion_client=notion_client)
                parent = {"page_id": home_node.page_id} if home_node else {"page_id": notion_client.database_id}
                new_page = notion_client.client.pages.create(
                    parent=parent,
                    properties={"title": [{"text": {"content": "Knowledge Graph"}}]},
                )
                page_id = new_page.get("id")

            # 2. Build blocks for active nodes & relationships
            active_nodes = self.query_active_knowledge(limit=30)
            blocks = [
                _build_notion_block(
                    content=f"🧠 **Ocean Knowledge Graph (Active Nodes: {len(active_nodes)})**",
                    block_type="callout",
                ),
            ]

            for node in active_nodes:
                summary_part = f": {node['summary']}" if node.get("summary") else ""
                text = f"[{node['entity_type']}] **{node['name']}**{summary_part}"
                blocks.append(_build_notion_block(content=text, block_type="bulleted_list_item"))

            # Append blocks to Notion Knowledge Graph page
            notion_client.client.blocks.children.append(block_id=page_id, children=blocks)
            logger.info("Successfully synced %d Knowledge Graph nodes to Notion page %s", len(active_nodes), page_id)
            return {
                "status": "ok",
                "count": len(active_nodes),
                "page_id": page_id,
                "reply_text": f"🧠 *Knowledge Graph Synced to Notion!*\n\nMirrored **{len(active_nodes)}** active entity nodes to **Knowledge Graph** page.",
            }
        except Exception as exc:
            logger.error("Failed to sync Knowledge Graph to Notion: %s", exc, exc_info=True)
            return {"status": "error", "message": str(exc), "reply_text": f"❌ Failed to sync Knowledge Graph to Notion: {exc}"}


# Global Singleton Instance
graph_memory = GraphMemoryService()
