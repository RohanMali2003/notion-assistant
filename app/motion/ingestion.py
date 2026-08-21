"""Automated Asynchronous Evidence Ingestion Engine for Ocean Motion v2.

Subscribes to Ocean background task completions, study logs, LeetCode practices,
and daily reflections, transforming them into atomic, verifiable EvidenceItems.
"""

from datetime import datetime, timezone
import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.motion.schemas import EvidenceIngestionEvent, EvidenceItem
from app.motion.storage import MotionStorage, motion_storage

import uuid

logger = logging.getLogger("notion-assistant.motion.ingestion")


def _generate_evidence_id(source_type: str, date_str: str, title: str) -> str:
    """Deterministic, collision-free evidence ID with unique suffix."""
    clean_title = re.sub(r"[^a-zA-Z0-9_]", "_", title.lower())[:30]
    unique_suffix = uuid.uuid4().hex[:6]
    return f"ev_{date_str.replace('-', '')}_{clean_title}_{unique_suffix}"


def _infer_category_and_tags(event: EvidenceIngestionEvent) -> tuple[str, List[str], float]:
    """Classify activity into strategic category, tags, and duration."""
    tags = list(event.tags)
    duration = event.duration_hours or 1.0
    text_corpus = f"{event.title} {event.description}".lower()

    if event.event_type == "leetcode_review" or "leetcode" in text_corpus or "algorithm" in text_corpus:
        category = "LeetCode & Algorithms"
        if "leetcode" not in tags:
            tags.append("leetcode")
        if not event.duration_hours:
            duration = 1.0

    elif event.event_type == "learning_milestone" or "research" in text_corpus or "study" in text_corpus or "paper" in text_corpus:
        category = "Learning & Research"
        if "learning" not in tags:
            tags.append("learning")
        if not event.duration_hours:
            duration = 1.5

    elif "ocean" in text_corpus or "agent" in text_corpus or "backend" in text_corpus or "build" in text_corpus or "code" in text_corpus or "deploy" in text_corpus:
        category = "Systems Engineering & Projects"
        if "engineering" not in tags:
            tags.append("engineering")
        if not event.duration_hours:
            duration = 2.0

    elif "cs" in text_corpus or "homework" in text_corpus or "exam" in text_corpus or "class" in text_corpus or "umass" in text_corpus:
        category = "Academics"
        if "academics" not in tags:
            tags.append("academics")
        if not event.duration_hours:
            duration = 1.5

    elif "bill" in text_corpus or "email" in text_corpus or "admin" in text_corpus or "schedule" in text_corpus or "errand" in text_corpus:
        category = "Admin & Operations"
        if "admin" not in tags:
            tags.append("admin")
        if not event.duration_hours:
            duration = 0.5

    else:
        category = "General Execution"
        if "execution" not in tags:
            tags.append("execution")
        if not event.duration_hours:
            duration = 1.0

    return category, tags, duration


class EvidenceIngestionEngine:
    """Asynchronously ingests raw execution events into structured EvidenceItem records."""

    def __init__(self, storage: Optional[MotionStorage] = None):
        self.storage = storage or motion_storage

    def ingest_event(self, event: EvidenceIngestionEvent) -> EvidenceItem:
        """Process an execution event, create an EvidenceItem, and store it."""
        date_str = event.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        category, tags, duration = _infer_category_and_tags(event)
        evidence_id = _generate_evidence_id(event.event_type, date_str, event.title)

        metrics = dict(event.metrics)
        metrics["category"] = category
        if "duration_hours" not in metrics:
            metrics["duration_hours"] = duration

        evidence = EvidenceItem(
            id=evidence_id,
            source_type=event.event_type,
            source_reference=event.source_ref,
            date=date_str,
            duration_hours=duration,
            tags=tags,
            metrics=metrics,
            description=f"{event.title}: {event.description}".strip(": "),
            raw_snippet=event.description[:300] if event.description else event.title,
        )

        self.storage.save_evidence(evidence)
        logger.info(
            "Ingested evidence '%s' [%s] category='%s' duration=%.1fh",
            evidence.id,
            event.event_type,
            category,
            duration,
        )
        return evidence

    def ingest_task_completion(
        self,
        task_title: str,
        notes: str = "",
        page_url: Optional[str] = None,
        duration_hours: Optional[float] = None,
        tags: Optional[List[str]] = None,
        date: Optional[str] = None,
    ) -> EvidenceItem:
        """Helper for task completion events."""
        event = EvidenceIngestionEvent(
            event_type="completed_task",
            source_ref=page_url,
            title=task_title,
            description=notes,
            date=date,
            duration_hours=duration_hours,
            tags=tags or [],
        )
        return self.ingest_event(event)

    def ingest_leetcode_review(
        self,
        problem_title: str,
        pattern_notes: str = "",
        page_url: Optional[str] = None,
        duration_hours: float = 1.0,
        date: Optional[str] = None,
    ) -> EvidenceItem:
        """Helper for LeetCode algorithm reviews."""
        event = EvidenceIngestionEvent(
            event_type="leetcode_review",
            source_ref=page_url,
            title=f"LeetCode: {problem_title}",
            description=pattern_notes,
            date=date,
            duration_hours=duration_hours,
            tags=["leetcode", "algorithms"],
            metrics={"problem": problem_title, "problems_solved": 1},
        )
        return self.ingest_event(event)

    def ingest_learning_milestone(
        self,
        topic_title: str,
        summary: str = "",
        page_url: Optional[str] = None,
        duration_hours: float = 1.5,
        date: Optional[str] = None,
    ) -> EvidenceItem:
        """Helper for learning curriculum milestones."""
        event = EvidenceIngestionEvent(
            event_type="learning_milestone",
            source_ref=page_url,
            title=f"Study: {topic_title}",
            description=summary,
            date=date,
            duration_hours=duration_hours,
            tags=["learning", "research"],
            metrics={"topic": topic_title},
        )
        return self.ingest_event(event)

    def ingest_daily_log(
        self,
        date_str: str,
        summary_text: str,
        page_url: Optional[str] = None,
        estimated_hours: float = 2.0,
    ) -> EvidenceItem:
        """Helper for daily journal / log entries."""
        event = EvidenceIngestionEvent(
            event_type="daily_log",
            source_ref=page_url,
            title=f"Daily Log for {date_str}",
            description=summary_text,
            date=date_str,
            duration_hours=estimated_hours,
            tags=["daily_log", "reflection"],
        )
        return self.ingest_event(event)


evidence_ingestion_engine = EvidenceIngestionEngine()
