"""Evidence Pipeline for Ocean Motion.

Implements the deterministic evidence processing pipeline:
Activity → Evidence Extractor → Observations → Conclusions → Strategic Model → Motion.
"""

from datetime import datetime, timezone
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

from app.motion.schemas import (
    ConfidenceLevel,
    EvidenceItem,
    MotionConclusion,
    MotionObservation,
    utc_now_iso,
)
from app.motion.storage import MotionStorage, motion_storage

logger = logging.getLogger("notion-assistant.motion.evidence")


def _generate_evidence_id(source_type: str, date: str, content: str) -> str:
    """Generate a deterministic ID for an evidence item."""
    raw = f"{source_type}:{date}:{content}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    clean_date = date.replace("-", "")
    return f"ev_{clean_date}_{digest}"


def _extract_duration_hours(text: str) -> Optional[float]:
    """Extract hours spent from textual descriptions like '4 hours', '2.5h', '90 mins'."""
    match_hours = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", text, re.IGNORECASE)
    if match_hours:
        try:
            return float(match_hours.group(1))
        except ValueError:
            pass

    match_mins = re.search(r"(\d+)\s*(?:mins?|minutes?)\b", text, re.IGNORECASE)
    if match_mins:
        try:
            return round(float(match_mins.group(1)) / 60.0, 2)
        except ValueError:
            pass
    return None


class EvidencePipeline:
    """Deterministic evidence extraction, observation building, and conclusion synthesis."""

    def __init__(self, storage: Optional[MotionStorage] = None):
        self.storage = storage or motion_storage

    # --- Step 1: Evidence Extractor ---
    def extract_evidence_from_raw_activity(
        self,
        activity_items: List[Dict[str, Any]],
    ) -> List[EvidenceItem]:
        """Convert raw activity dicts (tasks, daily logs, learning notes) into atomic EvidenceItems."""
        extracted: List[EvidenceItem] = []
        for item in activity_items:
            source_type = item.get("source_type", "daily_log")
            date_str = item.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            content = str(item.get("text") or item.get("title") or item.get("content") or "").strip()
            if not content:
                continue

            duration = item.get("duration_hours") or _extract_duration_hours(content)
            metrics = item.get("metrics") or {}
            source_ref = item.get("url") or item.get("page_id")

            ev_id = _generate_evidence_id(source_type, date_str, content)
            ev = EvidenceItem(
                id=ev_id,
                source_type=source_type,
                source_ref=source_ref,
                date=date_str,
                duration_hours=duration,
                metrics=metrics,
                description=content,
                raw_snippet=content[:200],
            )
            extracted.append(ev)

        logger.info("Extracted %d atomic evidence items from raw activity.", len(extracted))
        return extracted

    # --- Step 2: Observation Builder ---
    def build_observations_from_evidence(
        self,
        evidence_items: List[EvidenceItem],
        period_start: str,
        period_end: str,
    ) -> List[MotionObservation]:
        """Aggregate evidence items across a time window into factual MotionObservations."""
        if not evidence_items:
            return []

        # Group by category / source_type
        by_category: Dict[str, List[EvidenceItem]] = {}
        for ev in evidence_items:
            cat = "Execution"
            st_lower = ev.source_type.lower()
            desc_lower = ev.description.lower()

            if "leetcode" in st_lower or "algorithm" in desc_lower or "problem" in desc_lower:
                cat = "LeetCode"
            elif "learn" in st_lower or "research" in desc_lower or "paper" in desc_lower or "study" in desc_lower:
                cat = "Learning & Research"
            elif "course" in desc_lower or "school" in desc_lower or "umass" in desc_lower:
                cat = "Academics"
            elif "finance" in desc_lower or "budget" in desc_lower:
                cat = "Finances"

            by_category.setdefault(cat, []).append(ev)

        observations: List[MotionObservation] = []
        for cat, items in by_category.items():
            ev_ids = [it.id for it in items]
            freq = len(items)
            total_duration = sum(it.duration_hours or 0.0 for it in items)

            # Construct strictly factual statement
            if total_duration > 0:
                stmt = f"Recorded {freq} activities in {cat} totaling ~{total_duration:.1f} hours during {period_start} to {period_end}."
            else:
                stmt = f"Recorded {freq} distinct execution events in {cat} between {period_start} and {period_end}."

            obs_digest = hashlib.md5(f"{cat}:{period_start}:{period_end}:{freq}".encode("utf-8")).hexdigest()[:6]
            clean_date = period_end.replace("-", "")
            obs_id = f"obs_{clean_date}_{obs_digest}"

            obs = MotionObservation(
                id=obs_id,
                period_start=period_start,
                period_end=period_end,
                category=cat,
                statement=stmt,
                evidence_ids=ev_ids,
                frequency_count=freq,
            )
            self.storage.save_observation(obs)
            observations.append(obs)

        logger.info("Built and persisted %d MotionObservations.", len(observations))
        return observations

    # --- Step 3: Conclusion Builder ---
    def build_conclusions_from_observations(
        self,
        observations: List[MotionObservation],
    ) -> List[MotionConclusion]:
        """Synthesize observations into interpretations with rule-based confidence levels."""
        conclusions: List[MotionConclusion] = []
        for obs in observations:
            # Determine rule-based confidence
            if obs.frequency_count >= 7:
                conf = ConfidenceLevel.HIGH
            elif obs.frequency_count >= 3:
                conf = ConfidenceLevel.MEDIUM
            else:
                conf = ConfidenceLevel.LOW

            # Synthesize interpretation
            if obs.category == "Execution" and obs.frequency_count >= 5:
                stmt = f"High task throughput maintained in core development and system execution."
                rationale = f"Supported by {obs.frequency_count} recorded execution milestones."
            elif obs.category == "Learning & Research" and obs.frequency_count >= 2:
                stmt = f"Active research & study momentum sustained across core foundational topics."
                rationale = f"Supported by {obs.frequency_count} study notes and concept deep-dives."
            elif obs.category == "LeetCode" and obs.frequency_count >= 3:
                stmt = f"Steady algorithmic problem solving practice rhythm observed."
                rationale = f"Supported by {obs.frequency_count} algorithm review entries."
            else:
                stmt = f"Activity detected in {obs.category} with {conf.value} confidence."
                rationale = f"Derived from single observation window with {obs.frequency_count} items."

            conc_digest = hashlib.md5(f"{obs.id}:{stmt}".encode("utf-8")).hexdigest()[:6]
            conc_id = f"conc_{obs.id.replace('obs_', '')}_{conc_digest}"

            conc = MotionConclusion(
                id=conc_id,
                statement=stmt,
                derived_from_observation_ids=[obs.id],
                confidence_level=conf,
                rationale=rationale,
            )
            self.storage.save_conclusion(conc)
            conclusions.append(conc)

        logger.info("Built and persisted %d MotionConclusions.", len(conclusions))
        return conclusions


class EvidenceAttributionEngine:
    """Attribution helper for tracing reasoning chains."""

    def __init__(self, storage: Optional[MotionStorage] = None):
        self.storage = storage or motion_storage

    def trace_provenance(self, conclusion_id: str) -> Dict[str, Any]:
        """Traverse Conclusion -> Observation -> Evidence Items."""
        conclusions = self.storage.load_conclusions(limit=100)
        conc = next((c for c in conclusions if c.id == conclusion_id), None)
        if not conc:
            return {"error": f"Conclusion '{conclusion_id}' not found"}

        observations = self.storage.load_observations(limit=100)
        obs_map = {o.id: o for o in observations}

        supporting_obs = []
        for obs_id in conc.derived_from_observation_ids:
            obs = obs_map.get(obs_id)
            if obs:
                supporting_obs.append(obs.model_dump())

        return {
            "conclusion": conc.model_dump(),
            "supporting_observations": supporting_obs,
            "confidence": conc.confidence_level.value,
        }


evidence_pipeline = EvidencePipeline()
