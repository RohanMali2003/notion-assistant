"""Deep Multi-Hop Evidence Attribution Engine for Ocean Motion v2.

Traces causal provenance from Recommendations down through Conclusions,
Observations, and atomic EvidenceItems with Notion references.
"""

import logging
from typing import Any, Dict, List, Optional

from app.motion.schemas import (
    CausalAttributionTree,
    EvidenceItem,
    MotionConclusion,
    MotionObservation,
)
from app.motion.storage import MotionStorage, motion_storage

logger = logging.getLogger("notion-assistant.motion.attribution")


class MultiHopAttributionEngine:
    """Traverses and formats multi-hop causal trees for evidence-based belief explanations."""

    def __init__(self, storage: Optional[MotionStorage] = None):
        self.storage = storage or motion_storage

    def build_attribution_tree(self, target_id: str) -> CausalAttributionTree:
        """Construct a complete provenance tree for a recommendation, conclusion, or observation."""
        all_conclusions = {c.id: c for c in self.storage.load_conclusions(limit=100)}
        all_observations = {o.id: o for o in self.storage.load_observations(limit=200)}
        all_evidence = {e.id: e for e in self.storage.load_evidence(limit=500)}

        matched_conclusions: List[Dict[str, Any]] = []
        matched_observations: List[Dict[str, Any]] = []
        matched_evidence: List[EvidenceItem] = []
        sources_cited: List[str] = []
        recommendation_text: Optional[str] = None

        # Case 1: Target is a Decision Journal / Recommendation
        decision = None
        for d in self.storage.load_decision_journal():
            if d.id == target_id:
                decision = d
                break

        if decision:
            recommendation_text = decision.recommendation
            # Follow decision's derived conclusion IDs
            for conc_id in decision.derived_from_conclusion_ids:
                if conc_id in all_conclusions:
                    conc = all_conclusions[conc_id]
                    matched_conclusions.append(conc.model_dump())
                    for obs_id in conc.derived_from_observation_ids:
                        if obs_id in all_observations:
                            obs = all_observations[obs_id]
                            if obs.model_dump() not in matched_observations:
                                matched_observations.append(obs.model_dump())
                            for ev_id in obs.evidence_ids:
                                if ev_id in all_evidence and all_evidence[ev_id] not in matched_evidence:
                                    ev = all_evidence[ev_id]
                                    matched_evidence.append(ev)
                                    src = getattr(ev, "source_reference", None) or getattr(ev, "source_ref", None)
                                    if src and src not in sources_cited:
                                        sources_cited.append(src)

        # Case 2: Target is a Conclusion ID
        elif target_id in all_conclusions:
            conc = all_conclusions[target_id]
            matched_conclusions.append(conc.model_dump())
            for obs_id in conc.derived_from_observation_ids:
                if obs_id in all_observations:
                    obs = all_observations[obs_id]
                    matched_observations.append(obs.model_dump())
                    for ev_id in obs.evidence_ids:
                        if ev_id in all_evidence and all_evidence[ev_id] not in matched_evidence:
                            ev = all_evidence[ev_id]
                            matched_evidence.append(ev)
                            src = getattr(ev, "source_reference", None) or getattr(ev, "source_ref", None)
                            if src and src not in sources_cited:
                                sources_cited.append(src)

        # Case 3: Target is an Observation ID
        elif target_id in all_observations:
            obs = all_observations[target_id]
            matched_observations.append(obs.model_dump())
            for ev_id in obs.evidence_ids:
                if ev_id in all_evidence and all_evidence[ev_id] not in matched_evidence:
                    ev = all_evidence[ev_id]
                    matched_evidence.append(ev)
                    src = getattr(ev, "source_reference", None) or getattr(ev, "source_ref", None)
                    if src and src not in sources_cited:
                        sources_cited.append(src)

        # Case 4: Target is directly an Evidence ID
        elif target_id in all_evidence:
            ev = all_evidence[target_id]
            matched_evidence.append(ev)
            src = getattr(ev, "source_reference", None) or getattr(ev, "source_ref", None)
            if src:
                sources_cited.append(src)

        return CausalAttributionTree(
            target_id=target_id,
            recommendation_text=recommendation_text,
            conclusions=matched_conclusions,
            observations=matched_observations,
            evidence_items=matched_evidence,
            sources_cited=sources_cited,
        )

    def format_tree_as_markdown(self, tree: CausalAttributionTree) -> str:
        """Format an attribution tree into a clear, inspectable explanation of belief."""
        lines = [f"### 🔍 Causal Attribution for `{tree.target_id}`"]

        if tree.recommendation_text:
            lines.append(f"**Recommendation:** *\"{tree.recommendation_text}\"*")

        if tree.conclusions:
            lines.append("\n**1. Supporting Conclusions:**")
            for c in tree.conclusions:
                lines.append(f"- **{c.get('id')}** [{c.get('confidence')} Confidence]: {c.get('statement')}")
                lines.append(f"  *Reasoning:* {c.get('confidence_reasoning')}")

        if tree.observations:
            lines.append("\n**2. Factual Observations:**")
            for o in tree.observations:
                lines.append(
                    f"- **{o.get('id')}** ({o.get('time_window')}): {o.get('observation_summary')} "
                    f"[{o.get('frequency')} items, {o.get('total_duration_hours')}h]"
                )

        if tree.evidence_items:
            lines.append("\n**3. Underlying Atomic Evidence:**")
            for ev in tree.evidence_items[:10]:
                src_link = f" ([Source]({ev.source_reference}))" if ev.source_reference else ""
                lines.append(
                    f"- `{ev.date}` [{ev.source_type}] **{ev.description}** "
                    f"({ev.duration_hours or 1.0}h){src_link}"
                )
            if len(tree.evidence_items) > 10:
                lines.append(f"- *...and {len(tree.evidence_items) - 10} more evidence records.*")

        if tree.sources_cited:
            lines.append("\n**📚 Verified Sources:**")
            for src in tree.sources_cited:
                lines.append(f"- {src}")

        if not tree.conclusions and not tree.observations and not tree.evidence_items:
            lines.append("\n*No direct causal provenance records found for this identifier.*")

        return "\n".join(lines)


multi_hop_attribution_engine = MultiHopAttributionEngine()
