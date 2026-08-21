"""Strategic Context Retrieval Engine for Ocean Motion.

Dynamically assembles focused, high-signal strategic context for Motion prompts while
strictly excluding stale, archived, or irrelevant memory items.
"""

import logging
from typing import Any, Dict, List, Optional

from app.motion.schemas import (
    DecisionJournalEntry,
    HumanOverride,
    MotionConclusion,
    MotionContext,
    MotionIdentity,
    MotionInitiative,
    MotionObservation,
    MotionTrajectory,
    MotionWeeklyReview,
)
from app.motion.spec import DecisionStatus, OverrideStatus
from app.motion.storage import MotionStorage, motion_storage

logger = logging.getLogger("notion-assistant.motion.retrieval")


class StrategicContextRetriever:
    """Retrieves and formats query-relevant structured strategic context."""

    def __init__(self, storage: Optional[MotionStorage] = None):
        self.storage = storage or motion_storage

    def assemble_context(self, query: Optional[str] = None) -> MotionContext:
        """Dynamically assemble structured strategic context."""
        # 1. Identity (always loaded)
        identity: MotionIdentity = self.storage.load_identity()

        # 2. Current Trajectory (always loaded)
        trajectory: MotionTrajectory = self.storage.load_trajectory()

        # 3. Active Initiatives ONLY (exclude archived / completed / abandoned)
        all_initiatives = self.storage.load_initiatives()
        active_initiatives = [init for init in all_initiatives if init.status.upper() == "ACTIVE"]

        # 4. Open Human Overrides ONLY (status = ACTIVE)
        all_overrides = self.storage.load_overrides()
        active_overrides = [ovr for ovr in all_overrides if ovr.status == OverrideStatus.ACTIVE]

        # 5. Last 2 Weekly Reviews
        recent_reviews = self.storage.load_weekly_reviews(limit=2)
        review_dicts = [rev.model_dump() for rev in recent_reviews]

        # 6. Due / Pending Decision Journal Entries ONLY (exclude CLOSED)
        all_decisions = self.storage.load_decision_journal()
        relevant_decisions = [
            d for d in all_decisions if d.status in (DecisionStatus.DUE, DecisionStatus.PENDING)
        ]

        # 7. Recent active Conclusions & Observations (matching query keywords if provided)
        conclusions = self.storage.load_conclusions(limit=10)
        observations = self.storage.load_observations(limit=10)

        if query:
            query_tokens = [t.lower() for t in query.split() if len(t) > 3]
            if query_tokens:
                filtered_conc = [
                    c for c in conclusions
                    if any(tok in c.statement.lower() or tok in c.confidence_reasoning.lower() for tok in query_tokens)
                ]
                filtered_obs = [
                    o for o in observations
                    if any(tok in o.observation_summary.lower() or tok in o.category.lower() for tok in query_tokens)
                ]
                if filtered_conc:
                    conclusions = filtered_conc
                if filtered_obs:
                    observations = filtered_obs

        # 8. Latest Strategic Drift Report
        latest_drift = self.storage.load_latest_drift_report()

        context = MotionContext(
            identity=identity,
            trajectory=trajectory,
            active_initiatives=active_initiatives,
            active_overrides=active_overrides,
            recent_weekly_reviews=review_dicts,
            due_decisions=relevant_decisions,
            relevant_conclusions=conclusions,
            relevant_observations=observations,
            drift_report=latest_drift,
        )
        return context

    def format_context_for_prompt(self, context: MotionContext) -> str:
        """Format the MotionContext into a concise, structured markdown string for the system prompt."""
        sections: List[str] = []

        # 1. Identity
        id_lines = [
            f"**Education / Role:** {context.identity.education}",
            f"**Career Goals:** {', '.join(context.identity.career_goals) if context.identity.career_goals else 'N/A'}",
            f"**Long-term Ambitions:** {', '.join(context.identity.long_term_ambitions) if context.identity.long_term_ambitions else 'N/A'}",
            f"**Core Constraints:** {', '.join(context.identity.core_constraints) if context.identity.core_constraints else 'N/A'}",
        ]
        sections.append("### 1. User Identity (Stable Core)\n" + "\n".join(id_lines))

        # 2. Trajectory
        traj_lines = [
            f"**Current Phase:** {context.trajectory.current_phase}",
            f"**Current Direction:** {getattr(context.trajectory, 'current_direction', '') or getattr(context.trajectory, 'target_direction', '')}",
            f"**Momentum:** {context.trajectory.momentum.value if hasattr(context.trajectory.momentum, 'value') else context.trajectory.momentum}",
            f"**Biggest Opportunity:** {context.trajectory.biggest_opportunity}",
            f"**Biggest Risk:** {context.trajectory.biggest_risk}",
        ]
        sections.append("### 2. Strategic Trajectory\n" + "\n".join(traj_lines))

        # 3. Active Initiatives
        if context.active_initiatives:
            init_lines = [
                f"- **{init.title}** [{init.strategic_importance.value} / {init.horizon.value}]: {init.description} *(Momentum: {init.momentum.value})*"
                for init in context.active_initiatives
            ]
            sections.append("### 3. Active Strategic Initiatives\n" + "\n".join(init_lines))
        else:
            sections.append("### 3. Active Strategic Initiatives\nNone currently registered.")

        # 4. Strategic Drift & Alignment Report
        if context.drift_report:
            drift = context.drift_report
            drift_lines = [
                f"- **Strategic Alignment Score:** {drift.alignment_score:.1f}% ({drift.drift_severity.value})",
                f"- **Velocity Vector:** {drift.velocity_vector.value} ({drift.total_hours_analyzed}h analyzed)",
            ]
            if drift.neglected_initiatives:
                drift_lines.append(f"- **⚠️ Neglected Initiatives:** {', '.join(drift.neglected_initiatives)}")
            if drift.runaway_categories:
                drift_lines.append(f"- **⚠️ Runaway Categories:** {', '.join(drift.runaway_categories)}")
            sections.append("### 4. Strategic Drift & Alignment Index\n" + "\n".join(drift_lines))

        # 5. Open Human Overrides
        if context.active_overrides:
            ovr_lines = [
                f"- **Topic:** {ovr.recommendation_id_or_topic} | **User Decision:** {ovr.user_decision} | **Revisit Condition:** {ovr.review_trigger_condition}"
                for ovr in context.active_overrides
            ]
            sections.append("### 5. Active Human Overrides\n" + "\n".join(ovr_lines))

        # 6. Due / Pending Decisions
        if context.due_decisions:
            dec_lines = [
                f"- [ID: `{d.id}` | {d.status.value}] **Q:** {d.decision_title} | **Rec:** {d.recommendation} | **Scheduled:** {d.review_scheduled_for}"
                for d in context.due_decisions
            ]
            sections.append("### 6. Pending / Due Decisions in Journal\n" + "\n".join(dec_lines))

        # 7. Active Conclusions (Evidence-Backed)
        if context.relevant_conclusions:
            conc_lines = [
                f"- [{c.confidence.value} CONFIDENCE] {c.statement} *(Reasoning: {c.confidence_reasoning})*"
                for c in context.relevant_conclusions
            ]
            sections.append("### 7. Active Strategic Conclusions\n" + "\n".join(conc_lines))

        # 8. Recent Observations
        if context.relevant_observations:
            obs_lines = [
                f"- [{o.category} | n={o.frequency}] {o.observation_summary}"
                for o in context.relevant_observations
            ]
            sections.append("### 8. Recent Supporting Observations\n" + "\n".join(obs_lines))

        return "\n\n".join(sections)


strategic_context_retriever = StrategicContextRetriever()
