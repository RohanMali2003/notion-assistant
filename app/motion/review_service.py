"""Weekly and Daily Review Services for Ocean Motion.

Implements:
- Daily Review Routine: Converts daily logs/activities into atomic observations without giving unsolicited advice or modifying trajectory.
- Weekly Review Routine: Synthesizes weekly progress, detects strategic drift, evaluates decision triggers, updates trajectory, and generates MotionWeeklyReview reports.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import logging
from typing import Any, Dict, List, Optional

from app.motion.drift_detector import StrategicDriftDetector, drift_detector
from app.motion.evidence_pipeline import EvidencePipeline, evidence_pipeline
from app.motion.permissions import (
    MotionPermission,
    enforce_persona_permission,
    permission_engine,
)
from app.motion.schemas import (
    DecisionJournalEntry,
    MotionRecommendation,
    MotionWeeklyReview,
    utc_now_iso,
)
from app.motion.spec import DecisionStatus, PersonaType, TrajectoryMomentum
from app.motion.storage import MotionStorage, motion_storage
from app.motion.strategic_model import StrategicModelService, strategic_model_service
from app.motion.synthesis import MultiWindowSynthesisEngine, synthesis_engine

logger = logging.getLogger("notion-assistant.motion.review")


class MotionReviewService:
    """Orchestrates daily observation generation and weekly strategic reviews."""

    def __init__(
        self,
        storage: Optional[MotionStorage] = None,
        strategic_model: Optional[StrategicModelService] = None,
        pipeline: Optional[EvidencePipeline] = None,
        drift_engine: Optional[StrategicDriftDetector] = None,
        synth_engine: Optional[MultiWindowSynthesisEngine] = None,
    ):
        self.storage = storage or motion_storage
        self.strategic_model = strategic_model or strategic_model_service
        self.pipeline = pipeline or evidence_pipeline
        self.drift_detector = drift_engine or drift_detector
        self.synthesis_engine = synth_engine or synthesis_engine

    # --- Daily Review Routine ---
    @enforce_persona_permission(MotionPermission.READ_LOGS)
    def execute_daily_review(
        self,
        raw_activities: List[Dict[str, Any]],
        date_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process daily activity into atomic evidence and observations. Never alters trajectory or identity."""
        target_date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info("Executing Motion daily review for date: %s", target_date)

        # 1. Extract atomic evidence items
        evidence_items = self.pipeline.extract_evidence_from_raw_activity(raw_activities)

        # 2. Build and persist daily observations
        observations = self.pipeline.build_observations_from_evidence(
            evidence_items=evidence_items,
            period_start=target_date,
            period_end=target_date,
        )

        # 3. Build and persist conclusions with rule-based confidence
        conclusions = self.pipeline.build_conclusions_from_observations(observations)

        # 4. Trigger multi-window synthesis
        self.synthesis_engine.run_synthesis(reference_date=target_date)

        return {
            "status": "success",
            "date": target_date,
            "evidence_count": len(evidence_items),
            "observations_count": len(observations),
            "conclusions_count": len(conclusions),
            "observation_ids": [o.id for o in observations],
        }

    # --- Weekly Strategic Review Routine ---
    @enforce_persona_permission(MotionPermission.READ_STRATEGIC_MODEL)
    def execute_weekly_review(
        self,
        week_start: Optional[str] = None,
        week_end: Optional[str] = None,
    ) -> MotionWeeklyReview:
        """Execute weekly review: evaluates trajectory, detects drift, updates decisions, and generates report."""
        now = datetime.now(timezone.utc)
        if not week_end:
            week_end = now.strftime("%Y-%m-%d")
        if not week_start:
            week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")

        logger.info("Executing Motion weekly strategic review: %s to %s", week_start, week_end)

        # 1. Ingest context
        trajectory = self.strategic_model.get_trajectory()
        initiatives = self.strategic_model.get_initiatives(status="ACTIVE")
        observations = self.storage.load_observations(limit=50)
        conclusions = self.storage.load_conclusions(limit=50)
        decisions = self.strategic_model.get_decisions(status=DecisionStatus.PENDING)

        # 2. Evaluate Strategic Drift
        drift_report = self.drift_detector.evaluate_drift(
            window_days=7,
            reference_date=week_end,
            save_report=True,
        )

        # 3. Evaluate Decision Journal triggers: mark due decisions
        due_decision_ids = []
        for dec in decisions:
            try:
                self.strategic_model.transition_decision_state(
                    decision_id=dec.id,
                    target_status=DecisionStatus.DUE,
                )
                due_decision_ids.append(dec.id)
            except Exception as err:
                logger.debug("Decision %s not transitioned: %s", dec.id, err)

        # 4. Detect Wins, Regressions, and Strategic Drift
        wins = [
            f"Sustained execution across {len(initiatives)} active strategic initiatives.",
            f"Strategic Alignment Score: {drift_report.alignment_score:.1f}% ({drift_report.drift_severity.value}).",
        ]
        regressions = []
        if len(observations) == 0:
            regressions.append("Zero recorded activity logs in the past week (observability gap).")
        if drift_report.neglected_initiatives:
            regressions.append(f"Neglected initiatives: {', '.join(drift_report.neglected_initiatives)}.")
        if drift_report.runaway_categories:
            regressions.append(f"Runaway categories: {', '.join(drift_report.runaway_categories)}.")

        drift_analysis = (
            f"Strategic Alignment Index is {drift_report.alignment_score:.1f}% with {drift_report.velocity_vector.value} velocity. "
            + " ".join(drift_report.recommendations_for_rebalancing)
        )

        # 5. Generate Strategic Recommendations
        recommendations = [
            MotionRecommendation(
                id=f"rec_wreview_{int(now.timestamp())}",
                conclusion_ids=[c.id for c in conclusions[:2]],
                question_or_opportunity="Sustain momentum on primary initiative while protecting deep-work blocks",
                recommendation_text=drift_report.recommendations_for_rebalancing[0] if drift_report.recommendations_for_rebalancing else "Dedicate initial 3-hour morning blocks strictly to core project engineering.",
                rationale="Prevents cognitive switching costs and accelerates initiative milestone completion.",
                trade_offs=["Delayed responses to non-urgent incoming requests"],
                expected_outcome="Milestone delivery within the planned horizon.",
                review_trigger="Upcoming Sunday Weekly Review",
            )
        ]

        # 6. Propose and apply Trajectory update
        proposed_momentum = TrajectoryMomentum.HIGH if drift_report.alignment_score >= 80 else TrajectoryMomentum.MODERATE
        self.strategic_model.update_trajectory(
            momentum=proposed_momentum,
            next_review=f"Sunday Review ({(now + timedelta(days=7)).strftime('%Y-%m-%d')})",
        )

        review_id = f"rev_{now.strftime('%Y')}_w{now.isocalendar()[1]}"
        review = MotionWeeklyReview(
            id=review_id,
            week_start=week_start,
            week_end=week_end,
            wins=wins,
            regressions=regressions,
            strategic_drift=drift_analysis,
            opportunities=["High momentum on Ocean architecture completion"],
            recommendations=recommendations,
            decision_reviews_due=due_decision_ids,
        )

        self.storage.save_weekly_review(review)
        logger.info("Weekly Strategic Review %s generated and saved.", review_id)
        return review


motion_review_service = MotionReviewService()
