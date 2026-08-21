"""Strategic Drift & Trend Detection Engine for Ocean Motion v2.

Evaluates mathematical alignment between declared strategic initiatives and actual
time/effort distributions, detecting neglected goals, runaway tasks, and velocity trends.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, List, Optional, Tuple

from app.motion.schemas import (
    EvidenceItem,
    MotionInitiative,
    MotionTrajectory,
    StrategicDriftReport,
)
from app.motion.spec import (
    DriftSeverity,
    FRAGMENTATION_THRESHOLD_RATIO,
    NEGLECTED_INITIATIVE_DAYS,
    SLIDING_WINDOW_7D,
    VelocityVector,
)
from app.motion.storage import MotionStorage, motion_storage

logger = logging.getLogger("notion-assistant.motion.drift")


class StrategicDriftDetector:
    """Calculates Strategic Alignment Score (0-100%) and detects drift vectors."""

    def __init__(self, storage: Optional[MotionStorage] = None):
        self.storage = storage or motion_storage

    def evaluate_drift(
        self,
        window_days: int = SLIDING_WINDOW_7D,
        reference_date: Optional[str] = None,
        save_report: bool = True,
    ) -> StrategicDriftReport:
        """Run mathematical drift evaluation over the specified time window."""
        if not reference_date:
            reference_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")
        start_date = (ref_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")

        # 1. Load active initiatives and trajectory
        initiatives = self.storage.load_initiatives()
        active_inits = [i for i in initiatives if i.status == "ACTIVE"]
        trajectory = self.storage.load_trajectory()

        # 2. Load evidence in window
        evidence_items = self.storage.load_evidence(
            limit=500,
            start_date=start_date,
            end_date=reference_date,
        )

        total_hours = sum(e.duration_hours or 1.0 for e in evidence_items)

        if not evidence_items or total_hours == 0:
            # Baseline report with no data
            report = StrategicDriftReport(
                period_start=start_date,
                period_end=reference_date,
                alignment_score=50.0,
                drift_severity=DriftSeverity.LOW_DRIFT,
                velocity_vector=VelocityVector.STALLED,
                total_hours_analyzed=0.0,
                category_breakdown={},
                neglected_initiatives=[i.title for i in active_inits],
                runaway_categories=[],
                recommendations_for_rebalancing=[
                    "No activity evidence logged in this window. Log daily tasks and study sessions to calibrate alignment."
                ],
            )
            if save_report:
                self.storage.save_drift_report(report)
            return report

        # 3. Compute category effort breakdown
        category_hours: Dict[str, float] = {}
        for ev in evidence_items:
            cat = ev.metrics.get("category", "General Execution")
            category_hours[cat] = category_hours.get(cat, 0.0) + (ev.duration_hours or 1.0)

        category_pct: Dict[str, float] = {
            cat: round((hrs / total_hours * 100.0), 1)
            for cat, hrs in category_hours.items()
        }

        # 4. Map evidence to active initiatives
        init_hours: Dict[str, float] = {init.id: 0.0 for init in active_inits}
        for ev in evidence_items:
            ev_text = f"{ev.description} {' '.join(ev.tags)}".lower()
            for init in active_inits:
                init_keywords = [w.lower() for w in init.title.split() if len(w) > 3]
                if any(kw in ev_text for kw in init_keywords) or (init.id.lower() in ev_text):
                    init_hours[init.id] += (ev.duration_hours or 1.0)

        # 5. Check neglected initiatives (<= 1.0h in the window)
        neglected: List[str] = []
        for init in active_inits:
            hrs = init_hours.get(init.id, 0.0)
            if hrs <= 1.0:
                neglected.append(f"{init.title} ({hrs:.1f}h logged)")

        # 6. Check runaway categories (> 35% effort on Admin/General Execution)
        runaway: List[str] = []
        unplanned_hours = 0.0
        for cat, pct in category_pct.items():
            if cat in ("Admin & Operations", "General Execution"):
                unplanned_hours += category_hours.get(cat, 0.0)
                if pct > (FRAGMENTATION_THRESHOLD_RATIO * 100.0):
                    runaway.append(f"{cat} ({pct}% of time)")

        # 7. Calculate Strategic Alignment Score (0 - 100%)
        # Formula:
        # Base: 100
        # - Neglected penalty: -15 per neglected critical/high initiative
        # - Fragmentation penalty: -1.0 per % of runaway unplanned effort over 25%
        # - Effort bonus: +5 per active initiative receiving >= 4h
        score = 100.0

        if active_inits:
            neglected_ratio = len(neglected) / len(active_inits)
            score -= (neglected_ratio * 40.0)

        unplanned_pct = (unplanned_hours / total_hours * 100.0) if total_hours > 0 else 0.0
        if unplanned_pct > 25.0:
            excess_unplanned = unplanned_pct - 25.0
            score -= min(excess_unplanned * 0.8, 30.0)

        for init_id, hrs in init_hours.items():
            if hrs >= 4.0:
                score = min(score + 3.0, 100.0)

        score = max(round(score, 1), 0.0)

        # 8. Determine Drift Severity
        if score >= 80.0:
            severity = DriftSeverity.NORMAL
        elif score >= 65.0:
            severity = DriftSeverity.LOW_DRIFT
        elif score >= 45.0:
            severity = DriftSeverity.MODERATE_DRIFT
        else:
            severity = DriftSeverity.CRITICAL_DRIFT

        # 9. Determine Velocity Vector
        # Compare 7d vs prior 7d if available
        prior_start = (ref_dt - timedelta(days=window_days * 2)).strftime("%Y-%m-%d")
        prior_evidence = self.storage.load_evidence(
            limit=500,
            start_date=prior_start,
            end_date=start_date,
        )
        prior_hours = sum(e.duration_hours or 1.0 for e in prior_evidence)

        if total_hours >= prior_hours * 1.25 and total_hours >= 10.0:
            velocity = VelocityVector.ACCELERATING
        elif total_hours >= prior_hours * 0.85:
            velocity = VelocityVector.STEADY
        elif total_hours > 0:
            velocity = VelocityVector.SLOWING
        else:
            velocity = VelocityVector.STALLED

        # 10. Formulate Actionable Recommendations
        recs: List[str] = []
        if neglected:
            recs.append(f"Rebalance focus immediately toward neglected initiatives: {', '.join(neglected)}.")
        if runaway:
            recs.append(f"Contain runaway secondary tasks in {', '.join(runaway)} to free up deep work blocks.")
        if velocity in (VelocityVector.SLOWING, VelocityVector.STALLED):
            recs.append("Execution velocity has decelerated; schedule dedicated 2-hour strategic focus blocks.")
        if not recs:
            recs.append("Trajectory execution is well-aligned with core strategic priorities. Maintain current cadence.")

        report = StrategicDriftReport(
            period_start=start_date,
            period_end=reference_date,
            alignment_score=score,
            drift_severity=severity,
            velocity_vector=velocity,
            total_hours_analyzed=round(total_hours, 1),
            category_breakdown=category_pct,
            neglected_initiatives=neglected,
            runaway_categories=runaway,
            recommendations_for_rebalancing=recs,
        )

        if save_report:
            self.storage.save_drift_report(report)

        logger.info(
            "Drift evaluation [%s to %s]: Alignment Score = %.1f%% (%s, %s)",
            start_date,
            reference_date,
            score,
            severity.value,
            velocity.value,
        )
        return report


drift_detector = StrategicDriftDetector()
