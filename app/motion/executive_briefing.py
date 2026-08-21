"""Autonomous Executive Strategic Briefing Engine for Ocean Motion v3.

Synthesizes alignment indices, cognitive sustainability metrics, critical path milestones,
and high-leverage recommendations into a comprehensive weekly executive briefing.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid

from app.motion.drift_detector import StrategicDriftDetector, drift_detector
from app.motion.energy_monitor import CognitiveEnergyMonitor, energy_monitor
from app.motion.milestones import InitiativeMilestoneEngine, milestone_engine
from app.motion.schemas import (
    ExecutiveBriefing,
    MotionRecommendation,
)
from app.motion.storage import MotionStorage, motion_storage

logger = logging.getLogger("notion-assistant.motion.executive_briefing")


class ExecutiveBriefingEngine:
    """Generates autonomous weekly strategic executive briefings."""

    def __init__(
        self,
        storage: Optional[MotionStorage] = None,
        drift_eng: Optional[StrategicDriftDetector] = None,
        energy_eng: Optional[CognitiveEnergyMonitor] = None,
        ms_eng: Optional[InitiativeMilestoneEngine] = None,
    ):
        self.storage = storage or motion_storage
        self.drift_eng = drift_eng or drift_detector
        self.energy_eng = energy_eng or energy_monitor
        self.ms_eng = ms_eng or milestone_engine

    def generate_briefing(
        self,
        window_days: int = 7,
        reference_date: Optional[str] = None,
        save_briefing: bool = True,
    ) -> ExecutiveBriefing:
        """Synthesize all v1-v3 strategic telemetry into an executive briefing."""
        if not reference_date:
            reference_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")
        start_date = (ref_dt - timedelta(days=window_days - 1)).strftime("%Y-%m-%d")
        briefing_id = f"brf_{reference_date.replace('-', '')}_{uuid.uuid4().hex[:6]}"

        # 1. Gather Telemetry
        drift_report = self.drift_eng.evaluate_drift(
            window_days=window_days,
            reference_date=reference_date,
            save_report=True,
        )
        energy_report = self.energy_eng.evaluate_sustainability(
            window_days=window_days,
            reference_date=reference_date,
            save_report=True,
        )
        milestone_data = self.ms_eng.analyze_critical_path(reference_date=reference_date)
        trajectory = self.storage.load_trajectory()
        active_inits = [i for i in self.storage.load_initiatives() if i.status == "ACTIVE"]

        # 2. Extract Wins and Vulnerabilities
        key_wins: List[str] = []
        if drift_report.alignment_score >= 75.0:
            key_wins.append(f"Maintained high strategic discipline with {drift_report.alignment_score:.1f}% alignment.")
        if energy_report.flow_vs_thrash_ratio >= 1.5:
            key_wins.append(f"Exceptional deep work ratio ({energy_report.flow_vs_thrash_ratio:.1f} flow vs. fragmented).")
        if not key_wins:
            key_wins.append(f"Consistent execution across {drift_report.total_hours_analyzed:.1f} logged focus hours.")

        vulnerabilities: List[str] = []
        if drift_report.neglected_initiatives:
            for neg in drift_report.neglected_initiatives:
                vulnerabilities.append(f"Neglected Initiative: {neg}")
        if drift_report.runaway_categories:
            for run in drift_report.runaway_categories:
                vulnerabilities.append(f"Unplanned Time Sink: {run}")
        if energy_report.fatigue_risk_score >= 60.0:
            vulnerabilities.append(f"Cognitive Strain: {energy_report.sustainability_diagnosis}")
        if milestone_data.get("bottlenecks"):
            for b in milestone_data["bottlenecks"][:2]:
                vulnerabilities.append(f"Critical Path Bottleneck: {b}")

        # 3. Formulate Top Recommendations
        recommendations: List[MotionRecommendation] = []
        if drift_report.neglected_initiatives:
            recommendations.append(
                MotionRecommendation(
                    id=f"rec_brf_rebalance_{uuid.uuid4().hex[:4]}",
                    question_or_opportunity="Rebalance effort toward neglected strategic priorities",
                    recommendation_text=f"Schedule 2 dedicated focus blocks next week for '{drift_report.neglected_initiatives[0]}'.",
                    rationale="Prevents cumulative schedule slippage on active high-importance initiatives.",
                    expected_outcome="Restores strategic alignment index to >= 80%.",
                    review_trigger="Next Sunday Weekly Review",
                )
            )

        if energy_report.fatigue_risk_score >= 60.0:
            recommendations.append(
                MotionRecommendation(
                    id=f"rec_brf_decomp_{uuid.uuid4().hex[:4]}",
                    question_or_opportunity="Mitigate cognitive burnout risk",
                    recommendation_text=f"Enforce a {energy_report.recommended_decompression_hours:.1f}h restorative buffer block before high-intensity sprints.",
                    rationale=f"Fatigue score is at {energy_report.fatigue_risk_score:.1f}% ({energy_report.burnout_risk_level.value} tier).",
                    expected_outcome="Protects strategic decision clarity and reduces high-thrash context switching.",
                    review_trigger="Mid-week check-in",
                )
            )

        # 4. Proposed Trajectory Calibration
        proposed_calibration: Optional[Dict[str, Any]] = None
        if drift_report.velocity_vector.value != trajectory.momentum.value:
            proposed_calibration = {
                "momentum": drift_report.velocity_vector.value,
                "current_phase": trajectory.current_phase,
                "rationale": f"Observed velocity is {drift_report.velocity_vector.value} based on multi-window effort analysis.",
            }

        # 5. Build Formatted Executive Markdown Briefing
        lines = [
            f"# 🧭 Motion Executive Strategic Briefing",
            f"**Evaluation Window:** `{start_date}` to `{reference_date}`\n",
            "## 1. Strategic Health Telemetry",
            f"- 📊 **Strategic Alignment Index:** `{drift_report.alignment_score:.1f}%` ({drift_report.drift_severity.value})",
            f"- ⚡ **Velocity Vector:** `{drift_report.velocity_vector.value}` ({drift_report.total_hours_analyzed:.1f}h total logged)",
            f"- 🧠 **Cognitive Sustainability:** `{energy_report.fatigue_risk_score:.1f}%` Fatigue ({energy_report.burnout_risk_level.value} Risk, {energy_report.flow_vs_thrash_ratio:.1f} Flow Ratio)",
            "\n## 2. Key Achievements & Wins",
        ]
        for w in key_wins:
            lines.append(f"- 🏆 {w}")

        if vulnerabilities:
            lines.append("\n## 3. Strategic Vulnerabilities & Bottlenecks")
            for v in vulnerabilities:
                lines.append(f"- ⚠️ {v}")

        crit_path = milestone_data.get("critical_path", [])
        if crit_path:
            lines.append(f"\n## 4. Critical Path Milestones ({milestone_data.get('total_critical_hours', 0.0):.1f}h remaining)")
            for m in crit_path[:3]:
                lines.append(f"- 🎯 **{m['title']}** (`{m['status']}`, target: `{m.get('target_date', 'None')}`)")

        lines.append("\n## 5. Strategic Directives & Recommendations")
        for idx, r in enumerate(recommendations, 1):
            lines.append(f"{idx}. **{r.recommendation_text}**\n   *Rationale:* {r.rationale}")

        if proposed_calibration:
            lines.append(f"\n## 6. Proposed Trajectory Calibration\n- Calibrate momentum to `{proposed_calibration['momentum']}` ({proposed_calibration['rationale']})")

        formatted_md = "\n".join(lines)

        briefing = ExecutiveBriefing(
            id=briefing_id,
            period_start=start_date,
            period_end=reference_date,
            strategic_alignment_score=drift_report.alignment_score,
            drift_severity=drift_report.drift_severity,
            velocity_vector=drift_report.velocity_vector,
            fatigue_risk_score=energy_report.fatigue_risk_score,
            burnout_risk_level=energy_report.burnout_risk_level,
            milestone_summary=milestone_data,
            key_wins=key_wins,
            strategic_vulnerabilities=vulnerabilities,
            top_recommendations=recommendations,
            proposed_trajectory_calibration=proposed_calibration,
            formatted_markdown_briefing=formatted_md,
        )

        if save_briefing:
            self.storage.save_briefing(briefing)

        logger.info("Executive briefing generated for period %s to %s", start_date, reference_date)
        return briefing


executive_briefing_engine = ExecutiveBriefingEngine()
