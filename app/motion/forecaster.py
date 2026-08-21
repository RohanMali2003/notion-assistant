"""Predictive Trajectory Forecaster & Scenario Simulator for Ocean Motion v3.

Evaluates hypothetical time and focus reallocations against real-world historical velocity,
projecting milestone timeline shifts, alignment score changes, and second-order trade-offs.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional
import uuid

from app.motion.drift_detector import StrategicDriftDetector, drift_detector
from app.motion.schemas import (
    InitiativeMilestone,
    MotionInitiative,
    ScenarioSimulationRequest,
    ScenarioSimulationResult,
)
from app.motion.storage import MotionStorage, motion_storage

logger = logging.getLogger("notion-assistant.motion.forecaster")


class ScenarioSimulationEngine:
    """Simulates strategic what-if reallocations and projects trajectory impact."""

    def __init__(
        self,
        storage: Optional[MotionStorage] = None,
        drift_engine: Optional[StrategicDriftDetector] = None,
    ):
        self.storage = storage or motion_storage
        self.drift_engine = drift_engine or drift_detector

    def simulate_scenario(
        self,
        request: ScenarioSimulationRequest,
        save_result: bool = True,
    ) -> ScenarioSimulationResult:
        """Simulate a proposed time reallocation and compute trajectory forecasts."""
        sim_id = f"sim_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"

        # 1. Baseline Drift & Alignment Evaluation
        baseline_drift = self.drift_engine.evaluate_drift(window_days=7, save_report=False)
        current_alignment = baseline_drift.alignment_score
        baseline_hours = baseline_drift.total_hours_analyzed or 20.0
        baseline_breakdown = dict(baseline_drift.category_breakdown)

        active_inits = [i for i in self.storage.load_initiatives() if i.status == "ACTIVE"]
        active_milestones = self.storage.load_milestones()

        # 2. Compute Projected Category Breakdown & Alignment
        projected_breakdown = dict(baseline_breakdown)
        total_adj = 0.0

        for cat_or_init, delta_hrs in request.time_adjustments.items():
            matched_cat = self._resolve_category(cat_or_init, active_inits)
            current_pct = projected_breakdown.get(matched_cat, 0.0)
            current_hrs = (current_pct / 100.0) * baseline_hours
            new_hrs = max(0.0, current_hrs + delta_hrs)
            total_adj += (new_hrs - current_hrs)
            projected_breakdown[matched_cat] = new_hrs

        new_total_hours = max(1.0, baseline_hours + total_adj)
        # Recompute percentages
        for cat in list(projected_breakdown.keys()):
            projected_breakdown[cat] = round((projected_breakdown[cat] / new_total_hours * 100.0), 1)

        # Estimate Projected Alignment Score
        strategic_cats = set()
        for init in active_inits:
            for kw in [w.lower() for w in init.title.split() if len(w) > 3]:
                for cat in projected_breakdown.keys():
                    if kw in cat.lower() or "project" in cat.lower() or "algorithm" in cat.lower() or "learning" in cat.lower():
                        strategic_cats.add(cat)

        strat_pct = sum(pct for cat, pct in projected_breakdown.items() if cat in strategic_cats)
        projected_alignment = min(100.0, max(0.0, strat_pct))

        # 3. Compute Projected Completion Shifts (Days)
        completion_shifts: Dict[str, int] = {}
        trade_offs: List[str] = []
        bottlenecks: List[str] = []

        for target, delta_hrs in request.time_adjustments.items():
            weekly_rate = max(1.0, abs(delta_hrs))
            # Shift in days over simulation horizon: (delta_hrs * weeks) / avg_daily_rate
            total_delta_focus_hours = delta_hrs * request.timeframe_weeks
            # Positive delta accelerates (negative days shift), negative delta delays (positive days shift)
            days_shift = -int(round((total_delta_focus_hours / 2.0)))

            target_label = target
            matching_init = next((i for i in active_inits if target.lower() in i.title.lower() or target.lower() == i.id.lower()), None)
            if matching_init:
                target_label = matching_init.title
                completion_shifts[matching_init.id] = days_shift
            else:
                completion_shifts[target] = days_shift

            # Trade-off narrative
            if delta_hrs > 0:
                trade_offs.append(
                    f"Allocating +{delta_hrs:.1f}h/week to '{target_label}' accelerates progress by ~{abs(days_shift)} days over {request.timeframe_weeks} weeks."
                )
            elif delta_hrs < 0:
                trade_offs.append(
                    f"Reducing '{target_label}' by {abs(delta_hrs):.1f}h/week introduces a projected ~{days_shift} day delay."
                )
                if matching_init and matching_init.strategic_importance.value == "CRITICAL":
                    bottlenecks.append(f"Crucial initiative '{matching_init.title}' deprioritized by {abs(delta_hrs):.1f}h/week.")

        # 4. Synthesize Recommendation Verdict
        alignment_diff = projected_alignment - current_alignment
        if alignment_diff >= 5.0 and not bottlenecks:
            verdict = f"Favorable scenario (+{alignment_diff:.1f}% alignment). Increases focus on strategic priorities without creating critical bottlenecks."
        elif bottlenecks:
            verdict = f"High-risk scenario. Although alignment changes by {alignment_diff:+.1f}%, it deprioritizes critical path initiatives ({', '.join(bottlenecks)})."
        elif alignment_diff < -10.0:
            verdict = f"Unfavorable scenario ({alignment_diff:.1f}% alignment drop). Shifts effort away from core strategic initiatives."
        else:
            verdict = f"Neutral trade-off scenario (alignment shifts from {current_alignment:.1f}% to {projected_alignment:.1f}%)."

        result = ScenarioSimulationResult(
            id=sim_id,
            scenario_name=request.scenario_name,
            current_alignment_score=current_alignment,
            projected_alignment_score=projected_alignment,
            projected_completion_shifts=completion_shifts,
            trade_offs_identified=trade_offs,
            bottlenecks_flagged=bottlenecks,
            confidence_score=0.85,
            recommendation_verdict=verdict,
        )

        if save_result:
            self.storage.save_simulation(result)

        logger.info(
            "Simulation '%s' complete: alignment %0.1f%% -> %0.1f%%",
            request.scenario_name,
            current_alignment,
            projected_alignment,
        )
        return result

    def _resolve_category(self, target: str, active_inits: List[MotionInitiative]) -> str:
        """Resolve a category or initiative string to canonical category name."""
        target_lower = target.lower()
        if "leetcode" in target_lower or "algorithm" in target_lower:
            return "LeetCode & Algorithms"
        if "learning" in target_lower or "study" in target_lower or "academics" in target_lower:
            return "Learning & Deep Knowledge"
        if "ocean" in target_lower or "systems" in target_lower or "project" in target_lower:
            return "Systems Engineering & Projects"
        if "admin" in target_lower or "email" in target_lower:
            return "Administrative & Operations"
        return target


scenario_engine = ScenarioSimulationEngine()
