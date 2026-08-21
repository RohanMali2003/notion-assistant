"""Cognitive Energy & Sustainability Monitor for Ocean Motion v3.

Evaluates activity patterns, intensity bursts, flow-vs-thrash ratios, and consecutive
high-load days to compute Fatigue Risk Score (0-100%) and burnout tier classifications.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, List, Optional
import uuid

from app.motion.schemas import (
    BurnoutRiskLevel,
    CognitiveEnergyReport,
    EvidenceItem,
)
from app.motion.storage import MotionStorage, motion_storage

logger = logging.getLogger("notion-assistant.motion.energy_monitor")


class CognitiveEnergyMonitor:
    """Monitors cognitive sustainability, focus depth, and fatigue dynamics."""

    def __init__(self, storage: Optional[MotionStorage] = None):
        self.storage = storage or motion_storage

    def evaluate_sustainability(
        self,
        window_days: int = 7,
        reference_date: Optional[str] = None,
        save_report: bool = True,
    ) -> CognitiveEnergyReport:
        """Analyze activity dynamics over the specified window and formulate a sustainability report."""
        if not reference_date:
            reference_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")
        start_date = (ref_dt - timedelta(days=window_days - 1)).strftime("%Y-%m-%d")

        # 1. Load evidence in window
        evidence_items = self.storage.load_evidence(
            limit=300,
            start_date=start_date,
            end_date=reference_date,
        )

        report_id = f"energy_{reference_date.replace('-', '')}_{uuid.uuid4().hex[:6]}"

        if not evidence_items:
            # Baseline quiet/recovery report
            return CognitiveEnergyReport(
                id=report_id,
                period_start=start_date,
                period_end=reference_date,
                total_logged_hours=0.0,
                avg_daily_hours=0.0,
                consecutive_high_intensity_days=0,
                flow_vs_thrash_ratio=1.0,
                fatigue_risk_score=5.0,
                burnout_risk_level=BurnoutRiskLevel.LOW,
                sustainability_diagnosis="Minimal logged activity in period. Cognitive load is well within sustainable limits.",
                recommended_decompression_hours=0.0,
            )

        # 2. Daily aggregation
        daily_hours: Dict[str, float] = defaultdict(float)
        daily_item_counts: Dict[str, int] = defaultdict(int)

        for ev in evidence_items:
            dur = ev.duration_hours or 1.0
            daily_hours[ev.date] += dur
            daily_item_counts[ev.date] += 1

        total_hours = sum(daily_hours.values())
        avg_daily = round((total_hours / window_days), 2)
        active_days = len([d for d, h in daily_hours.items() if h > 0])
        avg_active_daily = round((total_hours / max(1, active_days)), 2)

        # 3. Calculate consecutive high intensity days (>= 7.0 hours/day)
        consecutive_high = 0
        current_streak = 0

        # Scan day by day chronologically
        for i in range(window_days):
            day_str = (ref_dt - timedelta(days=window_days - 1 - i)).strftime("%Y-%m-%d")
            hours_that_day = daily_hours.get(day_str, 0.0)
            if hours_that_day >= 7.0:
                current_streak += 1
                if current_streak > consecutive_high:
                    consecutive_high = current_streak
            else:
                current_streak = 0

        # 4. Compute Flow vs. Thrash Ratio
        # Deep sessions (dur >= 1.5h) vs fragmented context switching
        deep_hours = sum((e.duration_hours or 1.0) for e in evidence_items if (e.duration_hours or 1.0) >= 1.5)
        fragmented_hours = sum((e.duration_hours or 1.0) for e in evidence_items if (e.duration_hours or 1.0) < 1.5)

        if fragmented_hours == 0:
            flow_ratio = 2.5
        else:
            flow_ratio = round((deep_hours / fragmented_hours), 2)

        # 5. Compute Fatigue Risk Score (0 - 100%)
        fatigue_score = 0.0

        # Base from daily hours and active intensity
        if avg_daily >= 8.0 or avg_active_daily >= 8.0:
            fatigue_score += 45.0
        elif avg_daily >= 6.0 or avg_active_daily >= 6.5:
            fatigue_score += 35.0
        elif avg_daily >= 4.0:
            fatigue_score += 20.0
        else:
            fatigue_score += 10.0

        # Consecutive high intensity day penalty
        fatigue_score += min(35.0, consecutive_high * 7.0)

        # Thrash penalty (fragmented context switching increases mental fatigue)
        if flow_ratio < 0.6 and total_hours > 15.0:
            fatigue_score += 15.0
        elif flow_ratio >= 1.5:
            fatigue_score -= 5.0  # Flow state buffer

        fatigue_score = max(0.0, min(100.0, round(fatigue_score, 1)))

        # 6. Burnout Tier & Recommendation
        if fatigue_score >= 80.0:
            burnout_level = BurnoutRiskLevel.CRITICAL
            diagnosis = (
                f"Critical fatigue strain detected ({avg_daily:.1f}h/day avg, {consecutive_high} day high-intensity streak). "
                "Immediate schedule buffer and deliberate decompression recommended before strategic decision quality collapses."
            )
            decomp_hours = 6.0
        elif fatigue_score >= 60.0:
            burnout_level = BurnoutRiskLevel.HIGH
            diagnosis = (
                f"Elevated cognitive fatigue ({avg_daily:.1f}h/day avg). "
                "Sustained output is nearing capacity threshold. Protect evening recovery windows."
            )
            decomp_hours = 3.0
        elif fatigue_score >= 35.0:
            burnout_level = BurnoutRiskLevel.ELEVATED
            diagnosis = f"Moderate intensity ({avg_daily:.1f}h/day avg, flow ratio {flow_ratio}). Output is productive and sustainable with normal rest cycles."
            decomp_hours = 1.0
        else:
            burnout_level = BurnoutRiskLevel.LOW
            diagnosis = f"Cognitive energy is fresh and well-balanced ({avg_daily:.1f}h/day avg). High capacity available for deep focus blocks."
            decomp_hours = 0.0

        report = CognitiveEnergyReport(
            id=report_id,
            period_start=start_date,
            period_end=reference_date,
            total_logged_hours=round(total_hours, 1),
            avg_daily_hours=avg_daily,
            consecutive_high_intensity_days=consecutive_high,
            flow_vs_thrash_ratio=flow_ratio,
            fatigue_risk_score=fatigue_score,
            burnout_risk_level=burnout_level,
            sustainability_diagnosis=diagnosis,
            recommended_decompression_hours=decomp_hours,
        )

        if save_report:
            self.storage.save_energy_report(report)

        logger.info(
            "Energy evaluation complete: %0.1f%% fatigue score (%s tier)",
            fatigue_score,
            burnout_level.value,
        )
        return report


energy_monitor = CognitiveEnergyMonitor()
