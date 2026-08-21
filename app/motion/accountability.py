"""Proactive Accountability & Daily Strategic Check-In Engine for Ocean Motion v2.

Monitors Decision Journal review schedules, scans Human Override trigger conditions,
and crafts focused daily check-ins to maintain unbroken strategic alignment.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Tuple

from app.motion.schemas import (
    DailyCheckInPrompt,
    DecisionJournalEntry,
    HumanOverride,
    StrategicDriftReport,
)
from app.motion.spec import DecisionStatus, OverrideStatus
from app.motion.storage import MotionStorage, motion_storage

logger = logging.getLogger("notion-assistant.motion.accountability")


class AccountabilityMonitor:
    """Monitors decisions, overrides, and initiates proactive strategic check-ins."""

    def __init__(self, storage: Optional[MotionStorage] = None):
        self.storage = storage or motion_storage

    def scan_due_decisions(self, current_date: Optional[str] = None) -> List[DecisionJournalEntry]:
        """Identify Decision Journal entries whose review date has arrived and transition them to DUE."""
        if not current_date:
            current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        entries = self.storage.load_decision_journal()
        due_list: List[DecisionJournalEntry] = []
        updated = False

        for entry in entries:
            if entry.status == DecisionStatus.PENDING and entry.review_scheduled_for:
                if entry.review_scheduled_for <= current_date:
                    entry.status = DecisionStatus.DUE
                    due_list.append(entry)
                    updated = True
            elif entry.status == DecisionStatus.DUE:
                due_list.append(entry)

        if updated:
            self.storage.save_decision_journal(entries)
            logger.info("Transitioned %d decisions to DUE state as of %s.", len(due_list), current_date)

        return due_list

    def scan_override_triggers(self) -> List[Tuple[HumanOverride, str]]:
        """Scan active overrides against recent observations to detect potential trigger satisfaction."""
        overrides = self.storage.load_overrides()
        active_overrides = [o for o in overrides if o.status == OverrideStatus.ACTIVE]
        observations = self.storage.load_observations(limit=50)

        obs_text = " ".join((o.observation_summary or o.statement or "") for o in observations).lower()
        triggered_list: List[Tuple[HumanOverride, str]] = []

        for ovr in active_overrides:
            trigger = ovr.review_trigger_condition.lower()
            trigger_keywords = [w for w in trigger.split() if len(w) > 3]
            if any(kw in obs_text for kw in trigger_keywords):
                reason = f"Recent activity matching condition: '{ovr.review_trigger_condition}'"
                triggered_list.append((ovr, reason))

        return triggered_list

    def generate_daily_checkin(
        self,
        current_date: Optional[str] = None,
        drift_report: Optional[StrategicDriftReport] = None,
    ) -> DailyCheckInPrompt:
        """Generate a proactive, high-leverage daily status check-in message."""
        if not current_date:
            current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        trajectory = self.storage.load_trajectory()
        due_decisions = self.scan_due_decisions(current_date)
        triggered_overrides = self.scan_override_triggers()

        score_text = ""
        if drift_report:
            score_text = (
                f"📊 **Strategic Alignment:** {drift_report.alignment_score:.1f}% "
                f"({drift_report.drift_severity.value}, {drift_report.velocity_vector.value} velocity)"
            )
        else:
            score_text = "📊 **Strategic Alignment:** Active monitoring initialized."

        due_alerts = [f"Decision `{d.id}` (*{d.recommendation[:50]}...*)" for d in due_decisions]

        # Formulate highest-leverage question based on current phase and drift
        high_leverage_q = ""
        if due_decisions:
            high_leverage_q = f"What is the actual outcome of decision `{due_decisions[0].id}` that was due on {due_decisions[0].review_scheduled_for}?"
        elif drift_report and drift_report.neglected_initiatives:
            neg_init = drift_report.neglected_initiatives[0]
            high_leverage_q = f"What is the single biggest blocker preventing progress on '{neg_init}' today?"
        elif trajectory.biggest_risk:
            high_leverage_q = f"What proactive step will you take today to eliminate or mitigate our primary risk: '{trajectory.biggest_risk}'?"
        else:
            high_leverage_q = "What is the single highest-leverage priority you will complete today?"

        # Build formatted message
        lines = [
            f"🧭 **Motion Daily Check-In** (`{current_date}`)",
            f"\n**Trajectory Phase:** {trajectory.current_phase} (Direction: {trajectory.current_direction})",
            f"{score_text}",
        ]

        if due_decisions:
            lines.append("\n⚠️ **Decision Journal Reviews Due:**")
            for d in due_decisions:
                lines.append(f"- `{d.id}`: *{d.recommendation}*")

        if triggered_overrides:
            lines.append("\n🔄 **Override Conditions Met:**")
            for ovr, reason in triggered_overrides:
                lines.append(f"- Topic: *{ovr.recommendation_id_or_topic}* — {reason}")

        if drift_report and drift_report.runaway_categories:
            lines.append(f"\n⚠️ **Drift Alert:** Runaway focus on {', '.join(drift_report.runaway_categories)}")

        lines.append(f"\n🎯 **Strategic Focus for Today:**")
        lines.append(f"> {high_leverage_q}")

        reply_text = "\n".join(lines)

        return DailyCheckInPrompt(
            date=current_date,
            greeting=f"Daily Strategic Alignment for {current_date}",
            alignment_score_summary=score_text,
            due_decision_alerts=due_alerts,
            high_leverage_question=high_leverage_q,
            reply_text=reply_text,
        )


accountability_monitor = AccountabilityMonitor()
