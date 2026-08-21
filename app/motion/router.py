"""Persona Router for Ocean and Motion.

Intercepts messages BEFORE prompt construction or tool dispatch to decide whether
the Motion persona should be activated.
"""

import logging
import re
from typing import Optional, Tuple

from app.motion.accountability import accountability_monitor
from app.motion.attribution import multi_hop_attribution_engine
from app.motion.drift_detector import drift_detector
from app.motion.energy_monitor import energy_monitor
from app.motion.executive_briefing import executive_briefing_engine
from app.motion.forecaster import scenario_engine
from app.motion.mentorship_service import MotionMentorshipService, motion_mentorship_service
from app.motion.milestones import milestone_engine
from app.motion.permissions import PermissionEngine, permission_engine
from app.motion.schemas import ScenarioSimulationRequest
from app.motion.socratic import socratic_engine
from app.motion.spec import PersonaType

logger = logging.getLogger("notion-assistant.motion.router")


class PersonaRouter:
    """Ingress Persona Router intercepting messages prior to LLM invocation."""

    def __init__(self, mentorship_service: Optional[MotionMentorshipService] = None):
        self.mentorship = mentorship_service or motion_mentorship_service

    def is_motion_invoked(self, text: str) -> bool:
        """Check if message explicitly invokes the Motion persona."""
        if not text:
            return False
        clean = text.strip().lower()

        # Explicit tag/mention triggers
        if "@motion" in clean or clean.startswith("motion:") or clean.startswith("motion,") or clean.startswith("/motion"):
            return True

        # Explicit strategic keywords
        if clean.startswith("ask motion") or clean.startswith("motion "):
            return True

        return False

    def extract_motion_query(self, text: str) -> str:
        """Strip @motion tag / prefix to obtain the clean user inquiry."""
        clean = text.strip()
        # Remove @motion (case-insensitive)
        clean = re.sub(r"@motion\b", "", clean, flags=re.IGNORECASE).strip()
        # Remove leading "motion:" or "motion," or "/motion" or "ask motion"
        clean = re.sub(r"^(?:motion\s*[:,-]?|/motion\s*|ask\s+motion\s*)", "", clean, flags=re.IGNORECASE).strip()
        return clean or text.strip()

    def route_message(self, text: str) -> Tuple[PersonaType, str]:
        """Route message to target persona and return (persona, clean_text)."""
        if self.is_motion_invoked(text):
            clean_query = self.extract_motion_query(text)
            logger.info("Persona Router activated MOTION persona for query: '%s'", clean_query[:50])
            return PersonaType.MOTION, clean_query
        return PersonaType.OCEAN, text

    def process_motion_request(self, text: str, sender_id: Optional[str] = None) -> str:
        """Execute Motion consultation pipeline under PersonaType.MOTION context."""
        clean_query = self.extract_motion_query(text)
        lower_query = clean_query.lower()

        # Set active persona context
        PermissionEngine.set_current_persona(PersonaType.MOTION)
        try:
            # Check 1: Attribution inquiry ("why do you believe <id>" / "explain belief <id>")
            match_why = re.search(r"(?:why\s+do\s+you\s+believe|explain\s+belief|provenance)\s+([a-zA-Z0-9_\-]+)", lower_query)
            if match_why:
                target_id = match_why.group(1).strip()
                tree = multi_hop_attribution_engine.build_attribution_tree(target_id)
                return multi_hop_attribution_engine.format_tree_as_markdown(tree)

            # Check 2: Daily Check-In inquiry ("check in", "daily check in", "status report")
            if lower_query in ("check in", "daily check in", "daily check-in", "status report", "checkin", "morning briefing"):
                latest_drift = drift_detector.evaluate_drift(save_report=False)
                checkin = accountability_monitor.generate_daily_checkin(drift_report=latest_drift)
                return checkin.reply_text

            # Check 3: Drift report inquiry ("drift report", "alignment score", "show drift")
            if lower_query in ("drift report", "alignment score", "show drift", "check drift", "drift"):
                drift = drift_detector.evaluate_drift(save_report=True)
                lines = [
                    f"📊 **Motion Strategic Drift Report**",
                    f"- **Period:** {drift.period_start} to {drift.period_end}",
                    f"- **Strategic Alignment Score:** {drift.alignment_score:.1f}% ({drift.drift_severity.value})",
                    f"- **Velocity Vector:** {drift.velocity_vector.value}",
                    f"- **Hours Analyzed:** {drift.total_hours_analyzed:.1f}h",
                ]
                if drift.neglected_initiatives:
                    lines.append(f"\n⚠️ **Neglected Initiatives:** {', '.join(drift.neglected_initiatives)}")
                if drift.runaway_categories:
                    lines.append(f"\n⚠️ **Runaway Categories:** {', '.join(drift.runaway_categories)}")
                if drift.recommendations_for_rebalancing:
                    lines.append("\n🎯 **Rebalancing Actions:**")
                    for rec in drift.recommendations_for_rebalancing:
                        lines.append(f"- {rec}")
                return "\n".join(lines)

            # Check 4: Cognitive Energy & Fatigue ("energy", "fatigue", "burnout")
            if lower_query in ("energy", "fatigue", "burnout", "check energy", "energy report", "capacity"):
                energy = energy_monitor.evaluate_sustainability(window_days=7, save_report=True)
                lines = [
                    f"🧠 **Motion Cognitive Sustainability Report**",
                    f"- **Period:** {energy.period_start} to {energy.period_end}",
                    f"- **Fatigue Risk Score:** {energy.fatigue_risk_score:.1f}% ({energy.burnout_risk_level.value} Tier)",
                    f"- **Average Daily Hours:** {energy.avg_daily_hours:.1f}h/day ({energy.total_logged_hours:.1f}h total)",
                    f"- **High-Intensity Streak:** {energy.consecutive_high_intensity_days} consecutive days >= 7h",
                    f"- **Flow vs. Thrash Ratio:** {energy.flow_vs_thrash_ratio:.2f}",
                    f"\n**Diagnosis:** {energy.sustainability_diagnosis}",
                ]
                if energy.recommended_decompression_hours > 0:
                    lines.append(f"\n🧘 **Recommended Restorative Buffer:** {energy.recommended_decompression_hours:.1f} hours")
                return "\n".join(lines)

            # Check 5: Critical Path & Milestones ("milestones", "critical path", "bottlenecks")
            if lower_query in ("milestones", "critical path", "bottlenecks", "milestone status", "show milestones"):
                ms_data = milestone_engine.analyze_critical_path()
                lines = [
                    f"🎯 **Motion Critical Path & Milestone Analysis**",
                    f"- **Summary:** {ms_data['summary']}",
                    f"- **Critical Path Hours Remaining:** {ms_data['total_critical_hours']:.1f}h",
                ]
                crit_milestones = ms_data.get("critical_path", [])
                if crit_milestones:
                    lines.append("\n**Critical Path Milestones:**")
                    for m in crit_milestones:
                        lines.append(f"- `{m['id']}`: **{m['title']}** (Status: `{m['status']}`, Est: {m['estimated_hours']}h)")
                if ms_data.get("bottlenecks"):
                    lines.append("\n⚠️ **Active Bottlenecks:**")
                    for b in ms_data["bottlenecks"]:
                        lines.append(f"- {b}")
                return "\n".join(lines)

            # Check 6: Executive Strategic Briefing ("briefing", "executive briefing", "weekly briefing", "sunday briefing")
            if lower_query in ("briefing", "executive briefing", "weekly briefing", "sunday briefing", "generate briefing"):
                briefing = executive_briefing_engine.generate_briefing(window_days=7, save_briefing=True)
                return briefing.formatted_markdown_briefing

            # Check 7: Socratic Pre-Mortem ("pre-mortem <topic>", "socratic <topic>")
            match_socratic = re.match(r"(?:pre-mortem|premortem|socratic|inversion)\s+(.+)", lower_query)
            if match_socratic:
                target_topic = match_socratic.group(1).strip()
                inquiry = socratic_engine.conduct_inquiry(topic=target_topic)
                return socratic_engine.format_socratic_response(inquiry)

            # Check 8: Scenario Simulation ("scenario <topic>", "simulate <topic>")
            match_scenario = re.match(r"(?:scenario|simulate)\s+(.+)", lower_query)
            if match_scenario:
                sim_text = match_scenario.group(1).strip()
                # Parse simple delta or fallback to standard reallocation
                sim_req = ScenarioSimulationRequest(
                    scenario_name=sim_text.capitalize(),
                    description=sim_text,
                    time_adjustments={"LeetCode & Algorithms": 5.0, "Systems Engineering & Projects": -5.0},
                    timeframe_weeks=4,
                )
                sim_res = scenario_engine.simulate_scenario(sim_req, save_result=True)
                lines = [
                    f"🔮 **Motion Scenario Simulation Result** (`{sim_res.scenario_name}`)",
                    f"- **Alignment Score Impact:** {sim_res.current_alignment_score:.1f}% ➔ {sim_res.projected_alignment_score:.1f}%",
                    f"- **Recommendation Verdict:** {sim_res.recommendation_verdict}",
                    "\n**Projected Shifts & Trade-Offs:**",
                ]
                for to in sim_res.trade_offs_identified:
                    lines.append(f"- {to}")
                if sim_res.bottlenecks_flagged:
                    lines.append("\n⚠️ **Bottlenecks Flagged:**")
                    for bn in sim_res.bottlenecks_flagged:
                        lines.append(f"- {bn}")
                return "\n".join(lines)

            # Default: Full LLM strategic consultation
            consultation = self.mentorship.consult(user_message=clean_query, sender_id=sender_id)
            return consultation.reply_text
        finally:
            # Restore default Ocean persona
            PermissionEngine.set_current_persona(PersonaType.OCEAN)


persona_router = PersonaRouter()
