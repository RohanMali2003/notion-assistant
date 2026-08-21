"""Socratic Mentorship & Pre-Mortem Dialogue Engine for Ocean Motion v3.

Performs assumption deconstruction, pre-mortem failure mode analysis, inversion thinking,
and second-order consequence modeling on user dilemmas and strategic proposals.
"""

from datetime import datetime, timezone
import logging
from typing import List, Optional
import uuid

from app.motion.retrieval import StrategicContextRetriever, strategic_context_retriever
from app.motion.schemas import SocraticInquiryResult
from app.motion.storage import MotionStorage, motion_storage

logger = logging.getLogger("notion-assistant.motion.socratic")


class SocraticMentorshipEngine:
    """Deconstructs assumptions, performs pre-mortem failure analysis, and formulates Socratic challenges."""

    def __init__(
        self,
        storage: Optional[MotionStorage] = None,
        retriever: Optional[StrategicContextRetriever] = None,
    ):
        self.storage = storage or motion_storage
        self.retriever = retriever or strategic_context_retriever

    def conduct_inquiry(
        self,
        topic: str,
        user_context: Optional[str] = None,
    ) -> SocraticInquiryResult:
        """Analyze a strategic proposal or dilemma through the Socratic Inversion framework."""
        inquiry_id = f"soc_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"

        # Load active strategic model context
        trajectory = self.storage.load_trajectory()
        active_inits = [i for i in self.storage.load_initiatives() if i.status == "ACTIVE"]

        topic_lower = topic.lower()

        # 1. Deconstruct Implicit Assumptions
        assumptions: List[str] = []
        if "switch" in topic_lower or "drop" in topic_lower or "change" in topic_lower:
            assumptions.append("Assumes switching costs are low and context rebuild will be immediate.")
        if "more time" in topic_lower or "hours" in topic_lower or "double down" in topic_lower:
            assumptions.append("Assumes cognitive energy is elastic and additional hours scale linearly without fatigue.")
        if "parallel" in topic_lower or "both" in topic_lower or "all" in topic_lower:
            assumptions.append("Assumes multiple demanding initiatives can maintain high momentum without context switching penalties.")
        if not assumptions:
            assumptions.append(f"Assumes current execution velocity in '{trajectory.current_phase}' remains stable under new conditions.")
            assumptions.append("Assumes external academic and personal constraints will not introduce sudden friction.")

        # 2. Pre-Mortem Failure Analysis (6 Months Out)
        premortems: List[str] = [
            f"Scenario A (Execution Fragility): 6 months from now, '{topic[:40]}' stalled because daily focus became fragmented across secondary tasks.",
            f"Scenario B (Cognitive Exhaustion): Burnout or fatigue forced an unforced multi-week pause, derailing primary milestone timelines.",
            f"Scenario C (Strategic Drift): Progress occurred, but upon review, it failed to unlock the primary goal: '{trajectory.biggest_opportunity}'.",
        ]

        # 3. Second-Order Consequences
        second_order: List[str] = []
        for init in active_inits[:2]:
            second_order.append(f"Capacity diverted to '{topic[:30]}' directly reduces focus on active initiative '{init.title}'.")
        second_order.append("Increases decision complexity during weekly trajectory calibrations.")

        # 4. Inversion Analysis (Charlie Munger Principle)
        inversion = (
            f"To guarantee complete failure of '{topic[:30]}': spread focus thin across 4+ competing goals, ignore fatigue signals, "
            "and avoid committing to a single measurable milestone. Inverting this: protect one primary initiative and enforce clear stop-conditions."
        )

        # 5. Probing Socratic Questions
        questions: List[str] = [
            f"If you could ONLY complete one strategic objective this quarter between this and '{active_inits[0].title if active_inits else 'current trajectory'}', which provides 10x higher career leverage?",
            "What specific metric or condition will trigger an immediate rollback if this direction fails to show traction in 14 days?",
        ]

        result = SocraticInquiryResult(
            id=inquiry_id,
            topic=topic,
            unexamined_assumptions=assumptions,
            premortem_failure_scenarios=premortems,
            second_order_consequences=second_order,
            inversion_analysis=inversion,
            probing_questions=questions,
        )

        logger.info("Socratic inquiry completed for topic: '%s'", topic[:50])
        return result

    def format_socratic_response(self, inquiry: SocraticInquiryResult) -> str:
        """Format Socratic inquiry result into clean, impactful markdown for the user."""
        lines = [
            f"🏛️ **Motion Socratic Strategic Inquiry** (`{inquiry.topic}`)",
            "\n### 1. Unexamined Assumptions",
        ]
        for a in inquiry.unexamined_assumptions:
            lines.append(f"- 🔍 *{a}*")

        lines.append("\n### 2. Pre-Mortem Failure Analysis (6 Months Out)")
        for pm in inquiry.premortem_failure_scenarios:
            lines.append(f"- ⚠️ {pm}")

        lines.append("\n### 3. Second-Order Collateral Effects")
        for so in inquiry.second_order_consequences:
            lines.append(f"- 🌊 {so}")

        lines.append(f"\n### 4. Inversion Strategy\n> {inquiry.inversion_analysis}")

        lines.append("\n### 5. High-Leverage Socratic Challenge")
        for q in inquiry.probing_questions:
            lines.append(f"**❓ {q}**")

        return "\n".join(lines)


socratic_engine = SocraticMentorshipEngine()
