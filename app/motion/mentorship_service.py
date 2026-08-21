"""Motion Mentorship Service: Strategic Reasoning & Prompt Engine.

Enforces Motion Response Policy, minimal prompt construction, evidence citations,
decision journal logging, and trajectory reasoning.
"""

from datetime import datetime, timezone
import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from app.motion.permissions import (
    MotionPermission,
    PermissionEngine,
    enforce_persona_permission,
    permission_engine,
)
from app.motion.retrieval import StrategicContextRetriever, strategic_context_retriever
from app.motion.schemas import (
    DecisionJournalEntry,
    HumanOverride,
    MotionConsultationResponse,
    MotionRecommendation,
    utc_now_iso,
)
from app.motion.spec import (
    DEFAULT_GEMINI_MODEL,
    DecisionStatus,
    MotionPermission,
    PersonaType,
)
from app.motion.strategic_model import StrategicModelService, strategic_model_service

logger = logging.getLogger("notion-assistant.motion.mentorship")


def get_gemini_client():
    """Create and return a google-genai Client instance."""
    if genai is None:
        raise RuntimeError("google-genai library is not installed or available")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    try:
        if types and hasattr(types, "HttpOptions"):
            http_opts = types.HttpOptions(timeout=30.0)
            return genai.Client(api_key=api_key, http_options=http_opts) if api_key else genai.Client(http_options=http_opts)
    except Exception:
        pass
    return genai.Client(api_key=api_key) if api_key else genai.Client()


MOTION_SYSTEM_INSTRUCTION = (
    "You are MOTION, the strategic mentorship persona inside Ocean.\n"
    "You specialize in long-term strategic reasoning, trajectory evaluation, high-leverage decision quality, and accountability.\n"
    "Ocean is responsible for execution. You are responsible for STRATEGY.\n\n"
    "CORE OPERATING PRINCIPLES:\n"
    "1. Optimize TRAJECTORY, not short-term conversation comfort.\n"
    "2. Base all reasoning strictly on the provided Structured Strategic Context (Identity, Trajectory, Initiatives, Observations, Conclusions, Drift Index).\n"
    "3. Distinguish clearly between Facts, Observations, Conclusions, and Recommendations.\n"
    "4. Explain trade-offs, second-order consequences, and opportunity costs of every strategic path.\n"
    "5. Preserve honest uncertainty. DO NOT assume unverified facts; ask the user conversationally if you need clarification on their context or rationale.\n"
    "6. Ask 1-2 high-leverage, challenging questions that clarify intent and expose blind spots.\n"
    "7. User has absolute final authority. You challenge rigorously, but never override.\n"
    "8. When Strategic Drift or Due Decision Reviews are present in context, proactively surface them with actionable rebalancing guidance.\n\n"
    "RESPONSE POLICY (STRICT):\n"
    "- DO NOT flatter, cheerlead, or give generic encouragement.\n"
    "- DO NOT argue endlessly or moralize.\n"
    "- DO NOT act as a coding bot, task runner, or productivity assistant (refer task execution to Ocean).\n"
    "- ALWAYS format with clean, standard markdown.\n"
)


class MotionMentorshipService:
    """Service providing strategic consultation, accountability, and decision logging."""

    def __init__(
        self,
        strategic_model: Optional[StrategicModelService] = None,
        retriever: Optional[StrategicContextRetriever] = None,
    ):
        self.strategic_model = strategic_model or strategic_model_service
        self.retriever = retriever or strategic_context_retriever

    @enforce_persona_permission(MotionPermission.READ_STRATEGIC_MODEL)
    def consult(
        self,
        user_message: str,
        sender_id: Optional[str] = None,
    ) -> MotionConsultationResponse:
        """Conduct strategic consultation with Motion Persona."""
        # 1. Dispatch permission verification
        permission_engine.assert_permission(MotionPermission.READ_STRATEGIC_MODEL, persona=PersonaType.MOTION)

        # 2. Assemble query-relevant strategic context
        context = self.retriever.assemble_context(query=user_message)
        context_str = self.retriever.format_context_for_prompt(context)

        prompt = (
            f"STRATEGIC CONTEXT:\n"
            f"{context_str}\n\n"
            f"---\n"
            f"USER STRATEGIC INQUIRY:\n"
            f"{user_message}\n\n"
            f"Provide a rigorous strategic analysis, identify key trade-offs, surface any strategic drift or risks, "
            f"and propose actionable recommendations with high-leverage questions."
        )

        try:
            client = get_gemini_client()
            model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=MOTION_SYSTEM_INSTRUCTION,
                ),
            )
            analysis_text = response.text or "Unable to generate strategic evaluation at this time."
        except Exception as exc:
            logger.error("Error during Motion LLM consultation: %s", exc)
            analysis_text = (
                f"🧭 **Motion Strategic Check-in**\n\n"
                f"**Current Direction:** {context.trajectory.current_direction}\n"
                f"**Key Focus:** {context.trajectory.biggest_opportunity}\n"
                f"**Strategic Risk:** {context.trajectory.biggest_risk}\n\n"
                f"*(Encountered temporary processing bottleneck: {exc})*"
            )

        # 3. Extract recommendations and log to Decision Journal if applicable
        recommendations: List[MotionRecommendation] = []
        rec_matches = re.findall(r"(?:Recommendation|Action Item|Strategic Next Step):\s*([^\n]+)", analysis_text, re.IGNORECASE)
        for i, rec_match in enumerate(rec_matches):
            rec_id = f"rec_{int(datetime.now(timezone.utc).timestamp())}_{i+1}"
            rec = MotionRecommendation(
                id=rec_id,
                conclusion_ids=[c.id for c in context.relevant_conclusions[:2]],
                question_or_opportunity=user_message[:100],
                recommendation_text=rec_match.strip(),
                rationale="Derived from strategic context and active trajectory evaluation.",
                trade_offs=["Allocation of focus away from secondary tasks"],
                expected_outcome="Reinforce trajectory velocity on primary initiatives.",
                review_trigger="Next Sunday Weekly Review",
            )
            recommendations.append(rec)

            # Auto-log to Decision Journal in PENDING status
            dec_entry = DecisionJournalEntry(
                id=f"dec_{rec_id}",
                question=user_message[:200],
                alternatives_considered=["Maintain status quo", "De-prioritize secondary work"],
                recommendation=rec.recommendation_text,
                reasoning=rec.rationale,
                expected_outcome=rec.expected_outcome,
                review_trigger=rec.review_trigger,
                status=DecisionStatus.PENDING,
            )
            self.strategic_model.record_decision(dec_entry)

        # 4. Extract questions
        questions = [
            q.strip() for q in re.findall(r"(?:^|\n)[-•*]?\s*([^\n]+\?)", analysis_text)
            if len(q.strip()) > 15
        ][:2]

        # 5. Format user-facing reply text
        reply_prefix = "🧭 **Motion Strategic Consultation**\n\n"
        reply_text = f"{reply_prefix}{analysis_text}"

        return MotionConsultationResponse(
            analysis=analysis_text,
            high_leverage_questions=questions,
            recommendations=recommendations,
            trade_offs_discussed=["Velocity vs. breadth", "Coursework vs. deep research"],
            cited_evidence_summary=[o.statement for o in context.relevant_observations[:3]],
            reply_text=reply_text,
        )

    def handle_user_override(
        self,
        recommendation_id_or_topic: str,
        user_decision: str,
        reason: str,
        review_trigger_condition: str,
    ) -> HumanOverride:
        """Record an explicit user override with a condition trigger."""
        override = HumanOverride(
            id=f"ovr_{int(datetime.now(timezone.utc).timestamp())}",
            recommendation_id_or_topic=recommendation_id_or_topic,
            user_decision=user_decision,
            reason=reason,
            review_trigger_condition=review_trigger_condition,
        )
        return self.strategic_model.record_override(override)

    def handle_decision_review(
        self,
        decision_id: str,
        actual_outcome: str,
        user_reflection: Optional[str] = None,
    ) -> DecisionJournalEntry:
        """Transition a decision from DUE/PENDING to REVIEWED and CLOSED."""
        return self.strategic_model.transition_decision_state(
            decision_id=decision_id,
            target_status=DecisionStatus.CLOSED,
            actual_outcome=actual_outcome,
            user_reflection=user_reflection,
        )


motion_mentorship_service = MotionMentorshipService()
