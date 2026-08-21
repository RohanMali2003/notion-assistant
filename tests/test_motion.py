"""Unit and integration tests for Ocean Motion (v4.0 flagship strategic mentorship subsystem)."""

import os
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

from app.main import app
from app.motion.spec import (
    ConfidenceLevel,
    DecisionStatus,
    Horizon,
    MotionPermission,
    OverrideStatus,
    PersonaType,
    StrategicImportance,
    TrajectoryMomentum,
)
from app.motion.schemas import (
    DecisionJournalEntry,
    EvidenceItem,
    HumanOverride,
    MotionConclusion,
    MotionIdentity,
    MotionInitiative,
    MotionObservation,
    MotionTrajectory,
    MotionWeeklyReview,
)
from app.motion.permissions import (
    MotionPermissionError,
    PermissionEngine,
    enforce_persona_permission,
    permission_engine,
)
from app.motion.storage import MotionStorage
from app.motion.strategic_model import StrategicModelService
from app.motion.evidence_pipeline import EvidenceAttributionEngine, EvidencePipeline
from app.motion.retrieval import StrategicContextRetriever
from app.motion.mentorship_service import MotionMentorshipService
from app.motion.review_service import MotionReviewService
from app.motion.router import PersonaRouter


@pytest.fixture
def temp_motion_storage():
    """Create a temporary storage directory for isolated testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_motion_")
    storage = MotionStorage(base_dir=temp_dir)
    yield storage
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def motion_services(temp_motion_storage):
    """Instantiate all motion services with temporary isolated storage."""
    strat_model = StrategicModelService(storage=temp_motion_storage)
    pipeline = EvidencePipeline(storage=temp_motion_storage)
    attribution = EvidenceAttributionEngine(storage=temp_motion_storage)
    retriever = StrategicContextRetriever(storage=temp_motion_storage)
    mentorship = MotionMentorshipService(strategic_model=strat_model, retriever=retriever)
    review = MotionReviewService(storage=temp_motion_storage, strategic_model=strat_model, pipeline=pipeline)
    router = PersonaRouter(mentorship_service=mentorship)

    return {
        "storage": temp_motion_storage,
        "strategic_model": strat_model,
        "pipeline": pipeline,
        "attribution": attribution,
        "retriever": retriever,
        "mentorship": mentorship,
        "review": review,
        "router": router,
    }


# =========================================================
# --- 1. Tool Permission & Dispatch Enforcement Tests ---
# =========================================================

def test_permission_engine_allowed_and_forbidden():
    """Verify Motion and Ocean permissions are strictly enforced."""
    engine = PermissionEngine()

    # Ocean persona has full access
    assert engine.is_permission_allowed(MotionPermission.READ_LOGS, persona=PersonaType.OCEAN) is True
    assert engine.is_permission_allowed(MotionPermission.EXECUTE_TASKS, persona=PersonaType.OCEAN) is True
    assert engine.is_permission_allowed(MotionPermission.MODIFY_NOTION, persona=PersonaType.OCEAN) is True

    # Motion persona allowed permissions
    assert engine.is_permission_allowed(MotionPermission.READ_LOGS, persona=PersonaType.MOTION) is True
    assert engine.is_permission_allowed(MotionPermission.READ_STRATEGIC_MODEL, persona=PersonaType.MOTION) is True
    assert engine.is_permission_allowed(MotionPermission.READ_CALENDAR_TASKS, persona=PersonaType.MOTION) is True
    assert engine.is_permission_allowed(MotionPermission.READ_PROJECTS, persona=PersonaType.MOTION) is True
    assert engine.is_permission_allowed(MotionPermission.READ_DECISION_JOURNAL, persona=PersonaType.MOTION) is True

    # Motion persona forbidden permissions
    assert engine.is_permission_allowed(MotionPermission.EXECUTE_TASKS, persona=PersonaType.MOTION) is False
    assert engine.is_permission_allowed(MotionPermission.MODIFY_NOTION, persona=PersonaType.MOTION) is False
    assert engine.is_permission_allowed(MotionPermission.CREATE_TASKS, persona=PersonaType.MOTION) is False
    assert engine.is_permission_allowed(MotionPermission.WRITE_CODE, persona=PersonaType.MOTION) is False
    assert engine.is_permission_allowed(MotionPermission.RUN_SEARCHES, persona=PersonaType.MOTION) is False
    assert engine.is_permission_allowed(MotionPermission.INVOKE_CODING_TOOLS, persona=PersonaType.MOTION) is False


def test_permission_decorator_raises_error_for_motion():
    """Verify @enforce_persona_permission raises MotionPermissionError when forbidden."""
    @enforce_persona_permission(MotionPermission.EXECUTE_TASKS)
    def dangerous_task_execution():
        return "Executed"

    @enforce_persona_permission(MotionPermission.READ_STRATEGIC_MODEL)
    def safe_strategic_reading():
        return "Strategic Data"

    # Under Ocean persona: both succeed
    PermissionEngine.set_current_persona(PersonaType.OCEAN)
    assert dangerous_task_execution() == "Executed"
    assert safe_strategic_reading() == "Strategic Data"

    # Under Motion persona: safe succeeds, dangerous raises error
    PermissionEngine.set_current_persona(PersonaType.MOTION)
    try:
        assert safe_strategic_reading() == "Strategic Data"
        with pytest.raises(MotionPermissionError) as exc_info:
            dangerous_task_execution()
        assert "strictly forbidden" in str(exc_info.value)
    finally:
        PermissionEngine.set_current_persona(PersonaType.OCEAN)


# =========================================================
# --- 2. Strategic Model & Invariants Tests ---
# =========================================================

def test_identity_invariant_enforcement(motion_services):
    """Verify Identity only updates with explicit user confirmation."""
    strat = motion_services["strategic_model"]

    initial_id = strat.get_identity()
    assert initial_id.education != ""

    updated_id = MotionIdentity(
        education="MS CS at UMass Amherst",
        career_goals=["Principal AI Architect"],
        long_term_ambitions=["Lead AGI Systems Engineering"],
        core_constraints=["Time balance"],
        values=["Truth over convenience"],
    )

    # Rejects automated update without explicit confirmation
    with pytest.raises(ValueError) as exc:
        strat.update_identity(updated_id, explicit_user_confirmation=False)
    assert "Identity invariant violation" in str(exc.value)

    # Accepts explicit update
    saved = strat.update_identity(updated_id, explicit_user_confirmation=True)
    assert saved.career_goals == ["Principal AI Architect"]
    assert strat.get_identity().career_goals == ["Principal AI Architect"]


def test_trajectory_updates(motion_services):
    """Verify Trajectory reading and updating."""
    strat = motion_services["strategic_model"]

    current = strat.get_trajectory()
    assert current.current_phase != ""

    updated = strat.update_trajectory(
        current_phase="Phase 2: Hyper-Scale Execution",
        momentum=TrajectoryMomentum.HIGH,
        biggest_opportunity="Ocean v4.0 Launch",
    )
    assert updated.current_phase == "Phase 2: Hyper-Scale Execution"
    assert updated.momentum == TrajectoryMomentum.HIGH
    assert strat.get_trajectory().biggest_opportunity == "Ocean v4.0 Launch"


def test_initiatives_lifecycle(motion_services):
    """Verify Initiative creation, querying, and archiving."""
    strat = motion_services["strategic_model"]

    init = MotionInitiative(
        id="init_ai_voice",
        title="AI Voice Agent Architecture",
        description="Build real-time streaming voice agent.",
        status="ACTIVE",
        strategic_importance=StrategicImportance.CRITICAL,
        horizon=Horizon.NEAR_TERM,
        momentum=TrajectoryMomentum.HIGH,
        target_outcome="Latency under 300ms",
    )

    strat.create_or_update_initiative(init)
    active_inits = strat.get_initiatives(status="ACTIVE")
    assert len(active_inits) == 1
    assert active_inits[0].id == "init_ai_voice"

    # Archive initiative
    strat.archive_initiative("init_ai_voice", new_status="COMPLETED")
    assert len(strat.get_initiatives(status="ACTIVE")) == 0
    assert len(strat.get_initiatives(status="COMPLETED")) == 1


# =========================================================
# --- 3. Decision Journal State Machine Tests ---
# =========================================================

def test_decision_journal_state_machine(motion_services):
    """Verify PENDING -> DUE -> REVIEWED -> CLOSED lifecycle."""
    strat = motion_services["strategic_model"]

    entry = DecisionJournalEntry(
        id="dec_test_001",
        question="Should we prioritize Motion over UI polish?",
        alternatives_considered=["UI polish first", "Motion strategic core first"],
        recommendation="Prioritize Motion strategic core first.",
        reasoning="High leverage foundation for Ocean v4.0.",
        expected_outcome="Substantial boost in long-term decision quality.",
        review_trigger="End of August milestone",
    )

    saved = strat.record_decision(entry)
    assert saved.status == DecisionStatus.PENDING

    # Transition PENDING -> DUE
    due_entry = strat.transition_decision_state(saved.id, DecisionStatus.DUE)
    assert due_entry.status == DecisionStatus.DUE

    # Invalid transition: cannot close without outcome
    with pytest.raises(ValueError):
        strat.transition_decision_state(saved.id, DecisionStatus.CLOSED, actual_outcome=None)

    # Valid transition to CLOSED with outcome & reflection
    closed_entry = strat.transition_decision_state(
        saved.id,
        DecisionStatus.CLOSED,
        actual_outcome="Motion delivered ahead of schedule with 100% test coverage.",
        user_reflection="Strategy focus was indeed the right priority.",
    )
    assert closed_entry.status == DecisionStatus.CLOSED
    assert closed_entry.actual_outcome is not None
    assert closed_entry.closed_at is not None


# =========================================================
# --- 4. Human Overrides Tests ---
# =========================================================

def test_human_overrides_management(motion_services):
    """Verify Human Override recording and condition evaluation."""
    strat = motion_services["strategic_model"]

    override = HumanOverride(
        id="ovr_portfolio_001",
        recommendation_id_or_topic="Apply to 20 internships this week",
        user_decision="Focus exclusively on finishing Ocean v4.0 architecture first",
        reason="Better portfolio demo will yield 3x higher interview callback rate",
        review_trigger_condition="After Ocean v4.0 release demo is completed",
    )

    strat.record_override(override)
    active = strat.get_active_overrides()
    assert len(active) == 1
    assert active[0].review_trigger_condition == "After Ocean v4.0 release demo is completed"

    # Trigger review
    strat.trigger_override_review("ovr_portfolio_001", resolution_notes="Ocean v4.0 completed!")
    assert len(strat.get_active_overrides()) == 0

    # Resolve override
    resolved = strat.resolve_override("ovr_portfolio_001", resolution_notes="Applications resumed with demo.")
    assert resolved.status == OverrideStatus.RESOLVED


# =========================================================
# --- 5. Evidence Pipeline & Attribution Tests ---
# =========================================================

def test_evidence_extraction_and_observation_building(motion_services):
    """Verify deterministic Evidence -> Observation -> Conclusion pipeline."""
    pipeline = motion_services["pipeline"]
    attribution = motion_services["attribution"]

    raw_activities = [
        {"source_type": "daily_log", "date": "2026-08-20", "text": "Worked on Ocean v4 architecture for 4 hours."},
        {"source_type": "daily_log", "date": "2026-08-20", "text": "Refactored permissions and router for 2.5h."},
        {"source_type": "leetcode", "date": "2026-08-20", "text": "Solved 2 graph problems on Leetcode (45 mins)."},
        {"source_type": "learning_note", "date": "2026-08-20", "text": "Studied distributed consensus papers for 3 hours."},
    ]

    # 1. Extract evidence
    evidence = pipeline.extract_evidence_from_raw_activity(raw_activities)
    assert len(evidence) == 4
    assert any(e.duration_hours == 4.0 for e in evidence)

    # 2. Build observations
    observations = pipeline.build_observations_from_evidence(evidence, period_start="2026-08-20", period_end="2026-08-20")
    assert len(observations) >= 2
    for obs in observations:
        assert len(obs.evidence_ids) > 0
        assert obs.period_start == "2026-08-20"

    # 3. Build conclusions with rule-based confidence
    conclusions = pipeline.build_conclusions_from_observations(observations)
    assert len(conclusions) == len(observations)
    for conc in conclusions:
        assert conc.confidence_level in (ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH)
        assert len(conc.derived_from_observation_ids) > 0

    # 4. Attribution: "Why do I believe this?"
    target_conc_id = conclusions[0].id
    provenance = attribution.trace_provenance(target_conc_id)
    assert "conclusion" in provenance
    assert len(provenance["supporting_observations"]) > 0


# =========================================================
# --- 6. Dynamic Strategic Context Retrieval Tests ---
# =========================================================

def test_dynamic_context_retriever(motion_services):
    """Verify context retrieval includes active items and filters stale/closed data."""
    retriever = motion_services["retriever"]
    strat = motion_services["strategic_model"]

    # Add active initiative
    strat.create_or_update_initiative(
        MotionInitiative(
            id="init_active",
            title="Active Project",
            status="ACTIVE",
            strategic_importance=StrategicImportance.HIGH,
            horizon=Horizon.NEAR_TERM,
            momentum=TrajectoryMomentum.HIGH,
        )
    )
    # Add completed initiative (should be excluded)
    strat.create_or_update_initiative(
        MotionInitiative(
            id="init_closed",
            title="Old Closed Project",
            status="COMPLETED",
            strategic_importance=StrategicImportance.MEDIUM,
            horizon=Horizon.LONG_TERM,
            momentum=TrajectoryMomentum.STALLED,
        )
    )

    context = retriever.assemble_context(query="What is my primary focus on Active Project?")
    assert any(init.id == "init_active" for init in context.active_initiatives)
    assert not any(init.id == "init_closed" for init in context.active_initiatives)

    prompt_str = retriever.format_context_for_prompt(context)
    assert "Active Strategic Initiatives" in prompt_str
    assert "Active Project" in prompt_str
    assert "Old Closed Project" not in prompt_str


# =========================================================
# --- 7. Persona Router Tests ---
# =========================================================

def test_persona_router_detection_and_clean_query(motion_services):
    """Verify @motion and motion: triggers activate Motion persona."""
    router = motion_services["router"]

    # Positive triggers
    assert router.is_motion_invoked("@motion what are my biggest strategic bottlenecks?") is True
    assert router.is_motion_invoked("motion: evaluate my trajectory") is True
    assert router.is_motion_invoked("/motion review initiatives") is True
    assert router.is_motion_invoked("ask motion for advice") is True

    # Negative triggers (regular Ocean tasks)
    assert router.is_motion_invoked("create task buy milk") is False
    assert router.is_motion_invoked("what are my tasks for today?") is False
    assert router.is_motion_invoked("mark task done") is False

    # Clean query extraction
    clean = router.extract_motion_query("@motion what are my biggest opportunities?")
    assert clean == "what are my biggest opportunities?"

    clean_colon = router.extract_motion_query("motion: evaluate my current momentum")
    assert clean_colon == "evaluate my current momentum"


# =========================================================
# --- 8. Motion Mentorship Service & Decision Logging ---
# =========================================================

@patch("app.motion.mentorship_service.get_gemini_client")
def test_motion_mentorship_consultation(mock_client_factory, motion_services):
    """Verify Motion consultation runs under Motion persona and logs recommendations."""
    mentorship = motion_services["mentorship"]
    strat = motion_services["strategic_model"]

    # Mock Gemini response
    mock_genai_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "Based on your trajectory, you are in strong foundational readiness.\n\n"
        "Recommendation: Double down on the Ocean v4 core subsystem.\n\n"
        "What is the single most important technical unlock for this week?"
    )
    mock_genai_client.models.generate_content.return_value = mock_response
    mock_client_factory.return_value = mock_genai_client

    response = mentorship.consult(user_message="How should I balance Ocean vs secondary tasks?")
    assert response.reply_text != ""
    assert "Ocean v4 core subsystem" in response.analysis
    assert len(response.high_leverage_questions) > 0

    # Verify decision journal entry was automatically logged
    decisions = strat.get_decisions(status=DecisionStatus.PENDING)
    assert len(decisions) >= 1
    assert "Double down on the Ocean v4 core subsystem" in decisions[0].recommendation


# =========================================================
# --- 9. Daily and Weekly Review Routines ---
# =========================================================

def test_daily_review_routine(motion_services):
    """Verify Daily Review creates observations without modifying trajectory."""
    review_svc = motion_services["review"]
    strat = motion_services["strategic_model"]

    initial_trajectory = strat.get_trajectory()

    activities = [
        {"source_type": "daily_log", "date": "2026-08-21", "text": "Implemented Persona Router in 3 hours."},
        {"source_type": "leetcode", "date": "2026-08-21", "text": "Reviewed binary search tree problem."},
    ]

    res = review_svc.execute_daily_review(activities, date_str="2026-08-21")
    assert res["status"] == "success"
    assert res["observations_count"] > 0

    # Trajectory remains untouched
    assert strat.get_trajectory().current_phase == initial_trajectory.current_phase


def test_weekly_review_routine(motion_services):
    """Verify Weekly Review evaluates trajectory, marks due decisions, and saves report."""
    review_svc = motion_services["review"]
    strat = motion_services["strategic_model"]

    # Log a pending decision
    strat.record_decision(
        DecisionJournalEntry(
            id="dec_weekly_test",
            question="Architecture question",
            alternatives_considered=["A", "B"],
            recommendation="Choose A",
            reasoning="Reason",
            expected_outcome="Success",
            review_trigger="Weekly Review",
        )
    )

    weekly_report = review_svc.execute_weekly_review(week_start="2026-08-14", week_end="2026-08-21")
    assert isinstance(weekly_report, MotionWeeklyReview)
    assert len(weekly_report.wins) > 0
    assert len(weekly_report.recommendations) > 0

    # Verify pending decision was transitioned to DUE
    dec = strat.get_decision("dec_weekly_test")
    assert dec.status == DecisionStatus.DUE


# =========================================================
# --- 10. FastAPI Motion Endpoints Integration Tests ---
# =========================================================

def test_fastapi_motion_endpoints():
    """Verify FastAPI Motion REST API endpoints."""
    client = TestClient(app)

    # 1. Trajectory
    traj_res = client.get("/motion/trajectory")
    assert traj_res.status_code == 200
    assert "current_phase" in traj_res.json()

    # 2. Identity
    id_res = client.get("/motion/identity")
    assert id_res.status_code == 200
    assert "education" in id_res.json()

    # 3. Explicit Identity Update
    update_payload = {
        "education": "MS CS at UMass",
        "career_goals": ["AI Systems Engineer"],
        "long_term_ambitions": ["Build AGI infrastructure"],
        "core_constraints": ["Academic schedule"],
        "values": ["Engineering excellence"],
    }
    put_res = client.put("/motion/identity", json=update_payload)
    assert put_res.status_code == 200
    assert put_res.json()["career_goals"] == ["AI Systems Engineer"]

    # 4. Initiatives
    init_payload = {
        "id": "init_rest_test",
        "title": "REST API Test Initiative",
        "description": "Integration testing for Motion endpoints",
        "status": "ACTIVE",
        "strategic_importance": "HIGH",
        "horizon": "NEAR_TERM",
        "momentum": "HIGH",
        "target_outcome": "Pass 100% tests",
    }
    post_init_res = client.post("/motion/initiatives", json=init_payload)
    assert post_init_res.status_code == 200
    assert post_init_res.json()["id"] == "init_rest_test"

    get_inits_res = client.get("/motion/initiatives?status=ACTIVE")
    assert get_inits_res.status_code == 200
    assert any(i["id"] == "init_rest_test" for i in get_inits_res.json())

    # 5. Overrides
    ovr_payload = {
        "id": "ovr_rest_01",
        "recommendation_id_or_topic": "Take extra courses",
        "user_decision": "Self-study research papers instead",
        "reason": "Faster mastery of specialized agentic systems",
        "review_trigger_condition": "End of semester",
        "status": "ACTIVE",
    }
    ovr_res = client.post("/motion/overrides", json=ovr_payload)
    assert ovr_res.status_code == 200
    assert ovr_res.json()["id"] == "ovr_rest_01"

    # 6. Daily Review Endpoint
    daily_payload = {
        "activities": [
            {"source_type": "daily_log", "date": "2026-08-21", "text": "Tested Motion REST endpoints for 2 hours."}
        ],
        "date": "2026-08-21",
    }
    daily_res = client.post("/motion/review/daily", json=daily_payload)
    assert daily_res.status_code == 200
    assert daily_res.json()["status"] == "success"

    # 7. Weekly Review Endpoint
    weekly_res = client.post("/motion/review/weekly")
    assert weekly_res.status_code == 200
    assert "week_start" in weekly_res.json()
