"""Comprehensive Test Suite for Ocean Motion v3 Subsystem (Ocean v4.2).

Verifies:
1. Predictive Trajectory Forecaster & Scenario Simulator
2. Cognitive Energy & Fatigue Monitor
3. Initiative Milestones & Critical Path Engine
4. Socratic Mentorship & Pre-Mortem Inversion Dialogue
5. Autonomous Executive Strategic Briefing Engine
6. Motion v3 Router conversational triggers
7. FastAPI REST endpoints for Motion v3
"""

import os
import shutil
import tempfile
from typing import Generator
import pytest
from starlette.testclient import TestClient

from app.main import app
from app.motion.drift_detector import StrategicDriftDetector
from app.motion.energy_monitor import CognitiveEnergyMonitor
from app.motion.evidence_pipeline import EvidencePipeline
from app.motion.executive_briefing import ExecutiveBriefingEngine
from app.motion.forecaster import ScenarioSimulationEngine
from app.motion.ingestion import EvidenceIngestionEngine
from app.motion.mentorship_service import MotionMentorshipService
from app.motion.milestones import InitiativeMilestoneEngine
from app.motion.permissions import PermissionEngine
from app.motion.retrieval import StrategicContextRetriever
from app.motion.router import PersonaRouter
from app.motion.schemas import (
    BurnoutRiskLevel,
    InitiativeMilestone,
    MilestoneStatus,
    MotionInitiative,
    ScenarioSimulationRequest,
    StrategicImportance,
    TrajectoryMomentum,
)
from app.motion.socratic import SocraticMentorshipEngine
from app.motion.spec import PersonaType
from app.motion.storage import MotionStorage
from app.motion.strategic_model import StrategicModelService
from app.motion.synthesis import MultiWindowSynthesisEngine


@pytest.fixture
def test_motion_v3_env() -> Generator[dict, None, None]:
    """Isolated temporary storage environment for Motion v3 unit testing."""
    temp_dir = tempfile.mkdtemp(prefix="motion_v3_test_")
    storage = MotionStorage(base_dir=temp_dir)
    strat = StrategicModelService(storage=storage)
    ingestion = EvidenceIngestionEngine(storage=storage)
    synthesis = MultiWindowSynthesisEngine(storage=storage)
    drift = StrategicDriftDetector(storage=storage)
    energy = CognitiveEnergyMonitor(storage=storage)
    milestones = InitiativeMilestoneEngine(storage=storage)
    forecaster = ScenarioSimulationEngine(storage=storage, drift_engine=drift)
    retriever = StrategicContextRetriever(storage=storage)
    socratic = SocraticMentorshipEngine(storage=storage, retriever=retriever)
    briefing = ExecutiveBriefingEngine(storage=storage, drift_eng=drift, energy_eng=energy, ms_eng=milestones)
    mentorship = MotionMentorshipService(strategic_model=strat, retriever=retriever)
    router = PersonaRouter(mentorship_service=mentorship)

    # Initialize standard active initiatives
    strat.create_or_update_initiative(
        MotionInitiative(
            id="init_ocean_v4",
            title="Ocean v4.2 Motion v3 Subsystem Rollout",
            status="ACTIVE",
            strategic_importance=StrategicImportance.CRITICAL,
            momentum=TrajectoryMomentum.HIGH,
        )
    )
    strat.create_or_update_initiative(
        MotionInitiative(
            id="init_leetcode",
            title="Algorithm & System Design Mastery",
            status="ACTIVE",
            strategic_importance=StrategicImportance.HIGH,
            momentum=TrajectoryMomentum.MODERATE,
        )
    )

    yield {
        "storage": storage,
        "strat": strat,
        "ingestion": ingestion,
        "synthesis": synthesis,
        "drift": drift,
        "energy": energy,
        "milestones": milestones,
        "forecaster": forecaster,
        "socratic": socratic,
        "briefing": briefing,
        "router": router,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)
    PermissionEngine.set_current_persona(PersonaType.OCEAN)


def test_scenario_simulation_engine(test_motion_v3_env):
    """Verify what-if time and priority reallocation simulator."""
    ingestion = test_motion_v3_env["ingestion"]
    forecaster = test_motion_v3_env["forecaster"]

    # Ingest baseline activity
    for i in range(5):
        ingestion.ingest_task_completion(
            task_title=f"Ocean Systems Development {i}",
            duration_hours=3.0,
            date="2026-08-20",
        )
    for i in range(2):
        ingestion.ingest_leetcode_review(
            problem_title=f"Binary Search {i}",
            duration_hours=1.0,
            date="2026-08-20",
        )

    # Run simulation: Rebalance +6h to LeetCode, -6h from Systems
    req = ScenarioSimulationRequest(
        scenario_name="Aggressive LeetCode Sprint",
        description="Shift evening engineering hours toward algorithm mastery",
        time_adjustments={"LeetCode & Algorithms": 6.0, "Systems Engineering & Projects": -6.0},
        timeframe_weeks=4,
    )
    sim_res = forecaster.simulate_scenario(req, save_result=True)

    assert sim_res.scenario_name == "Aggressive LeetCode Sprint"
    assert len(sim_res.trade_offs_identified) >= 2
    assert "LeetCode & Algorithms" in sim_res.projected_completion_shifts or "init_leetcode" in sim_res.projected_completion_shifts
    assert sim_res.confidence_score > 0.0
    assert len(sim_res.recommendation_verdict) > 0


def test_cognitive_energy_monitor(test_motion_v3_env):
    """Verify fatigue risk score, consecutive high-intensity days, and flow-vs-thrash ratios."""
    ingestion = test_motion_v3_env["ingestion"]
    energy = test_motion_v3_env["energy"]

    # Simulate 5 consecutive high-intensity days (8h/day deep work)
    dates = ["2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21"]
    for d in dates:
        ingestion.ingest_task_completion(
            task_title=f"Deep Focus Architecture Session on {d}",
            duration_hours=4.0,
            date=d,
        )
        ingestion.ingest_task_completion(
            task_title=f"Core Systems Engineering on {d}",
            duration_hours=4.0,
            date=d,
        )

    report = energy.evaluate_sustainability(window_days=7, reference_date="2026-08-21", save_report=True)

    assert report.total_logged_hours == 40.0
    assert report.consecutive_high_intensity_days == 5
    assert report.fatigue_risk_score >= 60.0
    assert report.burnout_risk_level in (BurnoutRiskLevel.HIGH, BurnoutRiskLevel.CRITICAL)
    assert report.recommended_decompression_hours >= 3.0
    assert report.flow_vs_thrash_ratio >= 1.5


def test_initiative_milestone_and_critical_path(test_motion_v3_env):
    """Verify milestone DAG dependency resolution, critical path calculation, and blocker detection."""
    ms_engine = test_motion_v3_env["milestones"]

    # Create 4 milestones forming a DAG:
    # ms_1 (10h) -> ms_2 (20h) -> ms_4 (15h) [Total: 45h - Critical Path]
    # ms_3 (5h)  -> ms_4
    ms1 = ms_engine.record_milestone(
        InitiativeMilestone(
            id="ms_spec",
            initiative_id="init_ocean_v4",
            title="Design Motion v3 Spec",
            estimated_hours=10.0,
            completed_hours=10.0,
            status=MilestoneStatus.ACHIEVED,
            dependencies=[],
        )
    )
    ms2 = ms_engine.record_milestone(
        InitiativeMilestone(
            id="ms_core_engines",
            initiative_id="init_ocean_v4",
            title="Implement Core v3 Engines",
            estimated_hours=20.0,
            completed_hours=5.0,
            status=MilestoneStatus.IN_PROGRESS,
            dependencies=["ms_spec"],
            target_date="2026-08-25",
        )
    )
    ms3 = ms_engine.record_milestone(
        InitiativeMilestone(
            id="ms_docs",
            initiative_id="init_ocean_v4",
            title="Write Developer Docs",
            estimated_hours=5.0,
            completed_hours=0.0,
            status=MilestoneStatus.NOT_STARTED,
            dependencies=["ms_spec"],
        )
    )
    ms4 = ms_engine.record_milestone(
        InitiativeMilestone(
            id="ms_release",
            initiative_id="init_ocean_v4",
            title="Deploy Ocean v4.2 Release",
            estimated_hours=15.0,
            completed_hours=0.0,
            status=MilestoneStatus.NOT_STARTED,
            dependencies=["ms_core_engines", "ms_docs"],
            target_date="2026-08-30",
        )
    )

    analysis = ms_engine.analyze_critical_path(initiative_id="init_ocean_v4", reference_date="2026-08-21")

    assert len(analysis["critical_path"]) >= 2
    assert analysis["total_critical_hours"] >= 15.0
    crit_ids = [m["id"] for m in analysis["critical_path"]]
    assert "ms_release" in crit_ids


def test_socratic_premortem_inquiry(test_motion_v3_env):
    """Verify Socratic assumption deconstruction and pre-mortem failure mode synthesis."""
    socratic = test_motion_v3_env["socratic"]

    inquiry = socratic.conduct_inquiry(topic="Switching primary focus entirely to mobile app development")

    assert len(inquiry.unexamined_assumptions) >= 1
    assert len(inquiry.premortem_failure_scenarios) >= 3
    assert len(inquiry.probing_questions) >= 1
    assert "inverting" in inquiry.inversion_analysis.lower()

    formatted_md = socratic.format_socratic_response(inquiry)
    assert "Unexamined Assumptions" in formatted_md
    assert "Pre-Mortem Failure Analysis" in formatted_md


def test_executive_briefing_generation(test_motion_v3_env):
    """Verify autonomous executive weekly strategic briefing compilation."""
    ingestion = test_motion_v3_env["ingestion"]
    briefing_engine = test_motion_v3_env["briefing"]

    # Ingest representative activity
    for i in range(4):
        ingestion.ingest_task_completion(
            task_title=f"Ocean v4.2 Feature Dev {i}",
            duration_hours=3.0,
            date="2026-08-21",
        )

    briefing = briefing_engine.generate_briefing(window_days=7, reference_date="2026-08-21", save_briefing=True)

    assert briefing.strategic_alignment_score > 0.0
    assert len(briefing.key_wins) >= 1
    assert "Motion Executive Strategic Briefing" in briefing.formatted_markdown_briefing
    assert "Strategic Alignment Index" in briefing.formatted_markdown_briefing


def test_router_v3_intents(test_motion_v3_env):
    """Verify PersonaRouter handles energy, milestones, briefing, socratic, and scenario intents."""
    router = test_motion_v3_env["router"]
    ingestion = test_motion_v3_env["ingestion"]

    ingestion.ingest_task_completion("Router v3 Test Task", duration_hours=2.0, date="2026-08-21")

    # 1. Energy inquiry
    reply_energy = router.process_motion_request("motion: energy")
    assert "Cognitive Sustainability Report" in reply_energy

    # 2. Milestones inquiry
    reply_ms = router.process_motion_request("motion: milestones")
    assert "Critical Path & Milestone Analysis" in reply_ms

    # 3. Executive briefing inquiry
    reply_briefing = router.process_motion_request("motion: briefing")
    assert "Executive Strategic Briefing" in reply_briefing

    # 4. Socratic inquiry
    reply_socratic = router.process_motion_request("motion: pre-mortem dropping leetcode for research")
    assert "Socratic Strategic Inquiry" in reply_socratic

    # 5. Scenario simulation
    reply_scenario = router.process_motion_request("motion: scenario shift 5 hours to algorithms")
    assert "Scenario Simulation Result" in reply_scenario


def test_fastapi_motion_v3_endpoints():
    """Verify FastAPI Motion v3 REST API endpoints."""
    client = TestClient(app)

    # 1. POST /motion/simulate
    sim_payload = {
        "scenario_name": "Test REST Rebalance",
        "description": "Validating POST /motion/simulate endpoint",
        "time_adjustments": {"LeetCode & Algorithms": 4.0, "Systems Engineering & Projects": -4.0},
        "timeframe_weeks": 3,
    }
    post_sim_res = client.post("/motion/simulate", json=sim_payload)
    assert post_sim_res.status_code == 200
    assert "projected_alignment_score" in post_sim_res.json()

    # 2. GET /motion/energy
    energy_res = client.get("/motion/energy?window_days=7")
    assert energy_res.status_code == 200
    assert "fatigue_risk_score" in energy_res.json()

    # 3. POST & GET /motion/milestones
    ms_payload = {
        "id": "ms_rest_test_001",
        "initiative_id": "init_ocean_v4",
        "title": "REST API Test Milestone",
        "estimated_hours": 12.0,
        "completed_hours": 3.0,
        "status": "IN_PROGRESS",
    }
    post_ms_res = client.post("/motion/milestones", json=ms_payload)
    assert post_ms_res.status_code == 200
    assert post_ms_res.json()["id"] == "ms_rest_test_001"

    get_ms_res = client.get("/motion/milestones")
    assert get_ms_res.status_code == 200
    assert any(m["id"] == "ms_rest_test_001" for m in get_ms_res.json())

    # 4. GET /motion/milestones/critical-path
    cp_res = client.get("/motion/milestones/critical-path")
    assert cp_res.status_code == 200
    assert "critical_path" in cp_res.json()

    # 5. GET /motion/briefing
    brf_res = client.get("/motion/briefing")
    assert brf_res.status_code == 200
    assert "strategic_alignment_score" in brf_res.json()

    # 6. POST /motion/socratic
    soc_res = client.post("/motion/socratic", json={"topic": "Launching dual concurrent startup initiatives"})
    assert soc_res.status_code == 200
    assert "premortem_failure_scenarios" in soc_res.json()
