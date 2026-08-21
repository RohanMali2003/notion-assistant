"""Unit & Integration Test Suite for Ocean Motion v2 (Ocean v4.1).

Covers:
1. Automated Evidence Ingestion Engine
2. Multi-Window Observation & Conclusion Synthesis Engine
3. Strategic Drift & Trend Detection Engine (Alignment Index 0-100%)
4. Multi-Hop Causal Attribution Engine & Markdown Formatting
5. Proactive Accountability & Daily Check-In Engine
6. Persona Router Intent Recognition (Attribution, Drift, Daily Check-In)
7. FastAPI Motion v2 REST Endpoints
"""

import os
import shutil
import tempfile
from typing import Any, Dict, Generator
import pytest
from starlette.testclient import TestClient

from app.main import app
from app.motion import (
    AccountabilityMonitor,
    CausalAttributionTree,
    ConfidenceLevel,
    DecisionJournalEntry,
    DecisionStatus,
    DriftSeverity,
    EvidenceIngestionEngine,
    EvidenceIngestionEvent,
    EvidenceItem,
    Horizon,
    HumanOverride,
    MotionConclusion,
    MotionInitiative,
    MotionObservation,
    MotionStorage,
    MultiHopAttributionEngine,
    MultiWindowSynthesisEngine,
    OverrideStatus,
    PersonaRouter,
    PersonaType,
    StrategicDriftDetector,
    StrategicDriftReport,
    StrategicImportance,
    StrategicModelService,
    TrajectoryMomentum,
    VelocityVector,
)


@pytest.fixture(autouse=True)
def test_motion_v2_env(tmp_path) -> Generator[Dict[str, Any], None, None]:
    """Isolate Motion storage in a temporary directory for each test."""
    temp_dir = str(tmp_path / "motion_v2_test_data")
    storage = MotionStorage(base_dir=temp_dir)
    strat_model = StrategicModelService(storage=storage)
    ingestion = EvidenceIngestionEngine(storage=storage)
    synthesis = MultiWindowSynthesisEngine(storage=storage)
    drift = StrategicDriftDetector(storage=storage)
    attribution = MultiHopAttributionEngine(storage=storage)
    accountability = AccountabilityMonitor(storage=storage)
    router = PersonaRouter()

    # Initialize sample active initiatives
    strat_model.create_or_update_initiative(
        MotionInitiative(
            id="init_ocean_v4",
            title="Ocean v4.1 Motion Rollout",
            description="Autonomous strategic intelligence subsystem",
            status="ACTIVE",
            strategic_importance=StrategicImportance.HIGH,
            horizon=Horizon.NEAR_TERM,
            momentum=TrajectoryMomentum.HIGH,
            target_outcome="100% test coverage and autonomous ingestion",
        )
    )
    strat_model.create_or_update_initiative(
        MotionInitiative(
            id="init_leetcode",
            title="Algorithm & System Design Mastery",
            description="Daily LeetCode medium/hard practice",
            status="ACTIVE",
            strategic_importance=StrategicImportance.HIGH,
            horizon=Horizon.MEDIUM_TERM,
            momentum=TrajectoryMomentum.MODERATE,
            target_outcome="Consistent algorithmic intuition",
        )
    )

    yield {
        "storage": storage,
        "strategic_model": strat_model,
        "ingestion": ingestion,
        "synthesis": synthesis,
        "drift": drift,
        "attribution": attribution,
        "accountability": accountability,
        "router": router,
    }


def test_evidence_ingestion_engine(test_motion_v2_env):
    """Verify task, LeetCode, learning, and daily log ingestion."""
    ingestion = test_motion_v2_env["ingestion"]
    storage = test_motion_v2_env["storage"]

    # 1. Ingest task completion
    ev_task = ingestion.ingest_task_completion(
        task_title="Build Motion Ingestion Pipeline",
        notes="Implemented async event converter for background tasks",
        page_url="https://notion.so/task_123",
        duration_hours=2.5,
    )
    assert ev_task.source_type == "completed_task"
    assert ev_task.duration_hours == 2.5
    assert ev_task.metrics.get("category") == "Systems Engineering & Projects"

    # 2. Ingest LeetCode review
    ev_lc = ingestion.ingest_leetcode_review(
        problem_title="Course Schedule II",
        pattern_notes="Topological sort with Kahn's algorithm",
        page_url="https://notion.so/lc_course_schedule",
        duration_hours=1.0,
    )
    assert ev_lc.source_type == "leetcode_review"
    assert ev_lc.metrics.get("category") == "LeetCode & Algorithms"

    # 3. Ingest Learning milestone
    ev_learn = ingestion.ingest_learning_milestone(
        topic_title="Distributed Consensus in Raft",
        summary="Leader election, log replication, and commit index rules",
        page_url="https://notion.so/learn_raft",
        duration_hours=2.0,
    )
    assert ev_learn.source_type == "learning_milestone"
    assert ev_learn.metrics.get("category") == "Learning & Research"

    # 4. Ingest Daily log
    ev_log = ingestion.ingest_daily_log(
        date_str="2026-08-21",
        summary_text="Deep work on Ocean Motion v2 architecture and testing.",
        estimated_hours=3.0,
    )
    assert ev_log.source_type == "daily_log"

    # Verify storage loading
    loaded = storage.load_evidence(limit=10)
    assert len(loaded) == 4


def test_multi_window_synthesis(test_motion_v2_env):
    """Verify multi-window sliding synthesis creates Observations and Conclusions with rule-based confidence."""
    ingestion = test_motion_v2_env["ingestion"]
    synthesis = test_motion_v2_env["synthesis"]
    storage = test_motion_v2_env["storage"]

    # Ingest multiple evidence items
    for i in range(8):
        ingestion.ingest_task_completion(
            task_title=f"Ocean v4 Feature {i}",
            notes="Systems development on Ocean agent architecture",
            duration_hours=2.0,
        )

    for i in range(3):
        ingestion.ingest_leetcode_review(
            problem_title=f"LeetCode Graph Problem {i}",
            duration_hours=1.0,
        )

    observations, conclusions = synthesis.run_synthesis(reference_date="2026-08-21", save_records=True)

    assert len(observations) > 0
    assert len(conclusions) > 0

    # Verify high confidence conclusion for Systems Engineering (8 items >= 7)
    eng_conc = next((c for c in conclusions if "Systems Engineering" in c.statement or "Systems Engineering" in c.id), None)
    assert eng_conc is not None
    assert eng_conc.confidence == ConfidenceLevel.HIGH

    # Verify medium confidence conclusion for LeetCode (3 items >= 3)
    lc_conc = next((c for c in conclusions if "LeetCode" in c.statement or "leetcode" in c.id), None)
    assert lc_conc is not None
    assert lc_conc.confidence == ConfidenceLevel.MEDIUM


def test_strategic_drift_evaluation(test_motion_v2_env):
    """Verify Strategic Alignment Index calculation, neglected initiatives, and runaway tasks."""
    ingestion = test_motion_v2_env["ingestion"]
    drift = test_motion_v2_env["drift"]

    # Scenario A: High alignment (Focus on Ocean v4.1 Motion Rollout)
    for i in range(5):
        ingestion.ingest_task_completion(
            task_title="Ocean v4.1 Motion Rollout feature implementation",
            notes="Core backend architecture engineering",
            duration_hours=3.0,
        )
    for i in range(4):
        ingestion.ingest_leetcode_review(
            problem_title="Algorithm & System Design Mastery Binary Trees",
            duration_hours=1.0,
        )

    report_a = drift.evaluate_drift(window_days=7, reference_date="2026-08-21", save_report=True)
    assert report_a.alignment_score >= 80.0
    assert report_a.drift_severity == DriftSeverity.NORMAL
    assert len(report_a.neglected_initiatives) == 0

    # Scenario B: Neglected initiative & runaway admin tasks
    test_storage = test_motion_v2_env["storage"]
    # Clear evidence
    for f in os.listdir(test_storage.evidence_dir):
        os.remove(os.path.join(test_storage.evidence_dir, f))

    # Add only admin tasks (no Ocean, no LeetCode)
    for i in range(6):
        ingestion.ingest_event(
            EvidenceIngestionEvent(
                event_type="completed_task",
                title="Pay monthly utilities and admin bill",
                description="Sorting emails and billing spreadsheets",
                duration_hours=2.5,
                tags=["admin"],
            )
        )

    report_b = drift.evaluate_drift(window_days=7, reference_date="2026-08-21", save_report=True)
    assert report_b.alignment_score < 70.0
    assert report_b.drift_severity in (DriftSeverity.LOW_DRIFT, DriftSeverity.MODERATE_DRIFT, DriftSeverity.CRITICAL_DRIFT)
    assert len(report_b.neglected_initiatives) >= 1
    assert any("Admin & Operations" in r for r in report_b.runaway_categories)


def test_multi_hop_attribution(test_motion_v2_env):
    """Verify multi-hop causal graph traversal and markdown formatting."""
    ingestion = test_motion_v2_env["ingestion"]
    synthesis = test_motion_v2_env["synthesis"]
    strat = test_motion_v2_env["strategic_model"]
    attribution = test_motion_v2_env["attribution"]

    # 1. Ingest evidence
    ev = ingestion.ingest_task_completion(
        task_title="Build Motion Attribution Tree",
        notes="Created multi-hop provenance graph",
        page_url="https://notion.so/attribution_01",
        duration_hours=2.0,
    )

    # 2. Run synthesis
    observations, conclusions = synthesis.run_synthesis(reference_date="2026-08-21", save_records=True)
    assert len(conclusions) > 0

    # 3. Create a Decision Journal entry linked to conclusion
    decision = strat.record_decision(
        DecisionJournalEntry(
            id="dec_attribution_test",
            decision_title="Adopt Deep Multi-Hop Attribution",
            recommendation="Trace all strategic advice to verified Notion sources",
            derived_from_conclusion_ids=[conclusions[0].id],
            trade_offs_acknowledged=["Slightly higher storage usage"],
            expected_outcome="100% explainability",
            review_scheduled_for="2026-09-01",
        )
    )

    # 4. Build and format attribution tree
    tree = attribution.build_attribution_tree(decision.id)
    assert tree.target_id == decision.id
    assert tree.recommendation_text == decision.recommendation
    assert len(tree.conclusions) > 0
    assert len(tree.observations) > 0

    md_output = attribution.format_tree_as_markdown(tree)
    assert "Causal Attribution for `dec_attribution_test`" in md_output
    assert "Supporting Conclusions" in md_output
    assert "Factual Observations" in md_output


def test_accountability_and_daily_checkin(test_motion_v2_env):
    """Verify decision state transition to DUE and daily check-in prompt generation."""
    strat = test_motion_v2_env["strategic_model"]
    accountability = test_motion_v2_env["accountability"]

    # 1. Record decision due today
    strat.record_decision(
        DecisionJournalEntry(
            id="dec_due_today",
            decision_title="Switch to Async Ingestion",
            recommendation="Use non-blocking background tasks for evidence capture",
            trade_offs_acknowledged=["Requires task hooks"],
            expected_outcome="Zero webhook latency overhead",
            review_scheduled_for="2026-08-21",
        )
    )

    # 2. Scan due decisions
    due_list = accountability.scan_due_decisions(current_date="2026-08-21")
    assert any(d.id == "dec_due_today" for d in due_list)

    # 3. Generate daily check-in
    checkin = accountability.generate_daily_checkin(current_date="2026-08-21")
    assert "Motion Daily Check-In" in checkin.reply_text
    assert "dec_due_today" in checkin.reply_text
    assert checkin.high_leverage_question != ""


def test_router_v2_intents(test_motion_v2_env):
    """Verify PersonaRouter handles attribution, check-in, and drift report intents."""
    router = test_motion_v2_env["router"]
    ingestion = test_motion_v2_env["ingestion"]
    synthesis = test_motion_v2_env["synthesis"]

    # Ingest and synthesize sample data
    ingestion.ingest_task_completion("Test Task", duration_hours=1.0)
    synthesis.run_synthesis(reference_date="2026-08-21")

    # 1. Daily check-in intent
    reply_checkin = router.process_motion_request("motion: check in")
    assert "Motion Daily Check-In" in reply_checkin

    # 2. Drift report intent
    reply_drift = router.process_motion_request("@motion drift report")
    assert "Motion Strategic Drift Report" in reply_drift
    assert "Strategic Alignment Score" in reply_drift


def test_fastapi_motion_v2_endpoints():
    """Verify FastAPI Motion v2 REST API routes."""
    client = TestClient(app)

    # 1. Ingest evidence via POST /motion/evidence
    ev_payload = {
        "event_type": "completed_task",
        "title": "REST API Test Evidence",
        "description": "Validating POST /motion/evidence endpoint",
        "duration_hours": 2.0,
        "tags": ["testing", "engineering"],
    }
    post_ev_res = client.post("/motion/evidence", json=ev_payload)
    assert post_ev_res.status_code == 200
    assert "id" in post_ev_res.json()

    # 2. Query evidence via GET /motion/evidence
    get_ev_res = client.get("/motion/evidence?limit=10")
    assert get_ev_res.status_code == 200
    assert len(get_ev_res.json()) >= 1

    # 3. Trigger synthesis via POST /motion/synthesis
    synth_res = client.post("/motion/synthesis")
    assert synth_res.status_code == 200
    assert synth_res.json()["status"] == "ok"

    # 4. Query drift via GET /motion/drift
    drift_res = client.get("/motion/drift?window_days=7")
    assert drift_res.status_code == 200
    assert "alignment_score" in drift_res.json()

    # 5. Query daily checkin via GET /motion/checkin
    checkin_res = client.get("/motion/checkin")
    assert checkin_res.status_code == 200
    assert "high_leverage_question" in checkin_res.json()
