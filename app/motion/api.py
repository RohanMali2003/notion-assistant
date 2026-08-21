"""Ocean Motion REST API Router.

Encapsulates all Motion v1, v2, v3, and v4 REST endpoints:
- Strategic consultation & mentorship
- Trajectory & identity governance
- Initiatives, milestones & critical path DAG
- Multi-hop causal attribution & provenance
- Strategic drift, cognitive energy & sustainability
- Socratic inquiries & what-if scenario simulations
- Autonomous executive briefings & reviews
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from app.motion.accountability import accountability_monitor
from app.motion.attribution import multi_hop_attribution_engine
from app.motion.drift_detector import drift_detector
from app.motion.energy_monitor import energy_monitor
from app.motion.evidence_pipeline import evidence_pipeline
from app.motion.executive_briefing import executive_briefing_engine
from app.motion.forecaster import scenario_engine
from app.motion.ingestion import evidence_ingestion_engine
from app.motion.mentorship_service import motion_mentorship_service
from app.motion.milestones import milestone_engine
from app.motion.review_service import motion_review_service
from app.motion.schemas import (
    EvidenceIngestionEvent,
    HumanOverride,
    InitiativeMilestone,
    MotionIdentity,
    MotionInitiative,
    ScenarioSimulationRequest,
)
from app.motion.socratic import socratic_engine
from app.motion.spec import DecisionStatus
from app.motion.storage import motion_storage
from app.motion.strategic_model import strategic_model_service
from app.motion.synthesis import synthesis_engine

logger = logging.getLogger("notion-assistant.motion.api")

motion_api_router = APIRouter(prefix="/motion", tags=["Motion"])


@motion_api_router.post("/consult", response_model=Dict[str, Any])
async def motion_consult(request: Request):
    """Direct strategic consultation endpoint for Motion persona."""
    body = await request.json()
    message = body.get("message", "")
    sender_id = body.get("sender_id", "direct_api")
    if not message:
        raise HTTPException(status_code=400, detail="Missing required 'message' in request body.")

    response = await run_in_threadpool(
        motion_mentorship_service.consult,
        user_message=message,
        sender_id=sender_id,
    )
    return response.model_dump()


@motion_api_router.get("/trajectory", response_model=Dict[str, Any])
async def get_motion_trajectory():
    """Retrieve current strategic trajectory."""
    trajectory = await run_in_threadpool(strategic_model_service.get_trajectory)
    return trajectory.model_dump()


@motion_api_router.get("/identity", response_model=Dict[str, Any])
async def get_motion_identity():
    """Retrieve stable user identity."""
    identity = await run_in_threadpool(strategic_model_service.get_identity)
    return identity.model_dump()


@motion_api_router.put("/identity", response_model=Dict[str, Any])
async def update_motion_identity(identity: MotionIdentity):
    """Explicitly update user identity (never inferred)."""
    updated = await run_in_threadpool(
        strategic_model_service.update_identity,
        new_identity=identity,
        explicit_user_confirmation=True,
    )
    return updated.model_dump()


@motion_api_router.get("/initiatives", response_model=List[Dict[str, Any]])
async def get_motion_initiatives(status: Optional[str] = Query(None)):
    """Retrieve strategic initiatives, optionally filtered by status."""
    inits = await run_in_threadpool(strategic_model_service.get_initiatives, status=status)
    return [init.model_dump() for init in inits]


@motion_api_router.post("/initiatives", response_model=Dict[str, Any])
async def create_or_update_initiative(initiative: MotionInitiative):
    """Create or update a strategic initiative."""
    saved = await run_in_threadpool(strategic_model_service.create_or_update_initiative, initiative=initiative)
    return saved.model_dump()


@motion_api_router.get("/decision-journal", response_model=List[Dict[str, Any]])
async def get_decision_journal(status: Optional[str] = Query(None)):
    """Retrieve Decision Journal records."""
    target_status = DecisionStatus(status) if status else None
    decisions = await run_in_threadpool(strategic_model_service.get_decisions, status=target_status)
    return [d.model_dump() for d in decisions]


@motion_api_router.post("/decision-journal/{decision_id}/review", response_model=Dict[str, Any])
async def review_decision_journal_entry(decision_id: str, request: Request):
    """Submit actual outcome and user reflection to review and close a Decision Journal record."""
    body = await request.json()
    outcome = body.get("actual_outcome", "")
    reflection = body.get("user_reflection")
    if not outcome:
        raise HTTPException(status_code=400, detail="Missing required 'actual_outcome' field.")

    updated = await run_in_threadpool(
        motion_mentorship_service.handle_decision_review,
        decision_id=decision_id,
        actual_outcome=outcome,
        user_reflection=reflection,
    )
    return updated.model_dump()


@motion_api_router.post("/overrides", response_model=Dict[str, Any])
async def record_motion_override(override: HumanOverride):
    """Record an explicit human override with a condition-based review trigger."""
    saved = await run_in_threadpool(strategic_model_service.record_override, override=override)
    return saved.model_dump()


@motion_api_router.post("/review/weekly", response_model=Dict[str, Any])
async def trigger_motion_weekly_review(
    week_start: Optional[str] = Query(None),
    week_end: Optional[str] = Query(None),
):
    """Trigger generation of a weekly strategic review."""
    review = await run_in_threadpool(
        motion_review_service.execute_weekly_review,
        week_start=week_start,
        week_end=week_end,
    )
    return review.model_dump()


@motion_api_router.post("/review/daily", response_model=Dict[str, Any])
async def trigger_motion_daily_review(request: Request):
    """Trigger processing of daily activities into atomic evidence and observations."""
    body = await request.json()
    activities = body.get("activities", [])
    date_str = body.get("date")
    result = await run_in_threadpool(
        motion_review_service.execute_daily_review,
        raw_activities=activities,
        date_str=date_str,
    )
    return result


@motion_api_router.get("/provenance/{target_id}", response_model=Dict[str, Any])
async def get_motion_provenance(target_id: str):
    """Explain reasoning and trace provenance for a conclusion or observation ('Why do I believe this?')."""
    explanation = await run_in_threadpool(strategic_model_service.explain_belief, target_id=target_id)
    return explanation


@motion_api_router.get("/drift", response_model=Dict[str, Any])
async def get_motion_drift_report(
    window_days: int = Query(7, ge=1, le=60),
    reference_date: Optional[str] = Query(None),
):
    """Evaluate and retrieve current Strategic Drift and Alignment Report."""
    report = await run_in_threadpool(
        drift_detector.evaluate_drift,
        window_days=window_days,
        reference_date=reference_date,
        save_report=True,
    )
    return report.model_dump()


@motion_api_router.get("/evidence", response_model=List[Dict[str, Any]])
async def list_motion_evidence(
    limit: int = Query(50, ge=1, le=200),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Query discrete atomic evidence items."""
    evidence_items = await run_in_threadpool(
        motion_storage.load_evidence,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
    )
    return [e.model_dump() for e in evidence_items]


@motion_api_router.post("/evidence", response_model=Dict[str, Any])
async def ingest_motion_evidence(event: EvidenceIngestionEvent):
    """Ingest a single evidence event asynchronously or directly."""
    evidence = await run_in_threadpool(evidence_ingestion_engine.ingest_event, event=event)
    return evidence.model_dump()


@motion_api_router.post("/synthesis", response_model=Dict[str, Any])
async def trigger_motion_synthesis(reference_date: Optional[str] = Query(None)):
    """Trigger multi-window observation and conclusion synthesis."""
    observations, conclusions = await run_in_threadpool(
        synthesis_engine.run_synthesis,
        reference_date=reference_date,
        save_records=True,
    )
    return {
        "status": "ok",
        "observations_generated": len(observations),
        "conclusions_generated": len(conclusions),
        "observation_ids": [o.id for o in observations],
        "conclusion_ids": [c.id for c in conclusions],
    }


@motion_api_router.get("/attribution/{target_id}", response_model=Dict[str, Any])
async def get_motion_attribution(target_id: str):
    """Retrieve full multi-hop causal attribution tree for an item."""
    tree = await run_in_threadpool(multi_hop_attribution_engine.build_attribution_tree, target_id=target_id)
    return tree.model_dump()


@motion_api_router.get("/checkin", response_model=Dict[str, Any])
async def get_motion_daily_checkin(current_date: Optional[str] = Query(None)):
    """Generate daily proactive strategic check-in status prompt."""
    drift_report = await run_in_threadpool(
        drift_detector.evaluate_drift,
        window_days=7,
        reference_date=current_date,
        save_report=False,
    )
    checkin = await run_in_threadpool(
        accountability_monitor.generate_daily_checkin,
        current_date=current_date,
        drift_report=drift_report,
    )
    return checkin.model_dump()


@motion_api_router.post("/simulate", response_model=Dict[str, Any])
async def simulate_motion_scenario(request: ScenarioSimulationRequest):
    """Simulate what-if time and priority reallocation scenario."""
    result = await run_in_threadpool(
        scenario_engine.simulate_scenario,
        request=request,
        save_result=True,
    )
    return result.model_dump()


@motion_api_router.get("/energy", response_model=Dict[str, Any])
async def get_motion_energy_report(
    window_days: int = Query(7, ge=1, le=30),
    reference_date: Optional[str] = Query(None),
):
    """Retrieve cognitive sustainability and fatigue risk evaluation."""
    report = await run_in_threadpool(
        energy_monitor.evaluate_sustainability,
        window_days=window_days,
        reference_date=reference_date,
        save_report=True,
    )
    return report.model_dump()


@motion_api_router.get("/milestones", response_model=List[Dict[str, Any]])
async def get_motion_milestones(initiative_id: Optional[str] = Query(None)):
    """Retrieve active initiative milestones."""
    milestones = await run_in_threadpool(
        milestone_engine.get_milestones,
        initiative_id=initiative_id,
    )
    return [m.model_dump() for m in milestones]


@motion_api_router.post("/milestones", response_model=Dict[str, Any])
async def create_motion_milestone(milestone: InitiativeMilestone):
    """Create or update an initiative milestone."""
    saved_ms = await run_in_threadpool(milestone_engine.record_milestone, milestone=milestone)
    return saved_ms.model_dump()


@motion_api_router.get("/milestones/critical-path", response_model=Dict[str, Any])
async def get_motion_critical_path(
    initiative_id: Optional[str] = Query(None),
    reference_date: Optional[str] = Query(None),
):
    """Analyze critical path DAG and detect milestone schedule bottlenecks."""
    analysis = await run_in_threadpool(
        milestone_engine.analyze_critical_path,
        initiative_id=initiative_id,
        reference_date=reference_date,
    )
    return analysis


@motion_api_router.get("/briefing", response_model=Dict[str, Any])
async def get_motion_executive_briefing(
    window_days: int = Query(7, ge=1, le=30),
    reference_date: Optional[str] = Query(None),
):
    """Generate or retrieve autonomous weekly executive strategic briefing."""
    briefing = await run_in_threadpool(
        executive_briefing_engine.generate_briefing,
        window_days=window_days,
        reference_date=reference_date,
        save_briefing=True,
    )
    return briefing.model_dump()


@motion_api_router.post("/socratic", response_model=Dict[str, Any])
async def conduct_motion_socratic_inquiry(topic: str = Body(..., embed=True)):
    """Perform Socratic assumption deconstruction and pre-mortem failure analysis."""
    inquiry = await run_in_threadpool(socratic_engine.conduct_inquiry, topic=topic)
    return inquiry.model_dump()
