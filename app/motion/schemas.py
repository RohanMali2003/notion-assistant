"""Ocean Motion Pydantic Data Models & Schemas."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.motion.spec import (
    BurnoutRiskLevel,
    ConfidenceLevel,
    DecisionStatus,
    DriftSeverity,
    Horizon,
    MilestoneStatus,
    OverrideStatus,
    StrategicImportance,
    TrajectoryMomentum,
    VelocityVector,
)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class MotionIdentity(BaseModel):
    """Stable identity attributes. Changes ONLY via explicit user instruction."""
    education: str = Field(default="", description="Current academic status or background.")
    career_goals: List[str] = Field(default_factory=list, description="Primary career goals and target roles.")
    long_term_ambitions: List[str] = Field(default_factory=list, description="Multi-year long-term aspirations.")
    core_constraints: List[str] = Field(default_factory=list, description="Hard constraints: time, budget, visa, energy.")
    values: List[str] = Field(default_factory=list, description="Core guiding principles and priorities.")
    updated_at: str = Field(default_factory=utc_now_iso, description="ISO timestamp of last explicit modification.")


class MotionTrajectory(BaseModel):
    """Primary object Motion reasons from. Updated weekly."""
    current_phase: str = Field(default="Foundational Execution", description="Current overarching strategic phase.")
    current_direction: str = Field(default="", description="Primary direction and focus of efforts.")
    target_direction: Optional[str] = Field(default=None, description="Alias for current_direction.")
    momentum: TrajectoryMomentum = Field(default=TrajectoryMomentum.MODERATE, description="Current velocity & momentum.")
    biggest_opportunity: str = Field(default="", description="Highest-leverage immediate opportunity.")
    biggest_risk: str = Field(default="", description="Most significant strategic bottleneck or drift risk.")
    next_review: str = Field(default="", description="Target date or milestone for next trajectory review.")
    last_updated: str = Field(default_factory=utc_now_iso, description="Timestamp of last trajectory evaluation.")


class MotionInitiative(BaseModel):
    """Strategic initiative tracking major long-term efforts (not tactical tasks)."""
    id: str = Field(..., description="Unique initiative ID (e.g. init_ocean_v4).")
    title: str = Field(..., description="Initiative title.")
    description: str = Field(default="", description="Strategic objective and context.")
    status: str = Field(default="ACTIVE", description="ACTIVE, PAUSED, COMPLETED, or ABANDONED.")
    strategic_importance: StrategicImportance = Field(default=StrategicImportance.HIGH, description="Importance level.")
    horizon: Horizon = Field(default=Horizon.MEDIUM_TERM, description="Target timeframe horizon.")
    momentum: TrajectoryMomentum = Field(default=TrajectoryMomentum.MODERATE, description="Execution momentum.")
    target_outcome: str = Field(default="", description="Measurable definition of success.")
    created_at: str = Field(default_factory=utc_now_iso, description="Creation timestamp.")
    updated_at: str = Field(default_factory=utc_now_iso, description="Last update timestamp.")


class EvidenceItem(BaseModel):
    """Atomic measurable event extracted from raw workspace activity without interpretation."""
    id: str = Field(..., description="Unique evidence ID (e.g. ev_20260821_001).")
    source_type: str = Field(..., description="Source: daily_log, completed_task, leetcode, learning_note, etc.")
    source_ref: Optional[str] = Field(default=None, description="Notion page ID or URL reference.")
    source_reference: Optional[str] = Field(default=None, description="Alias for source_ref.")
    date: str = Field(..., description="Date of occurrence (YYYY-MM-DD).")
    duration_hours: Optional[float] = Field(default=None, description="Measurable time spent in hours if available.")
    tags: List[str] = Field(default_factory=list, description="Tags/categories associated with the evidence.")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Key-value metrics (e.g. problems_solved=3).")
    description: str = Field(..., description="Factual description of the measurable activity.")
    raw_snippet: Optional[str] = Field(default=None, description="Raw source log snippet.")


class MotionObservation(BaseModel):
    """Factual summary aggregating repeated evidence across time windows."""
    id: str = Field(..., description="Unique observation ID (e.g. obs_20260821_001).")
    date: str = Field(default_factory=utc_now_iso, description="Observation timestamp.")
    period_start: str = Field(default="", description="Start date of observation window (YYYY-MM-DD).")
    period_end: str = Field(default="", description="End date of observation window (YYYY-MM-DD).")
    category: str = Field(default="Execution", description="Category: Execution, Learning, Leetcode, Finances, etc.")
    statement: str = Field(default="", description="Factual synthesis of observed activity.")
    observation_summary: Optional[str] = Field(default=None, description="Detailed observation summary.")
    time_window: Optional[str] = Field(default=None, description="Human readable time window.")
    evidence_ids: List[str] = Field(default_factory=list, description="List of underlying EvidenceItem IDs (Provenance).")
    frequency_count: int = Field(default=1, description="Number of supporting evidence instances.")
    frequency: Optional[int] = Field(default=None, description="Alternative alias for frequency_count.")
    total_duration_hours: Optional[float] = Field(default=None, description="Total duration across supporting items.")


class MotionConclusion(BaseModel):
    """Interpretation derived from one or more observations."""
    id: str = Field(..., description="Unique conclusion ID (e.g. conc_20260821_001).")
    statement: str = Field(..., description="Strategic interpretation or hypothesis.")
    derived_from_observation_ids: List[str] = Field(default_factory=list, description="Supporting MotionObservation IDs (Provenance).")
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.LOW, description="Rule-based confidence level.")
    confidence: Optional[ConfidenceLevel] = Field(default=None, description="Alias for confidence_level.")
    rationale: str = Field(default="", description="Logical chain explaining why this conclusion is reached.")
    confidence_reasoning: Optional[str] = Field(default="", description="Reasoning for rule-based confidence.")
    created_at: str = Field(default_factory=utc_now_iso, description="Creation timestamp.")
    status: str = Field(default="ACTIVE", description="ACTIVE, SUPERSEDED, or RESOLVED.")


class MotionRecommendation(BaseModel):
    """Strategic advice generated by Motion, backed by conclusions."""
    id: str = Field(..., description="Unique recommendation ID (e.g. rec_20260821_001).")
    conclusion_ids: List[str] = Field(default_factory=list, description="Supporting MotionConclusion IDs (Provenance).")
    question_or_opportunity: str = Field(default="", description="The strategic challenge or opportunity addressed.")
    recommendation_text: str = Field(..., description="The core actionable advice.")
    rationale: str = Field(default="", description="Why this recommendation is made based on evidence.")
    trade_offs: List[str] = Field(default_factory=list, description="Explicit second-order trade-offs discussed.")
    expected_outcome: str = Field(default="", description="Anticipated strategic impact if followed.")
    review_trigger: str = Field(default="", description="Condition or date trigger for evaluating outcome.")
    alternatives_considered: List[str] = Field(default_factory=list, description="Alternative paths evaluated.")


class DecisionJournalEntry(BaseModel):
    """Append-only strategic decision log with mandatory review state machine."""
    id: str = Field(..., description="Unique decision ID (e.g. dec_20260821_001).")
    decision_title: Optional[str] = Field(default="", description="Short title of decision.")
    created_at: str = Field(default_factory=utc_now_iso, description="Timestamp recorded.")
    question: str = Field(default="", description="Strategic question or decision faced.")
    alternatives_considered: List[str] = Field(default_factory=list, description="Alternative options evaluated.")
    recommendation: str = Field(..., description="Motion's strategic recommendation.")
    reasoning: str = Field(default="", description="Evidence-backed reasoning.")
    trade_offs_acknowledged: List[str] = Field(default_factory=list, description="Trade-offs acknowledged.")
    expected_outcome: str = Field(default="", description="Expected results and success metrics.")
    derived_from_conclusion_ids: List[str] = Field(default_factory=list, description="Linked conclusion IDs.")
    review_trigger: str = Field(default="", description="Condition or timeframe triggering mandatory review.")
    review_scheduled_for: Optional[str] = Field(default=None, description="Scheduled review date (YYYY-MM-DD).")
    status: DecisionStatus = Field(default=DecisionStatus.PENDING, description="State: PENDING -> DUE -> REVIEWED -> CLOSED.")
    actual_outcome: Optional[str] = Field(default=None, description="User-reported actual outcome upon review.")
    user_reflection: Optional[str] = Field(default=None, description="Lessons learned and reflections.")
    closed_at: Optional[str] = Field(default=None, description="Timestamp closed.")


class HumanOverride(BaseModel):
    """Record of user overriding Motion's recommendation, with condition-based revisit trigger."""
    id: str = Field(..., description="Unique override ID (e.g. ovr_20260821_001).")
    recommendation_id_or_topic: str = Field(..., description="Referenced recommendation or strategic topic.")
    user_decision: str = Field(..., description="The alternative course of action chosen by user.")
    reason: str = Field(..., description="User's stated rationale for overriding.")
    date: str = Field(default_factory=utc_now_iso, description="Date override was made.")
    review_trigger_condition: str = Field(..., description="Condition under which Motion should revisit (not arbitrary date).")
    status: OverrideStatus = Field(default=OverrideStatus.ACTIVE, description="ACTIVE, TRIGGERED, REVIEWED, RESOLVED.")
    resolution_notes: Optional[str] = Field(default=None, description="Notes when condition was triggered and evaluated.")


class MotionWeeklyReview(BaseModel):
    """Synthesized weekly strategic review."""
    id: str = Field(..., description="Unique review ID (e.g. rev_2026_w34).")
    week_start: str = Field(..., description="Start of review week (YYYY-MM-DD).")
    week_end: str = Field(..., description="End of review week (YYYY-MM-DD).")
    wins: List[str] = Field(default_factory=list, description="Key strategic milestones and wins.")
    regressions: List[str] = Field(default_factory=list, description="Areas of backsliding or stalled momentum.")
    strategic_drift: Optional[str] = Field(default=None, description="Evaluation of alignment vs drift from trajectory.")
    opportunities: List[str] = Field(default_factory=list, description="High-leverage emerging opportunities.")
    recommendations: List[MotionRecommendation] = Field(default_factory=list, description="Generated recommendations.")
    trajectory_update: Optional[Dict[str, Any]] = Field(default=None, description="Proposed trajectory updates.")
    decision_reviews_due: List[str] = Field(default_factory=list, description="Decision journal IDs due for review.")
    created_at: str = Field(default_factory=utc_now_iso, description="Generation timestamp.")


class StrategicDriftReport(BaseModel):
    """Mathematical report evaluating alignment between actual activity and strategic initiatives."""
    period_start: str = Field(..., description="Start of evaluation window (YYYY-MM-DD).")
    period_end: str = Field(..., description="End of evaluation window (YYYY-MM-DD).")
    alignment_score: float = Field(..., description="Strategic alignment index (0.0 to 100.0%).")
    drift_severity: DriftSeverity = Field(default=DriftSeverity.NORMAL, description="Severity assessment of strategic drift.")
    velocity_vector: VelocityVector = Field(default=VelocityVector.STEADY, description="Velocity momentum trajectory.")
    total_hours_analyzed: float = Field(default=0.0, description="Total logged hours across the window.")
    category_breakdown: Dict[str, float] = Field(default_factory=dict, description="Percentage of effort per category.")
    neglected_initiatives: List[str] = Field(default_factory=list, description="Active initiatives with <= 1 hour logged.")
    runaway_categories: List[str] = Field(default_factory=list, description="Categories exceeding 35% unplanned effort.")
    recommendations_for_rebalancing: List[str] = Field(default_factory=list, description="Immediate corrective actions.")
    created_at: str = Field(default_factory=utc_now_iso, description="Timestamp generated.")


class CausalAttributionTree(BaseModel):
    """Complete causal provenance tree from Recommendation down to raw evidence and Notion sources."""
    target_id: str = Field(..., description="Root item ID (recommendation or conclusion).")
    recommendation_text: Optional[str] = Field(default=None, description="Recommendation summary if root is recommendation.")
    conclusions: List[Dict[str, Any]] = Field(default_factory=list, description="Supporting conclusions with confidence.")
    observations: List[Dict[str, Any]] = Field(default_factory=list, description="Supporting observations with frequencies.")
    evidence_items: List[EvidenceItem] = Field(default_factory=list, description="Underlying atomic evidence items.")
    sources_cited: List[str] = Field(default_factory=list, description="Clickable Notion URLs and dates.")


class DailyCheckInPrompt(BaseModel):
    """Structured daily check-in message from Motion to maintain accountability."""
    date: str = Field(default_factory=utc_now_iso, description="Date of check-in.")
    greeting: str = Field(..., description="Focused greeting referencing active trajectory.")
    alignment_score_summary: str = Field(..., description="Summary of current week's alignment index.")
    due_decision_alerts: List[str] = Field(default_factory=list, description="Decisions due for user review.")
    high_leverage_question: str = Field(..., description="The single highest-leverage question for today.")
    reply_text: str = Field(..., description="User-facing formatted markdown message.")


class EvidenceIngestionEvent(BaseModel):
    """Event payload for asynchronous background evidence ingestion."""
    event_type: str = Field(..., description="Event type: task_completed, leetcode_review, learning_milestone, daily_log.")
    source_ref: Optional[str] = Field(default=None, description="Notion page URL or ID.")
    title: str = Field(..., description="Task or activity title.")
    description: str = Field(default="", description="Detailed notes or description.")
    date: Optional[str] = Field(default=None, description="Event date (YYYY-MM-DD).")
    duration_hours: Optional[float] = Field(default=None, description="Estimated duration in hours.")
    tags: List[str] = Field(default_factory=list, description="Associated category or domain tags.")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Numerical metrics (e.g. problems_solved=2).")


class MotionContext(BaseModel):
    """Dynamically assembled strategic context for Motion prompt."""
    identity: MotionIdentity
    trajectory: MotionTrajectory
    active_initiatives: List[MotionInitiative] = Field(default_factory=list)
    active_overrides: List[HumanOverride] = Field(default_factory=list)
    recent_weekly_reviews: List[Dict[str, Any]] = Field(default_factory=list)
    due_decisions: List[DecisionJournalEntry] = Field(default_factory=list)
    relevant_conclusions: List[MotionConclusion] = Field(default_factory=list)
    relevant_observations: List[MotionObservation] = Field(default_factory=list)
    drift_report: Optional[StrategicDriftReport] = Field(default=None, description="Latest Strategic Drift Report.")


class MotionConsultationResponse(BaseModel):
    """Structured response from Motion Persona."""
    analysis: str = Field(..., description="Strategic evaluation and synthesis.")
    high_leverage_questions: List[str] = Field(default_factory=list, description="Targeted questions challenging assumptions.")
    recommendations: List[MotionRecommendation] = Field(default_factory=list, description="Evidence-backed recommendations.")
    trade_offs_discussed: List[str] = Field(default_factory=list, description="Key trade-offs highlighted.")
    cited_evidence_summary: List[str] = Field(default_factory=list, description="Observations and evidence cited.")
    reply_text: str = Field(..., description="User-facing formatted markdown message.")
    attribution_tree: Optional[CausalAttributionTree] = Field(default=None, description="Provenance tree if requested.")


# --- Motion v3 Schemas ---

class InitiativeMilestone(BaseModel):
    """Strategic milestone belonging to an active MotionInitiative."""
    id: str = Field(..., description="Unique milestone ID (e.g. ms_ocean_v4_2_forecaster).")
    initiative_id: str = Field(..., description="Parent MotionInitiative ID.")
    title: str = Field(..., description="Milestone title.")
    description: Optional[str] = Field(default="", description="Detailed definition of completion.")
    target_date: Optional[str] = Field(default=None, description="Target completion date (YYYY-MM-DD).")
    estimated_hours: float = Field(default=10.0, description="Estimated total focus hours required.")
    completed_hours: float = Field(default=0.0, description="Logged hours contributed toward this milestone.")
    status: MilestoneStatus = Field(default=MilestoneStatus.NOT_STARTED, description="Milestone progression state.")
    dependencies: List[str] = Field(default_factory=list, description="Prerequisite milestone IDs in DAG.")
    is_critical_path: bool = Field(default=False, description="Whether milestone lies on the critical path.")
    slippage_risk_days: int = Field(default=0, description="Projected schedule delay in days.")
    created_at: str = Field(default_factory=utc_now_iso, description="Creation timestamp.")
    updated_at: str = Field(default_factory=utc_now_iso, description="Last update timestamp.")


class ScenarioSimulationRequest(BaseModel):
    """Request payload to simulate a proposed time/focus reallocation."""
    scenario_name: str = Field(..., description="Descriptive scenario name (e.g. 'LeetCode Acceleration').")
    description: Optional[str] = Field(default="", description="Context behind the hypothetical shift.")
    time_adjustments: Dict[str, float] = Field(
        ...,
        description="Category/Initiative mapping to weekly hour delta (e.g. {'LeetCode & Algorithms': 5.0, 'Systems Engineering': -5.0}).",
    )
    timeframe_weeks: int = Field(default=4, description="Simulation horizon in weeks.")


class ScenarioSimulationResult(BaseModel):
    """Output analysis of a simulated strategic scenario."""
    id: str = Field(..., description="Unique simulation result ID.")
    scenario_name: str = Field(..., description="Simulated scenario name.")
    current_alignment_score: float = Field(..., description="Current baseline Strategic Alignment Index.")
    projected_alignment_score: float = Field(..., description="Simulated Strategic Alignment Index.")
    projected_completion_shifts: Dict[str, int] = Field(
        default_factory=dict,
        description="Initiative/Milestone ID -> shift in projected days (positive = delay, negative = accelerated).",
    )
    trade_offs_identified: List[str] = Field(default_factory=list, description="Second-order trade-offs identified.")
    bottlenecks_flagged: List[str] = Field(default_factory=list, description="Emerging critical path bottlenecks.")
    confidence_score: float = Field(default=0.85, description="Historical velocity statistical confidence (0-1).")
    recommendation_verdict: str = Field(..., description="Motion's strategic evaluation of this scenario.")
    created_at: str = Field(default_factory=utc_now_iso, description="Simulation timestamp.")


class CognitiveEnergyReport(BaseModel):
    """Cognitive sustainability, focus quality, and fatigue evaluation."""
    id: str = Field(..., description="Unique energy report ID.")
    period_start: str = Field(..., description="Evaluation start date (YYYY-MM-DD).")
    period_end: str = Field(..., description="Evaluation end date (YYYY-MM-DD).")
    total_logged_hours: float = Field(..., description="Total focus hours logged in period.")
    avg_daily_hours: float = Field(..., description="Average daily hours logged.")
    consecutive_high_intensity_days: int = Field(default=0, description="Consecutive days exceeding 7+ hours.")
    flow_vs_thrash_ratio: float = Field(default=1.0, description="Ratio of deep work to fragmented switching (>=1.5 is deep flow).")
    fatigue_risk_score: float = Field(..., description="Calculated fatigue index (0-100%).")
    burnout_risk_level: BurnoutRiskLevel = Field(..., description="Burnout severity tier.")
    sustainability_diagnosis: str = Field(..., description="Motion's diagnostic evaluation.")
    recommended_decompression_hours: float = Field(default=0.0, description="Suggested restorative buffer hours.")
    created_at: str = Field(default_factory=utc_now_iso, description="Report timestamp.")


class SocraticInquiryResult(BaseModel):
    """Socratic assumption deconstruction and pre-mortem failure analysis."""
    id: str = Field(..., description="Unique Socratic analysis ID.")
    topic: str = Field(..., description="Strategic dilemma or proposition analyzed.")
    unexamined_assumptions: List[str] = Field(default_factory=list, description="Identified implicit assumptions.")
    premortem_failure_scenarios: List[str] = Field(default_factory=list, description="Plausible 6-month failure modes.")
    second_order_consequences: List[str] = Field(default_factory=list, description="Downstream ripple effects.")
    inversion_analysis: str = Field(default="", description="Inverted thinking framework analysis.")
    probing_questions: List[str] = Field(default_factory=list, description="High-leverage challenging questions.")
    created_at: str = Field(default_factory=utc_now_iso, description="Timestamp.")


class ExecutiveBriefing(BaseModel):
    """Autonomous end-of-week strategic executive digest."""
    id: str = Field(..., description="Unique executive briefing ID.")
    period_start: str = Field(..., description="Start date (YYYY-MM-DD).")
    period_end: str = Field(..., description="End date (YYYY-MM-DD).")
    strategic_alignment_score: float = Field(..., description="Overall weekly alignment score.")
    drift_severity: DriftSeverity = Field(..., description="Current drift classification.")
    velocity_vector: VelocityVector = Field(..., description="Overall velocity vector.")
    fatigue_risk_score: float = Field(..., description="Cognitive fatigue score.")
    burnout_risk_level: BurnoutRiskLevel = Field(..., description="Burnout risk tier.")
    milestone_summary: Dict[str, Any] = Field(default_factory=dict, description="Active milestone completion metrics.")
    key_wins: List[str] = Field(default_factory=list, description="Major strategic achievements.")
    strategic_vulnerabilities: List[str] = Field(default_factory=list, description="Key risks and neglected priorities.")
    top_recommendations: List[MotionRecommendation] = Field(default_factory=list, description="High-leverage advice.")
    proposed_trajectory_calibration: Optional[Dict[str, Any]] = Field(default=None, description="Proposed trajectory adjustment.")
    formatted_markdown_briefing: str = Field(..., description="Ready-to-read executive briefing markdown.")
    created_at: str = Field(default_factory=utc_now_iso, description="Briefing creation timestamp.")
