"""Ocean Motion: Strategic Mentorship Subsystem.

Motion is a strategic mentorship persona within Ocean focused on trajectory optimization,
evidence-based reasoning, accountability, and long-term decision making.
"""

from app.motion.spec import (
    BurnoutRiskLevel,
    ConfidenceLevel,
    DecisionStatus,
    DriftSeverity,
    Horizon,
    MilestoneStatus,
    MotionPermission,
    OverrideStatus,
    PersonaType,
    StrategicImportance,
    TrajectoryMomentum,
    VelocityVector,
)
from app.motion.schemas import (
    CausalAttributionTree,
    CognitiveEnergyReport,
    DailyCheckInPrompt,
    DecisionJournalEntry,
    EvidenceIngestionEvent,
    EvidenceItem,
    ExecutiveBriefing,
    HumanOverride,
    InitiativeMilestone,
    MotionConclusion,
    MotionConsultationResponse,
    MotionContext,
    MotionIdentity,
    MotionInitiative,
    MotionObservation,
    MotionRecommendation,
    MotionTrajectory,
    MotionWeeklyReview,
    ScenarioSimulationRequest,
    ScenarioSimulationResult,
    SocraticInquiryResult,
    StrategicDriftReport,
)
from app.motion.permissions import (
    MotionPermissionError,
    PermissionEngine,
    enforce_persona_permission,
    permission_engine,
)
from app.motion.storage import MotionStorage, motion_storage
from app.motion.strategic_model import StrategicModelService, strategic_model_service
from app.motion.evidence_pipeline import (
    EvidenceAttributionEngine,
    EvidencePipeline,
    evidence_pipeline,
)
from app.motion.ingestion import EvidenceIngestionEngine, evidence_ingestion_engine
from app.motion.synthesis import MultiWindowSynthesisEngine, synthesis_engine
from app.motion.drift_detector import StrategicDriftDetector, drift_detector
from app.motion.attribution import MultiHopAttributionEngine, multi_hop_attribution_engine
from app.motion.accountability import AccountabilityMonitor, accountability_monitor
from app.motion.energy_monitor import CognitiveEnergyMonitor, energy_monitor
from app.motion.milestones import InitiativeMilestoneEngine, milestone_engine
from app.motion.forecaster import ScenarioSimulationEngine, scenario_engine
from app.motion.socratic import SocraticMentorshipEngine, socratic_engine
from app.motion.executive_briefing import ExecutiveBriefingEngine, executive_briefing_engine
from app.motion.retrieval import StrategicContextRetriever, strategic_context_retriever
from app.motion.mentorship_service import (
    MotionMentorshipService,
    motion_mentorship_service,
)
from app.motion.review_service import MotionReviewService, motion_review_service
from app.motion.router import PersonaRouter, persona_router

__all__ = [
    "BurnoutRiskLevel",
    "ConfidenceLevel",
    "DecisionStatus",
    "DriftSeverity",
    "Horizon",
    "MilestoneStatus",
    "MotionPermission",
    "OverrideStatus",
    "PersonaType",
    "StrategicImportance",
    "TrajectoryMomentum",
    "VelocityVector",
    "CausalAttributionTree",
    "CognitiveEnergyReport",
    "DailyCheckInPrompt",
    "DecisionJournalEntry",
    "EvidenceIngestionEvent",
    "EvidenceItem",
    "ExecutiveBriefing",
    "HumanOverride",
    "InitiativeMilestone",
    "MotionConclusion",
    "MotionConsultationResponse",
    "MotionContext",
    "MotionIdentity",
    "MotionInitiative",
    "MotionObservation",
    "MotionRecommendation",
    "MotionTrajectory",
    "MotionWeeklyReview",
    "ScenarioSimulationRequest",
    "ScenarioSimulationResult",
    "SocraticInquiryResult",
    "StrategicDriftReport",
    "MotionPermissionError",
    "PermissionEngine",
    "enforce_persona_permission",
    "permission_engine",
    "MotionStorage",
    "motion_storage",
    "StrategicModelService",
    "strategic_model_service",
    "EvidenceAttributionEngine",
    "EvidencePipeline",
    "evidence_pipeline",
    "EvidenceIngestionEngine",
    "evidence_ingestion_engine",
    "MultiWindowSynthesisEngine",
    "synthesis_engine",
    "StrategicDriftDetector",
    "drift_detector",
    "MultiHopAttributionEngine",
    "multi_hop_attribution_engine",
    "AccountabilityMonitor",
    "accountability_monitor",
    "CognitiveEnergyMonitor",
    "energy_monitor",
    "InitiativeMilestoneEngine",
    "milestone_engine",
    "ScenarioSimulationEngine",
    "scenario_engine",
    "SocraticMentorshipEngine",
    "socratic_engine",
    "ExecutiveBriefingEngine",
    "executive_briefing_engine",
    "StrategicContextRetriever",
    "strategic_context_retriever",
    "MotionMentorshipService",
    "motion_mentorship_service",
    "MotionReviewService",
    "motion_review_service",
    "PersonaRouter",
    "persona_router",
]
