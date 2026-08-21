"""Strategic Model Service for Ocean Motion.

Owns the high-level business logic, state machines, invariants, and provenance tracking
for the user's Strategic Model.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.motion.schemas import (
    DecisionJournalEntry,
    HumanOverride,
    MotionConclusion,
    MotionIdentity,
    MotionInitiative,
    MotionObservation,
    MotionRecommendation,
    MotionTrajectory,
    utc_now_iso,
)
from app.motion.spec import (
    DecisionStatus,
    Horizon,
    OverrideStatus,
    StrategicImportance,
    TrajectoryMomentum,
)
from app.motion.storage import MotionStorage, motion_storage

logger = logging.getLogger("notion-assistant.motion.strategic_model")


class StrategicModelService:
    """Service encapsulating strategic model state transitions, queries, and provenance."""

    def __init__(self, storage: Optional[MotionStorage] = None):
        self.storage = storage or motion_storage

    # --- Identity Operations ---
    def get_identity(self) -> MotionIdentity:
        """Fetch current stable identity."""
        return self.storage.load_identity()

    def update_identity(
        self,
        new_identity: MotionIdentity,
        explicit_user_confirmation: bool = False,
    ) -> MotionIdentity:
        """Update identity. Strictly enforces that identity only updates via explicit user edits."""
        if not explicit_user_confirmation:
            raise ValueError(
                "Identity invariant violation: Identity can only be updated via explicit user confirmation. "
                "Motion never automatically infers or updates identity."
            )
        new_identity.updated_at = utc_now_iso()
        self.storage.save_identity(new_identity)
        logger.info("Motion identity explicitly updated.")
        return new_identity

    # --- Trajectory Operations ---
    def get_trajectory(self) -> MotionTrajectory:
        """Fetch current strategic trajectory."""
        return self.storage.load_trajectory()

    def update_trajectory(
        self,
        current_phase: Optional[str] = None,
        current_direction: Optional[str] = None,
        momentum: Optional[TrajectoryMomentum] = None,
        biggest_opportunity: Optional[str] = None,
        biggest_risk: Optional[str] = None,
        next_review: Optional[str] = None,
    ) -> MotionTrajectory:
        """Update trajectory fields (normally during weekly reviews)."""
        current = self.get_trajectory()
        if current_phase is not None:
            current.current_phase = current_phase
        if current_direction is not None:
            current.current_direction = current_direction
        if momentum is not None:
            current.momentum = momentum
        if biggest_opportunity is not None:
            current.biggest_opportunity = biggest_opportunity
        if biggest_risk is not None:
            current.biggest_risk = biggest_risk
        if next_review is not None:
            current.next_review = next_review
        current.last_updated = utc_now_iso()
        self.storage.save_trajectory(current)
        logger.info("Motion trajectory updated.")
        return current

    # --- Initiatives Operations ---
    def get_initiatives(self, status: Optional[str] = None) -> List[MotionInitiative]:
        """Fetch strategic initiatives, optionally filtering by status (e.g. 'ACTIVE')."""
        all_inits = self.storage.load_initiatives()
        if status:
            return [init for init in all_inits if init.status.upper() == status.upper()]
        return all_inits

    def create_or_update_initiative(self, initiative: MotionInitiative) -> MotionInitiative:
        """Create or update a strategic initiative."""
        all_inits = self.storage.load_initiatives()
        existing_idx = next((i for i, item in enumerate(all_inits) if item.id == initiative.id), None)
        initiative.updated_at = utc_now_iso()
        if existing_idx is not None:
            all_inits[existing_idx] = initiative
        else:
            all_inits.append(initiative)
        self.storage.save_initiatives(all_inits)
        logger.info("Initiative '%s' (%s) saved.", initiative.title, initiative.id)
        return initiative

    def archive_initiative(self, initiative_id: str, new_status: str = "COMPLETED") -> bool:
        """Mark an initiative as COMPLETED or ABANDONED."""
        all_inits = self.storage.load_initiatives()
        for init in all_inits:
            if init.id == initiative_id:
                init.status = new_status
                init.updated_at = utc_now_iso()
                self.storage.save_initiatives(all_inits)
                return True
        return False

    # --- Decision Journal State Machine ---
    # Valid transitions: PENDING -> DUE -> REVIEWED -> CLOSED
    def record_decision(self, entry: DecisionJournalEntry) -> DecisionJournalEntry:
        """Append or update a strategic recommendation in the Decision Journal in PENDING state."""
        entries = self.storage.load_decision_journal()
        existing_idx = next((i for i, e in enumerate(entries) if e.id == entry.id), None)
        entry.status = DecisionStatus.PENDING
        if existing_idx is not None:
            entries[existing_idx] = entry
        else:
            entries.append(entry)
        self.storage.save_decision_journal(entries)
        logger.info("Strategic decision '%s' appended to Decision Journal in PENDING state.", entry.id)
        return entry

    def get_decisions(self, status: Optional[DecisionStatus] = None) -> List[DecisionJournalEntry]:
        """Fetch decision journal entries, optionally filtered by status."""
        entries = self.storage.load_decision_journal()
        if status:
            return [e for e in entries if e.status == status]
        return entries

    def get_decision(self, decision_id: str) -> Optional[DecisionJournalEntry]:
        """Fetch a specific decision by ID."""
        entries = self.storage.load_decision_journal()
        return next((e for e in entries if e.id == decision_id), None)

    def transition_decision_state(
        self,
        decision_id: str,
        target_status: DecisionStatus,
        actual_outcome: Optional[str] = None,
        user_reflection: Optional[str] = None,
    ) -> DecisionJournalEntry:
        """Execute a state machine transition on a Decision Journal entry."""
        entries = self.storage.load_decision_journal()
        entry_idx = next((i for i, e in enumerate(entries) if e.id == decision_id), None)
        if entry_idx is None:
            raise KeyError(f"Decision '{decision_id}' not found in Decision Journal.")

        entry = entries[entry_idx]
        current_status = entry.status

        # Transition validation
        allowed_transitions = {
            DecisionStatus.PENDING: [DecisionStatus.DUE, DecisionStatus.REVIEWED, DecisionStatus.CLOSED],
            DecisionStatus.DUE: [DecisionStatus.REVIEWED, DecisionStatus.CLOSED],
            DecisionStatus.REVIEWED: [DecisionStatus.CLOSED],
            DecisionStatus.CLOSED: [],
        }

        if target_status not in allowed_transitions.get(current_status, []):
            raise ValueError(
                f"Invalid Decision state transition: Cannot transition from {current_status.value} to {target_status.value}."
            )

        if target_status in (DecisionStatus.REVIEWED, DecisionStatus.CLOSED):
            if actual_outcome is not None:
                entry.actual_outcome = actual_outcome
            if user_reflection is not None:
                entry.user_reflection = user_reflection

        if target_status == DecisionStatus.CLOSED:
            if not entry.actual_outcome:
                raise ValueError("Cannot close decision without recording actual outcome.")
            entry.closed_at = utc_now_iso()

        entry.status = target_status
        entries[entry_idx] = entry
        self.storage.save_decision_journal(entries)
        logger.info("Decision '%s' transitioned from %s to %s.", decision_id, current_status.value, target_status.value)
        return entry

    # --- Human Overrides Management ---
    def record_override(self, override: HumanOverride) -> HumanOverride:
        """Record an explicit user override with a condition-based revisit trigger."""
        overrides = self.storage.load_overrides()
        existing_idx = next((i for i, o in enumerate(overrides) if o.id == override.id), None)
        override.status = OverrideStatus.ACTIVE
        if existing_idx is not None:
            overrides[existing_idx] = override
        else:
            overrides.append(override)
        self.storage.save_overrides(overrides)
        logger.info("Human override recorded for topic/rec '%s' with trigger: '%s'.", override.recommendation_id_or_topic, override.review_trigger_condition)
        return override

    def get_active_overrides(self) -> List[HumanOverride]:
        """Fetch all active human overrides."""
        all_ovrs = self.storage.load_overrides()
        return [o for o in all_ovrs if o.status == OverrideStatus.ACTIVE]

    def trigger_override_review(self, override_id: str, resolution_notes: Optional[str] = None) -> HumanOverride:
        """Mark an override as TRIGGERED when its review condition is met."""
        overrides = self.storage.load_overrides()
        for o in overrides:
            if o.id == override_id:
                o.status = OverrideStatus.TRIGGERED
                if resolution_notes:
                    o.resolution_notes = resolution_notes
                self.storage.save_overrides(overrides)
                return o
        raise KeyError(f"Override '{override_id}' not found.")

    def resolve_override(self, override_id: str, resolution_notes: str) -> HumanOverride:
        """Resolve a human override after evaluation."""
        overrides = self.storage.load_overrides()
        for o in overrides:
            if o.id == override_id:
                o.status = OverrideStatus.RESOLVED
                o.resolution_notes = resolution_notes
                self.storage.save_overrides(overrides)
                return o
        raise KeyError(f"Override '{override_id}' not found.")

    # --- Provenance Engine: "Why do I believe this?" ---
    def explain_belief(self, target_id: str) -> Dict[str, Any]:
        """Traverse provenance hierarchy for any recommendation, conclusion, or observation."""
        result: Dict[str, Any] = {
            "target_id": target_id,
            "chain": [],
            "status": "NOT_FOUND",
        }

        # Check if target is a conclusion
        conclusions = self.storage.load_conclusions(limit=100)
        matched_conc = next((c for c in conclusions if c.id == target_id), None)

        if matched_conc:
            result["status"] = "FOUND_CONCLUSION"
            result["conclusion"] = matched_conc.model_dump()
            obs_list = []
            for obs_id in matched_conc.derived_from_observation_ids:
                observations = self.storage.load_observations(limit=100)
                obs = next((o for o in observations if o.id == obs_id), None)
                if obs:
                    obs_list.append(obs.model_dump())
            result["supporting_observations"] = obs_list
            return result

        # Check if target is an observation
        observations = self.storage.load_observations(limit=100)
        matched_obs = next((o for o in observations if o.id == target_id), None)
        if matched_obs:
            result["status"] = "FOUND_OBSERVATION"
            result["observation"] = matched_obs.model_dump()
            result["evidence_ids"] = matched_obs.evidence_ids
            return result

        return result


strategic_model_service = StrategicModelService()
