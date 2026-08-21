"""Storage repository for Ocean Motion Strategic Model.

Maintains structured, file-backed storage in `data/motion/` for:
- identity.json
- trajectory.json
- initiatives.json
- decision_journal.json
- overrides.json
- observations/
- conclusions/
- weekly_reviews/
"""

import json
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

from app.motion.schemas import (
    CognitiveEnergyReport,
    DecisionJournalEntry,
    EvidenceItem,
    ExecutiveBriefing,
    HumanOverride,
    InitiativeMilestone,
    MotionConclusion,
    MotionIdentity,
    MotionInitiative,
    MotionObservation,
    MotionTrajectory,
    MotionWeeklyReview,
    ScenarioSimulationResult,
    StrategicDriftReport,
)
from app.motion.spec import DEFAULT_MOTION_DATA_DIR

logger = logging.getLogger("notion-assistant.motion.storage")


class MotionStorage:
    """File-backed storage manager for Motion Strategic Model."""

    def __init__(self, base_dir: str = DEFAULT_MOTION_DATA_DIR):
        self.base_dir = base_dir
        self.identity_file = os.path.join(self.base_dir, "identity.json")
        self.trajectory_file = os.path.join(self.base_dir, "trajectory.json")
        self.initiatives_file = os.path.join(self.base_dir, "initiatives.json")
        self.decision_journal_file = os.path.join(self.base_dir, "decision_journal.json")
        self.overrides_file = os.path.join(self.base_dir, "overrides.json")
        self.evidence_dir = os.path.join(self.base_dir, "evidence")
        self.observations_dir = os.path.join(self.base_dir, "observations")
        self.conclusions_dir = os.path.join(self.base_dir, "conclusions")
        self.weekly_reviews_dir = os.path.join(self.base_dir, "weekly_reviews")
        self.drift_reports_dir = os.path.join(self.base_dir, "drift_reports")
        self.milestones_dir = os.path.join(self.base_dir, "milestones")
        self.energy_reports_dir = os.path.join(self.base_dir, "energy_reports")
        self.briefings_dir = os.path.join(self.base_dir, "briefings")
        self.simulations_dir = os.path.join(self.base_dir, "simulations")
        self._init_storage()

    def _init_storage(self) -> None:
        """Create storage directories and initialize default files if missing."""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.evidence_dir, exist_ok=True)
        os.makedirs(self.observations_dir, exist_ok=True)
        os.makedirs(self.conclusions_dir, exist_ok=True)
        os.makedirs(self.weekly_reviews_dir, exist_ok=True)
        os.makedirs(self.drift_reports_dir, exist_ok=True)
        os.makedirs(self.milestones_dir, exist_ok=True)
        os.makedirs(self.energy_reports_dir, exist_ok=True)
        os.makedirs(self.briefings_dir, exist_ok=True)
        os.makedirs(self.simulations_dir, exist_ok=True)

        if not os.path.exists(self.identity_file):
            default_identity = MotionIdentity(
                education="MS in Computer Science candidate at UMass Amherst",
                career_goals=["AI/ML Systems Engineer", "Applied AI Researcher", "Founding Engineer"],
                long_term_ambitions=["Build impactful intelligent agent systems", "Publish novel AI systems research"],
                core_constraints=["Academic deadlines", "Time allocation between coursework, coding, and research"],
                values=["Depth over superficiality", "Rigorous execution", "Evidence-grounded strategy"],
            )
            self.save_identity(default_identity)

        if not os.path.exists(self.trajectory_file):
            default_trajectory = MotionTrajectory(
                current_phase="Foundational Execution & Research Readiness",
                current_direction="Deepening AI systems competence while maintaining steady academic velocity",
                momentum="MODERATE",
                biggest_opportunity="Ocean autonomous systems engineering and research project delivery",
                biggest_risk="Context fragmentation across too many concurrent secondary tasks",
                next_review="Upcoming Sunday Weekly Review",
            )
            self.save_trajectory(default_trajectory)

        if not os.path.exists(self.initiatives_file):
            self._write_json_atomic(self.initiatives_file, [])

        if not os.path.exists(self.decision_journal_file):
            self._write_json_atomic(self.decision_journal_file, [])

        if not os.path.exists(self.overrides_file):
            self._write_json_atomic(self.overrides_file, [])

    def _write_json_atomic(self, file_path: str, data: Any) -> None:
        """Write JSON data atomically using a temporary file to avoid corruption."""
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix="tmp_motion_")
        try:
            with open(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            shutil.move(temp_path, file_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _read_json_safe(self, file_path: str, default: Any) -> Any:
        """Read JSON data safely, returning default if file is missing or invalid."""
        if not os.path.exists(file_path):
            return default
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Error reading JSON file %s: %s", file_path, exc)
            return default

    # --- Identity ---
    def load_identity(self) -> MotionIdentity:
        """Load the user's stable identity."""
        data = self._read_json_safe(self.identity_file, {})
        return MotionIdentity.model_validate(data) if data else MotionIdentity()

    def save_identity(self, identity: MotionIdentity) -> None:
        """Save the user's stable identity. (Must only be invoked by explicit user edit)."""
        self._write_json_atomic(self.identity_file, identity.model_dump())

    # --- Trajectory ---
    def load_trajectory(self) -> MotionTrajectory:
        """Load current strategic trajectory."""
        data = self._read_json_safe(self.trajectory_file, {})
        return MotionTrajectory.model_validate(data) if data else MotionTrajectory()

    def save_trajectory(self, trajectory: MotionTrajectory) -> None:
        """Save updated strategic trajectory."""
        self._write_json_atomic(self.trajectory_file, trajectory.model_dump())

    # --- Initiatives ---
    def load_initiatives(self) -> List[MotionInitiative]:
        """Load all strategic initiatives."""
        raw_list = self._read_json_safe(self.initiatives_file, [])
        return [MotionInitiative.model_validate(item) for item in raw_list]

    def save_initiatives(self, initiatives: List[MotionInitiative]) -> None:
        """Save the list of strategic initiatives."""
        data = [init.model_dump() for init in initiatives]
        self._write_json_atomic(self.initiatives_file, data)

    # --- Decision Journal ---
    def load_decision_journal(self) -> List[DecisionJournalEntry]:
        """Load all decision journal entries."""
        raw_list = self._read_json_safe(self.decision_journal_file, [])
        return [DecisionJournalEntry.model_validate(item) for item in raw_list]

    def save_decision_journal(self, entries: List[DecisionJournalEntry]) -> None:
        """Save all decision journal entries."""
        data = [entry.model_dump() for entry in entries]
        self._write_json_atomic(self.decision_journal_file, data)

    # --- Human Overrides ---
    def load_overrides(self) -> List[HumanOverride]:
        """Load all human overrides."""
        raw_list = self._read_json_safe(self.overrides_file, [])
        return [HumanOverride.model_validate(item) for item in raw_list]

    def save_overrides(self, overrides: List[HumanOverride]) -> None:
        """Save all human overrides."""
        data = [ovr.model_dump() for ovr in overrides]
        self._write_json_atomic(self.overrides_file, data)

    # --- Observations ---
    def save_observation(self, observation: MotionObservation) -> str:
        """Save a discrete observation JSON file."""
        file_path = os.path.join(self.observations_dir, f"{observation.id}.json")
        self._write_json_atomic(file_path, observation.model_dump())
        return file_path

    def load_observations(self, limit: int = 50) -> List[MotionObservation]:
        """Load recent observations ordered by date descending."""
        obs_list = []
        if not os.path.exists(self.observations_dir):
            return obs_list
        files = sorted(os.listdir(self.observations_dir), reverse=True)
        for fname in files[:limit]:
            if fname.endswith(".json"):
                fpath = os.path.join(self.observations_dir, fname)
                data = self._read_json_safe(fpath, None)
                if data:
                    try:
                        obs_list.append(MotionObservation.model_validate(data))
                    except Exception as err:
                        logger.warning("Failed parsing observation %s: %s", fname, err)
        return obs_list

    # --- Conclusions ---
    def save_conclusion(self, conclusion: MotionConclusion) -> str:
        """Save a discrete conclusion JSON file."""
        file_path = os.path.join(self.conclusions_dir, f"{conclusion.id}.json")
        self._write_json_atomic(file_path, conclusion.model_dump())
        return file_path

    def load_conclusions(self, limit: int = 50) -> List[MotionConclusion]:
        """Load recent conclusions ordered by creation date descending."""
        conc_list = []
        if not os.path.exists(self.conclusions_dir):
            return conc_list
        files = sorted(os.listdir(self.conclusions_dir), reverse=True)
        for fname in files[:limit]:
            if fname.endswith(".json"):
                fpath = os.path.join(self.conclusions_dir, fname)
                data = self._read_json_safe(fpath, None)
                if data:
                    try:
                        conc_list.append(MotionConclusion.model_validate(data))
                    except Exception as err:
                        logger.warning("Failed parsing conclusion %s: %s", fname, err)
        return conc_list

    # --- Weekly Reviews ---
    def save_weekly_review(self, review: MotionWeeklyReview) -> str:
        """Save a synthesized weekly review JSON file."""
        file_path = os.path.join(self.weekly_reviews_dir, f"{review.id}.json")
        self._write_json_atomic(file_path, review.model_dump())
        return file_path

    def load_weekly_reviews(self, limit: int = 5) -> List[MotionWeeklyReview]:
        """Load recent weekly reviews ordered descending."""
        rev_list = []
        if not os.path.exists(self.weekly_reviews_dir):
            return rev_list
        files = sorted(os.listdir(self.weekly_reviews_dir), reverse=True)
        for fname in files[:limit]:
            if fname.endswith(".json"):
                fpath = os.path.join(self.weekly_reviews_dir, fname)
                data = self._read_json_safe(fpath, None)
                if data:
                    try:
                        rev_list.append(MotionWeeklyReview.model_validate(data))
                    except Exception as err:
                        logger.warning("Failed parsing review %s: %s", fname, err)
        return rev_list

    # --- Ingested Evidence ---
    def save_evidence(self, evidence: EvidenceItem) -> str:
        """Save a discrete evidence item JSON file."""
        file_path = os.path.join(self.evidence_dir, f"{evidence.id}.json")
        self._write_json_atomic(file_path, evidence.model_dump())
        return file_path

    def load_evidence(
        self,
        limit: int = 200,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[EvidenceItem]:
        """Load ingested evidence items, optionally filtered by date range."""
        ev_list: List[EvidenceItem] = []
        if not os.path.exists(self.evidence_dir):
            return ev_list
        files = sorted(os.listdir(self.evidence_dir), reverse=True)
        for fname in files:
            if fname.endswith(".json"):
                fpath = os.path.join(self.evidence_dir, fname)
                data = self._read_json_safe(fpath, None)
                if data:
                    try:
                        item = EvidenceItem.model_validate(data)
                        if start_date and item.date < start_date:
                            continue
                        if end_date and item.date > end_date:
                            continue
                        ev_list.append(item)
                        if len(ev_list) >= limit:
                            break
                    except Exception as err:
                        logger.warning("Failed parsing evidence %s: %s", fname, err)
        return ev_list

    # --- Strategic Drift Reports ---
    def save_drift_report(self, report: StrategicDriftReport) -> str:
        """Save a strategic drift evaluation report."""
        clean_period = f"{report.period_start.replace('-', '')}_{report.period_end.replace('-', '')}"
        file_path = os.path.join(self.drift_reports_dir, f"drift_{clean_period}.json")
        self._write_json_atomic(file_path, report.model_dump())
        return file_path

    def load_latest_drift_report(self) -> Optional[StrategicDriftReport]:
        """Load the most recent strategic drift report."""
        if not os.path.exists(self.drift_reports_dir):
            return None
        files = sorted(os.listdir(self.drift_reports_dir), reverse=True)
        for fname in files:
            if fname.endswith(".json"):
                fpath = os.path.join(self.drift_reports_dir, fname)
                data = self._read_json_safe(fpath, None)
                if data:
                    try:
                        return StrategicDriftReport.model_validate(data)
                    except Exception as err:
                        logger.warning("Failed parsing drift report %s: %s", fname, err)
        return None

    # --- Initiative Milestones (v3) ---
    def save_milestone(self, milestone: InitiativeMilestone) -> str:
        """Save an initiative milestone JSON file."""
        file_path = os.path.join(self.milestones_dir, f"{milestone.id}.json")
        self._write_json_atomic(file_path, milestone.model_dump())
        return file_path

    def load_milestones(self, initiative_id: Optional[str] = None) -> List[InitiativeMilestone]:
        """Load all initiative milestones, optionally filtered by initiative."""
        ms_list: List[InitiativeMilestone] = []
        if not os.path.exists(self.milestones_dir):
            return ms_list
        files = sorted(os.listdir(self.milestones_dir))
        for fname in files:
            if fname.endswith(".json"):
                fpath = os.path.join(self.milestones_dir, fname)
                data = self._read_json_safe(fpath, None)
                if data:
                    try:
                        ms = InitiativeMilestone.model_validate(data)
                        if initiative_id and ms.initiative_id != initiative_id:
                            continue
                        ms_list.append(ms)
                    except Exception as err:
                        logger.warning("Failed parsing milestone %s: %s", fname, err)
        return ms_list

    def delete_milestone(self, milestone_id: str) -> bool:
        """Delete a milestone file by ID."""
        file_path = os.path.join(self.milestones_dir, f"{milestone_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False

    # --- Cognitive Energy Reports (v3) ---
    def save_energy_report(self, report: CognitiveEnergyReport) -> str:
        """Save a cognitive energy & sustainability evaluation."""
        file_path = os.path.join(self.energy_reports_dir, f"{report.id}.json")
        self._write_json_atomic(file_path, report.model_dump())
        return file_path

    def load_latest_energy_report(self) -> Optional[CognitiveEnergyReport]:
        """Load the most recent cognitive energy report."""
        if not os.path.exists(self.energy_reports_dir):
            return None
        files = sorted(os.listdir(self.energy_reports_dir), reverse=True)
        for fname in files:
            if fname.endswith(".json"):
                fpath = os.path.join(self.energy_reports_dir, fname)
                data = self._read_json_safe(fpath, None)
                if data:
                    try:
                        return CognitiveEnergyReport.model_validate(data)
                    except Exception as err:
                        logger.warning("Failed parsing energy report %s: %s", fname, err)
        return None

    # --- Executive Briefings (v3) ---
    def save_briefing(self, briefing: ExecutiveBriefing) -> str:
        """Save an executive strategic briefing."""
        file_path = os.path.join(self.briefings_dir, f"{briefing.id}.json")
        self._write_json_atomic(file_path, briefing.model_dump())
        return file_path

    def load_latest_briefing(self) -> Optional[ExecutiveBriefing]:
        """Load the most recent executive briefing."""
        if not os.path.exists(self.briefings_dir):
            return None
        files = sorted(os.listdir(self.briefings_dir), reverse=True)
        for fname in files:
            if fname.endswith(".json"):
                fpath = os.path.join(self.briefings_dir, fname)
                data = self._read_json_safe(fpath, None)
                if data:
                    try:
                        return ExecutiveBriefing.model_validate(data)
                    except Exception as err:
                        logger.warning("Failed parsing briefing %s: %s", fname, err)
        return None

    # --- Scenario Simulations (v3) ---
    def save_simulation(self, sim: ScenarioSimulationResult) -> str:
        """Save a strategic scenario simulation result."""
        file_path = os.path.join(self.simulations_dir, f"{sim.id}.json")
        self._write_json_atomic(file_path, sim.model_dump())
        return file_path

    def load_simulations(self, limit: int = 20) -> List[ScenarioSimulationResult]:
        """Load historical scenario simulations."""
        sims: List[ScenarioSimulationResult] = []
        if not os.path.exists(self.simulations_dir):
            return sims
        files = sorted(os.listdir(self.simulations_dir), reverse=True)
        for fname in files:
            if fname.endswith(".json"):
                fpath = os.path.join(self.simulations_dir, fname)
                data = self._read_json_safe(fpath, None)
                if data:
                    try:
                        sims.append(ScenarioSimulationResult.model_validate(data))
                        if len(sims) >= limit:
                            break
                    except Exception as err:
                        logger.warning("Failed parsing simulation %s: %s", fname, err)
        return sims


motion_storage = MotionStorage()
