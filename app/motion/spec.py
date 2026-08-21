"""Ocean Motion Specification Constants, Enums, and Permission Definitions."""

from enum import Enum
import os
from typing import Set


class PersonaType(str, Enum):
    """Supported Ocean Persona Types."""
    OCEAN = "OCEAN"
    MOTION = "MOTION"


class ConfidenceLevel(str, Enum):
    """Rule-based, explainable confidence levels for observations & conclusions."""
    LOW = "LOW"        # Single observation / limited data point
    MEDIUM = "MEDIUM"  # Observed multiple times across days or sessions
    HIGH = "HIGH"      # Observed consistently across multiple review cycles


class DecisionStatus(str, Enum):
    """Decision Journal State Machine states."""
    PENDING = "PENDING"
    DUE = "DUE"
    REVIEWED = "REVIEWED"
    CLOSED = "CLOSED"


class OverrideStatus(str, Enum):
    """Human Override status states."""
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    REVIEWED = "REVIEWED"
    RESOLVED = "RESOLVED"


class StrategicImportance(str, Enum):
    """Importance tiers for strategic initiatives."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"


class Horizon(str, Enum):
    """Time horizon for initiatives."""
    NEAR_TERM = "NEAR_TERM"      # < 1 month
    MEDIUM_TERM = "MEDIUM_TERM"  # 1 - 6 months
    LONG_TERM = "LONG_TERM"      # > 6 months


class TrajectoryMomentum(str, Enum):
    """Trajectory momentum assessment."""
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    STALLED = "STALLED"
    REGRESSING = "REGRESSING"


class DriftSeverity(str, Enum):
    """Severity of strategic drift."""
    NORMAL = "NORMAL"                # > 80% alignment
    LOW_DRIFT = "LOW_DRIFT"          # 65% - 80% alignment
    MODERATE_DRIFT = "MODERATE_DRIFT" # 45% - 65% alignment
    CRITICAL_DRIFT = "CRITICAL_DRIFT" # < 45% alignment


class VelocityVector(str, Enum):
    """Week-over-week velocity trend."""
    ACCELERATING = "ACCELERATING"
    STEADY = "STEADY"
    SLOWING = "SLOWING"
    STALLED = "STALLED"


class BurnoutRiskLevel(str, Enum):
    """Cognitive sustainability and fatigue risk classification."""
    LOW = "LOW"            # Sustainable pace, balanced load
    ELEVATED = "ELEVATED"  # Minor fatigue accumulation / high intensity streak
    HIGH = "HIGH"          # Significant strain, high thrash / prolonged burst
    CRITICAL = "CRITICAL"  # Acute burnout danger, immediate decompression buffer required


class MilestoneStatus(str, Enum):
    """Initiative milestone status within critical path DAG."""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    AT_RISK = "AT_RISK"
    ACHIEVED = "ACHIEVED"


class MotionPermission(str, Enum):
    """Discrete system tools and capabilities controlled at dispatch layer."""
    # Motion Allowed
    READ_LOGS = "READ_LOGS"
    READ_STRATEGIC_MODEL = "READ_STRATEGIC_MODEL"
    READ_CALENDAR_TASKS = "READ_CALENDAR_TASKS"
    READ_PROJECTS = "READ_PROJECTS"
    READ_DECISION_JOURNAL = "READ_DECISION_JOURNAL"

    # Motion Forbidden (Ocean Only)
    EXECUTE_TASKS = "EXECUTE_TASKS"
    MODIFY_NOTION = "MODIFY_NOTION"
    CREATE_TASKS = "CREATE_TASKS"
    WRITE_CODE = "WRITE_CODE"
    RUN_SEARCHES = "RUN_SEARCHES"
    INVOKE_CODING_TOOLS = "INVOKE_CODING_TOOLS"


MOTION_ALLOWED_PERMISSIONS: Set[MotionPermission] = {
    MotionPermission.READ_LOGS,
    MotionPermission.READ_STRATEGIC_MODEL,
    MotionPermission.READ_CALENDAR_TASKS,
    MotionPermission.READ_PROJECTS,
    MotionPermission.READ_DECISION_JOURNAL,
}

MOTION_FORBIDDEN_PERMISSIONS: Set[MotionPermission] = {
    MotionPermission.EXECUTE_TASKS,
    MotionPermission.MODIFY_NOTION,
    MotionPermission.CREATE_TASKS,
    MotionPermission.WRITE_CODE,
    MotionPermission.RUN_SEARCHES,
    MotionPermission.INVOKE_CODING_TOOLS,
}

# --- Motion v2 Sliding Window Constants ---
SLIDING_WINDOW_3D = 3
SLIDING_WINDOW_7D = 7
SLIDING_WINDOW_14D = 14
SLIDING_WINDOW_30D = 30

# --- Strategic Drift Thresholds ---
NEGLECTED_INITIATIVE_DAYS = 7
FRAGMENTATION_THRESHOLD_RATIO = 0.35

DEFAULT_MOTION_DATA_DIR = os.path.join("data", "motion")
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
