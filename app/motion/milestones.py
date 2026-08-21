"""Initiative Milestone & Critical Path Engine for Ocean Motion v3.

Manages structured strategic milestones within initiatives, computes dependency DAGs,
identifies the critical path, and detects bottleneck slippage risks.
"""

from collections import defaultdict, deque
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from app.motion.schemas import (
    InitiativeMilestone,
    MilestoneStatus,
    MotionInitiative,
)
from app.motion.storage import MotionStorage, motion_storage

logger = logging.getLogger("notion-assistant.motion.milestones")


class InitiativeMilestoneEngine:
    """Manages initiative milestones and performs critical path DAG analysis."""

    def __init__(self, storage: Optional[MotionStorage] = None):
        self.storage = storage or motion_storage

    def record_milestone(self, milestone: InitiativeMilestone) -> InitiativeMilestone:
        """Create or update an initiative milestone."""
        milestone.updated_at = datetime.now(timezone.utc).isoformat()
        self.storage.save_milestone(milestone)
        logger.info("Saved milestone '%s' for initiative %s", milestone.title, milestone.initiative_id)
        return milestone

    def get_milestones(self, initiative_id: Optional[str] = None) -> List[InitiativeMilestone]:
        """Load milestones, optionally filtered by initiative."""
        return self.storage.load_milestones(initiative_id=initiative_id)

    def analyze_critical_path(
        self,
        initiative_id: Optional[str] = None,
        reference_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform topological sort and critical path DAG analysis across milestones."""
        if not reference_date:
            reference_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        milestones = self.storage.load_milestones(initiative_id=initiative_id)
        if not milestones:
            return {
                "critical_path": [],
                "total_critical_hours": 0.0,
                "bottlenecks": [],
                "milestones_by_status": {},
                "summary": "No milestones configured for analysis.",
            }

        ms_by_id: Dict[str, InitiativeMilestone] = {m.id: m for m in milestones}

        # 1. Build adjacency list and in-degrees for DAG
        graph: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = {m.id: 0 for m in milestones}

        for m in milestones:
            for dep_id in m.dependencies:
                if dep_id in ms_by_id:
                    graph[dep_id].append(m.id)
                    in_degree[m.id] += 1

        # 2. Topological Sort (Kahn's Algorithm)
        queue = deque([m_id for m_id, deg in in_degree.items() if deg == 0])
        topo_order: List[str] = []

        while queue:
            curr = queue.popleft()
            topo_order.append(curr)
            for neighbor in graph[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Fallback if cycle detected or disconnected nodes
        if len(topo_order) < len(milestones):
            for m in milestones:
                if m.id not in topo_order:
                    topo_order.append(m.id)

        # 3. Dynamic Programming for Longest Path (Critical Path)
        dist: Dict[str, float] = {m.id: (m.estimated_hours - m.completed_hours) for m in milestones}
        prev: Dict[str, Optional[str]] = {m.id: None for m in milestones}

        for u in topo_order:
            for v in graph[u]:
                rem_v = max(0.0, ms_by_id[v].estimated_hours - ms_by_id[v].completed_hours)
                if dist[u] + rem_v > dist[v]:
                    dist[v] = dist[u] + rem_v
                    prev[v] = u

        # Identify milestone with maximum distance
        end_node = max(dist.keys(), key=lambda k: dist[k]) if dist else None
        critical_path_ids: List[str] = []

        curr = end_node
        while curr is not None:
            critical_path_ids.append(curr)
            curr = prev.get(curr)

        critical_path_ids.reverse()
        critical_set = set(critical_path_ids)

        # 4. Check for Blockers, Slippage Risk, and Update Flags
        bottlenecks: List[str] = []
        status_counts: Dict[str, int] = defaultdict(int)

        ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")

        for m in milestones:
            m.is_critical_path = m.id in critical_set
            status_counts[m.status.value] += 1

            # Check if blocked by incomplete dependencies
            if m.dependencies and m.status != MilestoneStatus.ACHIEVED:
                incomplete_deps = [
                    ms_by_id[d].title for d in m.dependencies
                    if d in ms_by_id and ms_by_id[d].status != MilestoneStatus.ACHIEVED
                ]
                if incomplete_deps:
                    m.status = MilestoneStatus.BLOCKED
                    if m.is_critical_path:
                        bottlenecks.append(f"Critical path milestone '{m.title}' is blocked by: {', '.join(incomplete_deps)}")

            # Check slippage against target date
            if m.target_date and m.status not in (MilestoneStatus.ACHIEVED, MilestoneStatus.BLOCKED):
                try:
                    tgt_dt = datetime.strptime(m.target_date, "%Y-%m-%d")
                    days_remaining = (tgt_dt - ref_dt).days
                    rem_hours = max(0.0, m.estimated_hours - m.completed_hours)

                    if days_remaining <= 0 and rem_hours > 0:
                        m.status = MilestoneStatus.AT_RISK
                        m.slippage_risk_days = abs(days_remaining) + int(rem_hours / 2.0)
                        bottlenecks.append(f"Milestone '{m.title}' is overdue by {abs(days_remaining)} days ({rem_hours:.1f}h remaining).")
                    elif days_remaining > 0 and (rem_hours / max(1, days_remaining)) > 4.0:
                        m.status = MilestoneStatus.AT_RISK
                        m.slippage_risk_days = int(rem_hours / 2.0) - days_remaining
                        if m.is_critical_path:
                            bottlenecks.append(f"Critical path milestone '{m.title}' requires high burn ({rem_hours:.1f}h in {days_remaining}d).")
                except Exception:
                    pass

            self.storage.save_milestone(m)

        critical_path_milestones = [ms_by_id[m_id] for m_id in critical_path_ids if m_id in ms_by_id]
        total_crit_hours = sum(max(0.0, m.estimated_hours - m.completed_hours) for m in critical_path_milestones)

        summary = (
            f"Critical path contains {len(critical_path_ids)} milestones ({total_crit_hours:.1f} hours remaining). "
            f"Identified {len(bottlenecks)} active schedule bottlenecks."
        )

        return {
            "critical_path": [m.model_dump() for m in critical_path_milestones],
            "total_critical_hours": total_crit_hours,
            "bottlenecks": bottlenecks,
            "milestones_by_status": dict(status_counts),
            "summary": summary,
        }


milestone_engine = InitiativeMilestoneEngine()
