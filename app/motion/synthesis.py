"""Automated Multi-Window Observation & Conclusion Synthesis Engine for Motion v2.

Aggregates atomic EvidenceItems across sliding temporal windows (3d, 7d, 14d, 30d),
computes effort distributions, builds MotionObservations, and synthesizes MotionConclusions
with explainable rule-based confidence.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import logging
from typing import Dict, List, Optional, Tuple

from app.motion.schemas import (
    EvidenceItem,
    MotionConclusion,
    MotionObservation,
)
from app.motion.spec import (
    ConfidenceLevel,
    SLIDING_WINDOW_3D,
    SLIDING_WINDOW_7D,
    SLIDING_WINDOW_14D,
    SLIDING_WINDOW_30D,
)
from app.motion.storage import MotionStorage, motion_storage

logger = logging.getLogger("notion-assistant.motion.synthesis")


class MultiWindowSynthesisEngine:
    """Automates Observation and Conclusion generation from raw ingested evidence."""

    def __init__(self, storage: Optional[MotionStorage] = None):
        self.storage = storage or motion_storage

    def run_synthesis(
        self,
        reference_date: Optional[str] = None,
        save_records: bool = True,
    ) -> Tuple[List[MotionObservation], List[MotionConclusion]]:
        """Run multi-window synthesis across 3d, 7d, 14d, and 30d horizons."""
        if not reference_date:
            reference_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")
        start_30d = (ref_dt - timedelta(days=SLIDING_WINDOW_30D)).strftime("%Y-%m-%d")

        # Load all evidence in the last 30 days
        all_evidence = self.storage.load_evidence(limit=500, start_date=start_30d, end_date=reference_date)

        if not all_evidence:
            logger.info("No evidence found in window [%s to %s] for synthesis.", start_30d, reference_date)
            return [], []

        # 1. Bucket evidence by windows
        dt_3d = ref_dt - timedelta(days=SLIDING_WINDOW_3D)
        dt_7d = ref_dt - timedelta(days=SLIDING_WINDOW_7D)
        dt_14d = ref_dt - timedelta(days=SLIDING_WINDOW_14D)

        ev_3d = [e for e in all_evidence if datetime.strptime(e.date, "%Y-%m-%d") >= dt_3d]
        ev_7d = [e for e in all_evidence if datetime.strptime(e.date, "%Y-%m-%d") >= dt_7d]
        ev_14d = [e for e in all_evidence if datetime.strptime(e.date, "%Y-%m-%d") >= dt_14d]
        ev_30d = all_evidence

        generated_observations: List[MotionObservation] = []

        # 2. Build Observations for 7-day and 14-day windows
        for win_name, win_evidence, win_days in [
            ("7d_weekly", ev_7d, SLIDING_WINDOW_7D),
            ("14d_fortnightly", ev_14d, SLIDING_WINDOW_14D),
            ("3d_burst", ev_3d, SLIDING_WINDOW_3D),
        ]:
            if not win_evidence:
                continue

            # Group by category
            cat_groups: Dict[str, List[EvidenceItem]] = defaultdict(list)
            for ev in win_evidence:
                cat = ev.metrics.get("category", "General Execution")
                cat_groups[cat].append(ev)

            total_hours = sum(e.duration_hours or 1.0 for e in win_evidence)

            for cat, items in cat_groups.items():
                cat_hours = sum(it.duration_hours or 1.0 for it in items)
                pct = round((cat_hours / total_hours * 100.0), 1) if total_hours > 0 else 0.0
                obs_id = f"obs_{reference_date.replace('-', '')}_{win_name}_{cat.lower().replace(' ', '_')[:20]}"
                win_start = (ref_dt - timedelta(days=win_days)).strftime("%Y-%m-%d")

                summary = (
                    f"Over the last {win_days} days ({win_name}), logged {len(items)} activities in '{cat}' "
                    f"totalling {cat_hours:.1f} hours ({pct}% of total {total_hours:.1f} hours)."
                )

                obs = MotionObservation(
                    id=obs_id,
                    period_start=win_start,
                    period_end=reference_date,
                    time_window=f"Last {win_days} days ({win_name})",
                    category=cat,
                    statement=summary,
                    observation_summary=summary,
                    frequency_count=len(items),
                    frequency=len(items),
                    total_duration_hours=cat_hours,
                    evidence_ids=[it.id for it in items],
                )
                generated_observations.append(obs)
                if save_records:
                    self.storage.save_observation(obs)

        # 3. Build Conclusions from Observations
        generated_conclusions: List[MotionConclusion] = []

        # Group observations by category
        obs_by_cat: Dict[str, List[MotionObservation]] = defaultdict(list)
        for o in generated_observations:
            obs_by_cat[o.category].append(o)

        for cat, obs_list in obs_by_cat.items():
            distinct_evidence = set(ev_id for o in obs_list for ev_id in o.evidence_ids)
            total_freq = len(distinct_evidence)
            max_duration = max(o.total_duration_hours or 0.0 for o in obs_list)
            supporting_obs_ids = [o.id for o in obs_list]

            # Rule-based explainable confidence
            if total_freq >= 7:
                confidence = ConfidenceLevel.HIGH
                reasoning = f"Observed consistently across multiple temporal windows ({total_freq} data points, {max_duration:.1f}h max)."
            elif total_freq >= 3:
                confidence = ConfidenceLevel.MEDIUM
                reasoning = f"Observed recurring effort in {cat} ({total_freq} data points)."
            else:
                confidence = ConfidenceLevel.LOW
                reasoning = f"Preliminary observation with limited frequency ({total_freq} data points)."

            conc_id = f"conc_{reference_date.replace('-', '')}_{cat.lower().replace(' ', '_')[:20]}"
            conclusion_text = f"Consistent effort allocation detected in '{cat}' with strong momentum."

            conc = MotionConclusion(
                id=conc_id,
                statement=conclusion_text,
                confidence_level=confidence,
                confidence=confidence,
                rationale=reasoning,
                confidence_reasoning=reasoning,
                derived_from_observation_ids=supporting_obs_ids,
            )
            generated_conclusions.append(conc)
            if save_records:
                self.storage.save_conclusion(conc)

        logger.info(
            "Synthesis complete: generated %d observations and %d conclusions for %s.",
            len(generated_observations),
            len(generated_conclusions),
            reference_date,
        )
        return generated_observations, generated_conclusions


synthesis_engine = MultiWindowSynthesisEngine()
