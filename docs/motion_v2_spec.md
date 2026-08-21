# Ocean Motion v2 Specification (Ocean v4.1)
## Automated Evidence Pipeline, Strategic Drift Detection, and Advanced Attribution

**Status:** Approved Design Specification  
**Subsystem:** Ocean Motion v2  
**Target Release:** Ocean v4.1  
**Role:** Principal Systems & AI Engineer  

---

# 1. Objective & Scope

Motion v2 (Ocean v4.1) advances Motion from foundational structured reasoning (v1) into an **automated, proactive strategic intelligence system**.

Motion v2 delivers:
1. **Automated Evidence Ingestion Pipeline**: Asynchronous capture of real-world workspace activity (completed tasks, study logs, LeetCode practices, daily logs) into atomic evidence without manual daily triggers.
2. **Automated Observation $\rightarrow$ Conclusion Synthesis**: Multi-window aggregation and causal hypothesis synthesis with temporal recency weighting and rule-based confidence.
3. **Strategic Drift & Trend Detection Engine**: Mathematical comparison of real-world time/effort distribution against active Strategic Initiatives and Trajectory priorities, computing a Strategic Alignment Score (0–100%) and actionable drift alerts.
4. **Deep Evidence Attribution & Interactive "Why Do I Believe This?"**: Multi-hop graph traversal from Recommendations down to raw Notion sources, supporting inline citations.
5. **Proactive Decision & Override Triggers**: Automated evaluation of decision due dates and override condition triggers during consultations and review cycles.
6. **Query-Aware Semantic Strategic Retrieval**: High-signal context retrieval with recency decay and relevance ranking.

---

# 2. Design Invariants (Preserved from v1)

- **Persona Separation**: Motion remains a persona within Ocean. Strategy vs. execution separation is strictly preserved.
- **Pipeline Integrity**: Facts $\rightarrow$ Observations $\rightarrow$ Conclusions $\rightarrow$ Recommendations. No layer may skip another.
- **Dispatch Permission Enforcement**: All tools modifying Notion or executing code remain strictly forbidden for Motion.
- **Rule-Based Confidence**: Confidence levels remain strictly discrete (`LOW`, `MEDIUM`, `HIGH`), explainable, and derived from observation frequency and consistency across review cycles.
- **User Agency**: The user retains absolute authority. Motion challenges and alerts, but never overrides.

---

# 3. Motion v2 System Architecture

```
                    Ocean Execution Events
      (Tasks Completed, LeetCode, Learning Notes, Daily Logs)
                               │
                               ▼
                   [Async Evidence Ingestion]
                               │
                               ▼
                        Evidence Store
                               │
                               ▼
                 [Observation Aggregation Engine]
                  (3-day / 7-day sliding windows)
                               │
                               ▼
                 [Conclusion & Hypothesis Engine]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 [Strategic Drift Engine]              [Attribution Engine]
 (Alignment Score & Vectors)           ("Why do I believe this?")
            │                                     │
            └──────────────────┬──────────────────┘
                               │
                               ▼
                 [Motion Mentorship & Context]
              (Proactive Interventions & Advice)
```

---

# 4. Component Specifications

## 4.1 Automated Evidence Ingestion Engine (`app/motion/ingestion.py`)
- Subscribes to Ocean background task completion hooks:
  - `task_action_service.py` (task completed, in progress, postponed)
  - `leetcode_service.py` (problems reviewed, patterns identified)
  - `learning_service.py` (curriculum milestones created/read)
  - `daily_logs` / `mind` entries
- Converts each event into an immutable `EvidenceItem` with tags, category, duration, metrics, and Notion page reference.
- Saves asynchronously to `data/motion/evidence/` without blocking webhooks.

## 4.2 Automated Observation & Conclusion Engine (`app/motion/synthesis.py`)
- Automatically runs temporal analysis across sliding windows (3-day, 7-day, 14-day):
  - Calculates category distribution (e.g. 55% Systems Engineering, 25% Academics, 15% LeetCode, 5% Admin).
  - Detects velocity acceleration vs. stalled momentum.
  - Builds `MotionObservation` records with explicit supporting `evidence_ids`.
  - Synthesizes `MotionConclusion` records with confidence scoring (`LOW`, `MEDIUM`, `HIGH`) and linked `derived_from_observation_ids`.

## 4.3 Strategic Drift & Trend Detection Engine (`app/motion/drift_detector.py`)
- Compares real-world effort allocation against declared `MotionInitiative`s and `MotionTrajectory`.
- Calculates:
  - **Strategic Alignment Index (0 to 100%)**: Degree of convergence between planned strategic priorities and actual logged activity.
  - **Drift Vectors**: Identifies neglected initiatives ($0\text{ hours}$ in $\ge 7\text{ days}$) and runaway secondary tasks ($> 30\%\text{ time}$ on unplanned ad-hoc items).
  - **Momentum Trajectory Trends**: Detects improving, steady, or declining velocity over multi-week spans.

## 4.4 Advanced Evidence Attribution & Citation Formatting (`app/motion/attribution.py`)
- Deep traversal graph: `Recommendation` $\rightarrow$ `Conclusion` $\rightarrow$ `Observation` $\rightarrow$ `EvidenceItem` $\rightarrow$ `Notion URL / Date`.
- Generates human-readable citation trees.
- Supports inline citation keys (e.g., `[Evidence: 4h Ocean dev on Aug 20]`) in Motion consultation replies.

## 4.5 Enhanced Strategic Retrieval Engine (`app/motion/retrieval.py`)
- Adds temporal decay: Recent observations have higher ranking than stale ones.
- Adds semantic keyword clustering for topic-focused context injection.
- Preserves token efficiency while maximizing strategic signal.

## 4.6 Proactive Decision & Override Trigger Monitor (`app/motion/accountability.py`)
- Evaluates pending `DecisionJournalEntry` records and automatically flags due decisions for user review.
- Scans `HumanOverride` trigger conditions against recent observations to detect when review conditions are satisfied.
