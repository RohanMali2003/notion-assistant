# Ocean Motion v3 Specification (Ocean v4.2)
## Predictive Trajectory Simulation, Cognitive Sustainability, Critical Path Milestones & Socratic Mentorship

**Status:** Approved Design Specification  
**Subsystem:** Ocean Motion v3  
**Target Release:** Ocean v4.2  
**Role:** Principal Systems & AI Engineer  

---

# 1. Objective & Vision

Ocean Motion v3 completes the evolution of Motion from an observation and drift detector (v2) into a **predictive, proactive, and holistic strategic intelligence mentor**.

Motion v3 answers the highest-leverage questions for ambitious builders and researchers:
1. *"If I shift 5 hours/week from Project A to LeetCode, how does my milestone completion date and drift risk change?"* (**Predictive Trajectory Forecaster & Scenario Simulator**)
2. *"Am I sustaining high-yield execution or thrashing into cognitive fatigue and diminishing returns?"* (**Cognitive Energy & Fatigue Monitor**)
3. *"Which intermediate strategic milestones are on the critical path and blocking long-term trajectory?"* (**Milestone & Critical Path Engine**)
4. *"What unexamined assumptions and second-order failure modes exist in my current strategic direction?"* (**Socratic Reasoning & Pre-Mortem Dialogue**)
5. *"How do I receive an autonomous, executive-level weekly strategic briefing with clear trajectory calibration proposals?"* (**Autonomous Executive Briefing Generator**)

---

# 2. Design Invariants (Preserved Across All Versions)

1. **Persona Separation**: Motion is the strategic mentor within Ocean; Ocean is the execution engine. Motion never executes tasks, mutates Notion directly, or runs searches.
2. **Deterministic Provenance**: Facts $\rightarrow$ Observations $\rightarrow$ Conclusions $\rightarrow$ Recommendations $\rightarrow$ Simulated Projections. Every projection is grounded in real historical velocity.
3. **Dispatch Permission Enforcement**: All execution tools remain strictly forbidden for the Motion persona.
4. **Conversational Humility & Socratic Inversion**: Motion challenges assumptions, seeks user clarification on unknowns, and presents second-order trade-offs without assuming false certainty.
5. **Zero Memory Bloat**: Maintains bounded sliding horizons, structured models, and pruned historical summaries.

---

# 3. Motion v3 System Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         OCEAN INGRESS & ACTIVITY FEED                       │
 │  Completed Tasks  │  LeetCode Mastery  │  Learning Syllabus  │  Daily Logs  │
 └───────────────────────┬─────────────────────────────────────────────────────┘
                         │
                         ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                EVIDENCE INGESTION & MULTI-WINDOW SYNTHESIS (v2)             │
 │         Atomic EvidenceStore ➔ Observations ➔ Hypotheses ➔ Drift            │
 └───────┬──────────────────────────────┬──────────────────────────────┬───────┘
         │                              │                              │
         ▼                              ▼                              ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│  PREDICTIVE SCENARIO    │ │    COGNITIVE ENERGY     │ │  CRITICAL PATH          │
│  SIMULATOR & FORECASTER │ │    & SUSTAINABILITY     │ │  MILESTONE ENGINE       │
│ - Historical Velocity   │ │ - Burst Intensity Index │ │ - Milestone Dependency  │
│ - Trade-off Matrix      │ │ - Fatigue Risk Score    │ │   Graph (DAG)           │
│ - Timeline Shifts       │ │ - Flow vs Thrash Ratio  │ │ - Blocker Detection     │
│ - Capacity Modeling     │ │ - Sustainability Alert  │ │ - Slippage Projections  │
└────────┬────────────────┘ └───────────┬─────────────┘ └───────────┬─────────────┘
         │                              │                           │
         └──────────────────────┬───────┴───────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     EXECUTIVE BRIEFING & SOCRATIC MENTOR                    │
 │ - Autonomous Weekly Executive Strategic Briefing with Velocity Gauges       │
 │ - Socratic Mentorship: Assumption Inversion & Pre-Mortem Failure Analysis   │
 │ - Proactive Trajectory Calibration Proposals                                │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

# 4. Detailed Component Specifications

## 4.1 Predictive Trajectory Forecaster & Scenario Simulator (`app/motion/forecaster.py`)
- **Historical Velocity Metric**: Computes rolling hourly burn and completion throughput per category over 7d, 14d, and 30d.
- **Scenario Simulation**: Evaluates proposed time reallocations (e.g. `+X hours to Category A, -Y hours from Category B`):
  - Calculates updated milestone projected completion dates.
  - Predicts expected Strategic Alignment Score impact.
  - Computes trade-off severity and risk matrix.
- **What-If Analysis**: Answers complex scenario queries via `/motion scenario` and `POST /motion/simulate`.

## 4.2 Cognitive Energy & Sustainability Monitor (`app/motion/energy_monitor.py`)
- **Activity Dynamics Ingestion**: Analyzes daily distribution of logged hours, session timestamps, and consecutive high-intensity days.
- **Metrics Computed**:
  - `FatigueRiskScore` ($0–100\%$): High sustained hours ($\ge 8\text{h/day}$ for $\ge 5\text{ consecutive days}$) or late-night thrashing flags elevated fatigue.
  - `FlowVsThrashRatio`: Ratio of deep single-category focus sessions vs fragmented high-frequency context switching.
  - `BurnoutRiskLevel`: `LOW`, `ELEVATED`, `HIGH`, `CRITICAL`.
- **Proactive Intervention**: Suggests deliberate decompression buffers when fatigue exceeds sustainable thresholds.

## 4.3 Initiative Milestone & Critical Path Engine (`app/motion/milestones.py`)
- **Milestone Hierarchy**: `MotionInitiative` $\rightarrow$ `List[InitiativeMilestone]` (title, target_date, estimated_hours, dependencies, status).
- **Critical Path Analyzer**:
  - Builds milestone dependency DAG.
  - Computes the longest dependency chain determining completion dates.
  - Identifies bottleneck milestones that jeopardize downstream objectives.
  - Flags milestones with negative velocity variance (slippage risk).

## 4.4 Socratic Mentorship & Pre-Mortem Dialogue (`app/motion/socratic.py`)
- **Socratic Protocol**:
  1. *Assumption Extraction*: Identifies unproven premises in the user's plan.
  2. *Pre-Mortem Inversion*: Simulates plausible future failure scenarios (*"Imagine it is 6 months from now and this failed. What went wrong?"*).
  3. *Second-Order Consequences*: Traces ripple effects across academic, professional, and personal energy horizons.
  4. *High-Leverage Inquiries*: Challenges the user with 1–2 crisp, non-rhetorical questions to crystallize commitment.

## 4.5 Autonomous Executive Briefing Engine (`app/motion/executive_briefing.py`)
- Synthesizes all v1, v2, and v3 signals into an executive Sunday briefing:
  - Strategic Alignment Index & Drift Severity.
  - Cognitive Energy & Velocity Summary.
  - Milestone Progress & Critical Path Status.
  - Top 2 High-Leverage Strategic Recommendations.
  - Proposed Trajectory Calibrations with one-click acceptance payload.

---

# 5. Motion v3 REST & Messaging Interface

### REST Endpoints
- `POST /motion/simulate` - Run scenario simulation on proposed time reallocation
- `GET /motion/energy` - Retrieve cognitive sustainability and fatigue metrics
- `GET /motion/milestones` & `POST /motion/milestones` - Manage initiative milestones and dependencies
- `GET /motion/milestones/critical-path` - Retrieve critical path analysis and slippage risks
- `GET /motion/briefing` - Retrieve latest autonomous executive strategic briefing
- `POST /motion/socratic` - Run Socratic reasoning & pre-mortem analysis on a dilemma

### Chat / Messaging Ingress
- `motion: scenario <reallocation or dilemma>`
- `motion: energy` / `motion: fatigue`
- `motion: milestones`
- `motion: briefing`
- `motion: pre-mortem <initiative or goal>`
