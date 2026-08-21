# Ocean Motion
## Autonomous Implementation Specification (v1 → v3)

**Status:** Approved Design Specification  
**Role:** Principal Engineer responsible for implementing Motion inside the Ocean codebase.

---

# 1. Objective

Motion is a strategic mentorship subsystem inside Ocean.
Motion is **not** a separate application.
Motion is **not** another chatbot.
Motion is a persona operating within Ocean that specializes in long-term strategic reasoning.
Ocean remains responsible for execution.
Motion is responsible for strategy.
Your implementation must preserve this separation throughout the architecture.

---

# 2. Design Philosophy (Non-Negotiable)

These principles override implementation convenience.

## Principle 1
Motion optimizes **trajectory**, not conversations.
It exists to improve long-term decision quality.
It does **not** attempt to maximize user satisfaction or answer every question.

## Principle 2
Motion reasons from **structured strategic context**, never directly from raw conversations.

## Principle 3
Motion must distinguish:
- Facts
- Observations
- Conclusions
- Recommendations
These are different entities. They must never be merged.

## Principle 4
Observations require evidence.
Conclusions require observations.
Recommendations require conclusions.
No layer may skip another.

## Principle 5
The user always has authority.
Motion challenges.
Motion never overrides.

## Principle 6
Motion should become more useful over months.
It should never accumulate infinite memory.
It should maintain an evolving strategic model.

---

# 3. High-Level Architecture

```
                Ocean
                  │
          Persona Router
         ┌────────┴────────┐
      Ocean          Motion
         │                │
     Full Tools     Strategic Context
         │                │
         └──────┬─────────┘
                │
        Shared Ocean Services
```

Motion is a persona. It is NOT another agent.

---

# 4. Responsibilities

Motion owns:
- strategic planning
- trajectory evaluation
- accountability
- weekly reviews
- long-term priorities
- opportunity detection
- bottleneck detection

Motion never owns:
- task execution
- coding
- debugging
- research
- tutoring
- note editing
- task creation

These remain Ocean responsibilities.

---

# 5. Persona Routing

The routing layer—not the LLM—decides whether Motion is active.
Pipeline:
```
User Message
↓
Persona Router
↓
Detect @motion
↓
Load Motion Configuration
↓
Build Strategic Context
↓
Invoke LLM
```
The router must intercept before prompt construction.
The LLM should never decide which persona it is.

---

# 6. Tool Enforcement

This must NOT rely on prompt instructions.
Implement a dispatch-layer permission system.

```
Motion
Allowed:
✓ Read logs
✓ Read strategic model
✓ Read calendar
✓ Read projects
✓ Read decision journal

Forbidden:
✗ Execute tasks
✗ Modify Notion
✗ Create tasks
✗ Write code
✗ Run searches
✗ Invoke coding tools
```
Any forbidden tool invocation must be rejected before reaching the model.

---

# 7. Core Data Model

Motion owns a **Strategic Model**.
The Strategic Model is not conversation history.
It is the current understanding of the user's trajectory.

Files:
```
motion/
identity.json
trajectory.json
initiatives.json
decision_journal.json
overrides.json
weekly_reviews/
observations/
conclusions/
```

---

# 8. Identity
Contains stable information:
- education
- career goal
- long-term ambitions
- constraints
Identity changes only through explicit user edits. Never infer identity.

---

# 9. Trajectory
Trajectory is the primary object Motion reasons from:
- current phase
- current direction
- momentum
- biggest opportunity
- biggest risk
- next review
Trajectory is updated weekly.

---

# 10. Initiatives
Tracks major ongoing efforts:
- title
- status
- strategic importance
- horizon
- momentum
This is not task management. Only strategic initiatives belong here.

---

# 11. Evidence Pipeline
The pipeline must be deterministic:
```
Activity → Evidence Extractor → Observation → Conclusion → Strategic Model → Motion
```
- **Evidence Extraction**: converts raw activity into atomic facts (measurable events only).
- **Observation Builder**: summarizes repeated evidence.
- **Conclusion Builder**: interprets observations. Every conclusion must reference one or more observations.
- **Evidence Attribution**: Every observation and conclusion must contain provenance (`derived_from`). Motion must always be capable of explaining: *"Why do I believe this?"*

---

# 12. Strategic Context Retrieval
Load:
- Identity
- Current trajectory
- Active initiatives
- Open overrides
- Last two weekly reviews
- Relevant decision journal entries
- Observations related to current query

Do NOT load:
- archived initiatives
- old reviews
- completed decisions
- unrelated observations

---

# 13. Decision Journal
Every strategic recommendation creates a decision record:
- question
- alternatives considered
- recommendation
- reasoning
- expected outcome
- review trigger
- status
- actual outcome

State Machine: `Pending → Due → Reviewed → Closed`.
When a review becomes due, Motion asks the user to evaluate the outcome before closing.

---

# 14. Human Overrides
Users always have final authority:
- reason
- date
- review trigger condition (not arbitrary dates)
- status

---

# 15. Confidence
Rule-based confidence levels:
- `Low`: Single observation
- `Medium`: Observed multiple times
- `High`: Observed consistently across several review cycles

---

# 16. Weekly & Daily Reviews
- **Daily Review**: Generate observations (not advice, no trajectory update).
- **Weekly Review**: Update Strategic Model (wins, regressions, strategic drift, opportunities, recommendations).
