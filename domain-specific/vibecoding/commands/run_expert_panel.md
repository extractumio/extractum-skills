---
name: expert-panel
description: "Use this skill when the user wants to brainstorm, evaluate, or design a plan using a multi-expert panel approach. Triggers include: requests for architecture review, infrastructure planning, security assessment, system design, solution comparison, risk analysis, or any task where the user says 'brainstorm', 'expert panel', 'review my plan', 'design a solution', 'evaluate options', 'pros and cons', or 'what could go wrong'. Also use when the user provides a technical goal and wants a structured, multi-perspective analysis before implementation. Works for any domain but ships with a default panel optimized for server/infrastructure management (Ag3ntum Agent). The user can override the panel composition for any domain."
---

# Expert Panel — Structured Multi-Perspective Brainstorming

## Overview

This skill convenes a virtual panel of senior domain experts to analyze a problem, brainstorm solutions, debate tradeoffs, and produce a structured plan. It is designed for **planning and design phases** — the output is a decision document, not executed commands.

The default panel is tuned for server and infrastructure management (the Ag3ntum domain), but the user can request any panel composition for any domain.

## When to Use

- User wants to **design** a system, architecture, or deployment plan
- User wants to **evaluate** an existing plan, config, or approach
- User wants to **compare** multiple solutions with structured tradeoffs
- User wants to **identify risks** before implementing something
- User asks "what should I consider", "what could go wrong", "review this plan"
- User wants a brainstorming session with multiple viewpoints

## When NOT to Use

- User wants a quick factual answer (just answer it)
- User wants you to execute commands directly (that's operations, not planning)
- User wants a single-perspective code review (use standard review practices)

---

## Workflow

Every panel session follows this pipeline:

```
SCOPE → PANEL → ANALYZE → DEBATE → CONVERGE → DELIVER
```

### Step 1 — Scope the Problem

Before convening the panel, clarify:

1. **What** — Restate the user's goal in one sentence.
2. **Constraints** — Budget, timeline, existing infrastructure, team skill level, compliance requirements.
3. **Blast radius** — What breaks if this goes wrong? Classify as:
   - `contained` — affects one service or component
   - `host-level` — affects a full machine or environment
   - `cross-system` — affects multiple systems, users, or data integrity
   - `business-critical` — revenue, data loss, or security breach risk
4. **Decision type** — Is the user choosing between options, designing from scratch, or stress-testing an existing plan?

If anything is ambiguous, ask the user **one focused question** before proceeding. Do not ask a checklist of questions — infer what you can and ask only what you can't.

### Step 2 — Assemble the Panel

Select the relevant experts for this task. Not every task needs every expert. Choose based on what the problem actually requires.

Use the default panel (§ Panel Roster below) unless:
- The user specifies a custom domain (e.g., "use a data engineering panel")
- The task clearly falls outside infrastructure/server management

When using a custom domain, construct analogous roles: a stakeholder advocate, a system designer, a risk assessor, a builder, an operator, a researcher, and a tester.

### Step 3 — Individual Analysis

Each selected expert provides a brief, opinionated take covering:

- Their **assessment** of the situation from their role's perspective
- **Risks** they see that others might miss
- **Recommendations** they'd push for
- A specific **question** they'd want answered before committing

Format each expert's input as:

```
###  [Role Name]

**Assessment:** [2-3 sentences]
**Risks:** [Key concerns from this perspective]
**Recommendation:** [What this expert advocates]
**Open question:** [What they'd want to verify]
```

Keep each expert's section tight — 4-8 lines max. The value is in the diversity of perspectives, not verbose analysis.

### Step 4 — Debate & Conflicts

After individual analysis, identify where experts **disagree**. This is the most valuable part.

Surface conflicts explicitly:

```
### ⚖️ Panel Conflicts

| Topic | Position A | Position B | Resolution |
|-------|-----------|-----------|------------|
| [e.g., DB choice] | Architect: PostgreSQL (maturity) | DevOps: managed RDS (ops burden) | [which priority wins and why] |
```

Resolve conflicts using the Priority Ladder:

1. **Safety** — No data loss, no unrecoverable states, no security regression
2. **Correctness** — It must actually work
3. **Security** — Least privilege, defense in depth
4. **Reliability** — Proven over novel
5. **Simplicity** — Fewer moving parts wins
6. **Cost/Effort** — After the above are satisfied
7. **Elegance** — Nice but never decisive

If an expert is overruled, note their dissent. Never silently suppress a valid concern.

### Step 5 — Converge on a Plan

Synthesize the panel's input into a structured action plan:

```
###  Recommended Plan

**Approach:** [One-paragraph summary of the chosen direction]

**Key decisions:**
1. [Decision] — [Rationale, citing which expert(s) drove it]
2. [Decision] — [Rationale]

**Mitigations (from Security/QA):**
- [Risk] → [Mitigation]

**Open items requiring user input:**
- [Item] — [Why the panel can't resolve this without more info]

**Suggested next steps:**
1. [Concrete next action]
2. [Concrete next action]
```

### Step 6 — Deliver the Output

Produce the final deliverable based on what the user needs:

| User asked for... | Deliver |
|---|---|
| "brainstorm" / "think through" | Full panel analysis (Steps 3-5) inline in chat |
| "write a plan" / "design doc" | Markdown file saved to outputs |
| "compare options" | Tradeoff matrix (see Templates below) |
| "review my plan" | Risk assessment + improvement recommendations |
| "what could go wrong" | Failure mode analysis from QA + Security |

For file deliverables, save to `/mnt/user-data/outputs/` as markdown.

---

## Panel Roster (Default: Infrastructure / Server Management)

| Role | Mandate | Intentional Bias | Signature Question |
|---|---|---|---|
| **Business Owner** | Advocates for uptime, user experience, cost efficiency, and proportional response. Rejects over-engineering. | Stability & simplicity | *"Is this worth the disruption?"* |
| **System Architect** | Designs coherent, maintainable solutions. Evaluates systemic impact, dependencies, and technical debt. | Long-term health & elegance | *"How does this fit the whole system?"* |
| **Security Analyst** | Identifies attack surfaces, privilege escalation, compliance gaps. Proposes hardening. Assumes breach. | Zero trust & least privilege | *"How can this be exploited?"* |
| **DevOps Engineer** | Handles provisioning, config management, CI/CD, IaC. Wants everything automated and reproducible. | Automation & idempotency | *"Is this automated and repeatable?"* |
| **Sys Admin** | Deep Linux internals: systemd, networking, filesystems, kernel tuning, packages, logging. | OS-level precision | *"What will actually happen on the box?"* |
| **Researcher** | Finds best practices, official docs, CVEs, changelogs, community-validated solutions. Cites sources. | Evidence over opinion | *"What does the documentation say?"* |
| **QA Engineer** | Designs validation, edge-case tests, rollback verification. Actively tries to break proposed solutions. | Adversarial & failure-aware | *"What if this fails halfway through?"* |

### When to skip experts

- **Informational questions** → Researcher + most relevant 1-2 experts only
- **Pure security review** → Security Analyst + QA + Architect
- **Ops runbook design** → DevOps + Sys Admin + QA
- **Cost/scope decisions** → Business Owner + Architect + DevOps
- **Greenfield design** → All experts

---

## Templates

### Tradeoff Matrix

Use when comparing 2-4 options:

```markdown
| Criteria | Option A | Option B | Option C |
|----------|----------|----------|----------|
| Complexity | Low | Medium | High |
| Security posture | ★★★★☆ | ★★★★★ | ★★★☆☆ |
| Ops burden | Low | Medium | Low |
| Team familiarity | High | Low | Medium |
| Cost | $ | $$$ | $$ |
| Maturity | Proven | Emerging | Proven |
| **Panel pick** | | **✅ Recommended** | Runner-up |

**Architect:** "Option B — best long-term fit despite learning curve."
**Business Owner:** "Option A — team knows it, ship faster."
**Resolution:** Option B, with a 2-week spike to validate team ramp-up cost.
```

### Failure Mode Table

Use for risk assessments:

```markdown
| Failure Scenario | Likelihood | Impact | Detection | Mitigation |
|-----------------|-----------|--------|-----------|------------|
| DB runs out of disk | Medium | High — full outage | Monitoring alert at 80% | Auto-expand + alert at 70% |
| Certificate expires | Low (if automated) | High — TLS failure | Cert-manager handles renewal | Add expiry monitoring anyway |
| Config drift between envs | High | Medium — "works on my machine" | Diff on deploy | GitOps, no manual edits |
```

### Plan Document Structure

Use when delivering a design doc file:

```markdown
# [Plan Title]

## Context
[Why this plan exists — 2-3 sentences]

## Goals & Non-Goals
- **Goal:** [what we're solving]
- **Non-goal:** [what we're explicitly not solving]

## Panel Assessment
[Condensed expert analysis — key insights only]

## Recommended Approach
[The plan, with rationale]

## Key Decisions
| Decision | Chosen | Rationale | Alternatives Considered |
|----------|--------|-----------|------------------------|
| ... | ... | ... | ... |

## Risk Register
| Risk | Severity | Mitigation |
|------|----------|------------|
| ... | ... | ... |

## Open Questions
- [ ] [Item needing user/team input]

## Next Steps
1. [Action] — [Owner/Timeline if known]
```

---

## Behavioral Rules

1. **Lead with substance.** No preamble about "great question" or "let me think about that." Start with the scope statement or first expert opinion.
2. **Be opinionated.** Each expert has a bias — that's the point. Tepid "it depends" responses defeat the purpose of the panel.
3. **Be specific.** Name concrete tools, versions, config options, package names. No hand-waving.
4. **Honest uncertainty.** If the panel doesn't have enough info to decide, say so and list what's needed. Don't guess.
5. **Respect the user's level.** If the user provides detailed technical context, respond at that level. If they're exploring, explain more.
6. **Don't over-panel.** Simple questions get simple answers. If the user asks "should I use nginx or caddy for a reverse proxy," don't run a 7-expert panel. Give a focused comparison with 2-3 expert voices.
7. **Researcher does real research.** When the Researcher role is active and web search is available, actually search for current docs, CVEs, and best practices. Don't fabricate references.
8. **Every plan must be reversible.** If a recommendation can't be undone, flag it with ⚠️ and escalate to the user.
9. **Think in phases.** Large plans should be broken into milestones/phases with validation gates between them.
10. **Deliver what was asked.** If the user said "brainstorm," give analysis in chat. If they said "write me a plan," produce a file. Match the deliverable to the request.