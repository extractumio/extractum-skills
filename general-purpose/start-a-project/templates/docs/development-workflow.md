# Development Workflow

> Last updated: __DATE__

Detailed process rules for contributing to this repository. Referenced from [CLAUDE.md](../CLAUDE.md).

## Main Agent Role

The main agent inherits the full role and principles defined in [`.claude/agents/pragmatic.md`](../.claude/agents/pragmatic.md) and applies them to every request, plan, diff, and production issue — it reads that document as its own operating manual, not as something to delegate to. The `pragmatic` subagent remains available as an independent reviewer for fresh-context second opinions, but the main agent never uses it as an excuse to skip the thinking it should be doing itself.

## Expert Panel Before Major Changes

**BEFORE starting any new feature, global refactoring, or complex bugfix**, run a collaborative expert panel discussion to brainstorm and finalize the implementation plan. Do NOT skip this step — writing code without a reviewed plan leads to rework, architectural mismatches, and missed edge cases.

**Panel personas (project-specific — fill in with the domains that matter here):**

Every project has its own set of stakeholder perspectives. Replace the list below with the personas relevant to *this* codebase. The shape — one domain per persona, real disagreements expected — is what carries across projects.

- **[Primary domain expert]** — [e.g. Security Architect, Kernel Developer, ML Researcher, Database Engineer]
- **[Operational stakeholder]** — [e.g. Linux Administrator, SRE, DevOps Engineer]
- **[Business stakeholder]** — [e.g. CTO, Product Owner, Engineering Manager]
- **[Quality stakeholder]** — [e.g. QA Engineer, Accessibility Specialist]

**Common additional personas (add when relevant):**

- **QA Engineer** — test strategy, edge cases, regression risks, E2E scenario design
- **DevOps Engineer** — CI/CD, deploy pipeline, rollback strategy
- **Security Architect** — threat modeling, attack surface, policy implications
- **UX / Accessibility Specialist** — user-visible behavior, a11y, error surfaces
- **Performance Engineer** — hot paths, throughput, memory, tail latency

**Discussion format:**

- Each persona argues from their domain perspective — disagreements are expected and valuable
- The panel must consider trade-offs, not just pick the first idea
- The discussion must converge on a concrete, actionable plan with clear scope boundaries

**Required outputs (both files are mandatory):**

Plan and checklist files follow the structure defined in [`workflow-management.md`](workflow-management.md):

```
docs/ephemeral/plans/
├── <YYYY-MM-DD>-plan-<feature-name>.md        # single-stage plan
├── <YYYY-MM-DD>-checklist-<feature-name>.md   # matching checklist
│
└── <YYYY-MM-DD>-<feature-name>/               # grouped directory for multi-stage work
    ├── plan.md
    ├── checklist.md
    └── report.md
```

Use `docs/` only for durable reference material. Plans, checklists, reviews, proposals, QA notes, and other transient working material belong under `docs/ephemeral/`. The durable backlog inbox for tech debt and deferred fixes is [`TODO.md`](TODO.md), not ephemeral.

1. **Plan document** must include:
   - Problem statement and motivation
   - Options considered with pros/cons from each persona
   - Chosen approach with rationale
   - Affected files and components
   - Security implications (see [`security.md`](security.md))
   - Rollback strategy

2. **Implementation checklist** must include:
   - Step-by-step implementation tasks with checkboxes
   - Documentation updates keyed to the Documentation Map in `CLAUDE.md`
   - Deployment steps (see [`deploy-playbook.md`](deploy-playbook.md))
   - Real-world testing scenarios executed on the target environment (not hypothetical)
   - Verification commands with expected output
   - Each checkbox is checked off only after the step is actually completed and verified

## Workflow Orchestration

### Progress Tracking with Todos

**For any request with more than two discrete tasks, the main agent MUST create a todo list via `TaskCreate` before starting work, and update it via `TaskUpdate` as each task completes.** This is non-negotiable — it keeps progress visible to the user, prevents the main agent from losing track of steps mid-flight, and makes restarts after a context compression or long pause cheap.

Rules:

- **More than two tasks → todo list is mandatory.** One or two steps can be handled inline; three or more always get a todo list created up front.
- **Create before executing.** The todo list is the plan of record for the current session — it must exist before the first implementation step, not after.
- **Update as you go, never in batches.** Mark each task completed the moment it is done. Do not leave a trail of stale todos and fix them at the end.
- **Mirror the execution order and dependencies.** If tasks depend on each other, the todo list reflects that order so the user can see what is blocked on what.
- **Use todos *alongside* plans/tasks, not instead of them.** Todos track in-session progress. Plan files under `docs/ephemeral/plans/` and task files under `<plan>/tasks/backlog/` remain the durable record for non-trivial work per [`workflow-management.md`](workflow-management.md).
- **Subagents track their own progress with todos too.** Both [`task-executor`](../.claude/agents/task-executor.md) and [`pragmatic`](../.claude/agents/pragmatic.md) have access to `TaskCreate` / `TaskUpdate` / `TaskList` and must use them whenever their own work has more than two discrete steps, so progress stays visible and consistent across agent boundaries.

### Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, **STOP and re-plan immediately** — don't keep pushing a failing approach
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### Subagent Strategy

- Use subagents liberally to keep the main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution
- For multi-task plan execution, use **Orchestrated Execution** per
  [`workflow-management.md`](workflow-management.md#orchestrated-execution-subagent-fan-out):
  the main agent reads only `tasks/README.md`, creates a per-task
  git worktree, spawns
  [`task-executor`](../.claude/agents/task-executor.md) subagents
  (one per worktree, parallel where the DAG allows), ingests their
  YAML reports, cherry-picks each `done` commit onto the feature
  branch (first conflict serializes the pair, never resolve hunks),
  and spawns [`pragmatic`](../.claude/agents/pragmatic.md) per
  landed commit. The main agent never reads task files, diffs, test
  output, review transcripts, or conflict hunks. No subagent ever
  calls another subagent — all spawning is from the main loop.

### Pre-Execution Branch Setup

Mandatory before Phase 0. Run these in order, stop on any failure:

1. **`git status --porcelain` must be empty.** If not, stop and tell
   the user: *"Uncommitted changes in `<files>`. Commit or stash
   before running the plan."* Never auto-stash, auto-commit, or
   `checkout --`.
2. **`git fetch origin main`.** Fetch failure = hard stop with the
   exact git error.
3. **`git checkout -b <plan-dir-basename> origin/main`** where
   `<plan-dir-basename>` is e.g. `YYYY-MM-DD-<feature>`. If the
   branch already exists, ask the user (resume / rename / delete) —
   never force-reset with `-B`.
4. **Verify clean + at `origin/main`** before proceeding.

Every per-task worktree inherits this feature branch via `HEAD`; every
cherry-pick merge-back lands here, not on `main`. Merging the feature
branch into `main` is a separate explicit user step after the plan
ships.

### Orchestrated Execution Flow

Claude Code does **not** support context forking. Every spawned subagent
starts with a fresh context (CLAUDE.md + project skills + spawn prompt,
nothing else). The orchestration model below works around this by
writing cross-task facts to disk in a one-time exploration phase so
every executor can pull them in cheaply without the main agent ever
holding source code in its own window.

#### Diagram 1 — Roles and data flow

```
                               User
                                │
                                │  "execute the backlog"
                                ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                      MAIN  AGENT                             │
  │                 (persistent — owns the loop)                 │
  │                                                              │
  │   reads only:   tasks/README.md  +  structured YAML reports  │
  │                 + git exit codes + git status --porcelain    │
  │   writes only:  tasks/README.md status cells,                │
  │                 tasks/backlog/NN-<name>.context.md,          │
  │                 tasks/reports/NN-<name>.*.yaml,              │
  │                 user-facing running log                      │
  │   git actions:  worktree add, cherry-pick, cherry-pick       │
  │                 --abort, revert, worktree remove, branch -D  │
  │                 (never diff, never merge, never resolve)     │
  │   never reads:  task files, source, diffs, test logs,        │
  │                 shared_context.md, review transcripts,       │
  │                 conflict hunks                               │
  └──────────────────────────────────────────────────────────────┘
        │              │                 │                  │
        │ spawn        │ spawn × N       │ spawn            │ spawn
        │ (once)       │ (per round)     │ (per done task)  │ (on triage)
        ▼              ▼                 ▼                  ▼
  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐
  │ shared-  │  │ task-executor│  │   pragmatic   │  │  read-only   │
  │ explore  │  │  (N parallel)│  │   reviewer    │  │ triage agent │
  │ subagent │  │              │  │               │  │ (on failure) │
  │(ephemer.)│  │ (ephemeral)  │  │  (ephemeral)  │  │ (ephemeral)  │
  └─────┬────┘  └──────┬───────┘  └───────┬───────┘  └──────┬───────┘
        │              │                  │                 │
        │writes        │reads + writes    │reads            │reads
        ▼              ▼                  ▼                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                   DISK  (working tree + worktrees)           │
  │                                                              │
  │   plan.md                                                    │
  │   tasks/README.md                                            │
  │   tasks/shared_context.md  ← written in phase 0, read by all │
  │   tasks/backlog/NN-<name>.md                                 │
  │   tasks/backlog/NN-<name>.context.md    (optional per task)  │
  │   tasks/done/NN-<name>.md                                    │
  │   tasks/reports/NN-<name>.executor.yaml                      │
  │   tasks/reports/NN-<name>.review.yaml                        │
  │   source files, tests, git commits on feature branch         │
  │                                                              │
  │   .worktrees/NN-<name>/    ← ephemeral per-task checkouts    │
  │       └── (full repo copy on branch orch/NN-<name>,          │
  │            with .venv + node_modules symlinked from main,    │
  │            build artifacts kept local per worktree)          │
  │       created by the orchestrator before each executor       │
  │       spawn, removed after the task's commit is cherry-      │
  │       picked back onto the feature branch and the            │
  │       pragmatic review approves.                             │
  └──────────────────────────────────────────────────────────────┘
```

The only channels by which information flows between agents are:
(1) spawn prompts from the main agent, (2) structured YAML reports
back to the main agent, and (3) files on disk. No conversation history
is ever inherited.

#### Diagram 2 — Execution sequence over time

```
 time
  │
  │   MAIN AGENT                     SUBAGENTS                     DISK
  │   ──────────                     ─────────                     ────
  │
  │   receive "run the backlog"
  │        │
  │        ▼
  │   spawn plan-decompose  ───────► plan-decompose  ───────────►  backlog/*.md
  │                                    skill agent                   README.md
  │        ◄──────── ack ────────────────┘
  │        │
  │   ════════════════  PHASE 0  (once per plan)  ════════════════
  │        │
  │        ▼
  │   spawn exploration     ───────► shared-explore  ───────────►  shared_context.md
  │        │                             subagent
  │        ◄──────── ack ────────────────┘
  │        │
  │   ════════════════  PHASE 1  (scheduler loop)  ═══════════════
  │        │
  │        ▼
  │   read tasks/README.md  ◄──────────────────────────────────── tasks/README.md
  │        │
  │        ▼
  │   compute ready set = {T1, T2}   (DAG permits 2 in parallel)
  │        │
  │        ▼  filter out serialize:true tasks (generators, dep-mutators)
  │        │  → they schedule solo in the main checkout, not here
  │        │
  │        ▼  (optional: write per-task context packs for T1 / T2)
  │        │
  │        ▼  git worktree add -b orch/T1 .worktrees/T1 HEAD
  │        ▼  git worktree add -b orch/T2 .worktrees/T2 HEAD
  │        ▼  (symlink shared caches into each worktree)
  │        │                                                   ───► .worktrees/T1/
  │        │                                                   ───► .worktrees/T2/
  │        │
  │        ▼                        ┌── executor T1 ──┐
  │   spawn N in parallel  ────────►│  cwd=.worktrees/│ ───────►  code + commit(T1)
  │   each with worktree_path       │  T1 on orch/T1  │           on orch/T1 branch
  │                        ────────►│── executor T2 ──│ ───────►  code + commit(T2)
  │                                 │  cwd=.worktrees/│           on orch/T2 branch
  │                                 │  T2 on orch/T2  │
  │        │                        └────┬────────┬───┘
  │        ◄──────── YAML report(T1) ────┘        │
  │        ◄──────── YAML report(T2) ─────────────┘
  │        │          (each carries worktree_path, branch,
  │        │           commit sha, files_touched)
  │        ▼
  │   archive reports ──────────────────────────────────────────► reports/T1.executor.yaml
  │                                                                reports/T2.executor.yaml
  │        │
  │        ▼   merge-back phase (DAG order, not spawn order)
  │        │   git checkout <plan-feature-branch>
  │        │   git cherry-pick <sha T1>   ─── on clean apply ──►  feature branch
  │        │                                                      advances
  │        │   git cherry-pick <sha T2>   ─── on conflict ─────►  --abort, read
  │        │                                                      status --porcelain
  │        │                                                      (file list only),
  │        │                                                      mark T2 must-serialize,
  │        │                                                      re-queue next round
  │        ▼
  │        ▼                        ┌── pragmatic T1 ─┐
  │   spawn review × 2     ────────►│                 │ ───────► (read commit T1 on feature branch)
  │   (only for tasks that          │                 │
  │    made it through merge-back)  │                 │
  │                        ────────►│── pragmatic T2 ─│ ───────► (read commit T2 on feature branch)
  │        │                        └────┬────────┬───┘
  │        ◄──────── YAML verdict(T1) ───┘        │
  │        ◄──────── YAML verdict(T2) ────────────┘
  │        │
  │        ▼
  │   archive reviews  ─────────────────────────────────────────► reports/T*.review.yaml
  │        │
  │        ▼   on approve:
  │        │   git worktree remove .worktrees/T1
  │        │   git branch -D orch/T1
  │        │   (same for T2)
  │        ▼
  │   update status cells for T1, T2 ───────────────────────────► tasks/README.md
  │        │
  │        ▼
  │   next round: ready set = {T3, T4}  →  repeat worktree add → spawn → report
  │                                         → cherry-pick → review → remove → archive
  │        │
  │        ▼
  │   ... (deploy-gate tasks run solo, E2E task runs last solo) ...
  │        │
  │        ▼
  │   backlog empty
  │        │
  │        ▼
  │   round-up summary ─────────────────────────────────────────► to user
  ▼
```

#### Diagram 3 — Triage branches on failure

```
  executor returns report
         │
         ▼
   ┌───────────┐
   │  status?  │
   └─────┬─────┘
         │
    ┌────┴────┬─────────────┐
    │         │             │
    ▼         ▼             ▼
  "done"   "failed"      "blocked"
    │         │             │
    ▼         ▼             ▼
 cherry-   inspect        inspect
 pick onto report         report
 feature   fields         fields
 branch      │              │
    │        ▼              ▼
    │     decide:        fix precondition
    │     retry in       (missing dep,
    │     same work-     wrong plan range,
    │     tree / edit    missing context
    │     task /         pack, missing
    │     escalate       worktree_path)
    │        │           then respawn
    │        ▼              │
    ▼     respawn           │
 ┌──────┐ executor          │
 │cherry│ (once max)        │
 │ pick?│   │               │
 └──┬───┘   │               │
    │       │               │
 ┌──┴──┐    │               │
 │     │    │               │
 ▼     ▼    │               │
 ok   CONFLICT              │
  │     │                   │
  │     ▼                   │
  │  abort cherry-pick      │
  │  read status --porcelain│
  │  (file list only, no    │
  │   hunk content)         │
  │     │                   │
  │     ▼                   │
  │  mark task              │
  │  must-serialize,        │
  │  leave worktree,        │
  │  re-queue for next      │
  │  round on top of the    │
  │  commit that landed     │
  │  (first conflict only;  │
  │  second conflict on     │
  │  same pair → escalate)  │
  │                         │
  ▼                         │
 spawn                      │
pragmatic                   │
against                     │
commit                      │
 on feature                 │
 branch                     │
    │                       │
    ▼                       │
 ┌──────┐                   │
 │ver-  │                   │
 │dict? │                   │
 └──┬───┘                   │
    │                       │
 ┌──┴──┬────────┬──┐        │
 │     │        │  │        │
 ▼     ▼        ▼  ▼        ▼
"approve"  "approve_     "changes_
           with_         requested"
           conditions"
  │          │                │
  ▼          ▼                ▼
 mark     revert commit    revert
 done;    on branch; apply commit on
 update   small conditions branch; demote
 README;  as In scope      to failed;
 remove   bullets; respawn edit task
 worktree executor in      file; respawn
 + branch same worktree    in same
          (once max)       worktree
```

The main agent reads only structured YAML fields, git exit codes, and
`git status --porcelain`. It never reads subagent transcripts or
conflict hunks. Source artifacts (task file, plan range, commit diff)
are opened only when a report field names them as triage evidence.
Merge-back is strictly structural: clean cherry-pick → review;
conflict → serialize + re-queue. No branch reads diff content to
decide what to do next.

### Self-Improvement Loop

- After ANY correction from the user: update [`gotchas.md`](gotchas.md) with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for the relevant project

### Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between `main` and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

> **Full specification**: [`workflow-management.md`](workflow-management.md)
>
> Plans, tasks, decision flowcharts, execution protocol, and multi-stage
> workflows are all defined there. Follow it strictly.

## Documentation Placement

- Keep durable architecture, deployment, testing, and API references in `docs/`
- Keep plans, checklists, reviews, proposals, and one-off investigations in `docs/ephemeral/`
- The durable backlog ([`TODO.md`](TODO.md)) is **not** ephemeral — it persists across plans
- When moving or renaming durable docs, update `CLAUDE.md` and [`source-map.md`](source-map.md)
- When changing deployment behavior, update [`deploy-playbook.md`](deploy-playbook.md)

## Pre-Commit Workflow

**NEVER commit changes without running the project simplify/lint pass first.** This is a mandatory pre-commit step.

1. Complete the code changes
2. Run the project simplify/lint pass on all modified files — reviews for reuse, quality, and efficiency
3. Test the changed component(s) — exact commands live in [`how-to-run-tests.md`](how-to-run-tests.md)
4. Only after simplify + tests pass, create the commit

If the simplify pass introduces changes that break tests, fix the breakage before committing.

## Proactive Code Hardening

While implementing any requested feature or fix, **actively look for and fix adjacent weak, vulnerable, or incomplete code** in the files you touch. Not optional.

**What to look for:**
- **Security vulnerabilities**: injection, unvalidated input, missing auth checks, TOCTOU races, unsafe deserialization
- **Missing error handling**: unchecked errors, silent failures, panics on nil, missing context in error messages
- **Resource leaks**: unclosed files/connections, goroutine leaks, missing `defer`/`finally` cleanup
- **Logic bugs**: off-by-one, race conditions, incorrect nil/zero checks, missing locks
- **Dead code**: unused functions, unreachable branches, stale imports
- **Brittle patterns**: hardcoded values that should be configurable, magic numbers, string-typed enums

**How to handle it:**
- Fix small issues (< 10 lines) inline as part of the current change
- For larger issues, create a separate commit with a clear message
- If too large to fix now, add to [`TODO.md`](TODO.md) with file path, line number, description
- Never leave a file in worse shape than you found it

**Scope**: Only files you are already reading or modifying. Don't hunt through unrelated files — but if you open a file and see a problem, fix it.
