# Work Management: Plans, Tasks, and Workflow

> Last updated: __DATE__

Work management specification for Claude Code agents in this repository.

> **Related skills and agents** (do not duplicate their contents here —
> follow the links when in doubt):
>
> - [`.claude/skills/plan-decompose/SKILL.md`](../.claude/skills/plan-decompose/SKILL.md)
>   — how to break an approved plan into task files.
> - [`docs/how-to-write-tests.md`](how-to-write-tests.md) — the
>   test-design contract every task must honor.
> - [`.claude/agents/pragmatic.md`](../.claude/agents/pragmatic.md) — the
>   sparring-partner agent that must review non-trivial plans and task
>   sets before execution.

---

## Scope

This document owns:

- when plans, tasks, and optional checklists are required
- where plan/task artifacts live
- how plan-backed work moves from backlog to done

This document does **not** redefine:

- repo-wide engineering principles in [`../CLAUDE.md`](../CLAUDE.md)
- expert-panel or pre-commit flow in
  [`development-workflow.md`](development-workflow.md)
- test-design rules in
  [`how-to-write-tests.md`](how-to-write-tests.md)
- pragmatic reviewer behavior in
  [`../.claude/agents/pragmatic.md`](../.claude/agents/pragmatic.md)

---

## Overview

**IMPORTANT: This section defines how plan-backed work is planned,
tracked, and executed. Follow these rules strictly.**

### Two Artifacts

- **Plan** = design document (HOW to do something). Lives in
  `docs/ephemeral/plans/<plan-dir>/plan.md`. Contains step-by-step
  instructions, architecture decisions, nuances, tests, and doc updates.
- **Task** = one executable unit derived from a plan (WHAT to do next, with
  an explicit scope, test list, docs list, and final checklist). Tasks live
  **inside the plan directory** at
  `docs/ephemeral/plans/<plan-dir>/tasks/backlog/` and graduate to
  `.../tasks/done/` when shipped.

A plan says HOW. A task is one slice of the plan a fresh agent can pick up
and finish without re-reading the whole plan.

---

## Decision Flowchart

When the user requests work, size it by **context footprint**, not
wall-clock time. The question is always: can the full round-trip
(request → implement → test → verify → commit) fit inside a single
~150K-token context window with ≥ 20% headroom? See the
[`plan-decompose`](../.claude/skills/plan-decompose/SKILL.md) skill for
the formal sizing heuristic.

```
User requests work
│
├─ Trivial (≤ ~10 LOC, 1 file, no new tests, no design choices)?
│  YES → Just do it. No plan, no task.
│
├─ Non-trivial but the full round-trip fits in one context window
│  (cited sources + edited files + tests + verify output + headroom)?
│  │
│  ├─ Approach is obvious (no architectural or cross-cutting choices)?
│  │  YES → Just do it. No plan, no task.
│  │
│  └─ Needs design (multi-file, architectural, risky, affects contracts)?
│     YES → Create a plan. Execute it now. No task breakdown needed —
│          the plan + diff already fit in one window.
│
├─ Non-trivial AND the round-trip will NOT fit in one context window?
│  │  (too many files, too many tests, multi-phase, deploy gates)
│  │
│  ├─ Needs design?
│  │  YES → Create the plan, then run the plan-decompose skill to slice
│  │       it into context-window-sized tasks under
│  │       <plan-dir>/tasks/backlog/.
│  │
│  └─ Approach already clear?
│     YES → Create a minimal plan stub and let plan-decompose produce
│          one or more context-window-sized tasks.
│
└─ Did side-work surface while doing something else?
   YES → Capture it as a task inside the active plan's tasks/backlog/.
         If no active plan exists, create a stub plan first.
```

### Summary Table

| Scenario | Plan? | Tasks? |
|---|:---:|:---:|
| Trivial fix (≤ ~10 LOC, 1 file) | no | no |
| Non-trivial, fits one context window | **yes** | no |
| Non-trivial, exceeds one context window | **yes** | **yes** (via `plan-decompose`) |
| Deferred (future context window), regardless of size | **yes** (stub OK) | **yes** |
| Side-work discovered mid-flow | **yes** (stub or active plan) | **yes** (always captured) |
| Multi-phase plan | **yes** | **yes** (one task per phase + deploy gates) |

Every task is inside a plan directory. There are no top-level "orphan"
tasks — this keeps scope, context, and verification co-located.

---

## Plans

### What Goes in a Plan

A plan is a highly detailed implementation guide. It must include:

- Full context of the task (why are we doing this?)
- Step-by-step execution instructions (what to change, in what order)
- Implementation specifics and nuances (edge cases, gotchas)
- Files that will be touched
- What NOT to do (explicit out of scope)
- Test plan with **named** tests (not "add unit tests")
- Documentation updates keyed to the files that must change

Non-trivial plans must be reviewed with the
[`pragmatic`](../.claude/agents/pragmatic.md) agent before decomposition.
This document only defines **when** that review is required; the review
method and output format live in the agent file.

### Plan File Structure

Every non-trivial plan lives in its own dated directory. Flat single-file
plans are only acceptable for very small one-off work.

```
docs/ephemeral/plans/
├── <YYYY-MM-DD>-<plan-name>/
│   ├── plan.md                    # the plan document (PRD + tech spec)
│   ├── checklist.md               # optional verification checklist
│   ├── report.md                  # optional post-implementation report
│   └── tasks/
│       ├── README.md              # task list, DAG, workflow
│       ├── backlog/               # pending tasks
│       │   ├── 01-<short-name>.md
│       │   ├── 02-<short-name>.md
│       │   └── ...
│       └── done/                  # shipped tasks (moved from backlog)
│           └── ...
└── <YYYY-MM-DD>-<other-plan>/
    └── ...
```

Durable reference documentation does not belong here; it lives in `docs/`.
The durable backlog inbox for tech debt, refactor candidates, and deferred
fixes is [`docs/TODO.md`](TODO.md), not `docs/ephemeral/`.

### Committing plan artifacts

Every file under `docs/ephemeral/plans/<plan-dir>/` — `plan.md` (even
`Status: DRAFT`), `checklist.md`, `tasks/**`, `report.md`, archived
evidence like verification-result files — commits to the current
feature branch **the moment it is created**, never batched to
end-of-session. Working-copy state evaporates on the next `git clean`,
machine change, or stray `git checkout .`; plans are cheap to commit
and expensive to lose.

Unrelated drafts still go on the current branch — a thematically mixed
`docs/ephemeral/` commit that the reviewer ignores beats the draft
disappearing because you planned to commit it on some other branch
later.

**End-of-session audit:** `git status --short docs/ephemeral/plans/`
must return nothing.

### Checklists

Plans may include a `checklist.md` as a separate verification artifact when
the plan is complex enough that the per-task checklists do not cover every
cross-cutting concern (e.g. integration across multiple phases).

A checklist covers:
- A summary of what was implemented
- Detailed verification steps (file-by-file, behavior-by-behavior)
- Commands to run to confirm correctness

For most plans the per-task `## Checklist` section at the bottom of each
task file is sufficient — a separate `checklist.md` is optional.

### Multi-Phase Plans

Large plans are structured as phases or changes inside a single `plan.md`.
Each phase becomes one or more tasks when decomposed. Deploy + verify gates
between phases become **their own tasks** with no code — just integration
checks. This is enforced by the `plan-decompose` skill.

---

## Tasks

**How to create task files:** always run the
[`plan-decompose`](../.claude/skills/plan-decompose/SKILL.md) skill. It
owns the task file structure, naming rules, the context-window size
heuristic, and the template. Do not hand-author task files ad hoc and do
not restate the template here — the skill is the single source of truth.

### Project-specific rules

These are the rules layered on top of the generic skill for this
repository. They exist because this codebase has specific testing,
deployment, and review contracts:

- **Tests are named, not described.** Every task's Tests section must
  name exact test function identifiers and specify the tier runner
  per [`how-to-write-tests.md`](how-to-write-tests.md) and
  [`how-to-run-tests.md`](how-to-run-tests.md). Coverage standards live
  in the test docs; this file only requires tasks to point to concrete
  tests and runners.
- **Deploy + verify is its own task.** Never bundle "implement X" and
  "deploy + verify X" into one task. Deploy gates between phases use the
  repo verification flow from
  [`development-workflow.md`](development-workflow.md) and
  [`how-to-run-tests.md`](how-to-run-tests.md).
- **Doc rows are explicit.** Every task's Documentation updates table
  names specific files from the Documentation Map in `CLAUDE.md` (at
  minimum: `source-map.md` for new files, `architecture.md` for new
  behavior, the project-specific reference doc for new rules/fields,
  `gotchas.md` for new foot-guns, `CLAUDE.md` for new build targets).
- **Pragmatic review is non-optional** for non-trivial task sets and for
  any task's final diff. Run
  [`../.claude/agents/pragmatic.md`](../.claude/agents/pragmatic.md) and
  apply its findings before commit; do not restate its review framework
  inside task files.
- **Dependencies stay minimal.** `<plan-dir>/tasks/README.md` owns the
  DAG; tasks cite dependencies as plain `backlog/NN-name.md` or
  `done/NN-name.md` paths. Over-serialized graphs block parallel work —
  config-only or doc-only tasks almost never depend on anything.

### When to create a task

**Always create a task when:**

- A plan has just been finalized and its round-trip exceeds one
  ~150K-token context window (run the `plan-decompose` skill to slice
  it).
- Side-work is discovered mid-flow — capture it in the active plan's
  `tasks/backlog/`, continue main work.
- A multi-phase plan needs tracking — one task per phase plus one per
  deploy / verify gate.

**Do NOT create a task when:**

- The full round-trip (implement + test + verify + commit) fits in one
  context window and you are executing it now.
- It is a sub-step of an already-tracked task (add it to that task's
  checklist, do not spawn a new file).
- It is trivial (≤ ~10 LOC, 1 file, no new tests) — just do it inline.

---

## Execution Protocol

### Executing a Single Task

1. Pick the next task from `<plan-dir>/tasks/backlog/` respecting the
   dependency DAG in `<plan-dir>/tasks/README.md`.
2. Read the task file end to end. The task is self-contained — you should
   only need the plan's cited line range on top of it.
3. Implement the in-scope changes exactly as described.
4. Run the repo execution flow for this task scope: the project
   simplify/lint pass, test work per
   [`how-to-write-tests.md`](how-to-write-tests.md), verification
   evidence, deploy gates if applicable, and durable doc updates in the
   same commit. The detailed rules live in
   [`development-workflow.md`](development-workflow.md) and
   [`how-to-run-tests.md`](how-to-run-tests.md).
5. For any non-trivial change, run the
   [`pragmatic`](../.claude/agents/pragmatic.md) agent against the final
   diff and address its findings.
6. Move the file: `mv backlog/NN-name.md done/NN-name.md`.
7. Update the status column in `<plan-dir>/tasks/README.md`.
8. Commit.

### Executing a Backlog

When the user asks to execute the backlog of a plan:

1. **Read** `<plan-dir>/tasks/README.md` and the files in `backlog/`.
2. **Sort** by the dependency DAG, then by priority within each level.
3. **Present** the ordered list to the user with task titles and estimated
   size, before starting.
4. **Execute** each task using the single-task protocol above.
5. **Report** completion of each task before moving to the next. Compact
   context between tasks if needed.

### Dependency Resolution

- A task cannot start until **all** its dependencies are in `done/`.
- If a dependency is still in `backlog/`, execute it first (recursively).
- If a dependency seems half-done (partial implementation, stale PR), stop
  and ask the user rather than restarting.
- Circular dependencies are a bug — flag them and stop.

### Stage Execution for Multi-Task Runs

For each task in a multi-task execution run, execute this sequence
**without pausing between sub-steps** unless you hit a real blocker:

1. Read the task file
2. Implement in-scope changes
3. Run the repo execution flow from
   [`development-workflow.md`](development-workflow.md) for that task
4. Move task to `done/` and update README
5. Compact context if approaching limits
6. Move to the next task

---

## Orchestrated Execution (Subagent Fan-Out)

Multi-task plans produced by `plan-decompose` can be too large to ship
inline without bloating the main agent's context with per-task source
reads, test output, and review transcripts. **Orchestrated Execution**
is the repo's standard way to ship those plans: the main Claude Code
agent stays as a thin scheduler over `tasks/README.md`, and every task
is implemented by a fresh
[`task-executor`](../.claude/agents/task-executor.md) subagent that
starts with a clean context and returns a terse structured report.
The main agent then spawns a separate
[`pragmatic`](../.claude/agents/pragmatic.md) subagent to review the
resulting diff, also with a fresh context. All orchestration happens
from the main agent's loop. No subagent ever calls another subagent.

### What Claude Code does and does not give you

Be honest about the platform limits before designing around them:

- **No context forking.** Spawning a subagent never inherits the parent
  agent's conversation history, previously-read files, or tool-use
  state. Each subagent starts fresh with `CLAUDE.md`, MCP servers,
  project skills, and the spawn prompt — and that is all. The
  `context: fork` option on skills means "run this skill in an
  isolated subagent", not "fork parent state into a child".
- **No state sharing between subagents.** Spawning two task-executors
  in parallel gives you two independent fresh contexts. They share
  nothing except what lives on disk.
- **No resume of a prior subagent with new input.** Each invocation is
  a new lifespan.
- **Filesystem isolation is opt-in.** Parallel executors default to
  sharing one checkout, so in-flight edits leak between commits. The
  fix — **per-executor git worktrees** plus cherry-pick merge-back
  (the protocol below) — is non-optional for any parallel round.

The practical consequence: facts the main agent already knows cannot
be transferred to a child. They must be written to disk under
`<plan-dir>/tasks/` — every worktree inherits that path from the base
commit — and read by the executor there. The child still burns tokens
on the file, but only once and only the relevant slice, far cheaper
than re-crawling the repo from scratch. The orchestration model below
is designed around this reality.

### When to use it

Use Orchestrated Execution when **any** of the following is true:

- The plan has more than ~3 tasks that each touch non-trivial code.
- Any single task is expected to run close to its context budget on
  its own (full file reads + tests + deploy output).
- Tasks can be legitimately parallelized per the DAG in
  `tasks/README.md`.
- The user explicitly asks for "run the backlog", "execute the plan
  end to end", or similar.

Do **not** use it for:

- A single task you can finish comfortably inside the current context —
  inline execution is cheaper.
- Plans that are still in flux; stabilize the plan and task set first,
  then orchestrate.
- Tasks that genuinely need to negotiate with each other mid-flight
  (see the Agent Teams escalation below).

**Tasks that must be serialized** — run solo in the main checkout,
never in a parallel worktree:

- **Generator runs** — protobuf stub regeneration, migration revision
  generation, and similar. Two parallel worktrees each pick the same
  "next" revision/stub and the second commit lands with a silently
  wrong parent — no loud merge conflict to catch it. Worktree
  isolation makes this *harder* to detect, not easier.
- **Dependency-manifest edits** — `go.mod`, `go.sum`,
  `requirements.txt`, `package.json`, `package-lock.json`,
  `uv.lock`, `poetry.lock`, `yarn.lock`. The per-worktree shared
  caches (see Worktree Lifecycle below) cannot be mutated from two
  executors at once.

`plan-decompose` tags these with `serialize: true`; the orchestrator
filters them out of parallel rounds and schedules them solo.

### Roles

- **Main agent (orchestrator).** The top-level Claude Code agent
  serving the user. Owns the DAG, the ready set, spawning executors,
  spawning reviewers, ingesting reports, updating `tasks/README.md`,
  and deciding retry vs. escalate on failure. Never reads individual
  task files, source files, diffs, or test output itself unless
  triaging a failure.
- **Shared exploration subagent (one-time).** A one-off `general-purpose`
  or `Explore` subagent the main agent spawns once, before any tasks
  run, to crawl the repo and write a `shared_context.md` file to disk.
  This file captures cross-task facts that every executor benefits
  from: repo tour, key interfaces, relevant source file excerpts,
  environment facts, existing conventions, and gotchas. The main agent
  never ingests this file into its own window — it only tells each
  executor where to find it.
- **Task executor subagent.** A `task-executor`, one per task
  invocation. Owns the entire execution lifecycle for exactly one
  task: read, implement, self-review, test, deploy-verify if required,
  update docs, commit, move file, return YAML report. Never calls
  another subagent. Never runs review agents. Never decomposes. Its
  full transcript is discarded — only the structured report reaches
  the main agent.
- **Pragmatic reviewer subagent.** A `pragmatic`, one per completed
  task (or per batch of related tasks, at the main agent's
  discretion). The main agent spawns it against the diff of a task
  that returned `status: done` from its executor. It returns its own
  structured review. The main agent decides whether to accept, revise,
  or respawn the executor based on its verdict.
- **Context artifacts on disk.** Three file types, all under
  `<plan-dir>/tasks/`:
    - `shared_context.md` — written once by the exploration subagent,
      read by every executor
    - `backlog/NN-<name>.context.md` — optional per-task context pack
      written by the main agent when a task needs more starter context
      than the task file alone provides
    - `reports/NN-<name>.executor.yaml` and
      `reports/NN-<name>.review.yaml` — the terse structured reports
      the main agent archives for traceability
- **Worktrees.** Ephemeral per-task git worktrees under
  `.worktrees/NN-<name>/` at the repo root. Each one is a full checkout
  of the base commit plus a dedicated branch (`orch/NN-<name>`), with
  shared caches symlinked in (see Worktree Lifecycle below). Created by
  the orchestrator before spawning an executor; removed by the
  orchestrator after the resulting commit is cherry-picked back to the
  feature branch. The executor's sole working directory is the worktree
  path passed in its invocation prompt.

### Main agent loop

**Prerequisite:** before phase 0, the main agent must complete the
**Pre-Execution Branch Setup** from
[development-workflow.md](development-workflow.md#pre-execution-branch-setup)
— stop on uncommitted changes, `git fetch origin main`, and
`git checkout -b <plan-dir-basename> origin/main`. Phase 0 below
runs on the resulting feature branch, every worktree inherits it,
and every cherry-pick merge-back targets it (never `main`).

```
phase 0 (once per plan):
  a. Spawn shared-exploration subagent.
     Prompt: "Explore <plan-dir>/plan.md and the repo surfaces it
     touches. Write <plan-dir>/tasks/shared_context.md containing the
     repo tour, key interfaces, relevant source excerpts with file +
     line ranges, environment facts, and gotchas every task in this
     plan will need. Keep it focused — no narrative, no plan summary."
  b. Wait for the exploration subagent to return. It returns a short
     acknowledgement; the main agent does NOT read shared_context.md
     itself.

phase 1 (scheduler loop), repeat until backlog is empty or a fatal
failure:

  1. Read tasks/README.md only.

  2. Compute the ready set:
       ready = { t ∈ backlog
                 | all t.dependencies are in done/
                 and t.status = backlog }
     If ready is empty but backlog is not, stop — circular or stuck
     DAG. Report to user.

  3. Filter the ready set for serialization rules (see "When to use it"
     above and the generator/dep-file list). A task tagged as a
     generator or dependency-mutating task is pulled out of the ready
     set and scheduled solo for the next round; it runs alone in the
     main checkout, not a worktree. All other tasks are eligible for
     parallel worktree execution.

  4. Pick up to N ready tasks. N is whatever the DAG permits, capped
     at ~4 to keep parallel tool-call output digestible.

  5. For each picked task, optionally write or refresh
       <plan-dir>/tasks/backlog/NN-<name>.context.md
     with task-specific starter context beyond what shared_context.md
     already covers: source excerpts unique to this task, cited rule
     IDs, and verbatim report blobs from upstream dependencies this
     task materially relies on. A context pack is a shortcut, not a
     dump. Many tasks need none.

  6. For each picked task, create an ephemeral worktree:
       git worktree add -b orch/NN-<name> .worktrees/NN-<name> HEAD
     then wire the shared caches in (Worktree Lifecycle below).
     Record the worktree_path and branch name — they go into the
     executor's invocation prompt.

  7. Spawn N task-executor subagents in a single message (parallel
     tool calls). Each invocation prompt must contain:
        task_file          — absolute path to backlog/NN-<name>.md
        plan_file          — absolute path to plan.md
        worktree_path      — absolute path to .worktrees/NN-<name>
        worktree_branch    — orch/NN-<name>
        shared_context     — absolute path to tasks/shared_context.md
        context_pack       — absolute path (if one was written)
        depends_on_reports — verbatim YAML blobs from upstream tasks
                             this one materially depends on
        orchestrator       — short label identifying this main agent
        notes              — short free-form constraints

  8. Wait for all spawned executors to return. Ingest only the YAML
     report blocks. Archive each one to
       <plan-dir>/tasks/reports/NN-<name>.executor.yaml
     without reading source diffs or test output from the executor's
     transcript. The report must contain worktree_path,
     worktree_branch, commit, and files_touched — these are the
     inputs to the merge-back phase.

  9. Merge-back phase. For each executor report with
     executor.status = done, in dependency-DAG order (not spawn
     order):

        git checkout <plan-feature-branch>
        git cherry-pick <report.commit>

     `<plan-feature-branch>` is the feature branch from
     Pre-Execution Branch Setup
     (see development-workflow.md#pre-execution-branch-setup),
     named after the plan directory. Never cherry-pick onto `main`
     directly — `main` is advanced only by an explicit user merge
     of the feature branch after the whole plan ships.

     Clean cherry-pick → continue to review (step 10).

     Conflict:
       → `git cherry-pick --abort`. Do NOT read hunks. Read only
         `git status --porcelain=v1` to get the file list.
       → Evidence of a file-level collision with a task that
         already landed this round. Mark the failing task
         `must-serialize`, leave its worktree in place, and
         re-queue it for the next round on top of the advanced
         base. Do not retry in-round, do not resolve, do not
         inspect hunks.
       → Next round the re-cherry-pick will either apply cleanly
         (base has advanced) or fail again — a second conflict on
         the same pair means the task really did edit the same
         lines as a dependency it was supposed to build on. That
         is a task design error; escalate.

  10. For each task that made it through merge-back cleanly:

       executor.status = done (post-merge-back)
         → Spawn a pragmatic subagent against the landed commit.
           Prompt names the commit sha, task file path,
           shared_context path, and Out of scope list. Wait for
           the verdict, archive it to
           <plan-dir>/tasks/reports/NN-<name>.review.yaml.

           verdict = approve
             → Task is done. Update tasks/README.md, log the sha,
               and remove the worktree:
                 git worktree remove .worktrees/NN-<name>
                 git branch -D orch/NN-<name>

           verdict = approve_with_conditions
             → If conditions are small and in scope: revert the
               commit on the feature branch, keep the worktree,
               respawn the executor with the conditions inlined as
               In scope bullets in a refined context pack, and
               re-run the full cycle. If conditions expand scope,
               escalate.

           verdict = changes_requested
             → Revert the commit on the feature branch. Demote
               the executor result to failed. Keep the task in
               backlog. Decide: respawn once in the same worktree
               with a refined context pack, edit the task file,
               or escalate.

       executor.status = failed
         → No cherry-pick, no pragmatic. Read only the report
           fields (blockers, failure_evidence, diff_summary).
           Decide: retry once in the same worktree with a refined
           context pack, edit the task, or escalate. Remove the
           worktree only when the task is finally done or
           permanently abandoned.

       executor.status = blocked
         → No cherry-pick, no pragmatic. Fix the missing
           precondition (bad plan range, missing dependency
           report, missing context pack, missing worktree_path)
           and respawn in the same worktree. Never retry blindly.

  11. Loop.
```

The main agent never runs simplify, never runs tests, never reads
per-task diffs, and never calls pragmatic against the whole batch —
only per completed task. Its own writes are limited to
`tasks/README.md` status cells, per-task context packs, archived
reports, and the user-facing running log. Its only direct git actions
are `worktree add/remove`, `cherry-pick` (+ `--abort`), `revert`, and
`branch -D` on `orch/NN-<name>` — and it reads nothing from those
commands except exit codes and the porcelain file list.

### Worktree Lifecycle

Filesystem isolation between parallel executors is provided by
ephemeral git worktrees, one per task invocation. The orchestrator is
the only agent that creates or removes them.

**Creation (before every executor spawn):**

```
git worktree add -b orch/NN-<name> .worktrees/NN-<name> HEAD
# wire shared caches that are expensive to rebuild but read-mostly,
# project-specific — add the ones your repo has:
# ln -s ../../.venv                    .worktrees/NN-<name>/.venv
# ln -s ../../node_modules             .worktrees/NN-<name>/node_modules
# ln -s ../../<subproject>/node_modules .worktrees/NN-<name>/<subproject>/node_modules
```

| Cache | Sharing | Why |
|---|---|---|
| `GOCACHE` / `GOMODCACHE` | user-level already | content-addressed, parallel-safe |
| `.venv/` | symlinked | avoids per-worktree virtualenv bootstrap; `.pyc` writes are lazy and idempotent |
| `node_modules/` (including subproject copies) | symlinked | fresh `npm install` per worktree would destroy parallelism; node imports are read-only |
| Build artifacts (`.next/`, `__pycache__/`, `.pytest_cache/`, `target/`, etc.) | **per-worktree** | write-heavy; shared use corrupts them |

Sharing is safe only because executors are contractually forbidden
from running installers or dependency mutators (see task-executor.md
hard rules and the serialization list above). Tasks needing those run
serially in the main checkout.

**Removal (after successful cherry-pick + pragmatic approve):**

```
git worktree remove .worktrees/NN-<name>
git branch -D orch/NN-<name>
```

`git worktree remove` deletes the worktree directory including the
symlinks (but not their targets), along with every per-worktree build
artifact underneath it. User-level caches persist — they are designed
to be long-lived and self-pruning.

Removal is **mandatory** — not "when convenient." Worktree cleanup is
step 0 of the main agent's end-of-plan checklist, run before
archiving reports and before reporting the plan as done. An orphaned
worktree is a silent correctness hazard: the next run of this plan
cannot reuse the `orch/NN-<name>` branch name, and `git status`
noise hides real untracked files.

**Two creation paths, one cleanup rule.** Ephemeral worktrees can
arrive via two different mechanisms, and *both* must be cleaned up by
the orchestrator:

1. **Explicit `git worktree add` flow** (documented above). Path:
   `.worktrees/NN-<name>/`. Branch: `orch/NN-<name>`. This is the
   canonical orchestrator path when the orchestrator spawns executors
   as plain subagents and manages git itself.
2. **Agent-tool `isolation: "worktree"` flow.** When the orchestrator
   spawns a subagent via the `Agent` tool with
   `isolation: "worktree"`, the harness creates the worktree itself.
   Path: `.claude/worktrees/agent-<hash>/`. Branch:
   `worktree-agent-<hash>`. The harness auto-cleans only if the
   agent makes zero changes; if the agent commits anything, the path
   and branch are returned in the agent's result and the
   orchestrator owns the cleanup — exactly the same as flow (1).
   Missing this is how `.claude/worktrees/` accumulates dead
   checkouts across sessions. `.claude/worktrees/` is gitignored as
   a belt-and-braces so a missed cleanup does not dirty `git status`,
   but the gitignore does **not** replace the removal step — the
   disk, the worktree registry, and the orphan branches still need
   to go.

End-of-plan audit (run before declaring the plan done):

```
git worktree list             # should only show the main checkout
                              # and any user-owned long-lived trees
ls .claude/worktrees/ 2>/dev/null   # should be empty or absent
git branch | grep -E 'orch/|worktree-agent-'  # should return nothing
```

Any leftover from either flow gets the same treatment:

```
git worktree remove <path>
git branch -D <branch>
```

**Abandoned worktrees:** If a task is escalated to the user without
resolution, leave the worktree in place so a human can inspect it. The
orchestrator logs the worktree path in its user-facing summary. Never
`git worktree remove --force` without explicit user instruction — that
is a destructive action that discards in-flight debugging state.

### Context hygiene rules

These rules exist to keep the main agent's context bounded regardless
of how many task-executors and pragmatic reviewers run over the plan's
lifetime:

1. **Never read a task file in the main agent.** If triage requires the
   contents, spawn a read-only subagent to summarize it and return
   only the summary.
2. **Never read `shared_context.md` in the main agent.** It exists for
   the executors, not the main agent. Even during triage, read only
   the slice the failing report explicitly cites.
3. **Never read per-task diffs, test output, or pragmatic transcripts
   as raw artifacts.** The structured YAML reports are the only
   evidence channel the main agent ingests. Real artifacts live on
   disk and are inspected by follow-up subagents when a human asks
   for detail.
4. **Pass information through files, not conversation.** Context
   packs, shared context, and archived reports are files in the
   working directory — every subagent reads them on its own without
   the main agent relaying bytes.
5. **Ingest reports as-is.** Do not rewrite, expand, or narrate them
   back to the user until the whole round is done.
6. **Cap per-round output.** At the end of each round, summarize the
   round in ≤ 5 bullet points for the user and drop the raw reports
   from the main agent's scratchpad — they are archived on disk
   already.
7. **Compact between rounds.** If the main agent approaches its own
   context budget, compact before spawning the next round.
8. **Never read conflict hunks.** Cherry-pick failures give you an
   exit code and a `git status --porcelain` file list — that is the
   entire evidence channel. No `git diff`, no opening conflicted
   files, no subagent asked to "summarize the conflict." Collisions
   are handled structurally (serialize + re-queue), never
   semantically.

### Failure handling

- **One retry max per task.** If a task fails twice, escalate to the
  user with both structured reports and the task file path. Do not
  third-try blindly.
- **First cherry-pick conflict = immediate serialization, not a
  retry.** A conflict is discovered evidence that two worktrees
  touched the same file. Land the earlier-DAG task; re-queue the
  later one for the next round on top of the advanced base. Never
  resolve in place, never inspect hunks. A second conflict on the
  same pair after serialization means the task really did edit the
  same lines as a dependency it was supposed to build on — task
  design error, escalate.
- **No silent task edits.** If the main agent needs to change a task's
  scope, it must edit the task file explicitly, note the change in the
  running log, and mention it in the next user-facing summary.
- **Never mark a failed task done.** The executor's `status` field is
  authoritative for execution; the pragmatic reviewer's `verdict` is
  authoritative for review. A report the main agent cannot parse is a
  `blocked`.
- **Circular DAG.** Stop immediately, print the offending cycle, ask
  the user.
- **Reviewer overrides.** If pragmatic demands a change but the task
  file clearly forbids it (explicit Out of scope), escalate to the
  user — do not silently resolve the contradiction either way.
- **Abandoned worktrees stay on disk.** Never `git worktree remove` a
  worktree for a task that escalated without resolution. A human
  inspecting the failure needs the exact state the executor left
  behind. Log the worktree path in the user-facing summary instead.

### Escalation to Agent Teams (optional)

Agent Teams (Claude Code's experimental multi-session coordination)
give you native DAG scheduling with inter-teammate messaging and a
shared task list. They are the right tool **only** when tasks
legitimately need to negotiate with each other mid-flight — for
example, when two parallel tasks both propose changes to the same
public contract and need to converge before either ships.

Prefer Orchestrated Execution with `task-executor` subagents as the
default. Promote to Agent Teams only when:

- Multiple tasks share a contract and cannot be serialized without
  starving the graph, **and**
- The user has explicitly opted in to the experimental team flow,
  **and**
- The executors would otherwise need to negotiate through the main
  agent, creating a serial bottleneck.

Agent Teams have real costs: they are feature-flag gated, sessions do
not `/resume` cleanly, and cross-team messaging can pollute the lead's
context in ways that defeat the isolation benefits. If you reach for
them, document the reason in the plan's `report.md` so future work
can revisit the call.

---

## Backlog Rules

[`../TODO.md`](TODO.md) is the **durable** backlog inbox. It lives at
`docs/TODO.md` (not under `ephemeral/`) because tech debt and deferred
fixes outlive any single plan.

Categories tracked there:

- **Tech debt** — long-lived shortcuts that should be paid back; each entry names the cost.
- **Refactor candidates** — code that works but is hard to read, change, or test; each entry names the trigger that would justify the refactor.
- **Deferred fixes** — bugs observed but postponed; each entry names the workaround and the unblocking condition.
- **Follow-ups** — TODO comments, doc nits, tightenings that fell out of scope.

Rules:

- **Add the moment you discover it.** Do not trust memory.
- **One line, dated, with origin** (`PR # / issue # / file:line`).
- **No expansion in place** — discussion or design belongs in a plan, not in TODO.md.
- **Promote, don't expand** — when an item is ready to be worked on, move it to a new plan under `ephemeral/plans/<YYYY-MM-DD>-<slug>/` and remove it from TODO.md.
- **Prune regularly** — items that have aged out without being championed get archived or deleted.

---

## End-to-End Workflow Examples

### Example 1: Complex Feature (Multi-Phase, Deferred)

User asks: "Deliver feature X across all phases."

1. **Agent creates the plan** with the expert panel, lands it at
   `docs/ephemeral/plans/<YYYY-MM-DD>-<feature>/plan.md`.
   Runs the `pragmatic` agent against the plan. Applies feedback.
2. **Agent runs the `plan-decompose` skill** against the plan. The skill
   produces `tasks/backlog/01-…` through `tasks/backlog/NN-…`, a populated
   `tasks/README.md`, and an empty `tasks/done/`.
3. **Agent runs `pragmatic` against the task set** as the skill's final
   step (the skill already does this). Applies any revisions.
4. **Subsequent context windows execute one task at a time** using the
   single-task protocol above. Each task's per-file line references and
   test list keep every window self-contained.

### Example 2: Bug Found During Other Work

Agent is working on Phase 3 and spots an unrelated bug.

1. **Agent captures it as a task in the active plan's backlog**:
   `<plan-dir>/tasks/backlog/NN-fix-<bug>.md`. Priority `high`,
   tests named, docs rows filled, out-of-scope section explicit.
2. **Agent continues Phase 3** — does not get derailed.
3. **Later, user asks "what tasks are pending?"** → agent lists the task
   alongside the phase work.

If the bug is truly orthogonal to every active plan, create a stub plan
for it (single paragraph is fine) and a single task inside it. Keeps the
"every task belongs to a plan" invariant.

### Example 3: Non-trivial work that fits one context window

User asks: "Add feature Y to module Z."

1. Agent sizes the round-trip: ~3 source files, ~2 test files, one
   deploy check — well inside ~150K tokens with headroom.
2. **Agent creates a plan** — but does not run `plan-decompose`, since
   the plan + diff + tests fit in one window.
3. Agent executes the plan using the repo execution flow from
   `development-workflow.md`, then commits.

### Example 4: Trivial fix

User asks: "Fix the typo in the status field."

1. Agent sizes it: ≤ 10 LOC, 1 file, no new tests. Trivial.
2. **No plan, no task.** Just fixes it, runs tests, commits.

### Example 5: Orchestrated backlog run (mixed parallelism)

User asks: "Execute the backlog of
`docs/ephemeral/plans/<YYYY-MM-DD>-<feature>/`."

> **Worktree note.** Every executor spawn runs inside an ephemeral
> worktree (`git worktree add -b orch/NN-<name> .worktrees/NN-<name>`
> with shared caches symlinked in). Every `done` report is followed
> by `git cherry-pick` onto the feature branch before review;
> approved tasks end with `git worktree remove` + `git branch -D`.
> The narrative omits these — see Worktree Lifecycle.

1. **Phase 0 — shared exploration.** Main agent spawns one
   exploration subagent. Prompt tells it to crawl the plan and the
   repo surfaces it touches, then write
   `<plan-dir>/tasks/shared_context.md` with the repo tour, relevant
   source excerpts, IDs, environment facts, and gotchas that
   every task will need. The exploration subagent returns a short
   acknowledgement; the main agent does not read the file itself.
2. **Round 1.** Main agent reads only `tasks/README.md`, resolves
   the first ready set (Tasks 01 and 02 — two config-only tasks with
   no dependencies), and spawns two task-executor subagents in
   parallel. Each invocation prompt carries `task_file`, `plan_file`,
   `shared_context`, and a short `notes` field. Neither task needs a
   per-task context pack because shared_context.md already covers
   them.
3. **Ingest round 1 reports.** Both executors return
   `status: done`. Main agent archives the YAML reports to
   `tasks/reports/01-<name>.executor.yaml` and
   `tasks/reports/02-<name>.executor.yaml`. For each, main agent
   spawns a pragmatic subagent against the commit and archives its
   verdict to `tasks/reports/NN-<name>.review.yaml`. Both reviews
   return `approve`. Main agent updates the status cells in
   `tasks/README.md` and logs the commit shas.
4. **Round 2.** Ready set becomes Tasks 03 and 04 (they depend on
   01). Main agent writes a per-task context pack for Task 03
   (it needs a specific source excerpt shared_context did not cover
   in full) and spawns both executors in parallel. Both return
   `done`; pragmatic approves both. Round logged.
5. **Task 07 fails** with `status: failed`, `blockers:
   "loader rejects empty input"`. Main agent does not spawn
   pragmatic for a failed task. It reads only the executor's report
   fields, edits Task 07's context pack to inline the empty-input
   edge case as an explicit In scope bullet, and respawns the
   executor. The retry returns `done`. Pragmatic review then runs
   and returns `approve`.
6. **Task 09 returns `done` but pragmatic returns
   `changes_requested`** — the review notes a silent failure in an
   error branch the task file did not originally require. Main
   agent demotes the executor result to failed, edits the task
   file to add the missing failure-case test, and respawns the
   executor. The retry commits a new diff; pragmatic re-reviews
   and approves.
7. **Task NN is a deploy gate.** Its DAG position blocks downstream
   tasks. Main agent spawns it solo (no parallelism across deploy
   gates), waits for `done`, spawns pragmatic for the review,
   then resumes parallel execution for the downstream set.
8. **Final E2E task** runs solo after every other task is in
   `done/`. Executor returns `done`; pragmatic approves.
9. **Main agent produces a round-up summary** for the user listing
   commit shas, any `followups` surfaced by executors, and any
   reviewer conditions that were applied. The main agent's context
   remained bounded throughout: it never held a task file, a diff,
   a test log, a shared_context body, or a pragmatic transcript in
   its own window.

---
