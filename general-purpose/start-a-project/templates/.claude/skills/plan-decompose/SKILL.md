---
name: plan-decompose
description: >
  Decompose an approved plan document into self-contained task files under
  <plan-dir>/tasks/backlog/. Each task captures title, scope, out-of-scope,
  validation, tests, docs, and a final checklist — complete enough that a
  fresh agent can pick it up cold and ship it. Use when a plan is finalized
  and needs task breakdown before execution starts. Not for creating plans,
  not for doing the implementation work itself.
allowed-tools: Read, Write, Edit, Glob, Grep, Bash, Agent
model: opus
---

# Plan Decomposition

> Last updated: __DATE__

Convert an approved `plan.md` into a set of task files that another agent
(or a future you) can execute one at a time without re-reading the whole
plan each time.

## Goal

Given `docs/ephemeral/plans/<plan-dir>/plan.md`, produce:

1. `<plan-dir>/tasks/backlog/NN-<short-name>.md` — one file per task
2. `<plan-dir>/tasks/done/` — empty directory (tasks move here when finished)
3. `<plan-dir>/tasks/README.md` — dependency DAG, task table, workflow

Each task is self-contained: the executor should only need the task file
plus the referenced plan lines, never the whole plan.

## When to invoke

- A plan has just been finalized and an explicit breakdown is needed.
- The user says "decompose this plan", "break this into tasks", "turn this
  plan into a backlog".
- Re-decomposing after a plan revision — reconcile with any already-done
  tasks, do not overwrite them.

## When NOT to invoke

- The change is trivial (< 5 min, obvious approach) and needs no plan.
- There is no plan yet — creating the plan is a different workflow
  (expert panel + design, see
  [`docs/development-workflow.md`](../../../docs/development-workflow.md)).
- Tasks already exist and are partially in progress — edit them in place.
- You are being asked to *execute* tasks — that is also a different step.

## Inputs

Required:
- Absolute path to `plan.md` (or the directory containing it).

Optional:
- Scope hints ("Phase 2 only", "skip UI work", "ignore E2E for now").
- Priority override for specific sections.

## Outputs

- Task files under `<plan-dir>/tasks/backlog/`
- `<plan-dir>/tasks/README.md`
- Empty `<plan-dir>/tasks/done/`
- A short report listing: task count, dependency summary, total named
  test count, anything ambiguous the human still has to resolve.

## Decomposition algorithm

### Step 1 — Read the plan in full

Use `Read` on the plan top-to-bottom. Never decompose from a summary. Every
task must cite exact line numbers in the plan, and only a full read gives
you those.

### Step 2 — Identify natural task boundaries

Look for these structural signals, in order:

1. **Explicit phase / change / subphase headers** — `Phase 1`, `Phase 2a`,
   `Change 4b`. These usually map 1:1 to tasks.
2. **Deploy / verify gates between phases** — become their own tasks with
   no code, just integration checks. Never bundle "implement X" and
   "deploy and verify X" into one task.
3. **Independent parallel work** — UI-only changes, doc-only changes, or
   config-only changes that do not depend on code work.
4. **End-to-end verification** — scripted E2E becomes a final task at the
   bottom of the dependency graph.

A single-phase plan with one cohesive change becomes **one task**. Do not
fragment a five-line fix into three tasks just to hit a count.

**Size heuristic: context-window fit, not wall-clock time.** A good task is
one that can be implemented end-to-end inside a single ~150K-token context
window — meaning the executing agent can hold, at the same time:

- the full task file
- the cited plan line range
- every source file the task edits
- the test files it writes or extends
- the output of the test runner and any deploy/verify commands
- enough headroom (≥ 20%) for reasoning, tool results, and retries

If the combined footprint would blow past ~150K tokens, the task is too
big — **split it**. Typical splits: separate data model from consumers,
separate loader from rules, separate code change from deploy + verify,
separate config/rule additions from the engine code that evaluates them.

Conversely, if two adjacent tasks together are comfortably under the
budget and share most of the same files and tests, **merge them** — two
overhead-heavy tasks are worse than one cohesive one.

When in doubt, estimate: count edited source files × rough LOC, plus
test files × rough LOC, plus the cited plan range. If that number is
already more than a third of the budget before the agent even starts
reasoning, split.

### Step 3 — Determine dependencies

For each task, identify which other tasks must complete first. Common
reasons:

- File conflicts (two tasks edit overlapping regions of the same file)
- Logical ordering (config loading must exist before config consumers)
- Safety gates (deploy + verify must precede the next phase)
- Test ordering (rules or fixtures must exist before the integration
  test that asserts on them)

Dependencies go into the task file's `**Dependencies:**` line and into
the `tasks/README.md` dependency DAG.

**Keep dependencies minimal.** Over-serializing the graph blocks work that
could have run in parallel. Config-only or doc-only tasks rarely depend
on anything.

**Hard rule — serialize generator and dependency-mutating tasks.**
Per-task worktrees (see
[workflow-management.md](../../../docs/workflow-management.md#worktree-lifecycle))
isolate normal source edits, but some operations are globally ordered
and unsafe to parallelize even with isolation. The project's exact
serialize-true signal list lives in
[`docs/workflow-management.md`](../../../docs/workflow-management.md) —
consult it and adjust per the stack.

Common families (fill in concrete examples for your stack):

| Family | Serialize because |
|---|---|
| Code generators (IDL stubs, ORM models, schema-derived code) | Generated files diverge silently across parallel worktrees. |
| Migration revisions | Each worktree picks the same "next" revision; second commit lands with a wrong parent, no conflict to catch it. |
| Dependency manifest / lockfile edits | Package-manager caches and virtualenvs are often shared across worktrees; parallel installers corrupt them. |

Tag such tasks with `serialize: true` plus explicit upstream *and*
downstream edges so the orchestrator runs them solo in the main
checkout. If two tasks both need the same generator run, merge them
or split the generator step into its own task both depend on.

### Step 3.5 — Best-effort file-touch hints

Emit a `**Likely files:**` line in the task header. This is a
**scheduling hint, not a contract** — the executor is bound by In
scope, not this list, and the orchestrator only uses it to bias
pairing (prefer disjoint file sets per round).

Valid values: a list of concrete paths the plan names, or the literal
`unknown` (common and preferred) when the file set only becomes clear
after reading the code. Empty = `unknown`. Never guess — an inaccurate
hint is worse than an absent one because the orchestrator schedules
against it.

### Step 4 — Extract per-task test plan

For every task, apply the project's test-authoring contract (see
[`docs/how-to-write-tests.md`](../../../docs/how-to-write-tests.md)):

- At least 2 failure or edge-case assertions per happy-path assertion.
- Name every test. Never write "add unit tests" — list the exact test
  function names and what each proves.
- If the plan enumerates tests (e.g. `TestXxx_Foo` with a test number),
  copy the numbers and names **verbatim** — that is the contract between
  plan and task.
- If the plan does not enumerate tests, apply the project failure-mode
  catalog and name the cases yourself.
- Pick the correct tier per
  [`docs/how-to-run-tests.md`](../../../docs/how-to-run-tests.md).

### Step 5 — Extract documentation updates

If the plan has a "Documentation Updates" section, split it per task. If
not, infer from the change surface using the project's Documentation Map
in `CLAUDE.md`. Common signals:

| Change signal | Doc to update |
|---|---|
| New source file | `docs/source-map.md` |
| New build target / command | `CLAUDE.md` (Build Commands) |
| New config / rule / schema | project-specific reference doc |
| New subsystem / behavior | `docs/architecture.md` |
| New gotcha / foot-gun | `docs/gotchas.md` |
| New deploy step / env var | `docs/deploy-playbook.md` |
| New API endpoint / field | `docs/api-reference.md` |
| New test file or tier | `docs/testing.md` + `docs/how-to-write-tests.md` |
| Security-relevant change | `docs/security.md` |

### Step 6 — Write the task files

Use the template in [`references/task-template.md`](references/task-template.md).
Every field is required unless the template marks it optional. No "see
plan for details" handwaves except for the exact plan line range.

Naming: `NN-<kebab-short-name>.md` where `NN` is two-digit zero-padded
and matches execution order (01, 02, 03...). Keep names short but
unambiguous: `03-ring-buffer-size-bump.md` beats `03-phase2b.md`.

**Optional context packs.** For any task a fresh
[`task-executor`](../../agents/task-executor.md) subagent would not be
able to finish from the task file + cited plan lines alone, also emit a
sibling `backlog/NN-<name>.context.md` file and reference it from the
task's header block. The pack is a *focused* starter context, not a
dump. Include only what the executor legitimately cannot re-derive
cheaply on its own:

- exact excerpts (with file + line range) from source files the task
  must read but that are scattered across the repo
- rule IDs, config field names, or message/type names the task
  references implicitly through the plan's prose
- environment or deploy facts that are true at execution time but not
  captured anywhere durable
- verbatim YAML reports from prior tasks this one depends on, when the
  dependency is a contract decision rather than a file

Do **not** put into a context pack:

- anything the executor would naturally find by `Grep` in under a minute
- the full plan (the cited line range is enough)
- narrative background that repeats `CLAUDE.md` or `architecture.md`
- information the orchestrator may need to refresh at execution time —
  leave that field out and let the orchestrator add it on spawn

Config-only and trivial tasks almost never need a context pack.
Multi-file refactors, tasks with non-obvious cross-references, and
final E2E tasks usually do. When in doubt, skip it — the orchestrator
can always write one at spawn time.

### Step 7 — Write tasks/README.md

Use the template in [`references/readme-template.md`](references/readme-template.md).
It must include:

- Link back to the plan
- ASCII dependency DAG
- Numbered task table with columns: `# | Task | Phase | Priority | Status`
- Workflow section explaining how to run one task end-to-end
- Totals (task count, test count, lines-of-code estimate if known)

### Step 8 — Pragmatic review

Before declaring done, delegate the full set of task files to the
[pragmatic agent](../../agents/pragmatic.md) and ask it to stress-test:

- Is any task too big to implement inside a single ~150K-token context
  window (full task file + cited plan range + edited sources + test
  files + verify output + ≥ 20% reasoning headroom)?
- Is any task so small it is pure overhead (and should be merged)?
- Are the dependencies correct and **minimal**?
- Does every task have a concrete failure case a reviewer could reject?
- Is the out-of-scope section explicit or just implicit?
- Are deploy/verify gates separated from implementation tasks?
- Is any task missing the test plan or doc update rows?

Apply the agent's feedback **before** finishing. Do not ship a task set
the pragmatic agent has objections to.

## Task file structure

Every file in `tasks/backlog/` follows the template in
[`references/task-template.md`](references/task-template.md). The
mandatory sections are:

1. **Title** — `# Task NN: <Short Title>`
2. **Header block** — plan reference with line numbers, priority,
   dependencies, estimated size, `serialize: true|false` (default
   `false`; set `true` for generator / dep-mutating tasks per Step 3),
   and `Likely files:` hint per Step 3.5 (explicit file list or the
   literal `unknown`).
3. **Summary** — one paragraph: what, why, why-now.
4. **In scope** — specific files, line numbers, concrete changes. Include
   code snippets when the plan has them.
5. **Out of scope** — explicit list of things the task does NOT touch.
   This is load-bearing — it prevents scope drift during execution.
6. **Validation** — how to prove the task is done (unit tests, deploy +
   smoke, manual verification commands, API queries).
7. **Tests** — named test functions per the project test-authoring
   contract. Include exact file path, new-or-additions, and the command
   to run.
8. **Documentation updates** — table of files to update in the same
   commit as the code change.
9. **Checklist** — the final step-by-step execution checklist the agent
   ticks off.

## tasks/README.md structure

See [`references/readme-template.md`](references/readme-template.md).

## Quality bar

The task set passes only when **all** of these are true:

1. **Self-contained.** A fresh agent can execute any task without the
   whole plan — only the cited line range.
2. **Explicit out-of-scope.** Every task lists what it is not doing.
3. **Named tests.** Never "add unit tests"; always specific function
   names.
4. **Named docs.** Never "update relevant docs"; always the exact file
   and what to change.
5. **Minimal dependencies.** No over-serialization.
6. **Separated deploy gates.** Code + verify are never the same task.
7. **Right-sized.** Small plans produce one task, not artificial splits.
8. **Pragmatic-reviewed.** Not skipped, not batched after the fact.
9. **Reachable tests.** Every test in every task file can be run from
   the standard project runner per `docs/how-to-run-tests.md`.
10. **Generator and dep-mutating tasks tagged `serialize: true`.** Every
    generator-run and manifest-edit task has the tag and an explicit
    ordering dependency. No such task is buried inside a "normal"
    implementation task.
11. **Likely files hint set (or explicitly `unknown`).** Every task has
    a `Likely files` line. A guess is worse than `unknown`.

## Output contract

When done, report (under 200 words):

- Number of task files created and path to `tasks/README.md`
- Dependency graph summary (critical path length, parallel branches)
- Total named test count across all tasks
- Any ambiguity in the plan a human must resolve before execution starts

## References

- [`docs/workflow-management.md`](../../../docs/workflow-management.md) —
  plan/task decision rules this skill implements
- [`docs/how-to-write-tests.md`](../../../docs/how-to-write-tests.md) —
  test-design contract every task must honor
- [`.claude/agents/pragmatic.md`](../../agents/pragmatic.md) — review agent
  that must approve the task set before declaring done
- [`references/task-template.md`](references/task-template.md) — exact task
  file structure with placeholders
- [`references/readme-template.md`](references/readme-template.md) — exact
  tasks/README.md structure
