# tasks/README.md template

> Last updated: __DATE__

Copy this verbatim into `<plan-dir>/tasks/README.md`. Replace every
`<placeholder>`.

---

```markdown
# <Plan Title> — Task Breakdown

**Plan:** [`../plan.md`](../plan.md)
**Total tasks:** <N>
**Total tests:** <N> (<N> unit + <N> integration + <N> E2E)
**Rough scope:** <~N lines of code across <components>>

## Execution order & dependencies

\```
Phase 1 (independent):
  01 → <short name>

Phase 2 (sequential, ship together):
  02 → <short>
  03 → <short>    ← depends on 02
  04 → <short>    ← depends on 02
  05 → <short>    ← deploy + verify gate for Phase 2

...

E2E (after all phases):
  NN → <scripted verification>   ← depends on every phase gate
\```

## Task list

| # | Task | Phase | Priority | Status |
|---|---|---|---|---|
| 01 | [<Title>](backlog/01-<name>.md) | 1 | high | backlog |
| 02 | [<Title>](backlog/02-<name>.md) | 2 | high | backlog |
| ... |

## Workflow per task

Pick the next task from `backlog/` respecting the dependency order above,
then follow the task file's own Checklist section end to end.

- **Orchestrated runs:** the
  [`task-executor`](../../../../.claude/agents/task-executor.md) agent
  owns one task per invocation; the orchestrator handles review, status
  updates, and merge-back — see
  [`workflow-management.md`](../../../../docs/workflow-management.md#orchestrated-execution).
- **Solo runs:** work the checklist yourself, move the task file to
  `done/`, update the status column in this README, and commit. Run the
  [pragmatic agent](../../../../.claude/agents/pragmatic.md) against the
  diff before committing any non-trivial change.

## Scope summary

<2-3 sentences describing what the whole plan ships once all tasks land.>
```
