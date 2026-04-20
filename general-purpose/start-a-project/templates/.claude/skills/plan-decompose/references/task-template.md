# Task file template

> Last updated: __DATE__

Copy this verbatim for every file in `<plan-dir>/tasks/backlog/`. Replace
every `<placeholder>` and delete inline guidance comments (the italics
blocks) before saving.

---

```markdown
# Task NN: <Short Title>

**Plan reference:** [`../plan.md`](../plan.md) → <Section Name> (lines X–Y)
**Priority:** <critical | high | medium | low>
**Dependencies:** <none | `backlog/NN-name.md` | `done/NN-name.md`>
**Estimated size:** <~N lines of code + ~N tests | deploy gate, no code>
**Serialize:** <false | true — set true iff this task runs a code
generator, a migration revision, or edits a dependency manifest /
lockfile. See `docs/workflow-management.md` for the project's exact
serialize-true signal list. `true` tasks run solo in the main checkout,
never in a parallel worktree.>
**Likely files:** <`path/one.ext`, `path/two.ext` | unknown>

*Likely files is a best-effort scheduling hint for the orchestrator —
not a contract. The executor is bound by the In scope section, not
this line. Use `unknown` whenever the file set only becomes clear
after reading the code; a wrong guess is worse than `unknown`.*

**Context pack:** <none | `backlog/NN-<name>.context.md`>

*Context pack is optional. Add it only when a fresh executor would need
material beyond the task file and cited plan lines — e.g. excerpts from
other source files, rule IDs, environment facts, or prior task reports.
The orchestrator may generate or refresh it at execution time; the task
header just records that the pack exists.*

## Summary

<One paragraph: what this task accomplishes and why it must exist as its own
task (not merged into a neighbour). Lead with the user-visible outcome, not
the implementation detail.>

## In scope

*List every concrete change. Include file paths with line numbers, short
code snippets copied from the plan when the exact wording matters, and an
approximate line-count per change so the executor can size the work.*

### Change <Na>: <short label>
- **File:** `path/to/file.ext:LINE`
- <description of the change, ~N lines>
- <optional inline code snippet>

### Change <Nb>: <short label>
- **File:** `path/to/other.ext`
- <description>

## Out of scope

*Explicit list of things this task does NOT touch, especially things a
careful reader might wrongly assume are included. This section prevents
scope drift during execution — if it is missing, the task is not ready.*

- <e.g. No changes to module Y — that lands in Task 05>
- <e.g. No UI updates — Task 17 owns those>
- <e.g. No new config rules — only the loader>

## Validation

*How to prove the task is done. Mix the relevant items:*

- Unit tests green: `<exact command>`
- Project simplify/lint pass runs cleanly on all modified files
- Deploy to dev env: `<exact deploy command>`
- Manual smoke check: `<exact command and expected output>`
- API verification: `curl <endpoint>` returns `<expected field / value>`
- Event appears in log with `<specific label / decision>`

## Tests

Follow the project test-authoring contract
(see [`docs/how-to-write-tests.md`](../../../../docs/how-to-write-tests.md)).
Failure cases outnumber happy paths at least 2:1. Every test is named and
reachable from the standard runner for its tier.

**File:** `path/to/file_test.ext` (new | additions)

1. **`TestXxx_HappyPath`** — <one-line of what this verifies>
2. **`TestXxx_FailureMode1`** — <edge case or failure class>
3. **`TestXxx_FailureMode2`** — <another edge case>
4. ...

Run with: `<exact command>`

*For E2E tasks, swap the test list for the scripted verification commands
and the expected event fields.*

## Documentation updates

| Document | What to update |
|---|---|
| `docs/architecture.md` | <specific change> |
| `docs/source-map.md` | Add `<new files>` |
| `docs/<project-specific>.md` | Document `<new id / field>` |
| `docs/gotchas.md` | Note <discovered foot-gun> |
| `CLAUDE.md` | <only if a new build target or command was added> |

*Omit rows that do not apply. Do not leave placeholder rows.*

## Checklist

*Final step-by-step execution checklist. The executor ticks these off as
they go. Order matters: implement first, simplify next, test, then deploy
gate, then docs, then move and commit. Review is the caller's job, not
the executor's — see [`task-executor.md`](../../../../.claude/agents/task-executor.md)
and [`workflow-management.md`](../../../../docs/workflow-management.md#orchestrated-execution).*

- [ ] Implement in-scope changes exactly as described above
- [ ] Run the project simplify/lint pass on every modified file
- [ ] Write tests per the Tests section
- [ ] Run the relevant test suite and capture output
- [ ] Deploy + verify (if applicable)
- [ ] Update every document in the Documentation updates table
- [ ] Move this file: `mv backlog/NN-name.md done/NN-name.md`
- [ ] Commit on the assigned branch
```
