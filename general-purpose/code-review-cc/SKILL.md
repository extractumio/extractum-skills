---
name: code-review-cc
description: >-
  Multi-agent review of the current diff (or a PR/MR number, branch, or file
  target) for correctness bugs plus reuse, simplification, efficiency,
  altitude, and conventions cleanups, at a selectable effort level. Low/medium
  surface fewer, high-confidence findings; high/xhigh/max widen coverage
  (language pitfalls, security, concurrency, wrapper correctness) and may
  include uncertain findings. Use when the user asks to review a diff, branch,
  MR, or PR, or to hunt for bugs before committing. Pass --comment to post
  findings as inline PR/MR comments, or --fix to apply the surviving findings
  to the working tree after the review.
argument-hint: "[low|medium|high|xhigh|max] [--fix] [--comment] [<target>]"
---

`/code-review-cc → scope the diff → per-angle Find (parallel) → Verify (parallel) → (Sweep) → ranked findings`

Last updated: 2026-07-09

Review the current diff for **correctness bugs** and
**reuse / simplification / efficiency** cleanups. Unlike `/simplify-code`
(quality only), this hunts for bugs too.

## Host assumptions

This skill is agent-agnostic. It needs three capabilities, and degrades
cleanly when one is missing:

- **Shell** — to run `git diff` and, optionally, `gh` / `glab`.
- **File search and read** — to grep for symbols and read enclosing functions.
- **Parallel subagents** — wherever this skill says "run N finders/verifiers",
  dispatch them through whatever spawn mechanism the host provides (a
  subagent/task tool, a fan-out API, parallel sessions). If the host has no
  subagent mechanism, run each angle yourself, one at a time, in a fresh pass
  over the diff — treat each angle's instructions as its own self-contained
  prompt and do not let one angle's conclusions carry into the next.

Where the skill names a specific tool (`ReportFindings`, an inline-comment MCP
tool, `gh`, `glab`), that tool is optional: use it if the host exposes it,
otherwise take the stated fallback.

## Selecting the review

Parse the argument string the user passed when invoking the skill
(`$ARGUMENTS`, or the equivalent on your host):

- **Effort level** — the first token, one of `low` · `medium` · `high` ·
  `xhigh` · `max`. If no level is given, default to **medium**. Higher levels
  trade precision for recall: low/medium surface fewer, high-confidence
  findings; high→max widen coverage and may include uncertain findings.
- **`--fix`** — after producing the findings list, also apply them (see the
  *Applying fixes* appendix).
- **`--comment`** — post findings as inline PR/MR comments (see the *Posting
  review comments* appendix).
- **`<target>`** — anything left over (a PR/MR number, branch name, or file
  path) is the review target. If a target is given, prepend `Review target:
  \`<target>\`` to the run and review that instead of the default diff range.

`low` is a self-contained quick pass (next section). Every other level runs
the shared pipeline (Phases 0–3) with the parameters from this table:

| Level | Correctness angles | Quality angles | Cap/angle | Verify bias | Sweep | Max findings |
|---|---|---|---|---|---|---|
| `medium` | A B C | all 5 | 6 | precision | — | 8 |
| `high` | A B C D F | all 5 | 6 | recall | — | 10 |
| `xhigh` | A B C D E F G | all 5 | 8 | recall | 1 round | 15 |
| `max` | A B C D E F G | all 5 | 8 | recall | until dry (≤3 rounds) | 20 |

---

# Effort: low

`low effort → 1 diff pass → no verify → ≤4 findings`

## Turn 1 — read

One tool call: read the unified diff (`git diff @{upstream}...HEAD; git diff HEAD`
to cover both committed and uncommitted changes, or `git diff origin/main...HEAD` /
the target passed as an argument). Skip test/fixture
hunks (`test/`, `spec/`, `__tests__/`, `*_test.*`, `*.test.*`,
`fixtures/`, `testdata/`) — test-file changes are not reviewed at this level.
No subagents, no full-file reads.

## Turn 2 — findings

Flag runtime-correctness bugs visible from the hunk alone: inverted/wrong
condition, off-by-one, null/undefined deref where adjacent lines show the value
can be absent, removed guard, falsy-zero check, missing `await`,
wrong-variable copy-paste, error swallowed in a catch that should propagate.
Also flag — still from the hunk alone — new code that duplicates an existing
helper visible in the diff context, and dead code the diff leaves behind.

Do **not** flag style, naming, perf, missing tests, or anything outside the
hunk.

Output at most **4 findings**, most-severe first. If the host provides a
structured findings tool (`ReportFindings` or equivalent), report them with one
call (`level: "low"`, no `verdict` — no verify pass ran) and do not also print
them; otherwise print one line each:
`path/to/file.ext:123 — what's wrong and the concrete failure`. If nothing
qualifies, report an empty list / output exactly `(none)`.

---

# Shared pipeline (medium · high · xhigh · max)

State the selected level once at the start, then follow the phases below with
that level's row from the table.

**Review stance.** At `medium` you review for **precision**: every finding
you surface should be one a maintainer would act on. At `high` and above you
review for **recall**: catch every real bug a careful reviewer would catch —
a missed bug ships, so err on the side of surfacing.

## Phase 0 — Gather the diff

Run `git diff @{upstream}...HEAD` (or `git diff origin/main...HEAD` /
`git diff HEAD~1` if there's no upstream or HEAD is detached) to get the
unified diff under review. If there are uncommitted changes, or the range diff
is empty, also run `git diff HEAD` and include the working-tree changes in
scope — the review often runs before the commit. If a PR/MR number, branch
name, or file path was passed as an argument, review that target instead. If a
PR/MR number was passed but neither `gh` nor `glab` is available to fetch it,
say so and ask the user to check the branch out locally, then review the local
range. Treat this diff as the review scope.

**Exclude noise before fanning out.** Drop from scope: generated code
(`*.pb.go`, `*_pb2.py`, `*.gen.*`, generated mocks), lockfiles
(`package-lock.json`, `yarn.lock`, `go.sum`, `poetry.lock`, `Cargo.lock`),
vendored dependencies (`vendor/`, `node_modules/`), binary files, and
pure-formatting churn. Keep test files in scope — a weakened or deleted test
is a finding, not noise.

**Shard very large diffs.** If the remaining diff exceeds ~1,500 changed
lines or ~30 files, split the file list into shards of roughly equal size and
run Angles A and B once per shard (same instructions, different file subset)
instead of once total, so no finder skims. The other angles get the full file
list and read only what they need.

## Phase 1 — Find candidates (parallel finders)

Run the level's finder angles as **independent subagents, all dispatched at
once** so they run concurrently. Each surfaces up to the level's per-angle cap
of candidate findings with `file`, `line`, a one-line `summary`, and a concrete
`failure_scenario`.

**Each finder prompt must be self-contained.** Subagents do not share your
context: give each one the exact diff command (or paste the diff if small),
the list of in-scope files, its single angle's instructions verbatim, the
candidate cap, and the required output shape. Do NOT let one angle's
conclusions suppress another's — if two angles flag the same line for
different reasons, record both; dedup happens in Phase 2.

### Angle A — line-by-line diff scan

Read every hunk in the diff, line by line. Then read the enclosing function for
each hunk — bugs in unchanged lines of a touched function are in scope (the PR
re-exposes or fails to fix them). For every line ask: what input, state, timing,
or platform makes this line wrong? Look for inverted/wrong conditions,
off-by-one, null/undefined deref, missing `await`, falsy-zero checks,
wrong-variable copy-paste, error swallowed in catch, unescaped regex metachars.

### Angle B — removed-behavior auditor

For every line the diff DELETES or replaces, name the invariant or behavior it
enforced, then search the new code for where that invariant is re-established.
If you can't find it, that's a candidate: a removed guard, a dropped error
path, a narrowed validation, a deleted test that was covering a real case.
Also flag weakened tests: an assertion deleted or loosened, a tolerance
widened, real behavior replaced by a mock — coverage that silently stopped
protecting what it used to.

### Angle C — cross-file tracer

For each function the diff changes, find its callers (search the repo for the
symbol) and check whether the change breaks any call site: a new precondition, a
changed return shape, a new exception, a timing/ordering dependency. Also check
callees: does a parallel change in the same PR make a call unsafe?

### Angle D — language-pitfall specialist

Scan for the classic pitfalls of the diff's language/framework — for example:
JS falsy-zero, `==` coercion, closure-captured loop var; Python mutable default
args, late-binding closures; Go nil-map write, range-var capture; SQL injection;
timezone/DST drift; float equality. Flag any instance the diff introduces.

### Angle E — wrapper/proxy correctness

When the PR adds or modifies a type that wraps another (cache, proxy, decorator,
adapter): check that every method routes to the wrapped instance and not back
through a registry/session/global — e.g. a caching provider holding a
`delegate` field that resolves IDs via `session.get(...)` instead of
`delegate.get(...)` will re-enter the cache or recurse. Also check that the
wrapper forwards all the methods the callers actually use.

### Angle F — security & trust boundaries

Trace every place the diff lets external or user-controlled data cross a trust
boundary: injection (SQL/shell/regex/template/path traversal), missing or
weakened authn/authz checks on a reachable endpoint or handler, secrets or
tokens written to code/logs/errors, unsafe deserialization or `eval`-like
sinks, SSRF via user-supplied URLs, permissive TLS/verification flags,
overly-broad file or process permissions. Flag only what the diff introduces
or re-exposes, with the concrete attack path as the failure scenario.

### Angle G — concurrency & resource lifecycle

Look for what breaks under parallelism or over time: shared state written
without the lock that other paths use, lock scope narrowed, check-then-act
races (TOCTOU), missing context/timeout/cancellation on a new blocking call,
channel or queue that can deadlock or fill, and resources acquired without a
guaranteed release on error paths (files, sockets, transactions, goroutines,
listeners, timers). The failure scenario names the interleaving or leak.

### Reuse

The angles above hunt for bugs; this one and the four that follow hunt for
cleanup in the changed code. Flag new code that re-implements something the codebase
already has — search shared/utility modules and files adjacent to the change,
and name the existing helper to call instead.

### Simplification

Flag unnecessary complexity the diff adds: redundant or derivable state,
copy-paste with slight variation, deep nesting, dead code left behind. Name
the simpler form that does the same job.

### Efficiency

Flag wasted work the diff introduces: redundant computation or repeated I/O,
independent operations run sequentially, blocking work added to startup or
hot paths. Also flag long-lived objects built from closures or captured
environments — they keep the entire enclosing scope alive for the object's
lifetime (a memory leak when that scope holds large values); prefer a
class/struct that copies only the fields it needs. Name the cheaper
alternative.

### Altitude

Check that each change is implemented at the right depth, not as a fragile
bandaid. Special cases layered on shared infrastructure are a sign the fix
isn't deep enough — prefer generalizing the underlying mechanism over adding
special cases.

### Conventions (agent instruction files)

Find the instruction files that govern the changed code, under whatever names
this repo and host use — `AGENTS.md`, `CLAUDE.md`, `CLAUDE.local.md`,
`GEMINI.md`, `.cursor/rules/*`, `.cursorrules`,
`.github/copilot-instructions.md` — plus the host's user-level instruction file
if one exists (e.g. `~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`). Scope matters:
a file in a directory applies only to files at or below it, so collect the ones
in ancestor directories of each changed file, along with the repo-root and
user-level ones. Read each that exists, then check the diff for clear violations
of the rules they state.

Only flag a violation when you can quote the exact rule and the exact line
that breaks it — no style preferences, no vague "spirit of the doc"
inferences. In the finding, name the instruction file's path and quote the rule
so the report can cite it. If no instruction file applies, return nothing for
this angle.

Cleanup, altitude, and conventions candidates use the same
`file`/`line`/`summary` shape; in `failure_scenario`, state the concrete
cost (what is duplicated, wasted, harder to maintain, or which convention rule
is broken) instead of a crash. Correctness bugs always outrank cleanup,
altitude, and conventions findings when the output cap forces a cut.

Pass every candidate with a nameable failure scenario through — finders that
silently drop half-believed candidates bypass the verify step and are the
dominant cause of misses.

## Phase 2 — Verify (1-vote, 3-state)

Dedup candidates that point at the same line/mechanism, keeping the one with
the most concrete failure scenario. For each remaining candidate, run **one
verifier** subagent — dispatch all verifiers at once so they run in parallel,
each with a self-contained prompt (the diff, the relevant file paths, and the
candidate). Each returns exactly one of:

- **CONFIRMED** — can name the inputs/state that trigger it and the wrong
  output or crash. Quote the line.
- **PLAUSIBLE** — mechanism is real, trigger is uncertain (timing, env,
  config). State what would confirm it.
- **REFUTED** — factually wrong (code doesn't say that) or guarded elsewhere.
  Quote the line that proves it.

Keep candidates where the vote is CONFIRMED or PLAUSIBLE.

Cleanup, altitude, and conventions candidates go through the same verifier with
the states read against cost rather than crash: CONFIRMED when the duplication,
waste, or violation provably exists and the named helper or quoted rule is real;
REFUTED when the helper doesn't exist, the rule doesn't govern that file, or the
"simpler form" would change behavior.

**At `medium` (precision)** the verifier applies the definitions above
neutrally; when genuinely torn between PLAUSIBLE and REFUTED, lean REFUTED —
only findings a maintainer would act on should survive.

**At `high` and above (recall)** — PLAUSIBLE by default: do not refute a
candidate for being "speculative" or "depends on runtime state" when the
state is realistic: concurrency races, nil/undefined on a rare-but-reachable
path (error handler, cold cache, missing optional field), falsy-zero treated
as missing, off-by-one on a boundary the code does not exclude, retry storms
/ partial failures, regex/allowlist that lost an anchor. REFUTED only when
constructible from the code: factually wrong (quote the actual line);
provably impossible (type/constant/invariant — show it); already handled in
this diff (cite the guard); or pure style with no observable effect. A single
non-REFUTED vote carries the finding — do NOT drop on uncertainty.

## Phase 3 — Sweep for gaps (`xhigh` and `max` only)

Run **one more finder** as a fresh reviewer who has the verified list. Re-read
the diff and enclosing functions looking ONLY for defects not already listed.
Do not re-derive or re-confirm anything already there — the job is gaps. Focus
on what the first pass tends to miss: moved/extracted code that dropped a guard
or anchor; second-tier footguns (dataclass default evaluated once, `hash()`
non-determinism, lock-scope shrink, predicate methods with side effects);
setup/teardown asymmetry in tests; config defaults flipped.

Surface **up to 8 additional candidates**, each naming a defect not already on
the list, and verify them through Phase 2. If nothing new, return an empty
sweep — do not pad.

At `max`, repeat the sweep with a fresh finder until a round returns no new
verified finding, up to **3 rounds total**. At `xhigh`, one round only.

## Output

Rank surviving findings most-severe first using this order: (1) security
vulnerability or data loss/corruption, (2) crash or unhandled error on a
reachable path, (3) wrong result or broken behavior for realistic inputs,
(4) performance regression or resource leak, (5) reuse/simplification/
efficiency/altitude/conventions cleanups. If more survive than the level's
max-findings cap, keep the most severe.

If the host provides a structured findings tool (`ReportFindings` or
equivalent), report the final list with **one** call — `level` set to the
effort level, each finding carrying its `file`, `line`, `summary`,
`failure_scenario`, a `category` slug, and `verdict` from Phase 2 — and do not
also print the findings as text. Otherwise return them as a JSON array carrying
the same fields:

```json
[
  {
    "file": "path/to/file.ext",
    "line": 123,
    "summary": "one-sentence statement of the bug",
    "failure_scenario": "concrete inputs/state → wrong output/crash",
    "category": "correctness",
    "verdict": "CONFIRMED"
  }
]
```

If nothing survives verification, report/return an empty list.

---

# Appendix — Posting review comments (`--comment`)

The `--comment` flag was passed. After producing the findings list, post each
finding as an inline comment on the review target, one comment per finding;
include a suggestion block only when it fully fixes the issue.

- **GitHub PR** — use an inline-comment tool if the host exposes one (e.g.
  `mcp__github_inline_comment__create_inline_comment`), else
  `gh api repos/{owner}/{repo}/pulls/{pr}/comments` with the file path, line,
  and side.
- **GitLab MR** (check `git remote get-url origin` when the host is unclear) —
  use `glab api projects/:id/merge_requests/<mr>/discussions` with a
  `position` payload (`position_type=text`, `base_sha`/`head_sha`/`start_sha`
  from the MR's `diff_refs`, `new_path`, `new_line`). If inline positioning
  fails, fall back to one summary note via `glab mr note <mr>`.

If the target is not a PR/MR, or no posting path is available, print the
findings to the terminal and note that `--comment` was ignored.

# Appendix — Applying fixes (`--fix`)

The `--fix` flag was passed. After producing the findings list, apply the
findings to the working tree instead of stopping at the report: fix each one
directly — correctness bugs and reuse/simplification/efficiency cleanups alike.
Skip any finding whose fix would change intended behavior, require changes well
outside the reviewed diff, or that you judge to be a false positive — note the
skip rather than arguing with it.

After applying, validate: run the narrowest relevant check for the touched
code (the affected tests, or a build/typecheck when no test covers it) and
report the real result — if a fix breaks something, revert that fix and mark
the finding skipped. If findings were reported through a structured findings
tool, re-report them once with `outcome` set on each (`fixed` / `skipped` /
`no_change_needed`). Finish with a brief summary of what was fixed and what
was skipped.
