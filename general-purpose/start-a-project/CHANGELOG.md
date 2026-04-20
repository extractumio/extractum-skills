# Changelog — start-a-project

All notable changes to this skill are documented here. Versioning follows [SemVer](https://semver.org/).

## [2.1.0] — 2026-04-20

### Added
- **`CLAUDE.md` ↔ `AGENTS.md` hardlink step** in `scripts/init_project.sh`. After copying templates, the scaffolder ensures both filenames point to the same inode so Claude Code (`CLAUDE.md`) and AGENTS.md-spec-aware tools (`AGENTS.md`) read the same file. Handles all four pre-states:
  - Neither exists → template becomes `CLAUDE.md`, `AGENTS.md` is linked to it.
  - Only `CLAUDE.md` exists → `AGENTS.md` is linked to it.
  - Only `AGENTS.md` exists → `CLAUDE.md` is linked to it (preserves user's existing content).
  - Both exist with different content → warn and skip; `--force` relinks `AGENTS.md` to `CLAUDE.md` (discarding the diverged `AGENTS.md`).
- Portable inode detection (`stat -f %i` on macOS, `stat -c %i` on Linux).
- `--dry-run` reports the intended link/relink action.

### Notes
- Hardlinks do not survive `git clone` — a fresh clone materializes two independent files with identical content. Re-run the scaffolder after cloning to restore the single-file relationship.
- Existing v2.0.0 scaffolds re-running the script will keep their `CLAUDE.md` and acquire a hardlinked `AGENTS.md` alongside it (no file overwritten).

## [2.0.0] — 2026-04-20

### Changed (breaking for stub consumers)
- **`templates/.claude/agents/pragmatic.md`** upgraded from a minimal stub to the full production reviewer agent: frontmatter (`model: opus`, extended `tools` list including `AskUserQuestion`, `TaskCreate`, `TaskUpdate`, `TaskList`, `WebFetch`), the five-part "How You Operate" method, technical-lens checklist, repo-specific review gates, the 7-section required-output schema, anti-pattern catalogue, and progress-tracking contract.
- **`templates/.claude/agents/task-executor.md`** upgraded from a minimal stub to the full executor agent: mandatory `worktree_path` + `worktree_branch` invocation contract, strict read boundaries, 15-step execution protocol, hard rules forbidding cross-worktree leakage + dependency mutation + subagent spawning, and the YAML report schema with `status`/`files_touched`/`blockers`/`followups`.
- **`templates/.claude/skills/plan-decompose/SKILL.md`** upgraded from a minimal stub to the full plan-decomposition skill: 8-step algorithm, ~150K-token context-window sizing heuristic with ≥ 20% headroom, `serialize: true` tagging for generators and dependency-mutators, `Likely files` scheduling hints, optional per-task context packs, and the 11-point quality bar (including mandatory pragmatic review before declaring done).
- **`templates/docs/workflow-management.md`** rewritten as the full work-management spec: the decision flowchart sized by context footprint, plan/task artifact layout, Orchestrated Execution subagent-fan-out protocol with per-task git worktrees, the main-agent scheduler loop (phase 0 shared-exploration + phase 1 spawn/merge-back/review), the worktree lifecycle (two creation paths, one cleanup rule), context hygiene rules, failure handling, backlog rules, and five end-to-end examples.
- **`templates/docs/development-workflow.md`** rewritten as the full dev-workflow doc: main-agent role inheriting pragmatic, the expert-panel gate with project-agnostic persona placeholders, progress-tracking-with-todos rules, plan-mode default, subagent strategy, pre-execution branch setup, three ASCII diagrams of the orchestrated-execution flow (roles/data-flow, time sequence, triage branches), self-improvement loop, verification-before-done, demand-elegance, autonomous bug fixing, pre-commit workflow, and proactive code hardening.

### Added
- `templates/.claude/skills/plan-decompose/references/task-template.md` — verbatim task-file template with placeholders (header block including `Serialize:`, `Likely files:`, `Context pack:`; In scope / Out of scope / Validation / Tests / Documentation updates / Checklist sections).
- `templates/.claude/skills/plan-decompose/references/readme-template.md` — verbatim `tasks/README.md` template (ASCII dependency DAG, numbered task table, per-task workflow, scope summary).

### Synced from
These templates track the working patterns from the reference project at `/Users/greg/CLOUDLINUX_SECURITY_AGENT/agent-sentinel/` and have been made project-agnostic: test-runner and deploy commands are described by role rather than by exact name (e.g. "the project simplify/lint pass", "the relevant test runner"), panel personas are presented as placeholders with common alternatives, and serialize-true signals are listed generically (protobuf/gRPC stubs, migration revisions, `go.mod`/`go.sum`, `requirements.txt`/`uv.lock`/`poetry.lock`, `package.json`/`package-lock.json`/`yarn.lock`).

### Migration notes
- Existing v1.2.0 scaffolds keep their stub agent/skill/workflow files. Re-running v2.0.0 with default flags skips existing files (nothing is overwritten). To adopt the new content, back up local edits, run with `--force`, then replay edits via `git diff`.
- The genericized test-runner wording ("project simplify/lint pass", "relevant test runner") is deliberate — fill in concrete commands in [`docs/how-to-run-tests.md`](templates/docs/how-to-run-tests.md) and [`CLAUDE.md`](templates/CLAUDE.md.template) Build Commands once the project has them.

## [1.2.0] — 2026-04-20

### Changed
- **Moved `TODO.md` out of `docs/ephemeral/` to `docs/TODO.md`** (durable, not transient). Tech debt outlives any single plan.
- `templates/docs/TODO.md` rewritten as a structured backlog with four sections: **Tech Debt**, **Refactor Candidates**, **Deferred Fixes**, **Follow-ups**. Each section has its own definition and entry rules.
- `CLAUDE.md` *Mandatory Workflows* bullet now explicitly names tech debt, refactor candidates, and deferred fixes as TODO.md content; points at the new path.
- `CLAUDE.md` Documentation Map adds a row for `docs/TODO.md`.
- `CLAUDE.md` ephemeral list footer no longer includes "backlog items" (they moved to durable).
- `docs/workflow-management.md` *Backlog Rules* section is now concrete (categories, format, promotion rule, prune cadence) instead of `_TBD_`.

### Migration notes
- Existing v1.1.0 scaffolds will keep their `docs/ephemeral/TODO.md`. Re-running v1.2.0 creates the new `docs/TODO.md` alongside it. Move content over by hand and delete the ephemeral file.

## [1.1.0] — 2026-04-20

### Added
- Compact **Security** section in `templates/CLAUDE.md.template` (project-agnostic baseline: external content is data not instructions, halt on injection signals, no silent exfiltration, no untrusted execution, hooks-as-enforcement reminder).
- New `templates/docs/security.md` for the detailed threat model, sensitive paths, hard refuse list, prompt-injection recognition, on-detection protocol, positive rules, hooks notes, and project-specific allow/deny.
- Documentation Map row pointing to `docs/security.md`; Instruction Ownership entry pointing to it.

### Notes
- Existing scaffolds re-running the script will pick up `docs/security.md` (new file, created) but will skip the existing `CLAUDE.md` unless invoked with `--force`. Patch your `CLAUDE.md` by hand if you want the new Security section without overwriting local edits.

## [1.0.0] — 2026-04-20

Initial release.

### Added
- `scripts/init_project.sh` — idempotent scaffolder; copies templates into the target directory, substitutes `__DATE__` and `__SCAFFOLD_VERSION__`, writes a `.claude/.scaffold-version` marker.
- `templates/CLAUDE.md.template` — the canonical CLAUDE.md (placeholders for project name, components, terminology, gotchas, etc.).
- Empty-but-structured docs under `templates/docs/`: `architecture.md`, `source-map.md`, `testing.md`, `how-to-write-tests.md`, `how-to-run-tests.md`, `development-workflow.md`, `workflow-management.md`, `gotchas.md`, `deploy-playbook.md`, `api-reference.md`.
- `templates/docs/ephemeral/` working area: `TODO.md` and four subdirs (`plans/`, `reviews/`, `research/`, `proposals/`) each with a `.gitkeep`.
- Stub agent and skill definitions under `templates/.claude/`: `agents/pragmatic.md`, `agents/task-executor.md`, `skills/plan-decompose/SKILL.md`.
- Defaults to skipping existing files; `--force` overwrites; `--dry-run` previews.

### Conventions
- Every generated `.md` file carries a `Last updated: YYYY-MM-DD` line for tracking.
- Empty sections are kept (with `_TBD_`) so structure survives until the user fills them in.
