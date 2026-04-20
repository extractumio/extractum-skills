---
name: start-a-project
description: >
  Scaffold or update a project to match the standard CLAUDE.md structure — root
  CLAUDE.md, docs/ tree (architecture, source-map, testing trio, workflow,
  gotchas, deploy, api), docs/ephemeral/ working area, and .claude/ agent + skill
  stubs. Use when starting a new repo, adopting the standard layout in an existing
  repo, or repairing a partial scaffold. Skips existing files unless --force.
version: 2.1.0
argument-hint: [target-dir]
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Start A Project

Bootstraps the canonical `CLAUDE.md` + `docs/` + `.claude/` structure into a target directory. Idempotent: existing files are left alone unless explicitly forced.

## When to use

- Starting a fresh repo and you want the standard layout from day one.
- An existing repo is missing the layout (no `CLAUDE.md`, no `docs/`, etc.).
- You want to repair a partial scaffold — the script will fill in the gaps without disturbing what already exists.

## What gets created

```
<target>/
├── CLAUDE.md
├── AGENTS.md                      # hardlinked to CLAUDE.md (same inode, one file)
├── docs/
│   ├── architecture.md
│   ├── source-map.md
│   ├── testing.md
│   ├── how-to-write-tests.md
│   ├── how-to-run-tests.md
│   ├── development-workflow.md
│   ├── workflow-management.md
│   ├── gotchas.md
│   ├── deploy-playbook.md
│   ├── api-reference.md
│   └── ephemeral/
│       ├── TODO.md
│       ├── plans/.gitkeep
│       ├── reviews/.gitkeep
│       ├── research/.gitkeep
│       └── proposals/.gitkeep
└── .claude/
    ├── .scaffold-version          # records skill version + date
    ├── agents/
    │   ├── pragmatic.md
    │   └── task-executor.md
    └── skills/
        └── plan-decompose/SKILL.md
```

Every generated `.md` carries a `Last updated: YYYY-MM-DD` line. Empty sections are kept (with `_TBD_`) so the structure persists until the user fills them in.

## `CLAUDE.md` and `AGENTS.md` are the same file

After copying templates, the scaffolder hardlinks `AGENTS.md` to `CLAUDE.md` (same inode, one on-disk file, two directory entries). Claude Code reads `CLAUDE.md`; other tools following the [AGENTS.md spec](https://agents.md) read `AGENTS.md`. Editing one edits the other — there is no sync step, no drift.

Behavior:

| State before run | Result |
|---|---|
| Neither exists | Template → `CLAUDE.md`, then hardlink `AGENTS.md` to it |
| Only `CLAUDE.md` exists | Hardlink `AGENTS.md` to `CLAUDE.md` |
| Only `AGENTS.md` exists | Hardlink `CLAUDE.md` to `AGENTS.md` (preserves user's existing content) |
| Both exist, same inode | No-op |
| Both exist, different content | Warn and skip (use `--force` to re-link `AGENTS.md` → `CLAUDE.md`, discarding the diverged `AGENTS.md`) |

**Caveat:** hardlinks do not survive `git clone`. Git stores two independent blobs for the two paths. On a fresh clone the files will have identical content but be separate files until re-linked. Run the scaffolder again (or `rm AGENTS.md && ln CLAUDE.md AGENTS.md`) after cloning to restore the single-file relationship.

## How to invoke

### From the project root

```bash
bash /Users/greg/EXTRACTUM/SKILLS/general-purpose/start-a-project/scripts/init_project.sh
```

### Targeting another directory

```bash
bash /Users/greg/EXTRACTUM/SKILLS/general-purpose/start-a-project/scripts/init_project.sh --target /path/to/repo
```

### Options

| Flag | Effect |
|------|--------|
| `--target DIR` | Where to scaffold. Defaults to `$PWD`. |
| `--force` | Overwrite existing files. **Destroys local edits.** Default: skip. |
| `--dry-run` | Print what would be created/skipped; write nothing. |
| `-h`, `--help` | Show usage. |

## Procedure for Claude

1. **Confirm the target.** If the user did not name a directory, default to `$PWD`. Show the path and ask before scaffolding into a directory the user did not explicitly name.
2. **Survey existing state.** `ls` the target. If `CLAUDE.md`, `docs/`, or `.claude/` already exist, run `--dry-run` first and report the diff.
3. **Run the scaffolder.** Default mode (skip existing). Show the output verbatim — created/skipped/updated counts matter.
4. **Customize obvious placeholders if known.** If you already know the project's name, primary language, or component layout from context, offer to substitute the bracketed placeholders (`[PROJECT_NAME]`, `[ONE_LINE_DESCRIPTION]`, etc.) in the new `CLAUDE.md`. Do not guess — ask first.
5. **Stamp the version.** The script writes `.claude/.scaffold-version` automatically. Mention the version and date in the final summary.
6. **Do not** run with `--force` unless the user explicitly asks. Overwriting customized docs would destroy work.

## Updating an existing scaffold

When the skill version changes, re-running the script with default flags is safe — it only fills in *missing* files. To compare what shipped vs. what the user has now, run `--dry-run` and look at the `skip` lines: anything that exists is preserved.

If the user wants to forcibly refresh stub files they have not touched, advise them to:

1. Commit current state first.
2. Run with `--force`.
3. Use `git diff` to keep their edits and discard the rest.

## Versioning

This skill follows SemVer (`MAJOR.MINOR.PATCH`):

- **PATCH** — fixes to templates, no structural change.
- **MINOR** — new optional templates or new doc files.
- **MAJOR** — incompatible structure change (renamed files, removed sections).

Current version lives in `VERSION`. Changelog: [`CHANGELOG.md`](CHANGELOG.md). The version is stamped into every scaffolded project as `<target>/.claude/.scaffold-version` so future invocations can detect drift.

## Files in this skill

```
start-a-project/
├── SKILL.md                # this file
├── VERSION                 # semver, single line
├── CHANGELOG.md            # release notes
├── scripts/
│   └── init_project.sh     # the scaffolder
└── templates/              # source of truth for every generated file
    ├── CLAUDE.md.template
    ├── docs/...
    └── .claude/...
```

To add a new template: drop the file under `templates/<relative-path>` (use `__DATE__` where you want today's date and `__SCAFFOLD_VERSION__` for the skill version). The script picks it up on the next run — no other wiring needed.
