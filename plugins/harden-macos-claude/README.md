# harden-macos-claude

A Claude Code plugin that hardens a personal macOS workstation against secret
exfiltration, prompt injection, persistence, and hook-tampering.

Ships two hooks (`PreToolUse`, `PostToolUse`), a pattern library, a trusted-host
allowlist, and a 54-case regression test suite. Stdlib Python only — no pip
dependencies.

---

## What it blocks

**Exfiltration**
- `curl|sh`, `wget|bash`, `eval $(curl …)`, `bash <(curl …)` — always
- `curl|python` / `|node` / `|perl` / `|ruby` — unless every URL targets a
  host in `scripts/allowlist.txt` and no secret or sensitive path is present
- `env` / `printenv` / keychain dump piped to a network command
- Private-key and cloud-credential paths piped anywhere outbound
- Paste/webhook/tunnel domains (pastebin, ngrok, requestbin, transfer.sh, …)
- DNS-based tunneling (long subdomain probes)
- `ln -s` from a sensitive source, `$FOO_KEY`/`$FOO_TOKEN` dereferences
- Writing secrets into iCloud / Dropbox / OneDrive / Google Drive / Box folders
- Writing secrets into attachment-shaped filenames
- MCP outbound tools (gmail-send, slack-post, drive-upload, webhook) carrying
  secret material

**Persistence**
- Writes to `~/.zshrc`, `~/.bashrc`, LaunchAgents, cron, `/etc/profile*`

**Prompt injection**
- Post-tool scan of tool output for `ignore`-`previous`-`instructions`-class
  patterns (see `scripts/patterns.py` for the full list) and for private-key
  headers of the form `-----BEGIN ... KEY-----`; hard-blocks a private key
  in output, warns on the rest

**Self-defense**
- Write/Edit of the plugin's own scripts, `~/.claude/CLAUDE.md`, or
  `~/.claude/settings.json` is refused unless the session marker is present
  (see "Unlocking for plugin updates" below)
- Bash tamper attempts (`rm`, `sed -i`, `> redirect`, `mv/cp INTO`) targeting
  the same paths are blocked
- Malformed hook payloads fail closed (BLOCK) rather than fail open (allow)

---

## Install

### Option A — local plugin dir (fastest)

```bash
claude --plugin-dir ~/EXTRACTUM/SKILLS/plugins/harden-macos-claude
```

That's it. Hooks are wired automatically via `hooks/hooks.json`.

### Option B — Claude Code marketplace (if you publish it)

```bash
claude plugin install harden-macos-claude@<marketplace>
```

### Post-install: add the security charter

Claude Code plugins cannot merge into the user's global `CLAUDE.md`, so the
plugin ships an installer that manages its charter inside a delimited block:

```bash
~/EXTRACTUM/SKILLS/plugins/harden-macos-claude/install.sh
```

What it does:

- Creates `~/.claude/` if missing, and `~/.claude/CLAUDE.md` if missing.
- Writes a timestamped backup to `~/.claude/CLAUDE.md.bak-<ts>`.
- Inserts the charter between two HTML-comment markers:
  ```
  <!-- BEGIN harden-macos-claude charter (managed block — do not edit by hand) -->
  ... charter body ...
  <!-- END harden-macos-claude charter -->
  ```
- Leaves everything outside those markers untouched.
- On re-run: replaces the content between the markers (picks up plugin
  upgrades). **Idempotent** — two runs = one block.
- `install.sh --remove` strips the managed block and leaves your own content.

**Run it from your shell**, not from inside a Claude Code session — the
hooks (once enabled) will correctly block writes to `~/.claude/CLAUDE.md`.
If you must run it through Claude, first set the unlock marker (see
"Unlocking" below).

### Verify

```bash
/usr/bin/python3 ~/EXTRACTUM/SKILLS/plugins/harden-macos-claude/scripts/tests/run_tests.py
# → Passed: 54
```

---

## Unlocking for plugin updates

The plugin refuses Write/Edit of its own scripts, `~/.claude/CLAUDE.md`, and
`~/.claude/settings.json` — otherwise an attacker who hijacked a tool call
could weaken or disable it. To legitimately edit these, first create the
session marker:

```bash
touch ~/.claude/.harden-skill-active   # unlocks edits
# ... make your edits ...
rm   ~/.claude/.harden-skill-active    # re-lock (do this as two commands,
                                       # not `rm ... && ...`, to avoid the
                                       # tamper regex false-positive)
```

Alternatively, export `HARDEN_MACOS_CLAUDE_ACTIVE=1` in the session.

---

## Trusted-host allowlist

`scripts/allowlist.txt` lists hosts whose content can be piped into
`python` / `node` / `perl` / `ruby` without the remote-exec guard firing.
It ships with package registries, cloud APIs, and major AI provider endpoints.
Edit freely — the unlock procedure above applies, but this file itself is
*not* under absolute-deny; only the `.py` scripts are. Add domains like:

```
api.github.com
*.googleapis.com
registry.npmjs.org
pypi.org
```

Wildcards are prefix-matched against the leftmost label (`*.googleapis.com`
matches `searchconsole.googleapis.com` but not `googleapis.com`).

**Note:** `curl|sh` / `wget|bash` are *always* blocked regardless of host.
Only interpreter pipelines (`curl|python`, `curl|node`, …) can be allowlisted.

---

## Layout

```
harden-macos-claude/
├── .claude-plugin/plugin.json        # manifest
├── hooks/hooks.json                  # PreToolUse + PostToolUse wiring
├── scripts/
│   ├── pretool_guard.py              # pre-execution gate (exit 2 = block)
│   ├── posttool_guard.py             # post-execution scan + additionalContext
│   ├── patterns.py                   # regex library + allowlist loader
│   ├── alert.py                      # macOS notify/modal/log/say
│   ├── allowlist.txt                 # trusted hosts
│   └── tests/run_tests.py            # 54-case regression suite
├── CLAUDE.md                         # charter (copy into ~/.claude/CLAUDE.md)
└── README.md                         # this file
```

---

## What it does not do

- **Does not auto-install `~/.claude/CLAUDE.md`.** Plugins can't merge into
  the global charter. Append manually (see above).
- **Does not block all outbound traffic.** For a kernel-level egress gate, use
  `pfctl` or Little Snitch.
- **Does not rotate secrets.** If a real compromise happens, rotate them.
- **Does not replace human review.** Every block produces a notification + a
  line in `~/.claude/security.log` — check the log periodically.

---

## Troubleshooting

**"Test modals keep popping up"** — Quiet mode detects the test runner by
walking up the process tree and looking for `scripts/tests/run_tests.py` in
any ancestor's argv. If you invoke the tests through an unusual shim,
alerts may fire; run the tests via `python3 /abs/path/to/run_tests.py`.

**"Legitimate `curl ... | python3 ...` is blocked"** — Add the host to
`scripts/allowlist.txt`. Shell-interpreter pipes (`| sh` / `| bash`) are
never allowlistable.

**"I need to edit the plugin and it refuses"** — Create the marker
(`touch ~/.claude/.harden-skill-active`), edit, then remove it. See
"Unlocking" above.
