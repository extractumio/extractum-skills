# read-mailbox

A **strictly read-only** command-line interface (and [Claude Code](https://claude.com/claude-code)
skill) for the local Apple Mail store on macOS. Search, read, and extract
attachments from your Mail.app mailboxes — without ever modifying the store:
no read-marks, no flags, no deletions, not even an accidental SQLite journal
write.

Built for driving from an AI coding agent ("find the invoice from Acme and
save the PDF"), equally usable by hand.

## What it can do

```bash
SCRIPT=scripts/mail_ro.py
python3 $SCRIPT mailboxes                                   # accounts (with their addresses) + mailboxes
python3 $SCRIPT search --from acme --since 2w               # metadata search (subject/from/to/date/mailbox)
python3 $SCRIPT search --attachment invoice                 # by attachment filename
python3 $SCRIPT search --text IES --word                    # whole-word, avoids 'ser-IES' noise
python3 $SCRIPT search --content "wire transfer"            # full-body search (Spotlight or direct scan)
python3 $SCRIPT search --from acme --threads                # whole conversations, incl. your own replies
python3 $SCRIPT show 12345                                  # headers + decoded text body
python3 $SCRIPT thread 12345                                # the conversation around one message
python3 $SCRIPT attachments 12345 --save ~/Documents        # extract attachments (guarded)
python3 $SCRIPT open 12345 --user-requested                 # jump to the message in Mail.app
python3 $SCRIPT open --select 12345 12346 --user-requested  # highlight a set in Mail's viewer
python3 $SCRIPT status                                      # diagnostics: freshness, Spotlight, access
```

All subcommands take `--json` for machine-readable output; every payload is an
object carrying `_untrusted` (the data-not-instructions contract), a `warnings`
array, and the results (`messages`, `threads`, `mailboxes`, or the message
fields). See `SKILL.md` for the full flag reference.

## Requirements

- macOS with Apple Mail and local mail data (`~/Library/Mail/V*`).
- Python 3.9+ (stdlib only — no dependencies, no network).
- **Full Disk Access** for your terminal app (System Settings → Privacy &
  Security → Full Disk Access). Note this is a broad grant — it covers far
  more than Mail. Run `status` to check access.

## Install as a Claude Code skill

Copy this folder to `~/.claude/skills/read-mailbox/`. Claude picks up
`SKILL.md` automatically and will use the tool when you ask it to find or
read something in your mail.

## How it works

Apple Mail keeps an SQLite index at `~/Library/Mail/V*/MailData/Envelope
Index` (senders, subjects, dates, attachment names, short body snippets) and
the messages themselves as `.emlx` files. The store version directory is
auto-detected from `PersistenceInfo.plist`.

- **Default (`--db-mode immutable`)** opens the index with SQLite URI
  `mode=ro&immutable=1`: no locks taken, no writes possible — but mail still
  in the WAL (typically the last seconds-to-minutes) is not visible.
- **`--db-mode copy`** byte-copies the index to a private temp dir (0700,
  deleted on exit) and queries the copy: WAL-fresh, at the cost of briefly
  writing mail *metadata* outside the store.
- Full-body search tries Spotlight first; on many systems the live Mail
  store isn't in Spotlight's index, so it falls back to a bounded direct
  scan of decoded message bodies (~2 s per 1000 messages).
- Gmail-style accounts store each message once (in All Mail) with
  INBOX/label membership in a separate table — the tool resolves both, and
  collapses duplicates in results.
- Everything printed that came from mail is sanitised (control, invisible and
  bidi characters removed; one-line fields collapsed) and wrapped in a fence
  carrying a random per-run id, with a warning block above it when the content
  looks hostile. See **Safety model** below.

## Safety model

The premise: **an email is attacker-controlled input, and so is its
attachment's filename.** A mail-reading tool driven by an AI agent has two
things worth attacking — the agent's instructions, and the one code path that
writes a file. Both are treated as hostile-input problems.

- Store files are only ever opened read-only; the index is never opened
  writable.
- No network access of any kind; the only subprocesses are `/usr/bin/mdfind`,
  `/usr/bin/mdutil`, and — for the `open` subcommand — `/usr/bin/open`
  (`message:` URLs) and `/usr/bin/osascript` (read-only AppleScript:
  activate/select; needs a one-time Automation approval to control Mail).
- Honest exception: `open` shows messages in Mail's UI, and Mail itself marks
  a displayed message as read. The tool never writes to the store, but that
  UI side effect is unavoidable — so `open` is consent-gated: it refuses to
  run without `--user-requested`, a flag an AI agent may only pass when the
  user explicitly asked (in the current session) to see the message(s) in
  Mail. Every other subcommand never touches the Mail app.

### Attack vectors and what stops them

| Vector | Mitigation |
|---|---|
| Prompt injection in a body ("SYSTEM: …", "ignore previous instructions") | All mail text is printed inside a fence carrying a **random per-run id**; `SKILL.md` binds the agent to treat everything inside as data. A heuristic scanner prints a warning block *outside* the fence |
| Forging the end of the fence to make injected text look like tool output | The fence id is unpredictable and is redacted wherever mail text mentions it; fence-shaped lines in a body are rewritten with `[quoted] ` |
| Newlines in a subject/sender/snippet/filename forging extra output rows or a fake `note:` | One-line fields are collapsed to a single line and length-capped |
| ANSI escapes repainting or erasing the terminal | C0/C1 control characters stripped from every output mode (including `--raw`, which is emitted as sanitised text) |
| Text hidden from the human: `display:none`, zero font size, off-screen | Extracted and marked inline `⟪hidden: … ⟫`; warned about when substantial or injection-shaped, and scanned separately |
| Invisible smuggling: zero-width characters, direction overrides, Unicode tag characters | All removed. Overrides and tag characters always raise a warning (measured: zero occurrences in real mail); zero-width padding is removed quietly, since ordinary marketing mail is full of it |
| Injection obfuscated as base64 (including 76-column-wrapped) | Blobs are decoded and re-scanned; a warning fires only when the *decoded* text is injection-shaped, so ordinary base64 tracking tokens stay silent |
| Sender spoofing (`"billing@stripe.com" <ap@invoice-cdn.ru>`) | Display name is compared against the real address; mismatch is warned |
| Attachment named `../../.ssh/authorized_keys` | Reduced to a basename; path components cannot survive |
| Attachment named `.envrc` / `CLAUDE.md` / `settings.json` — config a tool reads on its own | Leading dot removed; known config names get a `.untrusted` suffix; every rename is reported |
| `invoice<U+202E>fdp.exe` displaying as `invoice.pdf` | Invisible/bidi characters stripped from the filename before it is used, exposing the real extension |
| Executable or macro-bearing attachment (`.command`, `.js`, `.xlsm`), decoy double extension (`report.pdf.command`) | `.untrusted` suffix appended; files are written mode 0600 with no exec bit |
| Saving into a location that executes or syncs off-machine | Two-tier destination check: credential/auto-run/system/`.git`/`.github`/`node_modules` paths refused non-overridably; cloud-sync, share and bare-home paths refused unless the user approved that exact path (`--dest-approved`). The destination must already exist unless `--mkdir` |
| Overwriting a file, or a pre-planted symlink at the destination | `O_EXCL|O_NOFOLLOW`, and a name already taken (even by a dangling symlink) gets a `-1`, `-2` … suffix |
| Disk-filling or memory-exhausting attachment/message | 100 MB per attachment (`--max-bytes`), 25 files per call, 64 MB parsed per `show`, 2 MB per message during a body scan |
| Sender-controlled `Message-ID` reaching `open`'s URL and AppleScript | Rejected unless it is plain printable ASCII without `<>` or spaces, ≤512 chars; passed as argv, never interpolated into the script; fully percent-encoded in the `message:` URL |

Known limits: the warning heuristics can miss a cleverly worded attack (absence
of a warning proves nothing); text hidden by matching colours, inside an image,
or in `alt`/`title` attributes is not detected (attribute text is dropped
outright, so it never reaches the agent); and `--raw` is sanitised text, not
byte-exact source.

## Caveats

- Tested on macOS 15 (Sequoia, Mail store V10). Apple changes the Envelope
  Index schema between major macOS versions; older systems may differ.
- Account email addresses shown by `mailboxes` are inferred from each
  account's Sent-mailbox senders (nothing is looked up outside the Mail
  store — but be aware the tool displays them).
- Attachments the user never opened in Mail.app may not exist on disk
  ("NOT downloaded locally"); the tool cannot and will not fetch them.
- If two externally-stored attachments can't be matched to their files
  unambiguously, the tool reports "not downloaded" rather than guessing —
  it never saves bytes it isn't sure about.

## License

MIT — see the [LICENSE](../../LICENSE) file at the repository root.
