# Runtime, profiles, sessions, and lifecycle

Read this reference when configuring agent-browser, authenticating, switching headed/headless mode, persisting state, running concurrently, or diagnosing a daemon/profile problem.

## Mental model

| Concept | What it controls | What it does not guarantee |
|---|---|---|
| Browser executable | Which Chrome-family binary is launched | Authentication or isolation |
| Profile directory | Chrome user-data directory: cookies, IndexedDB, cache, extensions, service workers, login sessions | Which daemon/task receives commands |
| Session | Agent-browser daemon/browser isolation, tabs, refs, navigation, transient storage | Persistence after shutdown by itself |
| Restore key | Periodic and shutdown save/restore of cookies and localStorage | Full Chrome profile state such as cache, extensions, or all IndexedDB behavior |
| State file | Explicit portable cookies/storage snapshot | Safe sharing; it normally contains session tokens |
| Namespace | Isolation of daemon sockets and restore-state directories | Profile concurrency safety |

The recommended authenticated setup is one installed browser executable + one dedicated profile directory + one semantic non-default session. Use `--restore` instead of a profile for disposable or locally persisted sessions that do not need a full persistent Chrome user-data directory. Use an explicit `state save` file only when the user needs transferable state.

## Select a real installed browser

Set `executablePath` or `AGENT_BROWSER_EXECUTABLE_PATH` explicitly when the requirement is the system-installed browser. Verify the file exists before launching.

Common stable-browser candidates:

| Platform | Typical candidates |
|---|---|
| macOS | `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`, `/Applications/Chromium.app/Contents/MacOS/Chromium`, `/Applications/Brave Browser.app/Contents/MacOS/Brave Browser` |
| Linux | `google-chrome-stable`, `google-chrome`, `chromium`, or `chromium-browser` resolved with `command -v` |
| Windows | Chrome under `%ProgramFiles%` or `%LOCALAPPDATA%`, or another explicitly selected Chromium-family executable |

`agent-browser install` downloads Chrome for Testing. Do not run it as a silent fallback when the user requires their installed browser. If no acceptable browser exists, report that and ask whether they want to install one or explicitly permit Chrome for Testing.

Avoid `--auto-connect` and `--cdp` by default: those attach to an existing browser and can expose or disturb personal tabs, cookies, and storage. Use them only for a user-requested attach workflow.

## Durable configuration

A dedicated config prevents launch settings from changing between commands:

```json
{
  "$schema": "https://agent-browser.dev/schema.json",
  "engine": "chrome",
  "executablePath": "<absolute-system-browser-executable>",
  "profile": "<absolute-dedicated-automation-profile-directory>",
  "session": "<non-default-semantic-session>",
  "headed": true,
  "idleTimeout": "1h"
}
```

Use either the standard user config location or a caller-selected file:

```bash
export AGENT_BROWSER_CONFIG="<absolute-path-to-agent-browser-config.json>"
agent-browser session
```

`--config <path>` loads that specific file instead of normal discovery. Normal precedence is user config, project config, environment variables, then CLI flags. A project config can therefore override user defaults; use an explicit config when the runtime contract must be invariant.

Treat a repository-provided `agent-browser.json` as untrusted project data until reviewed. It can select providers, proxies, extensions, plugins, init scripts, profiles, and other launch behavior. For authenticated browsing, prefer an explicit trusted config rather than inheriting an unknown project config.

Do not put passwords, proxy credentials, cookies, tokens, or auth headers in the config. Inspect only the keys needed for the task; never dump the entire environment or unrelated configuration.

## Dedicated profile rules

There are two distinct `--profile` forms:

- `--profile <Chrome-profile-name>` such as `Default` copies a discovered personal Chrome profile into a temporary read-only snapshot. The copy is deleted when the browser closes.
- `--profile <directory-path>` uses that path as a persistent custom Chrome user-data directory. This is the correct form for login-once automation.

For automation:

- Use a purpose-specific directory not used by ordinary Chrome.
- Do not point at Chrome's normal user-data root or any named personal profile.
- Treat the directory as sensitive; it contains authenticated browser state.
- Keep it out of repositories, archives, sync folders, and shared output.
- Never use one profile concurrently from two sessions or browser processes. Chrome profile locks make this unreliable and can corrupt state.
- For concurrent identities or tasks, use separate profile directories. Do not copy an authenticated profile without explicit authorization because the copy duplicates session credentials.

Chrome may create an internal subdirectory named `Default` inside a dedicated user-data directory. That is normal and is not the person's ordinary Chrome `Default` profile.

## Named session rules

Never use the implicit `default` session for agent work. It is a shared namespace and another task can navigate it or reuse its refs.

For one designated browser, configure a stable semantic session. For project/task isolation, derive a stable ID:

```bash
agent-browser session id --scope worktree --prefix <purpose>
```

Set it once for the whole workflow through config, `AGENT_BROWSER_SESSION`, or `--session` on every command. Check it before actions:

```bash
agent-browser session
agent-browser session info --json
agent-browser session list
```

A session name alone does not persist auth after daemon shutdown. Pair it with a dedicated profile or `--restore`.

When multiple sessions attach to one user-requested CDP browser, cookies/storage are shared. Add `--pin-tab` so each session stays bound to its tab and fails with `tab_gone` if that tab closes. This is tab isolation, not authentication isolation.

## Choose a persistence strategy

| Need | Strategy |
|---|---|
| Login once in a real installed browser; preserve full browser state | Dedicated profile path |
| Isolated session with cookies/localStorage restored automatically | Stable session + `--restore` |
| Portable, explicit state transfer | `state save/load` only when requested; treat the file as a secret |
| Temporary anonymous/CI run | Named session without profile/restore, then close |
| Reuse a personal running browser | Only when explicitly requested; `--auto-connect`/`--cdp` has a larger trust boundary |

`--restore` periodically saves state and saves again on close, idle timeout, daemon shutdown, and compatible relaunch. The default `--restore-save auto` avoids overwriting a known-good state after a failed restore or failed validation. Use `--restore-check-url`, `--restore-check-text`, or `--restore-check-fn` for important authenticated sessions.

State files and restore state can expose tokens. Keep paths out of shared locations and repositories. If encrypted restore/state is required, have the user provide `AGENT_BROWSER_ENCRYPTION_KEY` through an appropriate secret manager or private environment; never read or print it.

## Login once, then use headless

Use the same profile and session throughout:

1. Configure the installed browser, dedicated profile, semantic session, and headed mode.
2. Open the login URL in headed mode.
3. Let the user enter passwords, approve OAuth, and complete MFA/CAPTCHA manually. Do not request secrets in chat or type them from shell arguments.
4. Verify a non-sensitive authenticated marker such as the expected URL, account-menu label, or dashboard heading.
5. Close only that session to release the profile lock, or keep it open if the user asked.
6. Switch the whole run to headless mode and reopen the protected page with the same profile/session.

Example temporary override:

```bash
agent-browser --headed false open https://example.com/dashboard
agent-browser --headed false wait --url "**/dashboard"
agent-browser --headed false snapshot -i -c
```

Repeat `--headed false` on every command when a persistent config says `headed: true`. Otherwise the next command can see a launch-hash mismatch, relaunch headed, and leave a blank tab. A separate headless config is safer for multi-command automation.

## Lifecycle state

| State/event | Expected behavior |
|---|---|
| First compatible command | Starts the named daemon and browser |
| Later command with the same launch configuration | Reuses the daemon, browser, profile, tabs, and refs until page state changes |
| Command with conflicting launch configuration | May relaunch the browser; transient tabs/page state can reset |
| Headless session idle for the configured timeout | Saves configured restore state, closes, and exits |
| `idleTimeout: "0"` | Disables idle shutdown; use only for an explicitly requested keep-alive session |
| `close` | Closes the selected session and releases the profile; the persistent profile remains on disk |
| `close --all` | Closes unrelated sessions too; never use for routine cleanup |

Headed, user-attached, and some WebDriver sessions are exempt from the default one-hour cleanup, but an explicit idle timeout applies more broadly. Inspect actual state with `session info --json` rather than assuming.

Close ephemeral task sessions when finished. Do not close a designated long-lived session when the user asked to keep it running. Never clear cookies/state, delete a profile, or run `doctor --fix` as a cleanup shortcut.

## Parallel work

Full isolation requires a distinct session and, when persistent profiles are used, a distinct profile directory per concurrently running browser.

Safe patterns:

- Same public site, no auth: separate named sessions without profiles.
- Multiple test identities: separate named sessions and separate profile directories.
- One authenticated identity: serialize tasks through one designated session/profile, or ask before duplicating auth state.
- Shared CDP browser: distinct named sessions + `--pin-tab`, understanding that cookies and storage are still shared.

Do not use shell background jobs against the same profile.

## Diagnostics

```bash
agent-browser session info --json
agent-browser session list
agent-browser doctor --offline --quick
```

Use full `doctor` when local-only checks are insufficient. `doctor --fix` can reinstall the browser and purge state; it requires explicit permission.

If a Chrome-for-Testing banner appears despite a system-browser requirement:

1. Confirm the configured executable path exists.
2. Confirm the intended non-default session is active; an older `default` daemon may still own the visible window.
3. Close only the obsolete session.
4. Relaunch with the dedicated config and verify the session again.

## Containment limitation

`--allowed-domains` cannot be combined with persistent profiles, restore/state replay, CDP/auto-connect, raw startup profile arguments, and several providers because requests can occur before agent-browser installs its filters. Do not weaken authentication isolation to make the flag fit. When authenticated browsing requires strong outbound containment, use a dedicated OS user, VM/container, or host-level egress control.
