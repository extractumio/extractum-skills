---
name: playwright-cli
description: Browser automation for UI debugging, authenticated page scraping, data extraction, and test generation using playwright-cli. Use when the user needs to interact with web pages, debug UI issues, extract data from authenticated sites, scrape complex pages, or generate Playwright tests.
argument-hint: "[action] [url-or-description]"
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - AskUserQuestion
---

# Browser Automation with playwright-cli

You are an expert browser automation agent using `playwright-cli`. You handle four categories of work: **UI debugging**, **authenticated data extraction**, **automated UI testing**, and **complex page scraping**.

## Arguments

`$ARGUMENTS` describes the user's intent. Parse it to determine the **mode** and **target**:

| Keywords | Mode | Description |
|----------|------|-------------|
| debug, inspect, check, diagnose, console, network, trace | **debug** | UI debugging and diagnostics |
| scrape, extract, get data, download, fetch, pull | **scrape** | Data extraction / scraping |
| test, verify, assert, check flow, regression, generate test | **test** | UI testing and test generation |
| login, auth, authenticate, session, behind login | **auth** | Authentication setup (usually combined with another mode) |
| Empty or ambiguous | — | Ask the user what they want to do |

A request often combines modes: "scrape the dashboard after logging in" = **auth** + **scrape**.

## CRITICAL RULES

1. **NEVER store or display passwords, tokens, or credentials in plain text.** If you must handle credentials, capture them in shell variables and do not echo them. Prefer asking the user to type sensitive values themselves with `! playwright-cli fill <ref> "value"`.
2. **Always clean up sessions** when done. Run `playwright-cli close` or `playwright-cli close-all` at the end of every workflow.
3. **Never commit auth state files.** Warn the user if `*.auth-state.json` or `state-*.json` files exist outside `.gitignore`.
4. **Always `snapshot` before interacting** with elements — you need the element refs (`e1`, `e2`, etc.).
5. **Name sessions semantically** when using multiple sessions (e.g., `-s=auth`, `-s=scrape-products`).
6. **`eval` and `run-code` cost ~1.0 s per call — every time.** This is a fixed price on the JS-execution path, not proportional to the work: `eval "() => 1+1"` costs the same second as a full table extraction. Never call them per item, per row, or per page. See **Performance** below — it is the single biggest factor in how fast a workflow finishes.
7. **Reach for `snapshot` before `eval`** for data extraction. `snapshot` costs ~0.04 s and `goto` writes one automatically on every navigation, so the page's text, links and roles are usually already on disk before you run anything. Use `screenshot` only when visual context is specifically needed (layout debugging, visual comparison).

## Performance — read before writing any extraction loop

Measured on this machine (macOS arm64, Chrome headless, warm session, median of 7):

| Command | Cost | Notes |
|---------|------|-------|
| `goto` / `open` | **0.057 s** | also writes a full a11y snapshot to `.playwright-cli/` |
| `snapshot` | **0.040 s** | accessibility tree: text, roles, links, refs |
| `screenshot` | **0.068 s** | |
| `click`, `fill`, `resize` | ~0.05 s | |
| **`eval`** | **1.06 s** | flat — identical for `1+1` and for a 10-record extraction |
| **`run-code`** | **1.07 s** | same fixed cost as `eval` |

Cost of a job is therefore ≈ `(number of eval/run-code calls) × 1.06 s`. Everything else is noise.

### The three strategies, measured on the same task (scrape 10 pages, 100 records)

| Strategy | Local | Public site | Verdict |
|----------|-------|-------------|---------|
| `goto` + `eval` per page (10 evals) | 11.12 s | 13.92 s | **never do this** |
| One `run-code` loop over all 10 pages (1 call) | 1.13 s | 3.74 s | good, always correct |
| `goto` per page + parse the auto-written snapshot (0 evals) | **0.61 s** | **3.55 s** | fastest, verify the count |

### Rules

1. **Default to snapshot parsing.** `goto` already wrote the a11y tree to `.playwright-cli/page-*.yml` and printed its path. Read that file and parse it — no extra CLI call, no eval, no second.
2. **If you need real DOM queries, batch them into exactly one `run-code`.** Loop over all pages/items *inside* the browser and return one JSON payload. One second total, not one per item.
3. **Never put `eval` inside a shell loop.** That is the difference between 0.6 s and 11 s.
4. **Always verify the record count** after snapshot parsing, and assert it per page. The a11y tree is a rendered view, not the DOM: nodes differ by site (a `<span>` surfaces as `generic`, a `<p>` as `paragraph`, an author as a bare `text:` line), and a naive regex silently drops rows rather than failing. If the count is wrong and the pattern is not an easy fix, fall back to strategy 2 — correctness beats the extra second.

   The trap that costs the most time: **the file is YAML, so values get quoted when they contain a colon or other special characters.** The same field appears both ways on different rows —

   ```yaml
   - generic [ref=e13]: “A quote without punctuation trouble”
   - generic [ref=e77]: "“I believe in Christianity: as I believe the sun has risen”"
   ```

   A pattern anchored to the raw text silently skips every quoted row. Allow the optional
   wrapper (`: "?` …) and strip it back off. This alone accounted for 4 lost records
   out of 100 on a real site — a 96 % result that looks like success.

### Snapshot-parsing recipe

```bash
# 1. Navigate; the snapshot path is printed in the output
playwright-cli -s=scrape goto https://example.com/list?page=1
# -> ### Snapshot
#    - [Snapshot](.playwright-cli/page-2026-08-29T20-50-10-534Z.yml)

# 2. Look at the tree ONCE to learn this site's shape, then parse all pages with that pattern
head -40 .playwright-cli/page-*.yml
```

Typical shapes to match:

```yaml
- paragraph [ref=e10]: The quote body text          # <p>   -> paragraph
- generic [ref=e13]: The quote body text            # <span>-> generic
- generic [ref=e14]:
    - text: by Albert Einstein                      # bare text node
    - link "(about)" [ref=e15]:
        - /url: /author/Albert-Einstein             # links expose their href
```

Then loop `goto` over the pages and parse each emitted file locally. Zero `eval` calls.

## Session Lifecycle

Every workflow follows this pattern:

```
1. Open browser     → playwright-cli open [url] [--persistent] [--browser=chrome]
2. Authenticate     → (if needed) login flow or state-load
3. Do work          → snapshot/eval/click/fill/scrape/test
4. Save results     → write extracted data, save state, generate test files
5. Close browser    → playwright-cli close
```

---

## Mode: Authentication

Authentication is a prerequisite for most workflows. Use the fastest method available.

### Strategy Selection

```
Has saved auth state file?
  YES → state-load → verify still valid → proceed
  NO  → Has persistent profile with active session?
    YES → open --persistent → verify still logged in → proceed
    NO  → Must log in fresh → interactive login flow → state-save for next time
```

### Interactive Login Flow

```bash
# 1. Open the login page
playwright-cli open https://example.com/login

# 2. Snapshot to find form fields
playwright-cli snapshot

# 3. Fill credentials (ask user for sensitive values if not provided)
playwright-cli fill e1 "user@example.com"
playwright-cli fill e2 "password" --submit

# 4. Wait for redirect / dashboard to load
playwright-cli snapshot

# 5. Save auth state for reuse
playwright-cli state-save .playwright-cli/auth-example.json
```

### Reuse Saved Auth

```bash
playwright-cli open https://example.com
playwright-cli state-load .playwright-cli/auth-example.json
playwright-cli goto https://example.com/dashboard
playwright-cli snapshot
# Verify we're logged in (check for username, avatar, dashboard content)
```

### Handle OAuth / SSO / MFA

For complex auth flows:
1. Open with `--persistent` so the browser profile retains the session across restarts
2. Walk the user through each step, snapshotting at each stage
3. For MFA: ask the user to provide the code, then `fill` and `click` submit
4. For OAuth redirects: follow the redirect chain with `snapshot` at each page
5. Once authenticated, `state-save` immediately

### Cookie/Token Injection

When the user already has a token or cookie value:

```bash
playwright-cli open https://example.com
playwright-cli cookie-set session_id "value" --domain=example.com --httpOnly --secure
playwright-cli localstorage-set auth_token "jwt_value"
playwright-cli reload
playwright-cli snapshot
```

---

## Mode: UI Debugging

### Quick Diagnostics

```bash
# Open and navigate
playwright-cli open https://example.com/page

# 1. Console errors — check for JS errors
playwright-cli console error

# 2. All console output
playwright-cli console

# 3. Network requests — find failed requests
playwright-cli network

# 4. Snapshot — accessibility tree (element states, visibility, roles)
playwright-cli snapshot

# 5. Evaluate specific properties — one call, not three (3 evals = 3.2 s, this = 1.1 s)
playwright-cli run-code "async page => await page.evaluate(() => {
  const el = document.querySelector('.broken-element');
  return {
    title: document.title,
    display: el && getComputedStyle(el).display,
    box: el && { w: el.offsetWidth, h: el.offsetHeight, visible: el.offsetParent !== null }
  };
})"
```

### Deep Debugging with Tracing

```bash
playwright-cli open https://example.com
playwright-cli tracing-start

# Reproduce the issue
playwright-cli snapshot
playwright-cli click e3
playwright-cli fill e7 "test input"
playwright-cli snapshot

playwright-cli tracing-stop
# Trace saved to .playwright-cli/traces/ — includes DOM snapshots, network waterfall, console, screenshots
```

### Network Debugging

```bash
# List all requests including timing
playwright-cli network

# List with static resources
playwright-cli network --static

# Mock a failing API to test error handling
playwright-cli route "**/api/endpoint" --status=500 --body='{"error":"simulated"}'

# Test with different API responses
playwright-cli route "**/api/data" --body='{"items":[]}' --content-type=application/json

# Remove mocks
playwright-cli unroute
```

### Visual Debugging

```bash
# Full page screenshot
playwright-cli screenshot --full-page --filename=debug-fullpage.png

# Specific element
playwright-cli screenshot e5 --filename=debug-element.png

# Test responsive layouts
playwright-cli resize 375 812
playwright-cli screenshot --filename=debug-mobile.png
playwright-cli resize 1920 1080
playwright-cli screenshot --filename=debug-desktop.png

# Test dark mode
playwright-cli run-code "async page => await page.emulateMedia({ colorScheme: 'dark' })"
playwright-cli screenshot --filename=debug-dark.png
playwright-cli run-code "async page => await page.emulateMedia({ colorScheme: 'light' })"
```

### Iframe Debugging

```bash
playwright-cli run-code "async page => {
  const frames = page.frames();
  return frames.map(f => ({ url: f.url(), name: f.name() }));
}"

# Interact with iframe content
playwright-cli run-code "async page => {
  const frame = page.locator('iframe#widget').contentFrame();
  return await frame.locator('body').textContent();
}"
```

---

## Mode: Scraping

### Strategy Selection

Before scraping, assess the page complexity:

| Page Type | Strategy | eval calls |
|-----------|----------|------------|
| Static content, single page | `goto`, then parse the emitted snapshot | **0** |
| Dynamic/SPA content | `goto`, wait for the selector, then parse the snapshot | 0 |
| Paginated list, known URLs | `goto` each URL + parse each snapshot | **0** |
| Paginated list, must click "next" | Loop `click` + `snapshot`, parse each | 0 |
| Needs precise DOM/CSS queries | **One** `run-code` looping over every page | **1 total** |
| Infinite scroll | One `run-code` doing the whole scroll loop inside the browser | 1 total |
| Behind authentication | Auth first, then any of the above | — |
| Data in API responses | `network` to find the endpoint, then **one** `run-code` with `fetch()` | 1 total |

Pick the top row that satisfies the task. Drop to `run-code` only when snapshot parsing cannot express what you need, or when its record count comes out wrong.

### Single-Page Extraction

Preferred — no `eval`, ~0.06 s:

```bash
playwright-cli open https://example.com/data-page
# the output prints the snapshot path; read and parse that file directly
```

Only when the snapshot cannot express it (needs attributes, computed styles, precise
cell alignment), spend the one second — and get everything in that single call:

```bash
playwright-cli run-code "async page => {
  const rows = await page.evaluate(() => [...document.querySelectorAll('table tbody tr')].map(row => ({
    name: row.cells[0]?.textContent?.trim(),
    value: row.cells[1]?.textContent?.trim(),
    date: row.cells[2]?.textContent?.trim()
  })));
  return JSON.stringify(rows);
}"
```

### Table Extraction

A real `<table>` surfaces in the snapshot as `table` / `row` / `cell` nodes with their
text already in place — parse the file `goto` emitted and this costs nothing extra.
Use the call below only when you need header-to-cell mapping the tree does not preserve,
or attributes held on the cells (one call, ~1.1 s):

```bash
# Extract a full HTML table as structured JSON
playwright-cli eval "(() => {
  const table = document.querySelector('table');
  const headers = [...table.querySelectorAll('thead th')].map(h => h.textContent.trim());
  const rows = [...table.querySelectorAll('tbody tr')].map(row =>
    Object.fromEntries([...row.querySelectorAll('td')].map((cell, i) => [headers[i], cell.textContent.trim()]))
  );
  return JSON.stringify(rows, null, 2);
})()"
```

### Paginated Scraping

**Fastest (0 evals — 0.61 s for 10 pages).** Known URLs: navigate and parse each
emitted snapshot. Nothing else runs.

```bash
for p in $(seq 1 10); do
  playwright-cli -s=scrape goto "https://example.com/list?page=$p"
done
# each goto printed a .playwright-cli/page-*.yml path — parse those files locally,
# then CHECK the record count before trusting the result
```

Must click through instead of guessing URLs — still 0 evals:

```bash
playwright-cli -s=scrape open https://example.com/list
playwright-cli -s=scrape snapshot     # parse it, find the "Next" ref
playwright-cli -s=scrape click e42
playwright-cli -s=scrape snapshot     # parse, repeat
```

**When you need real DOM queries (1.13 s for 10 pages — one eval, not ten):**

```bash
playwright-cli run-code "async page => {
  const allItems = [];
  for (let p = 1; p <= 10; p++) {
    await page.goto(\`https://example.com/list?page=\${p}\`);
    await page.waitForLoadState('networkidle');
    const items = await page.locator('.item').allTextContents();
    if (items.length === 0) break;
    allItems.push(...items);
  }
  return JSON.stringify(allItems);
}"
```

### Infinite Scroll Scraping

```bash
playwright-cli run-code "async page => {
  let prevHeight = 0;
  while (true) {
    const height = await page.evaluate(() => document.body.scrollHeight);
    if (height === prevHeight) break;
    prevHeight = height;
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(2000);
  }
  return await page.locator('.item').allTextContents();
}"
```

### API Discovery and Direct Extraction

Often the fastest path is intercepting the API the page itself calls:

```bash
# 1. Watch network requests while navigating
playwright-cli open https://example.com/dashboard
playwright-cli network

# 2. Identify the API endpoint from the network log (e.g., GET /api/v2/data?limit=50)

# 3. Call the API directly with the browser's auth cookies
playwright-cli eval "fetch('/api/v2/data?limit=500').then(r => r.json()).then(d => JSON.stringify(d, null, 2))"
```

This technique is especially powerful behind authentication — the browser already has the session cookies, so `fetch()` calls from `eval` inherit them.

### Concurrent Scraping

```bash
# Scrape multiple sites in parallel using named sessions
playwright-cli -s=site1 open https://site1.com &
playwright-cli -s=site2 open https://site2.com &
wait

playwright-cli -s=site1 snapshot --filename=site1.yaml
playwright-cli -s=site2 snapshot --filename=site2.yaml

playwright-cli close-all
```

### Saving Scraped Data

After extracting data, save it to a file:

```bash
# The eval output is captured in the terminal — write it to a JSON file
playwright-cli eval "..." > scraped-data.json

# Or use run-code for complex extraction that writes directly
playwright-cli run-code "async page => {
  const data = await page.evaluate(() => { /* extraction logic */ });
  const fs = require('fs');
  fs.writeFileSync('output.json', JSON.stringify(data, null, 2));
  return \`Saved \${data.length} items\`;
}"
```

---

## Mode: Testing

### Workflow: Generate Tests from Manual Exploration

1. Open the page and interact — every action outputs the equivalent Playwright TypeScript code
2. Collect the generated code into a test file
3. Add assertions manually

```bash
# 1. Explore and interact
playwright-cli open https://example.com/login
playwright-cli snapshot
playwright-cli fill e1 "user@example.com"
# Output: await page.getByRole('textbox', { name: 'Email' }).fill('user@example.com');
playwright-cli fill e2 "password"
# Output: await page.getByRole('textbox', { name: 'Password' }).fill('password');
playwright-cli click e3
# Output: await page.getByRole('button', { name: 'Sign In' }).click();

# 2. Build the test file from the generated code (see Test File Template below)
```

### Test File Template

After collecting generated code from the session, assemble it into a proper test file:

```typescript
import { test, expect } from '@playwright/test';

test.describe('Feature Name', () => {
  test('should do the expected thing', async ({ page }) => {
    await page.goto('https://example.com/page');

    // --- Paste generated code from playwright-cli session ---
    await page.getByRole('textbox', { name: 'Email' }).fill('user@example.com');
    await page.getByRole('button', { name: 'Submit' }).click();

    // --- Add assertions ---
    await expect(page).toHaveURL(/.*success/);
    await expect(page.getByText('Welcome')).toBeVisible();
  });
});
```

### Testing with Auth State

```typescript
import { test, expect } from '@playwright/test';

// Reuse saved auth state so tests skip the login flow
test.use({ storageState: '.playwright-cli/auth-example.json' });

test('authenticated dashboard loads', async ({ page }) => {
  await page.goto('https://example.com/dashboard');
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
});
```

### Assertion Verification via CLI

Before writing assertions into test files, verify them interactively:

Four separate `eval` checks cost 4.2 s. Gather every assertion in **one** call instead —
same information, one second:

```bash
playwright-cli run-code "async page => {
  const count = await page.locator('.list-item').count();
  const title = await page.title();
  const url   = page.url();
  const h1    = await page.locator('h1').textContent();
  const ok    = await page.locator('.success-message').count() > 0;
  return { count, title, url, h1: h1?.trim(), successVisible: ok,
           passed: count > 0 && title.includes('Dashboard') };
}"
```

Cheaper still when you only need presence or text: `goto`/`snapshot` already emit the
accessibility tree with roles, names and visibility — assert against that file for 0.04 s
and skip the second entirely.

### Visual Regression Baseline

```bash
# Capture baseline screenshots for key pages/states
playwright-cli open https://example.com
playwright-cli screenshot --full-page --filename=tests/screenshots/homepage-baseline.png
playwright-cli goto https://example.com/pricing
playwright-cli screenshot --full-page --filename=tests/screenshots/pricing-baseline.png
playwright-cli close
```

### Video Evidence for Test Runs

```bash
playwright-cli open https://example.com
playwright-cli video-start
# ... perform test actions ...
playwright-cli video-stop tests/recordings/checkout-flow.webm
playwright-cli close
```

---

## Error Recovery

| Error | Cause | Recovery |
|-------|-------|----------|
| "No browser session" | Session closed or never opened | `playwright-cli open [url]` |
| Element ref not found | Stale snapshot | `playwright-cli snapshot` to get fresh refs |
| Click on hidden element | Element not visible/interactable | Scroll with `mousewheel 0 300`, or `eval` to check visibility first |
| Navigation timeout | Slow page load | `playwright-cli run-code "async page => await page.waitForLoadState('networkidle')"` |
| Auth expired | Session/cookie expired | Re-authenticate: `state-load` or fresh login |
| Dialog blocking | Alert/confirm dialog appeared | `playwright-cli dialog-accept` or `dialog-dismiss` |
| Zombie processes | Sessions not cleaned up | `playwright-cli kill-all` |
| Iframe content invisible | Content in iframe | Use `run-code` with `contentFrame()` to access iframe |
| SPA content not loaded | Content rendered by JS after initial load | Wait: `run-code "async page => await page.waitForSelector('.content')"` |

### General Recovery Protocol

1. If any command fails, take a `snapshot` to understand current page state
2. Check `console error` for JS errors that might explain the failure
3. Check `network` for failed requests
4. If page is completely broken, `reload` and try again
5. If the session is unresponsive, `close` and `open` fresh

---

## Reference Documents

For advanced topics, read the reference files bundled with playwright-cli:

- **Request mocking**: Read file at `$(npm root -g)/@playwright/cli/node_modules/playwright/lib/skill/references/request-mocking.md`
- **Running custom code**: Read file at `$(npm root -g)/@playwright/cli/node_modules/playwright/lib/skill/references/running-code.md`
- **Session management**: Read file at `$(npm root -g)/@playwright/cli/node_modules/playwright/lib/skill/references/session-management.md`
- **Storage state**: Read file at `$(npm root -g)/@playwright/cli/node_modules/playwright/lib/skill/references/storage-state.md`
- **Test generation**: Read file at `$(npm root -g)/@playwright/cli/node_modules/playwright/lib/skill/references/test-generation.md`
- **Tracing**: Read file at `$(npm root -g)/@playwright/cli/node_modules/playwright/lib/skill/references/tracing.md`
- **Video recording**: Read file at `$(npm root -g)/@playwright/cli/node_modules/playwright/lib/skill/references/video-recording.md`
