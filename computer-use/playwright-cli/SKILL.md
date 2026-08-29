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
6. **Prefer `eval` and `snapshot` over `screenshot`** for data extraction — they are more token-efficient and machine-readable. Use `screenshot` only when visual context is specifically needed (layout debugging, visual comparison).

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

# 5. Evaluate specific properties
playwright-cli eval "document.title"
playwright-cli eval "getComputedStyle(document.querySelector('.broken-element')).display"
playwright-cli eval "el => ({ width: el.offsetWidth, height: el.offsetHeight, visible: el.offsetParent !== null })" e5
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

| Page Type | Strategy |
|-----------|----------|
| Static content, single page | `snapshot` + `eval` for targeted extraction |
| Dynamic/SPA content | `snapshot` after waiting for network idle |
| Paginated list | Loop with `click` on next + `snapshot` each page |
| Infinite scroll | `mousewheel` loop + `eval` to check new content |
| Multi-page (known URLs) | `run-code` with page loop for efficiency |
| Behind authentication | Auth first, then any of the above |
| Data in API responses | `network` to find API endpoints, then `eval` with `fetch()` |

### Single-Page Extraction

```bash
playwright-cli open https://example.com/data-page
playwright-cli snapshot

# Extract specific data via eval
playwright-cli eval "JSON.stringify([...document.querySelectorAll('table tbody tr')].map(row => ({
  name: row.cells[0]?.textContent?.trim(),
  value: row.cells[1]?.textContent?.trim(),
  date: row.cells[2]?.textContent?.trim()
})))"
```

### Table Extraction

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

```bash
# Manual pagination — snapshot each page, click next
playwright-cli open https://example.com/list?page=1
playwright-cli snapshot
# ... extract data from snapshot ...
playwright-cli click e42  # "Next" button
playwright-cli snapshot
# ... repeat ...

# Efficient: use run-code for multi-page scraping
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

```bash
# Check element visibility
playwright-cli eval "document.querySelector('.success-message') !== null"

# Check text content
playwright-cli eval "document.querySelector('h1').textContent.trim()"

# Check URL after navigation
playwright-cli eval "window.location.href"

# Check element count
playwright-cli eval "document.querySelectorAll('.list-item').length"

# Complex assertions via run-code
playwright-cli run-code "async page => {
  const count = await page.locator('.item').count();
  const title = await page.title();
  const url = page.url();
  return { count, title, url, passed: count > 0 && title.includes('Dashboard') };
}"
```

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
