---
name: make-mvp
description: >
  Plan, prioritize, and sequence implementation work so every iteration ends
  with a working version the end-user can already benefit from. Applies a
  "progressive JPEG" principle — start with a thin core scaffold that already
  delivers real value, then add features in order of user-visible value. Bugs
  in already-shipped functionality come first; engineering for hypothetical
  future steps is rejected. Triggers on "what's next", "plan the next steps",
  "prioritize features", "order of implementation", "MVP plan", "roadmap",
  "what should ship first", or any other request that orders multi-step work.
author: Greg Z.
author_email: info@extractum.io
author_url: https://www.linkedin.com/in/gregzem/
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
---

# make-mvp — Value-First, Progressive Planning

Use this skill any time you plan, prioritize, or sequence implementation work. It enforces three non-negotiable rules on every plan:

1. **Each step ends in a working version** that an end-user can use right now — not a stub, not a half-built scaffold for later steps.
2. **The next step is whichever delivers the most user-visible value for the smallest scope** — not the most natural build order, not the most elegant abstraction, not what's easiest for the engineer.
3. **Every step is communicated to the user as a concrete deliverable** — a named artifact, the capability it unlocks in user terms, and a way the user can verify the value landed. No engineering-jargon step titles, no "phase 2 of the platform".

## When to invoke

- The user asks "what should I do next?", "plan the next steps", "prioritize the backlog", "what's the MVP?", "what should ship first?", "what's the order of implementation?".
- The user has a backlog, feature list, roadmap, or half-built project and needs sequencing.
- The user is starting a new project and wants to know how to phase it.
- You are about to propose a multi-step plan in any other context — apply this skill silently to validate the order before you present it.

If the request is a single feature or single bug with no sequencing involved, this skill does not apply.

## Core principles

### 1. Bugfix first

Open bugs in already-shipped functionality come **before** new features. A broken core experience destroys the value of every feature stacked on top of it. If users currently rely on something and it's broken, fix that before adding anything new.

Exception: a bug that affects a code path no real user reaches yet can wait — it's a defect in unshipped work, treat it as part of the relevant feature step.

### 2. Progressive JPEG

Like a progressive JPEG image, the project should be **usable at every level of detail** and refine over time:

- **Iteration 0 — Core scaffold.** The thinnest possible vertical slice that boots, accepts input, and produces *some* output the user finds valuable. A real, working, end-to-end path through the system, however narrow. Not a mock. Not a stub. Not "everything except the part that does the thing".
- **Iterations 1..N — Add value.** Each iteration adds one piece of real user-visible value (a feature, an input format, an integration, a fix). After each iteration the user can keep using the previous version *and* gets the new thing.

What this is NOT:

- Building scaffolding for features that won't ship in this iteration.
- Designing abstractions for "later" steps.
- Engineering for hypothetical future requirements.
- Refactoring before there is a working thing to refactor.

### 3. Value-first ordering

The order is determined by **value to the end-user per unit of scope**, not by:

- Technical elegance.
- "Natural" build order ("we should do auth first because everything will need it" — only if the *first* usable feature actually requires logged-in users).
- Internal team preference.
- What's easiest to build.

Always ask: *"If we shipped only this one next step, would the user get something they didn't have before that they actually want?"* If no, the step is wrong, too big, or out of order.

### 4. Limited scope per step

Each step must be small enough that:

- It can be implemented and verified in a short time (hours to a day, not weeks).
- It produces a working, runnable version at the end.
- It doesn't depend on a *future* step also being done to be useful.

If a step requires another step to also be done before any user can benefit, the two are actually one step — and probably too big.

## Procedure

### 0. Anchor on the end-user before sequencing anything

Value ranking is meaningless without a clear picture of *whose* value. Before you touch the backlog, get explicit on three things — ask the user if any are missing or unclear:

- **Who is the end-user?** Be specific. ("Solo indie developer using the CLI from a terminal" is useful; "developers" is not.) If there are multiple user types, name the *primary* one for this round of planning.
- **What outcome do they care about most?** The job they want done, in their words. Not "uses the export feature" but "gets a clean spreadsheet they can hand to their accountant".
- **What would make them say 'this was worth it' after the very first usable version?** This is your bar for Iteration 0.

Write these down in one short paragraph and reflect it back to the user before you sequence. If the user disagrees with your read, fix it now — every later prioritization decision rides on this. A wrong end-user picture produces a confidently wrong plan.

### 1. Inventory the candidate work

Pull from:

- The user's stated backlog or feature list.
- TODOs, issues, or tickets in the project if you can read them.
- Known bugs in the current codebase or in already-shipped features.

If the inventory isn't clear, ask the user to list the candidates before sequencing. Don't invent items.

### 2. Separate bugs from features

- **Fix-first pile:** open bugs in already-shipped functionality.
- **Value-rank pile:** features, enhancements, refactors, infrastructure work.

### 3. Sort the fix-first pile by impact

Within bugfixes, order by how broken the user's experience currently is:

1. Hard blockers (core feature unusable).
2. Frequent paper-cuts (works but degrades trust on every use).
3. Edge cases (rare path, low impact).

### 4. Sort the value-rank pile

For each candidate, write down five things:

- **User-value sentence** — one sentence in *user* terms. ("User can export results to CSV." Not: "Implement export module.")
- **Value tier** — High / Medium / Low, judged against the end-user's primary outcome from Step 0:
  - *High*: directly advances the primary outcome, or removes a blocker that prevents the user from getting it. Most users feel the difference.
  - *Medium*: improves an existing capability the user already gets value from (faster, more accurate, less friction), but they could live without it.
  - *Low*: nice-to-have, edge case, or only matters to a small subset of users. Often a candidate to drop, not just defer.
- **One-line value rationale** — *why* it has that tier, in user terms. ("Most users currently fall back to manual copy-paste — this removes the step they hate most.") This is what you'll show the user; if you can't write it, the value isn't really there.
- **Standalone or dependent?** — if it depends on another item, the dependency moves up *only if the dependency itself ships user value when done alone*. If the dependency is pure plumbing, the two are one step (and probably too big — cut scope).
- **Smallest scope that makes the value sentence true** — strip anything that isn't required for that sentence.

Rank by value-per-scope, with tier as the primary key and scope as the tiebreaker. High-value-small-scope wins the next slot. A High that requires huge scope may lose to a Medium that ships in a day — note the tradeoff explicitly so the user can decide.

### 5. Sanity-check every step

For each step in the proposed order, confirm:

- [ ] Is there still a working version of the app/service after this step? (must be yes)
- [ ] Does this step deliver value the user couldn't get before? (must be yes — otherwise it's scaffolding for later; push down or drop)
- [ ] Could this step be cut in half and still ship value? (if yes, cut it)
- [ ] Are we building anything here only because a *later* step will need it? (if yes, remove that part — add it when the later step actually arrives)

If any answer fails, rework the step before presenting.

### 6. Stress-test with the `pragmatic` agent

Before showing the plan to the user, delegate a review to the `pragmatic` agent (via the Agent tool, `subagent_type: "pragmatic"`). The pragmatic agent is the second pair of eyes that decides **what is truly valuable for the end-user** and flags over-engineering, premature abstraction, hidden scaffolding, and steps the user wouldn't actually pay for.

Brief the agent like a colleague who hasn't seen this conversation. Hand it:

- The candidate plan as you'd present it to the user (ordered steps, scope, working version per step).
- Any context it needs to judge user value: who the end-user is, what they're trying to accomplish, any stated constraints (deadline, budget, single-developer, etc.).
- The explicit ask: *"Stress-test this plan against real user value. For each step, tell me: is this what a user actually wants next, or are we engineering for ourselves? Flag anything over-scoped, anything that's scaffolding for later, and anything we could cut without losing the value sentence. If the order is wrong, say so."*

Apply the agent's findings:

- If it identifies steps with no real user pain point, push them down or drop them.
- If it flags over-engineering inside a step, cut that scope.
- If it proposes a simpler alternative that still ships the value sentence, take it.
- If it acknowledges a step is genuinely well-targeted, leave it as-is.

Do **not** treat the pragmatic agent's output as the final plan to present — integrate its critique, then move to step 7. If the agent's critique substantially changes the order, re-run step 5 (sanity-check) on the revised plan.

### 7. Present the plan

Open with a one-paragraph recap of who the end-user is and the outcome you're optimizing for (from Step 0). This frames every prioritization decision below — without it, the user can't tell whether you ranked things correctly.

Then present the result as an ordered list. For each step, communicate the deliverable in user terms — not engineering terms:

- **Step N — <one-line user-value sentence>** *(Value: High / Medium / Low)*
- **Deliverable** — the concrete artifact the user receives (e.g., "a `pdf2md` binary on macOS and Linux", "a hosted page at /convert", "a `--batch` flag added to the existing CLI"). Name the thing, don't describe the work.
- **What the user can now do** — the new capability stated as the user would say it. ("I can drop a folder of PDFs in and get a folder of `.md` files back.")
- **Why this rank** — the one-line value rationale from Step 4. This is the part that justifies the order; do not skip it.
- **In scope** — short bullets, only what's needed for the value sentence.
- **Out of scope** — what you are explicitly NOT doing in this step (so it doesn't sneak back in).
- **Working version at the end** — the state of the product after this step ships, in one line.
- **How the user verifies the value** — the smallest thing the user can do to confirm the deliverable lands. ("Run it on one of your own PDFs and check the Markdown is readable.") If you can't write this, the step is not really shipping value.

Keep the plan short. If it has more than about five steps, you are planning too far ahead — detail the next 2–3 with full structure above, and leave the rest as one-liners (value sentence + tier only) until they become the next 2–3.

End with an explicit decision prompt: *"Does this match what you want the user to get first? If the end-user picture or the top-ranked deliverable is wrong, say so before we start — re-ordering now is free, re-ordering after Step 1 isn't."*

### 8. Wait for the user's go-ahead

This skill produces a plan. It does **not** start implementing. The user picks which step to begin and gives the green light. If you produced this plan inside a larger task, surface it before writing code.

## Anti-patterns to reject

Call these out whenever they appear in a proposed plan — yours or anyone else's:

- **"First we'll set up the database schema, then the API, then the UI."** No — first ship a vertical slice with one entity, one endpoint, one screen. Add entities as features demand them.
- **"Step 1: build the auth system."** Only if the *very first* usable feature genuinely requires logged-in users. Otherwise auth arrives when a feature actually needs it.
- **"Let's add a plugin system so future features are easier."** No — add the next feature directly. Extract a plugin system if and when a third feature wants the same shape.
- **"We'll refactor X first to make adding Y easier."** Only if Y is the *immediate* next step and the refactor is genuinely required. Otherwise add Y the messy way and refactor when a third caller appears.
- **"Phase 1: foundations. Phase 2: features."** No phases. Each step ships value. Foundations grow as features demand them.
- **"This step doesn't ship anything user-visible but it unblocks the next three."** Wrong unit of work — bundle it with the smallest of the three so the combined step ships something.
- **"Let's get the architecture right before we start."** No — get *one working path* right and let the architecture emerge from real, shipped requirements.

## Examples

### Example 1 — PDF-to-Markdown tool from a feature wishlist

**User input:** "I want a tool that converts PDFs to Markdown, with OCR for scanned PDFs, batch processing, a CLI, a web UI, and a REST API. What should I build first?"

**Bad plan (rejected):**

1. Set up project structure and CI.
2. Build the PDF parsing core.
3. Add OCR engine.
4. Build the CLI.
5. Build the API.
6. Build the web UI.
7. Add batch processing.

After steps 1–3 the user has nothing usable. Steps 4–7 are bundled by surface, not by value.

**Good plan (progressive JPEG):**

*End-user: a technical writer who currently copy-pastes from PDF previews into their Markdown notes. Primary outcome: get clean Markdown out of a PDF without manual cleanup. First-version success bar: one PDF in, usable Markdown out, beats their current copy-paste workflow.*

- **Step 1 — User can convert a single text-PDF to Markdown from the command line.** *(Value: High)*
  - Deliverable: a `pdf2md` binary (macOS + Linux), invoked as `pdf2md input.pdf > out.md`.
  - What the user can now do: "I can turn a PDF I already have into a Markdown file in one command instead of copy-pasting."
  - Why this rank: replaces the exact manual step the primary user hates most; everything else assumes this works.
  - In scope: minimal CLI, text-PDF parsing only, single file.
  - Out of scope: OCR, batch, web, API, config files.
  - Working version: a binary that converts text PDFs.
  - How the user verifies the value: run it on one of their own PDFs and check the Markdown is readable without hand-editing.

- **Step 2 — User can convert scanned PDFs via OCR.** *(Value: High)*
  - Deliverable: same `pdf2md` binary, now auto-detects scanned PDFs and OCRs them.
  - What the user can now do: "I can run the same command on a scanned PDF and still get Markdown."
  - Why this rank: a large share of real-world PDFs are scans; without this, Step 1 silently fails for them.
  - In scope: detect image-only pages, route through OCR, same CLI interface.
  - Out of scope: language tuning, accuracy knobs, batch.
  - Working version: same CLI, now also handles scanned PDFs.
  - How the user verifies the value: run it on a scanned PDF and confirm text comes through.

- **Step 3 — User can convert a directory of PDFs in one command.** *(Value: Medium)*
  - Deliverable: `pdf2md ./folder/` produces one `.md` per PDF, with a progress line per file.
  - What the user can now do: "I can point it at a folder and walk away."
  - Why this rank: removes a real friction (running one command per file) but only matters once Steps 1–2 are reliable.
  - In scope: glob input, parallel processing, progress output.
  - Out of scope: resumable jobs, persistent queue, web.
  - Working version: CLI handles single files *and* directories.
  - How the user verifies the value: drop a folder of mixed PDFs in, get a folder of `.md` files out.

- **Step 4 — User can drop a PDF on a web page and get Markdown back.** *(Value: Medium)*
  - Deliverable: a self-hostable web page at `/` with a drop zone and a download button.
  - What the user can now do: "I can use this from a browser without installing anything, and share the URL with a colleague."
  - Why this rank: opens the tool to non-CLI users, but every CLI user is already served — so it's an expansion, not a fix.
  - In scope: thin web UI that calls the existing conversion code.
  - Out of scope: accounts, history, multi-file UI.
  - Working version: a self-hostable web UI on top of the existing CLI.
  - How the user verifies the value: open the page, drop a PDF, get Markdown back.

API and richer UI features come later, ranked by which one a real user asks for first.

### Example 2 — Bug present in shipped functionality

**User input:** "We've got two open bugs and three planned features. Plan the next sprint."

Outcome: the two bugs go *before* any feature, ordered by user impact. Only after the bug pile is cleared do the features get value-ranked. The plan is presented in that order with a one-line justification for each.

### Example 3 — Refactor that "everyone" wants

**User input:** "Before we add feature X, we should refactor the storage layer to make X cleaner."

Apply the sanity-check: does the refactor on its own ship anything the user can use? No. So bundle the refactor into Step X as the smallest slice that lets X ship — and only refactor what X actually touches. The "tidy up the rest of the storage layer" work stays out until a second caller appears.

## Files

```
make-mvp/
└── SKILL.md     # this file
```

This skill is pure procedure — no scripts, no templates. It runs in your head against the user's plan.
