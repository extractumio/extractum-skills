---
name: scrutinizer
description: Produce a complete, navigable project-state documentation set for any software project (mobile, web, desktop, backend, or monorepo) — architecture, data, flows, API, tests, security, state/sync, performance, observability, platform parity, config, and health. Use when the user wants a full-codebase audit, architecture documentation, an onboarding "helicopter view", a regenerable project snapshot to diff over time, or a grounded top-10 risks / action-items report. Orchestrates parallel "scout" agents in two waves plus a synthesis pass, emitting markdown plus self-contained interactive HTML per section.
---

# Task: Full Project-State Documentation
# (Architecture, Data, Flows, Tests, Security, Platform Parity, Health)

> **Universal template.** Works for any software project: mobile app, web app,
> desktop app, backend service, monorepo, or any mix of these.
>
> **How to adapt:**
> 1. Replace every `{PLACEHOLDER}` with your project's values.
> 2. Delete any section marked *(if applicable)* that doesn't apply
>    (e.g., platform parity when there is only one client).
> 3. Keep everything else as-is — the process is project-independent.

**Placeholders used below:**
- `{PROJECT_NAME}` — name of the project/system
- `{COMPONENTS}` — list of top-level components (e.g., "web client, mobile app,
  backend API at api.example.com, worker services")
- `{PLATFORMS}` — client platforms, if more than one (e.g., "iOS and Android",
  "web and desktop")
- `{DOCS_ROOT}` — output folder (e.g., `docs/ephemerals/reports/project-state/`)

---

## Objective
Produce a complete, navigable documentation set describing the **current state**
of {PROJECT_NAME} — all of {COMPONENTS} — readable top-down: a "helicopter view"
entry point that links into progressively deeper detail, down to individual
modules, flows, schemas, and tests. The output must double as a regenerable
snapshot so future runs can diff project state over time.

## Scope
- Every client codebase ({PLATFORMS}, if multiple)
- Backend / services: API layer, business logic, remote data stores
- Local (client-side) persistence and its sync/caching relationship with
  remote data *(if applicable)*
- All test suites (client and backend)
- Build system, CI/CD, dependencies, configs, environments

**Read-only task:** analyze and document only. Do not modify source code,
tests, or configuration.

## Writing style rules (apply to every document)

The documentation must be understandable by an engineer who has **zero prior
knowledge of this project** — and, for overview material, even by a
semi-technical reader.

- **Use simple, everyday words.** Prefer "saves data to disk" over "persists
  to the durable store". Short sentences. Active voice. Concise but information dense.
- **Explain in phases, from simple to detailed.** For every topic follow this
  order: (1) what it is, in one plain sentence → (2) why it exists /
  what problem it solves → (3) how it works, step by step → (4) where the
  code is (file references). Never start with implementation detail.
- **Define every technical or domain-specific term on first use**, in one
  plain-language sentence. Collect all such terms in the domain glossary
  (see the Docs & knowledge scout). Never assume the reader knows internal
  jargon, acronyms, or product-specific names.
- **Give enough context to read each document standalone.** A reader landing
  on any single file must find a 2–4 sentence intro saying what this document
  covers, how it fits into the whole system, and links to the overview and
  neighboring documents.
- **Use analogies for hard concepts** where they genuinely help
  (e.g., "the sync queue works like an outbox: letters wait there until the
  postal service — the network — is available").
- **Prefer a concrete example over an abstract description.** When explaining
  a flow, walk through one real, named scenario with real entity names.
- **Diagrams first, prose second.** Introduce each section with the picture,
  then explain it.

## Output location & structure
All output goes to `{DOCS_ROOT}`:

```
00-overview.md                     Helicopter view, C4 diagrams, module map,
                                   top-10 risks, per-module health scorecard,
                                   links to everything below
00-overview/index.html             Interactive whole-system picture (see HTML spec)
01-modules/<module-name>.md        One file per module (template below)
01-modules/index.html              Interactive module & dependency explorer
02-data/database.md                Local schema(s), remote schema(s),
                                   sync/replication or caching design
02-data/index.html                 Interactive schema + data-flow explorer
03-flows/<scenario-name>.md        Key user/system scenarios: happy path +
                                   failure-path variants
03-flows/index.html                Interactive flow/scenario explorer
04-api/api-surface.md              Endpoints/interfaces and client usage
04-api/index.html                  Interactive API catalog browser
05-tests/test-strategy.md          Test types, how they run, testability, gaps
05-tests/index.html                Interactive coverage visualization (treemap)
06-build/build-system.md           Targets, pipelines, dependencies, CI/CD
06-build/index.html                Interactive dependency & pipeline graph
07-security/security-posture.md    Auth, storage, transport, threat surface
07-security/index.html             Interactive threat-surface map
08-state/state-management.md       State architecture, offline, sync behavior
08-state/index.html                Interactive state/lifecycle diagram
09-performance/hotspots.md         Hot paths, memory, query, startup analysis
09-performance/index.html          Interactive hotspot map
10-observability/errors-logging.md Error propagation, logging coverage
10-observability/index.html        Interactive error-path / dark-zone map
11-platform/parity.md              Shared vs. platform code, parity matrix
                                   (if applicable — multi-platform projects)
11-platform/index.html             Interactive parity matrix
12-config/environments.md          Environments, flags, versioning/compat
12-config/index.html               Interactive environment matrix
13-meta/knowledge-gaps.md          Doc accuracy, tribal-knowledge hotspots,
                                   domain glossary
13-meta/index.html                 Interactive knowledge/glossary browser
99-action-items.md                 Ranked follow-ups tied to cited findings
99-action-items/index.html         Interactive, filterable action-item board
module-index.json                  Machine-readable index (spec below)
```

## Grounding rules (apply to every agent)
- Every claim and every diagram element must be grounded in actual code.
  Cite file paths next to each diagram: `path/to/file.ext:SymbolName`.
- When something is deduced rather than observed, mark it "(inferred)"
  and state the reasoning in one line.
- Every diagram gets a one-line caption: what it shows, which code it
  derives from.
- No aspirational architecture — document what IS, not what was intended.

## Visualization conventions

### Markdown diagrams
- Default: MermaidJS embedded in markdown (flowchart, sequenceDiagram,
  classDiagram, erDiagram, stateDiagram-v2).
- Keep each Mermaid diagram under ~25 nodes; split large graphs into one
  overview diagram plus detail diagrams.
- Consistent node naming across all documents (use the module names
  established in the module map from Wave 1).

### Per-section HTML visualization (mandatory, one per section)
Every numbered section (`00` through `13`, plus `99`) ships **one
self-contained `index.html`** that shows the whole picture of that section
at a glance. These are **complementary** to the markdown set, not a
replacement: markdown carries the full detail, citations, and prose;
the HTML carries the interactive, navigable overview.

Requirements for every `index.html`:
- **Self-contained single file**: inline JS/CSS, no external network
  requests, no build step — opens by double-clicking the file.
- **Shows the entire section in one view** (zoom/pan/collapse as needed),
  with drill-down on click/hover for detail.
- **Every visual element references its source files.** Clicking a node,
  row, or box reveals the underlying file path(s) (with a copy-to-clipboard
  affordance) and a link to the corresponding markdown document and heading.
- **Two-way linking**: each markdown document links to its section's HTML at
  the top; the HTML links back to the markdown documents it summarizes.
- **Same data, no divergence**: build the HTML from the same validated facts
  as the markdown (ideally from `module-index.json` plus the section's own
  extracted dataset), so the two can never contradict each other.
- **Consistent look across all sections**: shared color scheme, shared
  legend for module names/platforms, section title and generation date in
  a header. A legend explains every color/shape used.
- **Plain-language labels**: the same style rules apply — tooltips and
  labels use simple wording, not internal jargon.

Suggested visualization per section (adapt as needed):
| Section | Visualization |
|---|---|
| 00 overview | System map: components → modules, clickable into all other sections |
| 01 modules | Dependency graph explorer (filter by platform/layer, highlight cycles) |
| 02 data | ER diagram + animated/step-through sync or caching data flow |
| 03 flows | Scenario picker → step-through sequence view, toggle happy/failure path |
| 04 api | Filterable endpoint table + grouping by service/authz, request/response peek |
| 05 tests | Coverage treemap: modules sized by code volume, colored by test type |
| 06 build | Dependency graph with version/license/outdated flags + pipeline timeline |
| 07 security | Threat-surface map: trust boundaries, data stores, ranked risks overlay |
| 08 state | Interactive state machine (sync lifecycle, connectivity transitions) |
| 09 performance | Hotspot map over the module graph, severity color scale |
| 10 observability | Flow map with logging coverage overlay — dark zones highlighted |
| 11 platform | Parity matrix heatmap (feature × platform), drift risks flagged |
| 12 config | Environment × setting matrix with diff highlighting |
| 13 meta | Doc-health map + searchable glossary |
| 99 actions | Sortable/filterable board: severity, effort, owner-area, source finding |

Detailed spec for the test-coverage HTML is under "Templates & specs".

---

## Execution plan: two waves + synthesis

### Wave 1 — Core scouts (parallel)
Each scout owns one lens with non-overlapping outputs. Each scout also
produces its section's `index.html` (from its own validated dataset).

**Phase 1a — Recon:** each scout builds a structured inventory of its
domain (notes only, no final docs). The Structure scout's module map is
finalized first and becomes the shared vocabulary for all other scouts.

**Phase 1b — Deep dive:** scouts write their assigned documents and build
their section HTML.

1. **Structure scout** → `00-overview.md` skeleton + `01-modules/` skeletons
   - Module/package/target inventory across all components of {PROJECT_NAME}
   - Dependency graph, layering, orphan detection
   - C4-style diagrams: system context → containers → components

2. **Control-flow scout** → `03-flows/`
   - Application lifecycle (startup, background/idle, shutdown) per component
   - Key execution paths, concurrency model, error propagation paths
   - For each documented scenario, produce BOTH:
     a) happy-path flow/sequence diagram
     b) failure-path variant: network loss, auth expiry, data conflict,
        dependency outage — whichever failures apply to the scenario

3. **Data scout** → `02-data/database.md`
   - Local/client-side storage: schema (erDiagram), migrations, access layer
     *(if applicable)*
   - Remote data stores: schema, ownership per service
   - Local↔remote mapping: which entities sync or cache, transformation rules
   - Sync/caching sequence diagrams (initial load, incremental update,
     conflict or invalidation case)

4. **API scout** → `04-api/api-surface.md`
   - Endpoint/interface catalog: method, path/name, request/response shapes,
     authorization per endpoint
   - Client-side API layer: how requests are built, retried, cached
   - Versioning scheme and error-response conventions

5. **UI-flow scout** → UI sections in `03-flows/` + per-module docs
   *(if the project has a UI)*
   - Screen/page inventory and navigation graph, per platform
   - Screen-to-module mapping
   - Entry points: deep links, notifications, URL routes/schemes

6. **Test scout** → `05-tests/`
   - Enumerate all tests; classify: unit / integration / UI-snapshot / e2e
   - Map tests to modules; estimate per-module coverage and state the
     estimation method
   - Testability assessment: modules hard to test and why (coupling,
     singletons/globals, missing injection points)
   - Build `05-tests/index.html` (spec below)

7. **Security & privacy scout** → `07-security/security-posture.md`
   - Auth end-to-end: credential/token lifecycle, storage (secure storage
     usage), refresh flow
   - Transport security: TLS config, pinning, plaintext exceptions
   - Data-at-rest encryption status; secrets handling and hygiene
   - Runtime/OS permission usage vs. declared permissions *(if applicable)*
   - Backend: input validation, rate limiting, authorization model
   - Threat-surface diagram + ranked risk list

8. **State & sync scout** → `08-state/state-management.md`
   (behavioral counterpart to the Data scout's schema work)
   - Actual state-management pattern in use — and where it's violated
   - Offline/degraded capability map: what works without connectivity,
     queueing, replay *(if applicable)*
   - Conflict resolution strategy; sync edge cases: partial failure,
     retries, idempotency, clock skew
   - stateDiagram-v2 for the sync/session lifecycle and connectivity
     transitions

9. **Platform-parity scout** → `11-platform/parity.md`
   *(if the project ships on more than one platform)*
   - Shared vs. platform-specific code: percentage + file map
   - Feature parity matrix (per-platform-only / all platforms)
   - Divergent implementations of the same feature — drift risks

### Wave 2 — Context-dependent scouts (parallel, AFTER Wave 1)
These scouts read Wave 1 outputs as context before analyzing code —
they are significantly sharper with the module map, flows, and risk
findings already available.

10. **Build & dependency scout** → `06-build/build-system.md`
    - Build definition: targets/projects, configurations, build variants
    - Dependency graph per package manager: versions, licenses,
      outdated/abandoned flags
    - CI/CD: what runs, when, duration, gaps in automation
    - Signing/release credentials setup (documented, never secret values)

11. **Performance scout** → `09-performance/hotspots.md`
    - UI-thread / hot-path risks: synchronous I/O, heavy work in
      latency-sensitive paths
    - Memory: leak-prone patterns, caching policy, large object graphs
    - Data-store query patterns: N+1s, missing indexes (local and remote)
    - Startup path sequence diagram: what runs at launch, in what order
    - Backend: likely slow endpoints, payload sizes

12. **Error & observability scout** → `10-observability/errors-logging.md`
    - Error taxonomy: propagation paths, where errors are swallowed
    - Logging/analytics/crash-reporting coverage map — which flows are dark
    - User-facing error UX per failure class *(if there is a UI)*

13. **Config & environments scout** → `12-config/environments.md`
    - Environment matrix: dev/staging/prod; feature flags; remote config
    - How clients select their backend/service target
    - API versioning, minimum client version, forced-update mechanism,
      version migration story *(as applicable)*

14. **Docs & knowledge scout** → `13-meta/knowledge-gaps.md`
    - Audit existing READMEs/comments/docs: accurate / stale / missing
    - Tribal-knowledge hotspots: high-complexity code with no comments/tests
    - **Domain glossary**: plain-language definition of every internal or
      domain-specific term used anywhere in this documentation set, plus
      entity naming conventions. Every other document links terms here.

### Phase 3 — Synthesis (single synthesizer agent)
- Write final `00-overview.md`: C4 set, module map, top-10 risks
  aggregated across all scouts, per-module health scorecard
  (coverage %, complexity indicator, doc status, risk flags)
- Build `00-overview/index.html`: the whole-system interactive map,
  linking into every section's HTML and markdown
- Resolve contradictions between scouts (flag unresolvable ones explicitly)
- Ensure all cross-links between documents resolve; verify every Mermaid
  block renders; verify every section HTML opens standalone and its links
  back to markdown resolve
- Write `99-action-items.md` (+ its HTML board): ranked, concrete
  follow-ups; each item must cite the finding (document + section) it
  derives from — no free-floating opinions
- Generate `module-index.json`

---

## Model tiering & routing policy

Three tiers. Route by the nature of the subtask, not by scout identity —
every scout contains both mechanical and judgment work.

### Tier definitions
- **T1 — Light** (small/fast model): deterministic-ish extraction and
  enumeration. Wrong answers are cheap to detect and retry.
- **T2 — Mid** (mid model): structured analysis with local judgment;
  summarization with fidelity requirements; diagram and HTML generation
  from already-extracted facts.
- **T3 — Heavy** (frontier model): cross-cutting reasoning, contradiction
  resolution, risk ranking, anything where the output shapes other agents'
  work or the final conclusions.

### Routing rules by work type

T1 (light):
- File/target/dependency inventories; directory walking
- Endpoint enumeration, test enumeration and classification by pattern
- Schema extraction (tables, columns, types) from migrations/models
- Screen/page inventory, permission/config listing
- Grep-style hotspot candidate lists (sync I/O calls, unsafe unwraps/casts,
  singleton/global usage) — candidates only, no verdicts
- Rendering module-index.json from already-validated data
- Mermaid syntax validation, link checking, HTML link/anchor checking

T2 (mid):
- Writing per-module docs from T1 inventories (in the plain-language style
  defined above)
- Drawing flow/sequence/state diagrams for a single scenario
- Mapping tests→modules; estimating coverage with a stated method
- Building each section's index.html from that section's validated dataset
- Summarizing existing docs as accurate/stale/missing
- Parity matrix assembly from T1 file maps

T3 (heavy):
- Structure scout Phase 1a: finalizing the module map and boundaries
  (this decision propagates to all other scouts — never economize here)
- Sync/conflict-resolution analysis: inferring the actual consistency
  model from code behavior
- Security scout verdicts: threat-surface reasoning, risk ranking
  (T1 may collect the raw evidence first)
- Failure-path flow analysis (what ACTUALLY happens on network loss /
  auth expiry — requires tracing non-obvious paths)
- Performance verdicts from T1 candidate lists (real N+1 vs. false positive)
- ALL of Phase 3 synthesis: contradiction resolution, top-10 risks,
  health scorecard, 99-action-items.md

### Pipeline pattern (mandatory within each scout)
Structure every scout as: T1 extract → T2 compose → T3 judge (if needed).
The T3 step receives compact, structured T1/T2 output — never raw code
dumps. Heavy models read conclusions and evidence snippets, not file trees.

### Escalation triggers (T1/T2 must hand up, not guess)
Escalate one tier when:
- Confidence is low or evidence is contradictory
- The finding would appear in top-10 risks or 99-action-items.md
- Two scouts' outputs conflict
- The subtask requires tracing behavior across ≥3 modules
An escalation must include: the question, the evidence gathered so far,
and the specific decision needed — not "please redo my task."

### De-escalation rule
Never use T3 for anything whose correctness is mechanically checkable
(does the file exist, does the Mermaid render, does the JSON validate,
does the HTML open and do its links resolve). Validation is T1 work even
when the artifact was produced by T3.

### Budget expectation
Healthy distribution for this task: roughly 60–70% of calls on T1,
25–35% on T2, ≤10% on T3. If T3 usage exceeds ~15%, the T1 extraction
layer is under-delivering structured evidence — fix that first rather
than paying for more T3.

---

## Templates & specs

### Per-module document (`01-modules/*.md`)
0. Plain-language intro: what this module is and why it exists,
   in 2–3 jargon-free sentences a newcomer can understand
1. Purpose (2–3 sentences)
2. Public interface: types, protocols/interfaces, entry points
   (file references)
3. Dependencies: uses / used-by (Mermaid graph)
4. Internal structure: key classes and data structures
5. Interactions: sequence diagrams for main collaborations
6. Persistence/network touchpoints
7. Platform notes (per-platform differences, if any)
8. Test coverage summary + testability notes (link to 05-tests)
9. Known risks/gaps found by any scout (linked)
10. Link to `01-modules/index.html` with this module pre-highlighted
    (via URL fragment, e.g. `index.html#module=<name>`)

### Test-coverage visualization (`05-tests/index.html`)
- Treemap or grid of modules, sized by code volume
- Color-coded by covering test type: unit / integration / UI-snapshot /
  e2e / uncovered — distinct colors, with legend
- Legend also states, in plain language, how coverage was estimated
- Hover/click per module reveals: covering test files (paths), test counts,
  notable gaps, link to the module's markdown doc
- Self-contained single HTML file (inline JS/CSS), no external requests

### `module-index.json` schema
```json
{
  "generated": "<ISO date>",
  "project": "<project name>",
  "modules": [{
    "name": "...",
    "platform": "<platform-or-component id, e.g. web | mobile | shared | backend>",
    "files": ["..."],
    "dependsOn": ["..."],
    "usedBy": ["..."],
    "coverage": { "unit": 0.0, "integration": 0.0, "ui": 0.0, "e2e": 0.0 },
    "risks": ["<short id refs into docs>"],
    "docs": ["01-modules/<name>.md"]
  }]
}
```
Purpose: future runs regenerate this file and diff it to track project
state over time. Keep keys stable across runs. The section HTML files
should consume this file's data (inlined at generation time) so visuals
and markdown never diverge.

---

## Quality bar
- Every module in the codebase appears in the module map — no orphans
- Every scenario in 03-flows has both happy-path and failure-path diagrams
- Diagrams reflect actual code; uncertain elements marked "(inferred)"
- No secret values ever appear in output (names/locations of secrets only)
- A new engineer can go from 00-overview.md to any implementation file
  in ≤3 clicks
- **Plain-language check:** every document opens with a jargon-free intro;
  every domain term is defined on first use and linked to the glossary;
  a reader with no project context can follow each document top to bottom
- **HTML check:** every section has exactly one self-contained index.html;
  it opens offline by double-click; every visual element exposes its source
  file references; markdown↔HTML links resolve in both directions
- Action items are traceable: each one cites its source finding
- module-index.json validates against the schema and covers every module