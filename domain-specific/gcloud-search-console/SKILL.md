---
name: gcloud-search-console
description: Query Google Search Console (Webmasters) performance data from the CLI using gcloud-issued credentials. Use when the user wants to pull clicks, impressions, CTR, or position data for their sites — e.g. "top queries for example.com", "pages ranking for X", "search performance last week".
author: Greg Z.
author_email: info@extractum.io
author_url: https://www.linkedin.com/in/gregzem/
---

# Google Search Console via gcloud

Pull Search Console (GSC) data from the CLI using the official REST API at `searchconsole.googleapis.com`, authenticated via `gcloud` Application Default Credentials.

## Prerequisites

1. **`gcloud` is installed and a project is set.** If not, run the `gcloud-setup` skill first.
2. **The Google account being used owns or has access to the site in Search Console.** Check at https://search.google.com/search-console — if the site is not listed there, this skill cannot help; the user needs to verify the property first.
3. **The Search Console API is enabled on the project:**
   ```bash
   gcloud services enable searchconsole.googleapis.com
   ```

## Step 1: Authenticate ADC with the Search Console scope

The scope is `https://www.googleapis.com/auth/webmasters.readonly` (read-only) or `https://www.googleapis.com/auth/webmasters` (read/write). Read-only is sufficient for querying performance data.

⚠️ `gcloud auth login --scopes=...` does NOT work. Use `application-default login`:

```bash
SCOPES="openid"
SCOPES="$SCOPES,https://www.googleapis.com/auth/userinfo.email"
SCOPES="$SCOPES,https://www.googleapis.com/auth/cloud-platform"
SCOPES="$SCOPES,https://www.googleapis.com/auth/webmasters.readonly"
gcloud auth application-default login --scopes="$SCOPES"
```

When the browser opens, log in with the Google account that has access to the site, and **check every scope box** on the consent screen (if `cloud-platform` is unchecked the ADC will be unusable).

## Step 2: List accessible sites

```bash
TOKEN=$(gcloud auth application-default print-access-token)
PROJECT=$(gcloud config get-value project)

curl -s \
  -H "Authorization: Bearer $TOKEN" \
  -H "x-goog-user-project: $PROJECT" \
  https://searchconsole.googleapis.com/webmasters/v3/sites | jq
```

Sample output:
```json
{
  "siteEntry": [
    {"siteUrl": "sc-domain:example.com", "permissionLevel": "siteOwner"},
    {"siteUrl": "https://example.com/", "permissionLevel": "siteOwner"}
  ]
}
```

### Site URL formats

- **Domain property** (covers all subdomains + schemes): `sc-domain:example.com` — URL-encode the colon as `%3A` when placing it in a URL path: `sc-domain%3Aexample.com`.
- **URL-prefix property**: the full URL, e.g. `https://example.com/`. URL-encode the whole thing with `jq -rn --arg s "$SITE" '$s|@uri'` or `python -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$SITE"`.

## Step 3: Query search analytics

Endpoint: `POST /webmasters/v3/sites/{siteUrl}/searchAnalytics/query`.

```bash
SITE="sc-domain:example.com"
SITE_ENC=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$SITE")

curl -s -X POST \
  -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
  -H "x-goog-user-project: $(gcloud config get-value project)" \
  -H "Content-Type: application/json" \
  "https://searchconsole.googleapis.com/webmasters/v3/sites/${SITE_ENC}/searchAnalytics/query" \
  -d '{
    "startDate": "2026-04-07",
    "endDate":   "2026-04-14",
    "dimensions": ["query"],
    "rowLimit": 25
  }' | jq
```

### Request body fields

| Field | Notes |
|---|---|
| `startDate`, `endDate` | `YYYY-MM-DD`. GSC data has ~2-day lag; yesterday's data may be missing. |
| `dimensions` | Array. Any combination of `query`, `page`, `country`, `device`, `date`, `searchAppearance`. |
| `rowLimit` | Max 25,000 per call. Default 1,000. |
| `startRow` | For pagination. Increment by `rowLimit`. |
| `dimensionFilterGroups` | Filter results. See examples below. |
| `type` | `web` (default), `image`, `video`, `news`, `discover`, `googleNews`. |
| `dataState` | `final` (default) or `all` (includes fresh, unsettled data). |

### Common query recipes

**Top queries, last 7 days:**
```json
{"startDate":"2026-04-07","endDate":"2026-04-14","dimensions":["query"],"rowLimit":25}
```

**Top pages, last 28 days:**
```json
{"startDate":"2026-03-17","endDate":"2026-04-14","dimensions":["page"],"rowLimit":50}
```

**Daily click/impression series:**
```json
{"startDate":"2026-03-17","endDate":"2026-04-14","dimensions":["date"]}
```

**Queries that contain "llm" and rank in top 10:**
```json
{
  "startDate":"2026-03-17","endDate":"2026-04-14",
  "dimensions":["query"],
  "dimensionFilterGroups":[{
    "filters":[
      {"dimension":"query","operator":"contains","expression":"llm"}
    ]
  }],
  "rowLimit":100
}
```

**Performance for a specific page:**
```json
{
  "startDate":"2026-03-17","endDate":"2026-04-14",
  "dimensions":["query"],
  "dimensionFilterGroups":[{
    "filters":[
      {"dimension":"page","operator":"equals","expression":"https://example.com/some-page/"}
    ]
  }],
  "rowLimit":100
}
```

Filter operators: `equals`, `notEquals`, `contains`, `notContains`, `includingRegex`, `excludingRegex`.

## Step 4 (optional): Other Search Console endpoints

| Goal | Method & Path |
|---|---|
| List sitemaps | `GET /webmasters/v3/sites/{site}/sitemaps` |
| Submit a sitemap | `PUT /webmasters/v3/sites/{site}/sitemaps/{feedpath}` (needs `webmasters` scope, not read-only) |
| Inspect a URL | `POST /v1/urlInspection/index:inspect` on `https://searchconsole.googleapis.com` — body: `{"inspectionUrl":"...","siteUrl":"..."}` |

## Step 5 (optional): Bulk export to BigQuery

For large date ranges (API caps at 50k rows per query and samples heavily above that), use Search Console's **native BigQuery bulk export**:

1. In Search Console UI → Settings → Bulk data export → link to your GCP project.
2. Google creates a dataset `searchconsole` in that project and writes daily snapshots.
3. Query with `bq`:
   ```bash
   bq query --use_legacy_sql=false '
     SELECT query, SUM(clicks) AS clicks, SUM(impressions) AS impressions
     FROM `YOUR_PROJECT.searchconsole.searchdata_site_impression`
     WHERE data_date BETWEEN "2026-04-01" AND "2026-04-14"
     GROUP BY query
     ORDER BY clicks DESC
     LIMIT 50
   '
   ```

## One-liner "is this working" check

```bash
curl -s \
  -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
  -H "x-goog-user-project: $(gcloud config get-value project)" \
  https://searchconsole.googleapis.com/webmasters/v3/sites | jq -r '.siteEntry[]?.siteUrl'
```

If this lists the user's sites, the skill is fully working.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `accessNotConfigured` / `SERVICE_DISABLED` | API not enabled on quota project | `gcloud services enable searchconsole.googleapis.com` |
| `API requires a quota project` | Missing header | Add `-H "x-goog-user-project: $(gcloud config get-value project)"` |
| `Request had insufficient authentication scopes` | ADC created without `webmasters.readonly` | Re-run Step 1 with the correct scope list |
| `User does not have sufficient permission for site 'X'` | Logged-in Google account lacks GSC access to that property | Log in as the owning account, or have the owner add this account in Search Console → Settings → Users and permissions |
| Empty `rows` in the response | No data for that dimension/date range, OR data lag (GSC is ~2 days behind) | Widen the date range, or set `"dataState":"all"` |
| 400 `invalid filter` | Wrong operator name or malformed body | See the filter operator list above |

---

# Part 2 — Search Console Operator Playbook (2025-2026)

Everything below is for **analysis and action**, not for calling the API. Use it once you can pull data (Part 1) to decide *what* to pull, *why*, and *what to do with it*.

## A. Ranking model — what Google actually weighs

| Signal | How measured | Weight | What to watch |
|---|---|---|---|
| **Relevance** | Neural matching + RankEmbed vectors; BERT/DeepRank re-ranks top 20-30 | Must-pass gate | Query-doc semantic match, entity overlap, intent class |
| **Helpful content / site quality** | Sitewide classifier, folded into core March 2024 | Very high; sitewide demotion possible | Originality, depth, firsthand experience, % unhelpful pages |
| **E-E-A-T** | Rater guidelines + off-site entity signals | High across all topics (not just YMYL) | Named authors + bios + `sameAs`, cited sources, editorial mentions |
| **Navboost (click signals)** | 13-month rolling GoodClicks / BadClicks / LastLongestClicks, per device × country × language | Top-tier (per Pandu Nayak testimony) | SERP CTR vs position, dwell, pogo-stick back-to-SERP |
| **Backlinks** | SpamBrain-filtered graph; topical relevance > DR | Still major but bar raised | Editorial, body-of-page, topically matched links; avoid PBNs/anchor spam |
| **Brand mentions (linked + unlinked)** | Corroboration/co-occurrence | Rising fast; now ~3× links for AI visibility | Ahrefs 75k brands: mentions Spearman 0.664, links 0.218 |
| **Core Web Vitals** | CrUX p75 field data, 28-day window | Tiebreaker; heavier post-Dec 2025 CU | LCP ≤2.5s, INP ≤200ms, CLS ≤0.1 |
| **UX / ad experience** | `clutterScore`, interstitial policy demotions | Can single-handedly tank a site | Ad density, popups, unclosable video, intrusive interstitials |
| **Freshness (QDF)** | Query-deserved-freshness classifier | Query-dependent | Real content diffs, not date-swapping |
| **HTTPS / mobile** | Booleans | Baseline | Valid cert, mobile-usable (mobile-first indexing default) |

Key shift: **ranking is comparative.** Your page is scored against the live competing set for each query — rankings can drop with zero on-site changes.

## B. Core algorithms / systems (2025-2026 state)

- **Helpful Content system** — continuous, sitewide. Integrated into core updates since March 2024. Tanks scaled/fringe/AI-at-scale editorial. Recoveries are slow and land on Core Update cycles.
- **SpamBrain / spam updates** — Aug 2025 + Oct 2025 targeted scaled content abuse and site-reputation abuse (a.k.a. "parasite SEO"). Mostly silent devaluation now; manual actions are the exception.
- **Core updates** — 2025: March, June, December. 2026: Feb Discover-only, March (completed Apr 8, 2026). Unannounced "tremors" run between announced CUs — don't panic-edit mid-rollout.
- **Navboost** — user-interaction re-ranker; domain-level click patterns contribute to site-wide quality signals.
- **Neural matching / RankEmbed / BERT / DeepRank** — semantic retrieval + top-N re-ranking. DeepRank decides positions 1-10.
- **Reviews system** — continuous; rewards firsthand testing, original photos/video, pros/cons, comparisons; tanks aggregator rewrites.
- **AI Mode / AI Overviews** — powered by Gemini 3 Pro (AI Mode since Nov 2025, Gemini 3 Flash default Dec 2025). Query fan-out: one user query decomposes into multiple sub-queries retrieved and synthesized (confirmed Google I/O 2025).
- **Twiddlers** — post-retrieval re-rankers (QualityBoost, freshness, diversity) that can promote/demote after main scoring.

## C. GSC reports — what each exposes and what to do with it

### Performance → Search results (the workhorse)

Metrics and real meaning:

| Metric | Real definition | Watch out for |
|---|---|---|
| **Impressions** | URL appeared in SERP for query, even below fold; AIO citations count | Sept 2025 `num=100` removal + a May-Oct 2025 logging bug inflated/deflated impressions — don't over-read mid-2025 deltas |
| **Clicks** | User left SERP for your URL | Multi-click sessions aggregate to 1 |
| **CTR** | clicks ÷ impressions | Noisy at low volume; SERP features (AIO, snippet, packs) consume clicks |
| **Average position** | Impression-weighted mean of topmost rank per query-page-day-country-device | Max 1 position per scope/day. Adding long-tail can *lower* average while clicks rise. Directional only. |

- Data lag ~2-3 days. "Last 24 hours" view gives provisional same-day.
- UI exports cap at 1,000 rows; API at 50,000 per call. For unsampled data use **BigQuery bulk export**.
- Tabs: Queries / Pages / Countries / Devices / Search appearance / Dates. "Compare" mode is the main diagnostic lens.

**High-ROI query-level recipes:**

| Goal | Filter / view |
|---|---|
| Striking-distance keywords | Queries → `Position 8.0-20.0`, sort by impressions desc |
| High-impression low-CTR | Queries → `Impressions > 500`, sort CTR asc (cross-check for AIO on the SERP) |
| Declining queries | Compare last 3mo vs previous, Queries sort Click-diff asc |
| Rising queries | Compare, sort Click-diff desc — expand coverage |
| Cannibalization | Filter 1 query → switch to Pages tab; multiple URLs with traffic = consolidate |
| Brand vs non-brand | Regex filter on Queries — `Doesn't match` brand regex |

### URL Inspection

- **Indexed view** = what Google has on file (coverage, canonical selection, last crawl, user-declared vs Google-selected canonical). Use for canonical mismatch forensics.
- **Live test** = Googlebot runs now (rendered HTML, screenshot, JS console). **Does not evaluate quality/canonical/duplicate** — only tech.
- **Request indexing** is rate-limited (~10-15/day). Don't use it as a bulk strategy — fix the root cause and update the sitemap.

### Indexing → Pages (why URLs aren't indexed)

| Reason | Action |
|---|---|
| **Crawled - currently not indexed** | Quality signal. If spiking after a CU/HCU or on AI-content-heavy sites: prune thin/derivative pages, consolidate, improve E-E-A-T. Sudden spike on small site = possible hack. |
| **Discovered - currently not indexed** | Crawl-budget / perceived low priority. Improve internal links from authoritative pages, faster server response, include in sitemap. |
| **Duplicate, Google chose different canonical** | Audit canonical signals (hreflang, internal links, redirects, content similarity). |
| **Soft 404** | Return real 404/410 or add real content. Common on empty category/search pages. |
| **Excluded by 'noindex'** | Audit — CMS plugins often add noindex inadvertently. |
| **Page indexed without content** | JS rendering failure. Live-test and move critical content server-side. |
| **5xx server error** | Block-priority fix; audit logs, edge workers, capacity. |

### Sitemaps

- Every URL in sitemap must be 200, canonical, indexable. Remove noindex/redirected URLs — they pollute discovery and dilute the report.
- Split sitemaps per content type (products/articles/categories) so Pages → "All submitted pages" pinpoints which segment fails.
- "Discovered URLs" ≠ indexed. Cross-reference via Pages report.

### Experience → Core Web Vitals

- Source: CrUX field data, p75 per URL group, 28-day window. Lab data (Lighthouse/PSI) is for debugging only.
- A group's status = its **worst** metric. Fix at template level, not per URL.
- Report lag: ~28 days after fix before field data shifts. Use PSI per-URL for instant validation.
- Low-traffic URLs don't appear (insufficient CrUX samples).

### Enhancements (rich results)

What's gone or restricted as of 2025-2026:
- **HowTo** — fully retired Sept 2023.
- **FAQPage rich results** — restricted to authoritative gov/health sites since Aug 2023 for Search display (but still high-value for AI answer engines — see Section E).
- **Mobile Usability** standalone report — retired Dec 2023; mobile signal surfaces only in URL Inspection.
- **June 2025 simplification** — Practice Problems, Nutrition Facts variants, Vehicle Listings nearby offers, TV Season Selector, Local Bikeshare, Today's Doodle phased out. BigQuery export fields for deprecated types return NULL after Oct 1, 2025; report/API support stops Jan 2026.

What invalidates a valid result: missing required field, malformed JSON-LD, markup not matching visible content, content behind unrendered JS, expired dates, wrong currency/units.

### Security & Manual actions

Manual-action categories: user-generated spam, thin content, unnatural links to/from site, cloaking, sneaky redirects, pure spam, hidden text/keyword stuffing, sneaky mobile redirects, **site reputation abuse** (clarified Jan 2025 — noindex alone insufficient; moving to subdomain counts as evasion).

Reconsideration request must: admit violation, detail every remediation step, document outcomes (sample fixed URLs, removed link counts, disavow upload). Generic copy-paste requests have ~70% higher rejection. Don't resubmit early — slows review. Ranking recovery is not guaranteed or instant.

### Links

- External tab (Top linked pages, Top linking sites, Top anchors) is sampled. Use "Latest links" export (~100k rows) for fuller data. Look for: sudden foreign-TLD spikes, anchor concentration on money keywords.
- Internal tab → confirm important commercial pages have sufficient internal links; thin counts = restructure nav/breadcrumbs/contextual links.
- **Disavow tool:** separate URL; **does NOT support Domain properties** (URL-prefix only). In 2025 Google ignores most spammy backlinks algorithmically — reserve disavow for confirmed schemes or pending unnatural-links actions.

### Insights (and 2025-2026 additions)

- **Query Groups** (Oct 27, 2025 rollout) — AI-clustered semantically similar queries with Top / Trending up / Trending down buckets. UI-only (no API/BigQuery export).
- **AI-powered configuration** (Dec 2025) — natural-language filter setup in Performance report ("striking-distance queries on mobile in Germany last 28 days"). Always review suggested filters before relying on output.
- **Annotations** (Nov 17, 2025, 120-char limit, Performance report) — mark update/deploy dates for future correlation.

### BigQuery bulk export (why it beats the API)

- Three tables: `searchdata_site_impression` (site-level aggregates; no anonymized queries), `searchdata_url_impression` (adds URL dim + booleans for AIO/AMP/rich results/Discover — **this is where AI Overview impression data exists**), `ExportLog`.
- No 50k row cap, no 16-month cutoff, partitioned by `data_date`.
- Cost discipline: never `SELECT *`, always `WHERE data_date BETWEEN ...`, set partition expiration (e.g., 16 months), aggregate into derived tables.

## D. Content, technical, off-page — compressed playbook

### Content (on-page)

- **Title**: primary keyword near front, brand at end, 50-60 chars, unique, match dominant SERP intent. Don't anchor-stuff.
- **Meta description**: not a ranking factor; writes feed CTR → Navboost indirectly.
- **H1 (one) + H2s as question-form** → harvests People-Also-Ask + AI Overview passages.
- **Internal linking**: Shepard 23M-link study — ~10 varied internal links/page optimal; **anchor diversity matters more than count**. Over-varied = manipulative.
- **Schema priority** (JSON-LD): `Organization` + `sameAs` (Wikidata QID, Wikipedia, LinkedIn), `Article` w/ `author` → `Person` + `sameAs`, `Product`/`Review`/`AggregateRating`, `BreadcrumbList`, `VideoObject`. FAQ/HowTo dead for Search rich results but **+30% citation lift / 3.2× AI Overview appearance in GEO** — keep for AI engines. Schema with populated attributes: 61.7% citation rate vs 41.6% minimally populated vs 59.8% no schema.
- **Intent matching** is the #1 ranking failure. SERP-check the target query, mirror the dominant format (list/guide/comparison/PDP/video) or pick a differentiated format backed by intent evidence.
- **Featured snippet / AI Overview capture**: 40-60 word definitional answer under each question-form H2; ideal passage 134-167 words.
- **Information Gain patent (June 2024)** — net-new info vs prior seen docs. AI-rewritten consensus content scores ~0. Add original data, firsthand experience, expert quotes, unique POV, original media.

### Technical

- **JavaScript** (Dec 2025 change): pages returning non-200 may be excluded from render queue. SPAs serving 200 on 404 routes risk mass deindexing. Prefer SSR/SSG/ISR for core templates.
- **Canonicals on JS sites**: must match in raw HTML *and* post-render. If JS changes canonical, omit from raw HTML rather than conflict.
- **Core Web Vitals 2026 targets** (p75): LCP ≤2.5s, INP ≤200ms, CLS ≤0.1. INP is the hardest — 43% of sites fail. Dec 2025 CU impact: LCP >3s = -23% traffic, INP >300ms = -31%, CLS >0.15 = -19%.
- **Crawl budget**: audit logs, kill infinite calendars/session params/facet explosions, robots.txt-block sort/filter params, fix redirect chains (max 1 hop).
- **hreflang**: bidirectional; server-rendered (not JS-injected); in `<head>` or XML sitemap.
- **Pagination**: `rel=next/prev` deprecated. Self-canonical each page, unique titles, don't canonicalize page 2+ → page 1.

### Off-page

- **Link quality ranking (2026)**: topical relevance > editorial body placement > entity strength of linking domain > DR alone > anchor context.
- **Toxic patterns**: exact-match anchor spam, PBN fingerprints, site-wide footer links, guest-post farms, link-velocity spikes w/o PR/news trigger.
- **Digital PR** (Mueller: "as critical as technical SEO"): earned editorial > guest posts. Nofollow from top-tier pub ≈ dofollow in practice.
- **Unlinked brand mentions** now correlate ~3× higher than backlinks for AI visibility — reclaim, amplify, seed.

## E. GEO / AISEO — optimizing for AI answer engines

AI Overviews trigger rate: ~6.5% (Jan 2025) → >60% of US queries (Nov 2025). AIO causes ~-58 to -61% organic CTR when present. **Being cited** inside AIO = +35% organic clicks vs uncited peers in same SERP.

### How AI engines select sources (signals, roughly in order of weight)

1. **Brand search volume** — strongest single predictor (Ahrefs Spearman 0.334).
2. **Branded web mentions** (linked + unlinked) — Spearman 0.664.
3. **Corroboration across independent sources** — ≥4 non-affiliated forum mentions = 2.8× ChatGPT inclusion.
4. **Entity resolution** — Wikidata QID, Wikipedia, consistent `sameAs` across authoritative profiles.
5. **Recency** — Perplexity 3.2× citation lift for content updated <30 days; 50% of top-cited AI content is <13 weeks old.
6. **Technical perf** — FCP <0.4s → 3× more ChatGPT citations.
7. **Rank in underlying index** — 76% of AIO citations from Google top-10; 87% of ChatGPT Search citations from Bing top-10 (so **Bing SEO is now a GEO lever**).
8. **Structured data** — attribute-rich schema, FAQPage + HowTo still earn citations even though Search deprecated them.

### Citation skews per engine

| Engine | Index | Avg citations | Top sources |
|---|---|---|---|
| Google AI Overviews | Google | 3-10 | YouTube (#1), Reddit, Wikipedia, Quora, LinkedIn |
| ChatGPT Search | Bing + SearchGPT | ~7.9 | Wikipedia ~48%, Reddit, G2/Capterra/Gartner (SaaS) |
| Perplexity | Bing + own crawler | ~21.9 | Reddit ~46.5%, niche directories (Zocdoc, TripAdvisor) |
| Claude | Training + fetch | Low | Authority sites |
| Gemini / AI Mode | Google + Knowledge Graph | Mid | Brand-owned, Medium, Google properties |

### Content structure AI tools prefer

- **BLUF / inverted pyramid** — direct 40-60 word answer in first 1-2 sentences.
- **TL;DR 50-70 words** up top of long pages.
- **Q&A formatting** — literal questions as H2/H3, direct answer paragraph below.
- **Semantic chunks** — each chunk self-contained (RAG retrieves passages, not pages). No pronouns resolving across paragraphs.
- **Lists, tables, pros/cons, comparison matrices, numbered HowTo steps.**
- **Princeton GEO paper (KDD 2024) validated lifts:** Citing sources +40% visibility (+115% for low-ranking pages); quotation addition +37%; statistics addition +22-40%; authoritative tone +.
- **Opinion / first-person reviews** cited preferentially — AI latches onto pre-formed frameworks.

### Crawler control (default 2025 policy)

| Allow (gives you AI visibility) | Block-to-consider (training-only) |
|---|---|
| Googlebot, Bingbot, OAI-SearchBot, ChatGPT-User, PerplexityBot, Perplexity-User, ClaudeBot (search), Claude-SearchBot, Claude-User, Applebot | GPTBot, CCBot, Bytespider, Meta-ExternalAgent, Amazonbot |

- **Google-Extended** blocks Gemini/Vertex training only — does NOT affect AI Overviews or Search rankings. Safe to block if you don't want training use.
- Blocking Bingbot = killing ChatGPT Search + Copilot citations. Don't do it.

### llms.txt reality check

Adoption claim vs log reality: major AI crawlers rarely hit `/llms.txt`. No major provider officially supports it. Low-cost to deploy, essentially unproven ROI. Ship it if trivial; don't build strategy around it.

### Entity optimization (slow, compounding)

- Wikidata QID with `instance of`, `official website`, `inception`, external IDs (Crunchbase, LinkedIn, ISNI, GLEIF, OpenCorporates), sitelinks.
- Wikipedia page (requires independent secondary coverage — earn via PR).
- Knowledge Panel typically lands 3-6 months after consistent entity signal build.
- Author entities: `Person` schema + external `sameAs` + consistent bylines.

### Brand surfaces for AI citation

- **Reddit** — #1 Perplexity citation source (46.5%), 21% of Google AIO. Authentic presence in category threads, not spam.
- **YouTube** — #1 domain in AIO per Ahrefs Brand Radar. Title + transcript + description indexed.
- **Quora** — 14.3% of Google AIO citations.
- **G2 / Capterra / Gartner / PCMag** — critical for SaaS/B2B ChatGPT citations.
- **LinkedIn Pulse** — can rank in AIO within hours (piggybacks LinkedIn's authority).
- **Industry directories** — Zocdoc, TripAdvisor, etc.

### Measurement tools (2025-2026)

Profound, Peec AI, Otterly, Semrush AI Visibility Toolkit, Ahrefs Brand Radar, Scrunch AI, HubSpot AI Search Grader, Qforia (query-fan-out simulator). Only statistical aggregates are meaningful (AI is stochastic — SparkToro: <1/100 identical brand lists across runs). Demand methodology transparency.

### GEO vs SEO tensions

| Dimension | SEO | GEO |
|---|---|---|
| Content length | Long-form, engagement | Chunk-friendly, direct-answer-first |
| Opening | Hook / narrative | BLUF in 1-2 sentences |
| Links vs mentions | Links dominate | Unlinked mentions correlate 3× higher |
| Brand | Helps | Essential |
| FAQ schema | Deprecated for SERP | +30% AI citation lift |
| Ranking metric | Position | Consideration-set inclusion rate |

Warning (Lily Ray): chasing manufactured AI visibility with spam schema / AI content / fake authority triggers Helpful Content demotion — kills both SEO and the AI-retrieval layer. Real SEO + real brand + real expertise is the foundation.

## F. Diagnostic playbooks (by symptom)

### Traffic drop after a Core Update

1. **Pin dates** from Google Search Status Dashboard. Compare 28 days pre-rollout-start vs 28 days post-rollout-end.
2. **Pages tab**, sort Click-diff asc → identify impacted directories.
3. **Queries tab** with page-prefix filter → did whole intent clusters drop (HCU/relevance shift) or did rankings slide uniformly (broader quality)?
4. **Metric sequence**: impressions move first, clicks follow. CTR is noisy early — ignore.
5. **Segment by page type** (regex page filter): blogs vs products vs categories. HCU damage concentrates on thin/AI-feel editorial; product/category often spared.
6. **SERP-check** top dropped queries manually: did SERP composition change (forums, video, AIO)? That's intent/feature change, not necessarily your quality.
7. **Cross-check Pages report** — "Crawled - currently not indexed" surge correlated with rollout = sitewide quality signal.
8. **AIO caveat**: GSC tracks AIO/AI Mode impressions but doesn't break them out. Triangulate with Query Groups + manual SERP audits + BigQuery `searchdata_url_impression` AIO flags.
9. **Annotate** dates and deploys; read signal only *after* rollout completes + 2 weeks.

### Indexing decay / "Crawled - not indexed" spike

- Not usually a technical bug post-HCU — site-wide quality signal.
- Action: prune thin/duplicate/AI-rewritten pages, consolidate cannibalizers, improve remaining content depth + E-E-A-T + internal links.
- Sudden spike on a small site = **audit for hack** (injected URLs).

### Cannibalization

- GSC Performance → filter 1 query → Pages tab → multiple URLs ranking with non-trivial impressions.
- Consolidate: 301 lesser URLs to the best-performing, merge unique content, rebuild internal links to the winner.

### Ranking in striking distance (pos 8-20) but not climbing

- Intent mismatch (#1 cause): SERP-check and realign format.
- Content depth deficit: information-gain audit vs top-10.
- Internal link starvation: add 3-5 contextual internal links from cluster siblings.
- E-E-A-T: add author with credentials + `sameAs`, citations, updated date with real diff.
- UX: ad density / CWV regression check.

### AIO appears but you're not cited

- Do you rank top-10 on underlying query? If no, fix SEO first.
- Schema audit: add FAQPage/HowTo + Article + author.
- Q&A structure: rewrite opening as 40-60 word direct answer.
- Info-gain: add original stat/quote/data point that's citation-worthy.
- Brand signal gap: build Reddit/YouTube/Quora presence; earn mentions in category listicles.

### Manual action

- Address root cause across site, not just examples.
- Reconsideration: (a) admit violation, (b) detail every remediation, (c) document outcomes (sample URLs, link removal counts, disavow file). Give weeks per round.

## G. "Reach top-10 / get cited by AI" priority sequence

**Week 1-4 (fast, controllable):**
1. SERP intent audit on target queries; mirror dominant format.
2. Schema: `Organization` + `sameAs`, `Article` + author, `FAQPage`, `Product`/`Review`. Validate in Rich Results Test.
3. Rewrite top 20 pages: 40-60 word direct answer + TL;DR + question-form H2s + comparison tables.
4. Add citeable stats + authoritative citations + direct quotations (Princeton: +22-40%).
5. CWV template fixes; aim FCP <0.4s (3× ChatGPT citation lift).
6. Robots.txt audit — allow search-retrieval AI crawlers; allow Bingbot; decide on training crawlers.
7. Server-render core content; eliminate JS-only critical paths.
8. Update/republish high-traffic pages with real diffs.

**Month 2-3 (mid-leverage):**
9. Wikidata QID + properties + external IDs.
10. Digital PR for category listicles, data studies, expert commentary.
11. Original research / proprietary data (citation magnet + Info Gain).
12. Authentic Reddit/Quora presence in category threads.
13. YouTube program (transcripts indexed).
14. Set up GEO measurement (Profound/Peec/Otterly/Brand Radar).
15. Consolidate cannibalizers, noindex thin pages.

**Month 4-12+ (slow, compounding):**
16. Wikipedia page (requires earned secondary coverage).
17. Sustained digital PR — linked + unlinked mentions.
18. Multi-platform repurposing saturating corroboration.
19. G2/Capterra/Trustpilot/Gartner presence for SaaS/B2B.
20. Quarterly audits: Wikidata, schema, robots.txt, crawl logs, CWV, links.

## H. Common pitfalls & misreads

- **"Crawled - currently not indexed" exploding ≠ tech bug.** Post-HCU or on AI-content-heavy sites, Google declines to index thin/derivative pages. Pruning + consolidation, not "Request indexing," is the answer.
- **Impression deltas May-Oct 2025.** Pre-Sept 2025 SERP scrapers using `&num=100` inflated impressions at deep ranks. Post-Sept 2025 logging-error correction reduced apparent impressions. Both create non-real movement — don't declare victory or disaster off this window.
- **Page-2 impressions ≠ opportunity.** A query at position 23 is likely judged tangential. Striking-distance = positions 8-20.
- **Average position misreads.** Impression-weighted, capped at one rank per query/page/day; long-tail expansion can lower it even as clicks rise. Always read alongside clicks + impressions.
- **Live test as proof of indexing.** Only proves technical crawl/render. Doesn't predict canonical selection, duplicate handling, or quality acceptance.
- **Routine disavow.** In 2025 Google ignores most spammy backlinks algorithmically; reflexive disavow can suppress legitimate citations.
- **Sitemap "Discovered URLs" counted as indexed.** They aren't — reconcile via Pages report → "All submitted pages."
- **Data gaps around updates.** Google occasionally pauses or rebases reporting during major launches; don't read the first 5-7 days of a rollout.
- **Chasing "AI rank position" metrics.** AI outputs are non-deterministic. Only statistical aggregates (citation rate across N runs) are meaningful.
- **Over-optimization as the new spam.** Over-varied internal anchors, site-reputation abuse workarounds (subdomain moves), and mass FAQ-stuffing all correlate with post-update losses.
- **Mid-rollout panic edits.** CU rollouts take weeks; tremors and reversals happen. Document changes, wait for rollout-complete + 2 weeks, then read.

---

**Authoritative sources to consult directly** (not a full bibliography — these are the primary feeds to monitor):
- Google Search Central blog + Search Off The Record podcast + Search Status Dashboard
- Google Quality Rater Guidelines (updated 2024-2025)
- Glenn Gabe (gsqi.com) — core update post-mortems
- Lily Ray (lilyraynyc.substack.com) — GEO / HCU analysis
- Cyrus Shepard (zyppy.com) — large-sample SEO studies
- Aleyda Solis (learningseo.io)
- Mike King / iPullRank — The AI Search Manual, Relevance Engineering
- Barry Schwartz (Search Engine Roundtable) — daily tracking
- Ahrefs blog + Brand Radar studies
- Princeton GEO paper (arXiv:2311.09735 / KDD 2024)
- SparkToro / Datos — State of Search reports
