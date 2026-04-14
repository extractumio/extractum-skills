# Research Dimensions — Detailed Guide

This file contains the detailed research methodology for each analytical dimension
and the comprehensive Source Registry. For each dimension, follow the specific
investigation steps and capture all data points listed.

---

## Source Registry — Master Reference

This is the authoritative list of all sources to consult during product analysis.
Every source listed here should be checked for each product. Sources are organized
by category with specific guidance on what to extract from each.

### Search Engines & AI Research Tools

Use **multiple** search approaches — no single engine catches everything.

| Source | URL / Access | Best For | How to Query |
|--------|-------------|----------|-------------|
| **Google Search** | google.com | Broadest index; `site:` targeting, exact-match, date filters | `"product name" site:example.com`, use `Tools > Past year` for freshness |
| **Perplexity AI** | perplexity.ai | AI-synthesized summaries with source citations | Natural language: "What are the pros and cons of [product]?", "Compare [A] vs [B]" |
| **Google Scholar** | scholar.google.com | Academic papers, whitepapers citing the product | `"[product name]" OR "[company name]"` — useful for security/ML products |

### Tech Communities & Forums

| Source | URL / Access | What to Extract | Search Method |
|--------|-------------|----------------|---------------|
| **Hacker News** | news.ycombinator.com | Show HN posts, user debates, founder comments, technical critiques | `site:news.ycombinator.com "[product]"` or hn.algolia.com |
| **Reddit** | reddit.com | r/programming, r/devops, r/netsec, r/sysadmin, r/cybersecurity, r/SaaS, r/selfhosted, r/startups, + niche subs | `site:reddit.com "[product]"`, check subreddit-specific search |
| **Lobsters** | lobste.rs | Developer-focused technical discussions, often higher signal-to-noise than HN | `site:lobste.rs "[product]"` |
| **Stack Overflow** | stackoverflow.com | Integration questions, error patterns, adoption signals | `site:stackoverflow.com "[product]"` or `[product] tag` |
| **Stack Exchange** | stackexchange.com | Network sites: Software Recommendations, Information Security, DevOps, Server Fault | `site:*.stackexchange.com "[product]"` |
| **Dev.to** | dev.to | Developer blog posts, tutorials, experience reports | `site:dev.to "[product]"` |
| **HackerNoon** | hackernoon.com | Long-form tech articles, product reviews, comparison pieces | `site:hackernoon.com "[product]"` |
| **InfoQ** | infoq.com | Architecture case studies, conference presentations, expert interviews | `site:infoq.com "[product]"` |
| **DZone** | dzone.com | Developer tutorials, tool reviews, integration guides | `site:dzone.com "[product]"` |
| **Hashnode** | hashnode.com | Developer blog posts, often from product users | `site:hashnode.com "[product]"` |
| **Discourse forums** | Various | Many products host their own Discourse community | Check product's website for "Community" or "Forum" link |

### Review & Comparison Platforms

| Source | URL / Access | What to Extract | Signal Quality |
|--------|-------------|----------------|---------------|
| **G2** | g2.com | Verified enterprise reviews, pro/con structure, scores by category | High — reviews verified against LinkedIn |
| **Capterra** | capterra.com | SMB-focused reviews, feature comparisons, pricing data | High — verified purchasers |
| **TrustRadius** | trustradius.com | Detailed long-form reviews, TrustMaps, vendor comparisons | High — no vendor-influenced scoring |
| **GetApp** | getapp.com | Feature comparison matrices, category rankings | Medium — useful for feature checklists |
| **Software Advice** | softwareadvice.com | Buyer guides, advisor recommendations | Medium — acquisition-oriented reviews |
| **SaaSWorthy** | saasworthy.com | SaaS-specific reviews and awards | Medium |
| **Product Hunt** | producthunt.com | Launch reactions, upvotes, maker comments, early adopter sentiment | High for launch reception; check launch date |
| **AlternativeTo** | alternativeto.net | Competitor mapping, user-suggested alternatives, popularity votes | High for competitor discovery |
| **Slant** | slant.co | Community-voted comparisons ("What are the best X tools?") | Medium — crowd-sourced pros/cons |
| **Gartner Peer Insights** | gartner.com/reviews | Enterprise buyer reviews, overall ratings, "willingness to recommend" | High — enterprise-focused |

### Business & Funding Intelligence

| Source | URL / Access | What to Extract | Notes |
|--------|-------------|----------------|-------|
| **Crunchbase** | crunchbase.com | Funding rounds, investors, team size, acquisitions, IPO status | Primary source for funding data |
| **PitchBook** | pitchbook.com | Deeper funding data, valuations, investor details | Often paywalled; extract what's visible |
| **CB Insights** | cbinsights.com | Market maps, competitive landscapes, analyst commentary | Check free reports and blog posts |
| **LinkedIn** (company page) | linkedin.com/company/ | Employee count, growth rate, department distribution, key hires | Check "Insights" tab; track employee trend |
| **AngelList / Wellfound** | wellfound.com | Startup profile, team, open roles, funding | Strong for early-stage companies |
| **Owler** | owler.com | Revenue estimates, competitor mapping, news aggregation | Free tier has useful estimates |
| **Wikipedia** | wikipedia.org | Company history, notable events, third-party citations | Check for existence — absence is also a signal |
| **SEC filings** (Edgar) | sec.gov/edgar | Revenue, customers, risk factors — for public companies or pre-IPO S-1 | Only for public companies or IPO candidates |

### News, Press & Analyst Coverage

| Source | URL / Access | What to Extract |
|--------|-------------|----------------|
| **TechCrunch** | techcrunch.com | Funding announcements, product launches, market analysis |
| **VentureBeat** | venturebeat.com | AI/ML product coverage, enterprise tech news |
| **The Hacker News** (security) | thehackernews.com | Security product coverage, vulnerability news, tool announcements |
| **Dark Reading** | darkreading.com | Cybersecurity product reviews, industry analysis |
| **InfoSecurity Magazine** | infosecurity-magazine.com | Security product news, expert commentary |
| **ZDNet** | zdnet.com | Enterprise tech reviews, product comparisons |
| **Ars Technica** | arstechnica.com | Deep technical reviews, analysis |
| **The Register** | theregister.com | Opinionated tech coverage, product critiques |
| **Wired** | wired.com | Consumer/prosumer tech, trend pieces |
| **Gartner** (reports/mentions) | gartner.com | Magic Quadrant placement, Market Guide mentions, Hype Cycle position |
| **Forrester** (reports/mentions) | forrester.com | Wave reports, Total Economic Impact studies, analyst notes |
| **IDC** | idc.com | MarketScape evaluations, market sizing, forecasts |

### Developer Platforms & Package Registries

| Source | URL / Access | What to Extract | Key Metrics |
|--------|-------------|----------------|-------------|
| **GitHub** | github.com | Repos, stars, forks, issues, PRs, contributors, commit frequency, Discussions | Stars, contributor count, issue response time, last commit date |
| **GitLab** | gitlab.com | Same as GitHub — some companies prefer GitLab | Same metrics |
| **npm** | npmjs.com | JavaScript/Node package adoption | Weekly downloads, dependents count |
| **PyPI** | pypi.org | Python package adoption | Monthly downloads (via pypistats.org) |
| **crates.io** | crates.io | Rust package adoption | Downloads, recent downloads |
| **Docker Hub** | hub.docker.com | Container image adoption | Pull count, star count, last updated |
| **VS Code Marketplace** | marketplace.visualstudio.com | Editor extension adoption | Install count, ratings, last updated |
| **JetBrains Marketplace** | plugins.jetbrains.com | IDE plugin adoption | Downloads, ratings, compatibility |
| **Homebrew** | formulae.brew.sh | CLI tool adoption on macOS/Linux | Install count (30/90/365 day) |
| **Chrome Web Store** | chromewebstore.google.com | Browser extension adoption | Users, rating, reviews |

### Social Media & Real-Time Community

| Source | URL / Access | What to Extract |
|--------|-------------|----------------|
| **Twitter / X** | x.com | Founder activity, product announcements, user complaints/praise, community size |
| **LinkedIn** (posts) | linkedin.com | Thought leadership, company announcements, employee advocacy |
| **YouTube** | youtube.com | Product demos, tutorials, conference talks, review videos |
| **Discord** | discord.com | Product-specific servers — check product website for invite link |
| **Slack communities** | Various | Niche communities where product is discussed — search "[niche] slack community" |
| **Mastodon / Bluesky** | Various | Alternative social platforms, especially for open-source / privacy communities |

### Company Health & Talent Signals

| Source | URL / Access | What to Extract | Why It Matters |
|--------|-------------|----------------|---------------|
| **Glassdoor** | glassdoor.com | Employee reviews, CEO approval, culture ratings, salary data | Internal health → product quality signal |
| **Careers page** (company site) | company.com/careers | Open roles, tech stack requirements, team structure | Hiring velocity = growth; job descriptions reveal stack |
| **LinkedIn Jobs** | linkedin.com/jobs | Job postings mentioning the product or from the company | Technology requirements, team expansion areas |
| **Wellfound Jobs** | wellfound.com | Startup-focused job listings | Compensation transparency, equity, stage |
| **Levels.fyi** | levels.fyi | Compensation data for known companies | Company maturity and financial health signal |

### Traffic & Technology Detection

| Source | URL / Access | What to Extract |
|--------|-------------|----------------|
| **SimilarWeb** | similarweb.com | Monthly traffic estimates, traffic sources, geography, bounce rate |
| **BuiltWith** | builtwith.com | Technology stack detection (frontend, analytics, hosting, CDN) |
| **Wappalyzer** | wappalyzer.com | Technology stack detection — browser extension for real-time |
| **StackShare** | stackshare.io | Self-reported tech stacks, tool decisions, comparisons |

### Marketplace & Integration Listings

| Source | What to Check | Signal |
|--------|--------------|--------|
| **AWS Marketplace** | Product listing, pricing, reviews | Enterprise adoption, AWS partnership level |
| **GCP Marketplace** | Product listing, pricing | Google Cloud partnership |
| **Azure Marketplace** | Product listing, pricing | Microsoft partnership |
| **Atlassian Marketplace** | Jira/Confluence plugins | Enterprise dev-tool ecosystem presence |
| **Salesforce AppExchange** | App listing | Enterprise CRM ecosystem presence |
| **Zapier** | Integration available? Zap count? | Automation ecosystem presence |
| **Make (Integromat)** | Integration modules | Automation ecosystem presence |
| **GitHub Marketplace** | Actions, Apps | Developer workflow integration |

### Security, Compliance & Trust

| Source | What to Check | Signal |
|--------|--------------|--------|
| **HackerOne** | Public bug bounty program? | Security maturity; severity of reported issues |
| **Bugcrowd** | Public program? | Same as above |
| **CVE / NIST NVD** | Known vulnerabilities? | Track record of security issues |
| **Trust/Security page** | SOC2, ISO 27001, HIPAA, FedRAMP, GDPR, PCI-DSS | Compliance certifications → target market signal |
| **Status page** | status.example.com | Uptime history, incident communication quality |

### Historical & Archival

| Source | URL / Access | When to Use |
|--------|-------------|-------------|
| **Wayback Machine** | web.archive.org | Check past pricing pages, removed features, historical claims, pivots |
| **Google Cache** | `cache:url` in Google | Recent snapshots of changed pages |
| **Google Patents** | patents.google.com | Patented technology — reveals IP strategy and technical differentiation |

### Public Roadmaps & Feature Tracking

| Source | What to Check |
|--------|--------------|
| **ProductBoard** (public boards) | Feature voting, planned features, user requests |
| **Canny** (public boards) | Feature requests, upvotes, planned/in-progress/completed |
| **GitHub Projects / Issues** | Roadmap boards, milestone tracking, feature request labels |
| **Changelog** (product site) | Release frequency, feature cadence, what shipped recently |
| **Product blog** | "What's new" posts, quarterly updates, roadmap announcements |

---

## Dimension 1 — Value Proposition & Problem-Solution Fit

### What to Investigate

- **Core problem statement:** What pain does this product address? Is it a vitamin (nice-to-have) or painkiller (must-have)?
- **Solution approach:** How does the product solve the problem? What is unique about their approach?
- **Value delivery mechanism:** Where does the customer realize value? (time saved, risk reduced, cost avoided, capability gained, compliance achieved)
- **Customer outcomes:** What measurable results do customers achieve? Look for case studies, testimonials, ROI claims
- **Before/after narrative:** What does the customer's world look like without vs. with this product?

### Research Queries

- `"[product name]" "helps" OR "enables" OR "allows" OR "solves"`
- `"[product name]" "before" OR "without" OR "used to"`
- `"[product name]" case study OR results OR ROI`
- `"[product name]" "why we chose" OR "why we switched" OR "why we use"`
- `"[product name]" problem OR challenge OR pain point`

### Data Points to Capture

| Data Point | Source Priority |
|-----------|----------------|
| Product tagline / hero statement | Product website |
| Problem statement (in their words) | About page, blog, pitch decks |
| Solution description (in their words) | Features page, docs |
| Customer-stated problem (in users' words) | Reviews, Reddit, HN |
| Measurable outcomes / ROI claims | Case studies, testimonials |
| Use case categories | Docs, marketing pages, reviews |

---

## Dimension 2 — Target Audience & Market Positioning

### What to Investigate

- **Primary buyer persona:** Who makes the purchase decision? (CISO, VP Engineering, DevOps lead, individual developer, etc.)
- **Primary user persona:** Who uses the product daily? (same as buyer, or different?)
- **Company size targeting:** SMB, mid-market, enterprise, or all? Look at pricing tiers and customer logos for signals
- **Industry verticals:** Any specific industry focus? (fintech, healthcare, government, etc.)
- **Geographic focus:** Global, US-only, EU-specific? Check for compliance certifications as indicators
- **Maturity targeting:** Startups, growth-stage, established enterprises? Check messaging tone and feature emphasis

### Research Queries

- `"[product name]" "for teams" OR "for enterprises" OR "for developers" OR "for startups"`
- `"[product name]" customer OR client OR "who uses"`
- `"[product name]" industry OR vertical OR sector`
- `"[product name]" SOC2 OR HIPAA OR FedRAMP OR GDPR` (compliance = audience signal)
- `"[product name]" site:linkedin.com` (check who engages, company size of followers)

### Audience Signals to Capture

| Signal | What It Tells You | Where to Find |
|--------|------------------|---------------|
| Customer logos on website | Target company size & industry | Home/customers page |
| Pricing page structure | SMB vs enterprise positioning | Pricing page |
| "Book a demo" vs "Sign up free" | Sales-led vs self-serve | CTA buttons |
| Job postings mentioning product | Who adopts internally | LinkedIn, job boards |
| Content topics on blog | Target persona interests | Company blog |
| Conference sponsorships/talks | Community alignment | Events page, YouTube |
| Compliance certifications | Regulated industry targeting | Security/compliance page |

---

## Dimension 3 — Feature & Capability Analysis

### What to Investigate

- **Core features:** The 3-5 features that define the product
- **Feature breadth:** How many distinct capabilities does it offer?
- **Feature depth:** How sophisticated is each capability? Surface-level or deep?
- **Integration ecosystem:** What other tools/platforms does it connect with?
- **API/SDK availability:** Can users build on top of it?
- **Platform coverage:** What environments, languages, frameworks, CI systems does it support?
- **Extensibility:** Plugins, custom rules, scripting, webhooks?
- **Roadmap signals:** What's coming next? Check changelogs, blog posts, GitHub issues tagged "enhancement"
- **Feature gaps:** What do users commonly request that doesn't exist yet?

### Research Queries

- `"[product name]" features OR capabilities OR "what it does"`
- `"[product name]" integration OR "works with" OR "connects to"`
- `"[product name]" API OR SDK OR webhook OR plugin`
- `"[product name]" roadmap OR "coming soon" OR "planned" OR changelog`
- `"[product name]" "wish it had" OR "missing feature" OR "would be nice" OR "feature request"`
- `"[product name]" documentation site:docs.*`

### Feature Inventory Template

For each product, compile:

```
Core Features:
  1. [Feature name] — [One-line description] — [Maturity: GA/Beta/Alpha]
  2. ...

Integrations:
  - Native: [list]
  - Via API: [list]
  - Via third-party (Zapier, etc.): [list]

Platform Support:
  - Languages/Frameworks: [list]
  - CI/CD: [list]
  - Cloud providers: [list]
  - OS/Environment: [list]

Notable Gaps:
  - [Gap] — Source: [where this was noted]
```

---

## Dimension 4 — Technology & Architecture

### What to Investigate

- **Programming languages:** Primary and secondary languages used
- **Frameworks and libraries:** Web framework, ML frameworks, testing frameworks
- **Infrastructure:** Cloud provider, containerization, orchestration
- **Data storage:** Databases, caches, message queues
- **Architecture pattern:** Monolith, microservices, serverless, edge, hybrid
- **AI/ML components:** If applicable — models used, training approach, inference infrastructure
- **Security architecture:** How they handle customer data, authentication, encryption
- **Open-source footprint:** What parts are open-source? What license?
- **Build and release cadence:** How often do they ship? (GitHub commits, changelog frequency)

### Where to Find Technical Details

| Source | What It Reveals |
|--------|----------------|
| GitHub/GitLab repos | Languages, dependencies, architecture, code quality |
| `package.json`, `go.mod`, `Cargo.toml`, `requirements.txt` | Direct dependency list |
| `Dockerfile`, `docker-compose.yml` | Infrastructure choices |
| Job postings (careers page, LinkedIn) | Stack requirements reveal internal tech |
| Blog posts titled "How we built..." | Architecture decisions |
| Conference talks (YouTube, SlideShare) | Technical deep dives |
| BuiltWith, Wappalyzer, StackShare | Frontend/hosting tech detection |
| Security/compliance pages | Data handling, encryption, certifications |

### Research Queries

- `"[product name]" "built with" OR "tech stack" OR "architecture"`
- `"[product name]" site:github.com`
- `"[product name]" open source OR github OR repository`
- `"[company name]" careers OR jobs` (look for required tech skills)
- `"[product name]" "how we built" OR "engineering blog"`
- `"[company name]" site:stackshare.io`
- `"[product name]" infrastructure OR cloud OR AWS OR GCP OR Azure`

---

## Dimension 5 — Business Model, Pricing & Funding

### What to Investigate

- **Pricing model:** Free, freemium, subscription, usage-based, per-seat, enterprise-only, open-core
- **Pricing tiers:** What's included at each level? What are the limits?
- **Free tier:** Does one exist? What are its meaningful limitations?
- **Enterprise pricing:** Is it transparent or "contact sales"?
- **Billing cadence:** Monthly, annual, or both? Discounts for annual?
- **Funding history:** Seed, Series A/B/C? How much? Who invested? When?
- **Revenue signals:** Any public revenue figures, ARR milestones, customer counts?
- **Business model type:** Product-led growth, sales-led, community/open-source-led
- **Monetization trajectory:** What's the likely path to increased monetization? (features behind paywall, usage limits, enterprise add-ons)

### Research Queries

- `"[product name]" pricing OR cost OR "how much" OR plan OR tier`
- `"[product name]" free OR freemium OR "open source" OR community edition`
- `"[company name]" funding OR raised OR series OR seed OR investors`
- `"[company name]" site:crunchbase.com`
- `"[company name]" revenue OR ARR OR customers OR growth`
- `"[company name]" "business model" OR monetization`
- `"[product name]" "enterprise" pricing OR plan OR contact`

### Pricing Data Template

```
Model: [Freemium / Subscription / Usage-based / Open-core / Enterprise-only]

Tiers:
  - Free:       [features, limits]
  - Tier 1:     $X/mo — [features, limits]
  - Tier 2:     $X/mo — [features, limits]
  - Enterprise: Contact sales — [notable features]

Billing: [Monthly / Annual / Both — annual discount %]

Funding:
  - [Date]: [Round] — $[Amount] — Led by [Investor]
  - Total raised: $[Amount]

Revenue Signals:
  - [Any public figures or estimates]
```

---

## Dimension 6 — Competitive Landscape & Differentiation

### What to Investigate

- **Direct competitors:** Same problem, same audience, same approach
- **Indirect competitors:** Same problem, different approach (e.g., open-source alternative, platform-native feature, managed service vs. self-hosted)
- **Adjacent products:** Different primary problem but overlapping use cases
- **Substitutes:** Non-product alternatives (manual processes, internal tooling, consulting services)
- **Market category definition:** What "category" does this product create or belong to? Are they category creators or category followers?
- **Differentiation axis:** On what dimension does each product primarily compete? (price, features, ease of use, performance, security, integrations, support)
- **Switching costs:** How hard is it to switch from one to another?

### Research Queries

- `"[product name]" vs OR versus OR alternative OR competitor OR "compared to"`
- `"[product name]" OR "[product name] alternative" site:alternativeto.net`
- `"[product name]" OR "[product name] alternative" site:g2.com`
- `"best [product category] tools [year]"`
- `"[product category]" market landscape OR comparison OR guide`
- `"[product name]" "switched from" OR "moved from" OR "replaced"`
- `"[product name]" "better than" OR "worse than" OR "instead of"`

### Competitive Mapping Template

```
Direct Competitors:
  1. [Name] — [One-line] — Key differentiator: [X]
  2. ...

Indirect Competitors:
  1. [Name] — [One-line] — Overlap area: [X]
  2. ...

Substitutes:
  1. [Approach] — When teams choose this instead: [scenario]
  2. ...

Differentiation Matrix:
  | Dimension       | [Product] | [Competitor 1] | [Competitor 2] |
  |----------------|-----------|----------------|----------------|
  | Primary focus   |           |                |                |
  | Deployment      |           |                |                |
  | Pricing model   |           |                |                |
  | Key strength    |           |                |                |
  | Key weakness    |           |                |                |
```

---

## Dimension 7 — Strengths, Weaknesses & Risks

### Methodology

This dimension is a synthesis layer — it draws from all other dimensions rather than
requiring its own primary research. However, the following additional queries can
surface explicit positive and negative sentiment:

### Research Queries — Strengths

- `"[product name]" "love" OR "great" OR "excellent" OR "game changer" OR "best"`
- `"[product name]" "easy to" OR "simple" OR "intuitive" OR "well designed"`
- `"[product name]" "saved us" OR "reduced" OR "improved" OR "transformed"`

### Research Queries — Weaknesses

- `"[product name]" "frustrating" OR "disappointing" OR "confusing" OR "broken"`
- `"[product name]" "doesn't support" OR "can't do" OR "missing" OR "lacks"`
- `"[product name]" "switched away" OR "stopped using" OR "cancelled" OR "churned"`
- `"[product name]" "expensive" OR "overpriced" OR "not worth"`
- `"[product name]" bug OR issue OR problem OR error`

### Assessment Framework

For each product, classify findings into:

| Category | Definition | Evidence Standard |
|----------|-----------|-------------------|
| **Confirmed Strength** | Positive attribute validated by 2+ independent sources | Product site + user review/community post |
| **Claimed Strength** | Product claims it, but limited external validation | Product site only |
| **Confirmed Weakness** | Limitation noted by 2+ independent sources | Multiple user reports/reviews |
| **Potential Weakness** | Single-source concern or inferred from technical analysis | One review, or logical inference |
| **Risk Factor** | Not a current problem but a foreseeable concern | Market analysis, funding stage, competitive pressure |

### Sentiment Aggregation

For community sentiment, check at minimum:
- 3+ Reddit threads mentioning the product
- Hacker News discussions (if any)
- G2/Capterra reviews (overall score + read low-star reviews carefully)
- Product Hunt comments (if launched there)
- Twitter/X mentions (sentiment, not just volume)

Summarize as: **Enthusiastic** / **Positive** / **Mixed** / **Critical** / **Sparse** (not enough data)
