# Phase 6 — Report Format & Output Templates

Use these exact formats for the final analytical report. When analyzing multiple
products, produce one unified report with clear per-product sections and a
comparative analysis section.

---

## Report Structure

The final report must follow this section order. Every section is mandatory — if
data is unavailable for a section, include the section header with an explicit
"No public information found" note explaining what was searched.

```
1. Executive Summary
2. Per-Product Deep Dive (repeated for each product)
   2.1 Product Overview
   2.2 Problem & Value Proposition
   2.3 Target Audience
   2.4 Feature Analysis
   2.5 Deployment & Delivery
   2.6 Technology & Architecture
   2.7 Pricing & Business Model
   2.8 Strengths & Weaknesses
3. Competitive Landscape
4. Comparative Analysis (if multiple products)
5. Source Index
```

---

## Section Templates

### 1. Executive Summary

A concise synthesis — the "TL;DR" for a busy decision-maker. Maximum 300 words.

```markdown
## Executive Summary

**Products analyzed:** [Product 1], [Product 2], ...
**Analysis date:** [YYYY-MM-DD]
**Category:** [Market category these products belong to]

### Key Findings

- [Most important finding — the headline takeaway]
- [Second key finding]
- [Third key finding]

### Bottom Line

[2-3 sentences synthesizing the overall picture: what these products represent in
the market, who they serve best, and the most critical differentiators between them]
```

### 2.1 Product Overview

```markdown
## [Product Name]

| Attribute | Detail |
|-----------|--------|
| **Company** | [Legal entity / parent company] |
| **Website** | [URL] |
| **Founded** | [Year] |
| **Headquarters** | [Location] |
| **Team size** | [Approximate — from LinkedIn, Crunchbase, or about page] |
| **Funding** | [Total raised, last round, key investors] |
| **Category** | [Product category] |
| **Tagline** | [Their own tagline / hero text] |
| **Status** | [GA / Beta / Alpha / Pre-launch] |
```

### 2.2 Problem & Value Proposition

```markdown
### Problem Statement

[What specific problem does this product address? Describe the pain point in
concrete terms — not marketing language. Include who experiences this pain and
what the consequences of inaction are.]

### Solution Approach

[How does the product solve this problem? What is their specific technical or
methodological approach? What makes it different from how others solve the same
problem?]

### Value Delivered

| Value Type | Description | Evidence |
|-----------|-------------|----------|
| [Time saved / Risk reduced / Cost avoided / Capability gained / Compliance] | [Specific claim] | [Source: case study, review, product page] |
| ... | ... | ... |
```

### 2.3 Target Audience

```markdown
### Target Audience

**Primary buyer:** [Role — e.g., CISO, VP Engineering, DevOps Manager]
**Primary user:** [Role — if different from buyer]
**Company size:** [SMB / Mid-market / Enterprise / All]
**Industries:** [Specific verticals or "horizontal/cross-industry"]
**Maturity stage:** [Startups / Growth / Enterprise / All]

### Audience Evidence

| Signal | Observation | Source |
|--------|-------------|--------|
| [Customer logos] | [Names of companies shown] | [Product website] |
| [Pricing structure] | [What it implies about target] | [Pricing page] |
| [Content topics] | [What audience they write for] | [Blog] |
| [Conference presence] | [Where they present/sponsor] | [Events] |
| [Compliance certs] | [What regulated industries they target] | [Security page] |
```

### 2.4 Feature Analysis

```markdown
### Core Features

| # | Feature | Description | Maturity |
|---|---------|-------------|----------|
| 1 | [Name] | [What it does — one line] | GA / Beta / Alpha |
| 2 | ... | ... | ... |

### Integration Ecosystem

| Integration Type | Supported Platforms |
|-----------------|-------------------|
| Native integrations | [List] |
| API/SDK | [Languages, availability] |
| CI/CD | [Supported systems] |
| Cloud platforms | [AWS, GCP, Azure, etc.] |
| Identity providers | [SSO, SAML, etc.] |
| Other | [Webhooks, plugins, etc.] |

### Notable Feature Gaps

| Gap | Source | Impact |
|-----|--------|--------|
| [Missing capability] | [Where this was noted — review, forum, GitHub issue] | [Why it matters] |
| ... | ... | ... |
```

### 2.5 Deployment & Delivery

```markdown
### Deployment Model

| Model | Available? | Details |
|-------|-----------|---------|
| SaaS (hosted) | Yes/No | [Details — multi-tenant, single-tenant, region options] |
| Self-hosted / On-prem | Yes/No | [Details — Docker, Kubernetes, VM, bare metal] |
| Hybrid | Yes/No | [Details] |
| CLI tool | Yes/No | [Details — install method, OS support] |
| IDE plugin/extension | Yes/No | [Details — VS Code, JetBrains, etc.] |
| Browser extension | Yes/No | [Details] |
| GitHub Action / CI plugin | Yes/No | [Details] |
| API-only | Yes/No | [Details] |
| Agent / daemon | Yes/No | [Details — what it monitors, where it runs] |

### Setup Complexity

[Brief assessment: how long does it take to go from signup to value?
Minutes, hours, days? What's required?]
```

### 2.6 Technology & Architecture

```markdown
### Technology Stack

| Layer | Technologies | Confidence |
|-------|-------------|------------|
| **Languages** | [Primary, secondary] | Confirmed / Likely |
| **Backend framework** | [Framework] | Confirmed / Likely |
| **Frontend** | [Framework, if applicable] | Confirmed / Likely |
| **Database** | [DB technology] | Confirmed / Likely |
| **Infrastructure** | [Cloud provider, orchestration] | Confirmed / Likely |
| **AI/ML** | [Models, frameworks — if applicable] | Confirmed / Likely |
| **Key libraries** | [Notable dependencies] | Confirmed / Likely |

### Open Source Footprint

| Repository | URL | Stars | Language | License | Last Active |
|-----------|-----|-------|----------|---------|-------------|
| [Repo name] | [URL] | [Count] | [Primary lang] | [License] | [Date of last commit] |
| ... | ... | ... | ... | ... | ... |

### Architecture Notes

[Any known architectural patterns, design decisions, or technical constraints.
Note the source of each claim.]
```

### 2.7 Pricing & Business Model

```markdown
### Pricing Model

**Type:** [Freemium / Subscription / Usage-based / Open-core / Enterprise-only / Free]
**Billing:** [Monthly / Annual / Both — note annual discount if any]

### Pricing Tiers

| Tier | Price | Key Inclusions | Key Limits |
|------|-------|---------------|------------|
| Free | $0 | [What's included] | [What's limited] |
| [Tier name] | $X/mo or $X/yr | [What's included] | [What's limited] |
| [Tier name] | $X/mo or $X/yr | [What's included] | [What's limited] |
| Enterprise | Contact sales | [What's included] | [Custom] |

### Funding & Financial Context

| Event | Date | Amount | Investors | Source |
|-------|------|--------|-----------|--------|
| [Seed / Series A / etc.] | [Date] | [$Amount] | [Lead investor, others] | [Crunchbase, press release] |
| **Total raised** | | **$[Amount]** | | |

### Business Model Assessment

[Is this product-led growth or sales-led? Open-source community-driven or
commercial-first? What's the likely trajectory of monetization? Any
signals about financial health or runway?]
```

### 2.8 Strengths & Weaknesses

```markdown
### Strengths

| # | Strength | Evidence | Confidence |
|---|----------|----------|------------|
| 1 | [Strength] | [Source(s)] | Confirmed / Claimed |
| 2 | ... | ... | ... |

### Weaknesses & Limitations

| # | Weakness | Evidence | Confidence | Impact |
|---|----------|----------|------------|--------|
| 1 | [Weakness] | [Source(s)] | Confirmed / Potential | High / Medium / Low |
| 2 | ... | ... | ... | ... |

### Risk Factors

| Risk | Description | Likelihood | Impact |
|------|-------------|-----------|--------|
| [Risk type] | [What could go wrong] | High / Medium / Low | High / Medium / Low |
| ... | ... | ... | ... |

### Community Sentiment

**Overall:** [Enthusiastic / Positive / Mixed / Critical / Sparse]

| Platform | Sentiment | Sample Size | Notable Themes |
|----------|-----------|-------------|----------------|
| Reddit | [Tone] | [# of threads reviewed] | [Key themes] |
| Hacker News | [Tone] | [# of threads reviewed] | [Key themes] |
| G2/Capterra | [Score, # reviews] | [# of reviews read] | [Key themes] |
| Product Hunt | [Score, # upvotes] | [# of comments read] | [Key themes] |
| Twitter/X | [Tone] | [Volume estimate] | [Key themes] |
```

---

## 3. Competitive Landscape

```markdown
## Competitive Landscape

### Market Map

| Category | Products |
|----------|----------|
| **Direct competitors** | [Product 1, Product 2, ...] |
| **Indirect competitors** | [Product 1, Product 2, ...] |
| **Substitutes** | [Approach 1, Approach 2, ...] |

### Competitor Profiles

#### [Competitor Name]
- **Website:** [URL]
- **What they do:** [1-2 sentences]
- **How they differ:** [Key difference from analyzed products]
- **Market position:** [Larger / Smaller / Comparable — any notable context]

[Repeat for each significant competitor — minimum 5, aim for 8-12]
```

---

## 4. Comparative Analysis (Multi-Product)

Only include this section when analyzing 2+ products.

```markdown
## Comparative Analysis

### Head-to-Head Comparison

| Dimension | [Product A] | [Product B] | ... |
|-----------|------------|------------|-----|
| **Core focus** | | | |
| **Primary audience** | | | |
| **Deployment model** | | | |
| **Pricing approach** | | | |
| **Free tier** | | | |
| **Tech stack** | | | |
| **Open source** | | | |
| **Integration depth** | | | |
| **Maturity** | | | |
| **Funding** | | | |
| **Community sentiment** | | | |
| **Key strength** | | | |
| **Key weakness** | | | |

### Where Each Product Wins

**[Product A] is the better choice when:**
- [Scenario 1]
- [Scenario 2]

**[Product B] is the better choice when:**
- [Scenario 1]
- [Scenario 2]

### Key Differentiators

[Narrative synthesis — what fundamentally separates these products? Is it
technical approach, market positioning, pricing philosophy, feature depth,
or something else?]
```

---

## 5. Source Index

```markdown
## Sources

All sources consulted during this analysis, grouped by type.

### Product Websites
- [Product Name] — [URL] (accessed [date])

### Review Platforms
- [Platform] — [URL to specific review/page]

### Community Discussions
- [Platform] — [Thread title] — [URL]

### News & Press
- [Publication] — [Article title] — [URL] — [Date]

### Technical Sources
- [GitHub repo / blog post / talk] — [URL]

### Business Intelligence
- [Crunchbase / LinkedIn / etc.] — [URL]
```

---

## Confidence & Evidence Standards

Apply these markers consistently throughout the report:

| Marker | Meaning | Minimum Evidence |
|--------|---------|-----------------|
| **Confirmed** | High confidence — verified across sources | Product site + 1 independent source |
| **Likely** | Moderate confidence — strong signals | Single credible source or multiple indirect signals |
| **Unverified** | Low confidence — mentioned but not confirmed | Single mention, inference, or dated source (>2 years) |
| **Not Found** | Searched but no information available | Documented what was searched |

---

## Report Quality Checklist

Before delivering the report, verify:

- [ ] Every section has content or an explicit "Not Found" note
- [ ] Every factual claim has a source citation
- [ ] Confidence markers are applied to all non-obvious claims
- [ ] Pricing information is current (check the date on the pricing page)
- [ ] Competitor list includes at least 5 entries
- [ ] Strengths and weaknesses are balanced (not a marketing brochure)
- [ ] Community sentiment is based on actual reviewed posts, not assumptions
- [ ] Technology claims are sourced (not inferred from product category)
- [ ] The executive summary accurately reflects the detailed findings
- [ ] Multi-product comparison is fair and balanced across all products
- [ ] All URLs in the source index are real (not fabricated)
