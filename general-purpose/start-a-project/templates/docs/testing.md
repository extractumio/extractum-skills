# Testing Map

> Last updated: __DATE__  
> Status: empty — fill in as the system takes shape.

The orientation document for testing in this project. Owns goals, tier boundaries, project expectations, and which doc owns what.

## Goals

_TBD — what does testing protect us from? Regressions, contract drift, performance, security? Order them by importance._

## Tier Boundaries

_TBD — define the tiers used in this project (e.g. unit / integration / e2e / smoke / load) and the boundary for each. A test in the wrong tier wastes CI minutes or hides bugs._

| Tier | Scope | Speed | When it runs |
|------|-------|-------|--------------|
| _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Project Expectations

_TBD — coverage targets, mandatory tier per change type (e.g. "every API change needs an integration test"), what blocks merge, what blocks deploy._

## Doc Ownership

| Topic | Document |
|-------|----------|
| How to **write** tests (placement, observables, naming) | [`how-to-write-tests.md`](how-to-write-tests.md) |
| How to **run** tests (commands, env, verification flows) | [`how-to-run-tests.md`](how-to-run-tests.md) |
| Why a test exists / what it protects | inline `#` comment in the test file |

## Anti-patterns

_TBD — list test patterns this project rejects and why (e.g. "no DB mocks at integration tier"). Each entry: pattern, reason, alternative._
