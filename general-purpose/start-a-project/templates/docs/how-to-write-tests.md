# How to Write Tests

> Last updated: __DATE__  
> Status: empty — fill in as the system takes shape.

The hands-on guide for authoring a test in this project. Pair with [`testing.md`](testing.md) (orientation) and [`how-to-run-tests.md`](how-to-run-tests.md) (execution).

## Test Placement

_TBD — directory layout for tests, naming conventions (`test_*.py`, `*.spec.ts`, `*_test.go`), per-tier locations._

| Tier | Path | Naming |
|------|------|--------|
| _TBD_ | _TBD_ | _TBD_ |

## Required Observables

_TBD — what every test must assert on (return value, side effect, persisted state, emitted event). A test that only checks "it didn't crash" is not a test._

## Coverage Expectations by Tier

_TBD — per tier: coverage target, what counts, what is excluded._

## Fixtures and Factories

_TBD — where shared fixtures live, the convention for project-specific factories, and what may be reused vs. what must be local to a test file._

## Test Doubles Policy

_TBD — when mocks/stubs/fakes are allowed and which boundaries they may cross. Reference the anti-patterns list in [`testing.md`](testing.md)._

## Hermeticity

_TBD — every test must be deterministic and self-contained. List the project's rules: clock control, network policy, temp dirs, DB isolation._

## Naming a Test

_TBD — preferred naming pattern. Aim for "given/when/then" or "what it protects" rather than "test_function_name"._
