# Backlog — Tech Debt & Follow-ups

> Last updated: __DATE__  
> Status: empty — fill in as items accumulate.

The **durable** backlog inbox. Tech debt, refactor candidates, deferred fixes, and follow-ups discovered while working on something else — all land here.

Items here are **not** plans. They are notes-to-future-self. When an item is ready to be worked on, promote it: create a new plan under `docs/ephemeral/plans/<YYYY-MM-DD>-<slug>/` and remove the item from this file.

## Rules

- **Add the moment you discover it.** Do not trust yourself to remember after the current task.
- **One line per item, dated, with origin.** Origin = the PR, issue, or `file:line` that prompted it. Without origin you cannot judge relevance later.
- **No expansion in place.** Discussion, design, or sub-tasks belong in a plan, not here.
- **Prune regularly.** Items that have aged out without being championed get archived or deleted.

Format:

```
- YYYY-MM-DD — <one-line description> (origin: <PR # / issue # / path/to/file.ext:LINE>)
```

## Tech Debt

_Long-lived shortcuts that should be paid back. Each entry should name the cost (what's slow, brittle, or risky because of it)._

_(none yet)_

## Refactor Candidates

_Code that works but is hard to read, change, or test. Each entry should name the trigger that would make the refactor worth doing._

_(none yet)_

## Deferred Fixes

_Bugs that were observed but explicitly postponed (workaround applied, low priority, blocked on something). Each entry should name the workaround and the unblocking condition._

_(none yet)_

## Follow-ups

_Smaller items: TODO comments lifted from code, tightening that fell out of scope, doc nits noticed in passing._

_(none yet)_
