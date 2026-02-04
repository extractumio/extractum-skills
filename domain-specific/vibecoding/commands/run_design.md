---
name: run_design
description: Design and plan a new feature or change with comprehensive analysis, user consultation, and detailed implementation plan
---

# Feature Design and Planning Workflow

Rigorous design process for implementing new features or changes. Follow each phase sequentially. Do not skip steps.

**Input:** `$ARGUMENTS` — Feature description or change request

---

## Design Principles

These principles apply to every phase. Violations are grounds for rejecting an approach.

### Completeness

- Every feature in the design is mandatory. No "nice to have" sections.
- Each step must be complete. No stubs, placeholders, or "enhance later."
- Reduce scope rather than deliver incomplete work.
- Tests are part of implementation, not a follow-up task.
- Security is implemented alongside functionality, never deferred.

### Code Quality

- **Follow existing patterns** — match style and conventions in the codebase.
- **Reuse existing code** — import existing utilities, constants, configs.
- **Type safety** — all code must be properly typed.
- **Be complete** — no TODOs, no placeholders.
- **Fail-fast** — avoid capturing and suppressing exceptions; throw so errors are visible.
- **No backward-compatibility hacks** — unless explicitly required, do not add fallbacks. Remove unused leftovers from previous implementations.

### Anti-Patterns

| Anti-Pattern | What to Do Instead |
|--------------|-------------------|
| Jump to implementation | Complete all phases first |
| Single solution focus | Explore 3+ approaches |
| Ignore existing patterns | Study codebase first |
| Incomplete tests | Tests are mandatory for every function |
| Optional features | Make mandatory or remove from scope |
| Vague implementation steps | Provide specific code examples |
| Skip user consultation | Always get approval before implementation |
| Scope creep in improvements | Only include changes that directly benefit the feature |

---

## Phase 1: Context Gathering

### 1.1 Study Project Documentation

Read and internalize:

1. **Project README / CLAUDE.md** — structure, patterns, gotchas
2. **Architecture documentation** — system design, component relationships
3. **Security documentation** — security model, audit checklists

Identify which architectural layers the feature touches. Scan documentation directories for relevant files.

### 1.2 Scope Determination

Map the feature to relevant documentation areas:

| If Feature Involves | Study |
|---------------------|-------|
| Security, permissions, sandboxing | Security model docs, permission configs |
| File operations, paths | Path validation/resolution docs |
| Environment variables, secrets | Secrets management docs |
| Events, streaming, real-time | Event/SSE architecture docs |
| Hooks, callbacks, lifecycle | Event hook system docs |
| User interaction, prompts | Human-in-the-loop / interaction docs |
| Web UI, frontend | Frontend architecture docs |
| Task execution, queuing | Task queue / scheduling docs |
| External service integration | Integration / proxy docs |
| Request filtering, rate limiting | WAF / middleware docs |
| Database operations | Schema, migration, ORM docs |

Read ALL relevant documents before proceeding. Summarize key constraints and patterns that apply.

### 1.3 Identify Constraints

Document constraints from three categories:

1. **Security** — which security requirements apply (see [Appendix A](#appendix-a-security-requirements))
2. **Performance** — latency, throughput, resource limits
3. **Compatibility** — existing APIs, data formats, migration needs

---

## Phase 2: Solution Brainstorming

### 2.1 Generate Multiple Approaches

Generate at least 3 distinct implementation approaches. For each:

```
APPROACH [N]: [Name]
====================

Overview:
[1-2 sentence description]

Technical Strategy:
- [How it works at high level]
- [Key components involved]
- [Integration points]

Pros:
+ [Advantage 1]
+ [Advantage 2]
+ [Advantage 3]

Cons:
- [Disadvantage 1]
- [Disadvantage 2]
- [Disadvantage 3]

Risks:
! [Technical risk]
! [Security risk]
! [Maintenance/complexity risk]

Effort Estimate:
- Files to modify: [count]
- New files: [count]
- Test coverage: [scope]

Dependencies:
- [Existing code/libs to leverage]
- [New dependencies — MUST be vetted]
```

### 2.2 Evaluation Criteria

Evaluate each approach against:

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | Security posture — maintains or enhances the security model | Critical |
| 2 | Attack surface — minimizes new attack vectors | Critical |
| 3 | Least privilege — requests only necessary permissions | Critical |
| 4 | Fail-closed — denies by default on errors | Critical |
| 5 | Pattern alignment — follows established codebase conventions | High |
| 6 | DRY — maximizes reuse of existing code | High |
| 7 | Testability — can be comprehensively tested | High |
| 8 | Maintainability — easy to understand and modify | Medium |
| 9 | Performance — no unacceptable latency or resource cost | Medium |
| 10 | User experience — impact on end users | Medium |

### 2.3 Rejection Criteria

**Reject any approach that:**

- Trusts user input without validation
- Uses deprecated or vulnerable dependencies
- Stores secrets in code, environment variables, command line, or logs
- Bypasses existing security controls (path validators, command filters, etc.)
- Allows arbitrary writes outside designated areas
- Requires disabling security profiles or protections
- Runs with unnecessarily elevated privileges

Add project-specific rejection criteria from the security documentation.

---

## Phase 3: User Consultation

### 3.1 Present Alternatives

```
I've analyzed [N] approaches for implementing [feature]:

**Approach 1: [Name]** — [One-line summary]
[Key differentiator]
Security: [green/yellow/red]

**Approach 2: [Name]** — [One-line summary]
[Key differentiator]
Security: [green/yellow/red]

**Approach 3: [Name]** — [One-line summary]
[Key differentiator]
Security: [green/yellow/red]

My recommendation: [Approach N] because [reason].

Which direction would you like to explore?
```

### 3.2 Gather User Input

Collect user preference. Follow up on ambiguities:

- Scope boundaries (what's in/out)
- Priority trade-offs
- Integration preferences
- Constraints not yet known
- Security trade-offs (user must explicitly acknowledge any)

**Do not proceed to Phase 4 until user explicitly approves an approach.**

---

## Phase 4: Detailed Design

### 4.1 Component Breakdown

Expand the selected approach:

1. **Component breakdown** — every file to create or modify
2. **Interface design** — function signatures, class interfaces, API contracts
3. **Data flow** — how data moves through the system
4. **State management** — what state changes, where it's stored
5. **Error handling** — failure modes and recovery strategies

### 4.2 Security Design

Perform the security analysis defined in [Appendix A](#appendix-a-security-requirements):

1. Build the threat model (assets, actors, vectors, mitigations)
2. Map to project security checklist items
3. Complete the security controls checklist
4. Verify no rejection criteria are violated

### 4.3 Alternative Implementations

Even within the selected approach, explore variations:

- Different data structures or decomposition strategies
- Different API designs
- Different test strategies
- Edge-vs-layer validation trade-offs

Document why specific choices were made.

### 4.4 Improvement Opportunities

Identify changes that **directly benefit** the feature:

1. **Refactoring** — code that should be restructured to support the feature
2. **Technical debt** — existing issues worth fixing alongside this work
3. **Consolidation** — duplicated code that could be unified
4. **Pattern alignment** — deviations from codebase conventions

Do not scope-creep.

---

## Phase 5: Design Document

Create at the project's designated plans location (e.g., `docs/plans/<feature-name>.md`).

### 5.1 Document Template

```markdown
# [Feature Name] Implementation Plan

## 1. Overview
[Problem statement, goals, success criteria]

## 2. Security Summary
- Attack surface change: [increased/decreased/unchanged]
- New permissions required: [list or "none"]
- Sensitive data handling: [yes/no — if yes, describe]
- Security issues identified: [list or "none"]

## 3. Architecture Impact
[Which layers are affected, integration diagram]

## 4. Detailed Design

### 4.1 Component Changes
[For each file to modify:]
- File: `path/to/file`
- Changes: [description]
- Reason: [why needed]

### 4.2 New Components
[For each new file:]
- File: `path/to/new_file`
- Purpose: [what it does]
- Dependencies: [what it imports/uses]

### 4.3 Configuration Changes
[For each config file:]
- File: `path/to/config`
- Changes: [exact additions/modifications]

### 4.4 API Changes
[If any endpoints added/modified:]
- Endpoint: [METHOD /path]
- Request/Response schemas
- Error codes
- Authentication: [required/optional]
- Authorization: [permissions needed]
- Rate limiting: [if applicable]
- Input validation: [validations applied]

### 4.5 Database Changes
[If any schema changes:]
- Table: [name]
- Migration: [description]

## 5. Security Design
[Threat model, security controls, checklist mapping — per Appendix A]

## 6. Implementation Steps

### Step 1: [Name]
**Files:** [list]
**Changes:**
```
# Actual code or changes — be specific
```
**Rationale:** [why this step first]

No backward-compatibility hacks unless explicitly required. Follow fail-fast principle.

### Step 2: [Name]
...

## 7. Test Plan

### 7.1 Unit Tests
[For each component:]
- Test file: `tests/.../test_*`
- Coverage: [scenarios]

### 7.2 Integration Tests
[Cross-component tests]

### 7.3 Security Tests
[Per Appendix A — test malicious inputs, boundary conditions, bypasses]

## 8. Rollback Strategy
[How to revert if something goes wrong. Migration rollback if applicable.]

## 9. Validation Commands
```
[Project-specific build/test commands]
```

## 10. Documentation Updates
[Which docs need updating and what to add]

## 11. Validation Checklist

### Implementation
- [ ] All implementation steps completed
- [ ] All configuration changes applied
- [ ] Unit tests written and passing
- [ ] Integration tests written and passing
- [ ] Build succeeds
- [ ] All tests pass
- [ ] Feature works as specified (manual verification)
- [ ] No regressions in existing functionality
- [ ] Documentation updated
- [ ] Code follows existing patterns
- [ ] DRY principle followed

### Security
- [ ] Threat model documented and mitigations implemented
- [ ] Security controls checklist completed (Appendix A)
- [ ] Security tests written and passing
- [ ] No CRITICAL or HIGH severity issues introduced
```

### 5.2 Test Quality Requirements

Tests must:

1. **Be comprehensive** — happy path, edge cases, error conditions
2. **Reuse existing fixtures** — use the project's test infrastructure
3. **Follow existing patterns** — match test style in the codebase
4. **Be specific** — each test tests one thing
5. **Include assertions** — clear assertions with meaningful error messages

---

## Phase 6: Documentation Updates

Identify which project documentation needs updates based on the feature:

| Change Type | Documents to Update |
|-------------|-------------------|
| New architectural component | Architecture docs |
| New security mechanism | Security model docs |
| New event types | Event/streaming docs |
| New UI patterns | Frontend docs |
| New API endpoints | API reference docs |
| New tools / plugins | Tool-specific docs |
| Configuration changes | Config reference docs |

Include specific additions/changes in the design document. Follow each document's existing style.

---

## Execution Protocol

1. **Present the design document** to the user for final review
2. **Wait for approval** before any implementation begins
3. **Execute step-by-step** following the implementation steps
4. **Run validation** after each major step
5. **Complete the checklist** item by item
6. **Final verification** with full build and test suite

---

## Appendix A: Security Requirements

This appendix consolidates all security requirements. Reference it from phases rather than duplicating inline.

### A.1 Security Principle

Security is not optional. Every feature must:

- Maintain or enhance the existing security model
- Follow the fail-closed principle (deny by default)
- Be resilient against privilege escalation
- Protect against common attack vectors (OWASP Top 10)
- Never introduce secrets exposure risks
- Preserve isolation guarantees

### A.2 Threat Model Template

Required for every feature in Phase 4.2.

```
THREAT MODEL
============

Assets at Risk:
- [What data/resources could be compromised?]

Threat Actors:
- [Who might attack? Malicious user, compromised component, external attacker?]

Attack Vectors:
- [How could they attack?]

Mitigations:
- [How does the design prevent each attack?]
```

### A.3 Security Controls Checklist

Complete during Phase 4.2. Mark items as N/A when they genuinely don't apply.

**Input Validation:**
- [ ] All user input validated
- [ ] Type checking enforced
- [ ] Length limits applied
- [ ] Character restrictions where applicable

**Output Encoding:**
- [ ] Output encoded for its context (HTML, SQL, shell, etc.)
- [ ] Parameterized queries for database operations
- [ ] Command arguments properly escaped (prefer list args over shell strings)

**Access Control:**
- [ ] Authentication required where applicable
- [ ] Authorization checked
- [ ] Least privilege enforced
- [ ] File path validation for all file operations
- [ ] Command filtering for all command executions

**Secrets Protection:**
- [ ] No secrets in code
- [ ] No secrets in logs
- [ ] No secrets in error messages
- [ ] Secrets loaded from secure source (secret manager, encrypted config)

**Infrastructure Security:**
- [ ] Minimal privileges / capabilities
- [ ] Security profiles applied (if applicable)
- [ ] No unnecessary mounts or exposed paths
- [ ] Non-root execution (where applicable)
- [ ] Resource limits enforced

### A.4 Security Test Requirements

Security tests are mandatory, not optional. They must cover:

1. **Input validation** — malformed, oversized, special character inputs
2. **Injection attacks** — SQL, command, LDAP, XPath, path traversal
3. **Authentication bypass** — missing tokens, invalid tokens, expired tokens
4. **Authorization bypass** — access resources without permission
5. **Privilege escalation** — attempt to gain higher privileges
6. **Information disclosure** — error messages don't leak sensitive data
7. **DoS resilience** — resource limits are enforced

### A.5 Security Anti-Patterns

| Anti-Pattern | What to Do Instead |
|--------------|-------------------|
| Trust user input | Validate ALL inputs at system boundaries |
| Log sensitive data | Never log secrets, tokens, passwords |
| Fail-open on errors | Always fail-closed |
| Hardcode secrets | Use secure secret storage |
| Run with elevated privileges | Use least-privilege execution |
| Disable security profiles | Keep all security profiles active |
| Skip security tests | Security tests are mandatory |
| Defer security hardening | Security ships with the feature |
| SQL string concatenation | Use parameterized queries |
| Shell command strings | Use subprocess with list args |
| Unvalidated file paths | Validate all paths through path validator |
| Trust error messages from dependencies | Sanitize all error outputs to users |
| Excessive permissions/capabilities | Drop all, add only what's required |

### A.6 Severity Reference

| Severity | Description | Action |
|----------|-------------|--------|
| CRITICAL | Immediate exploitation, direct unauthorized access, data breach | **Must fix before merge** |
| HIGH | Privilege escalation, significant security weakness | **Must fix before merge** |
| MEDIUM | Defense-in-depth issue, requires additional vulnerabilities to exploit | Should fix, document if deferred |
| LOW | Best practice, minimal direct security impact | Fix if low effort |

**The design must not introduce any new CRITICAL or HIGH severity issues.**

---

Now begin with Phase 1 for the following feature request:

**Feature:** $ARGUMENTS
