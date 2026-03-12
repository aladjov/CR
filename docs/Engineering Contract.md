# Engineering Contract

This plan refactors the pipeline to produce a single point-in-time snapshot per `as_of` and reuse it across multiple model types and training tiers, while keeping changes small, predictable, and test-driven.

The plan is executed in phases. Each phase is intentionally narrow: one conceptual change, minimal files, explicit tests, and a clear “definition of done”.

---

## Engineering Contract for Every Phase

### 1) Locate the best insertion point before editing
- Before making changes, identify where the concept already exists (or almost exists).
- Prefer enhancing existing abstractions over adding parallel logic.
- Avoid duplication across pandas vs PySpark templates by factoring shared logic where practical.

### 2) Ask questions before making architectural decisions
- If the spec leaves multiple valid design choices, stop and ask targeted questions.
- Do not silently pick a new abstraction boundary, naming convention, or lifecycle change.
- When unclear, implement the smallest change that preserves backwards compatibility and makes future intent explicit via tests.

### 3) Test-driven development is mandatory
- Write tests first from the phase specification.
- Coverage target: ≥90% for changed modules.
- Include edge cases and anti-pattern tests, not only happy paths.
- Any bug discovered must get a dedicated test first, then a fix.
- After implementation, re-check behavior against the specification.

### 4) CI discipline
- Run ruff and ensure no new violations.
- Fix any ruff errors that are not explicitly ignored in CI.

### 5) Code style constraints
- No comments; use descriptive names instead.
- Prefer small single-responsibility functions.
- Avoid side effects; prefer pure functions.
- Keep functions on one abstraction level.
- Start with public/high-level APIs, then private helpers.
- Prefer compact code (avoid variables used only once).
- Do not spread method arguments across many lines unless >5.
- Avoid repetition; use inheritance only when it reduces duplication and remains clear.
- Make everything testable.

---

# Phase Template (Use This For Every Agent Task)

Each phase is delivered as a single agent task with the following structure:

1) Phase goal (1–2 sentences)
2) Questions (only if the phase introduces an unavoidable choice)
3) Best insertion point analysis (where to implement to avoid duplication)
4) Tests to add (written first)
   - happy path tests
   - edge case tests
   - anti-pattern tests (regressions)
5) Implementation steps (minimal file set, explicit edits)
6) Validation steps
   - run test suite
   - run ruff
   - re-check spec compliance
7) Definition of done (objective checklist)

---
