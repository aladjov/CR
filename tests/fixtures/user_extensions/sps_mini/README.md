# sps_mini — synthetic 3-dataset SPS-shaped fixture

Tier 1 fixture for `tests/integration/test_mini_sps_smoke.py`. Exercises
generator + generation_manifest + kill-switch end-to-end. **Findings-only
tier** — no raw CSV/parquet data; the generator writes scripts but nothing
runs against real rows. Upgrade to tier 2 (real trimmed SPS data) once
NB08 has finished producing a stable `recommendations.yaml` and raw
sources are trimmable.

## Shape

| Dataset | Granularity | Role | Landing user-extension |
|---|---|---|---|
| `account` | entity-level | primary entity (holds `churn` target) | — |
| `request` | event-level | events per account | landing filter: `amount > 0` |
| `contract` | entity-level | second non-event dataset | — |

Aggregation windows mirror SPS conventions: `7d`, `30d`, `90d`.

## Files

- `multi_dataset_findings.yaml` — 3-dataset roster with `request` declared
  in `event_datasets` and two relationships pointing at `account`.
- `account_findings.yaml`, `request_findings.yaml`,
  `contract_findings.yaml` — per-dataset findings.
- `recommendations.yaml` — minimal registry with one landing filter on
  `request` (`amount > 0`) + one bronze null-impute on `account.age`.

## Do not edit to silence test failures

Any diff against this fixture means the generator's contract with
registry input has changed. Investigate the code change that produced
the diff before touching the fixture.
