# Baseline snapshot — user-extensions design

Frozen at branch point `user-extensions-baseline-2026-04-17` (tag), on
top of `master` commit `6476fe7` (`Bump version to 1.01.2a8`).

Purpose: the inputs Phase 1–8 tests compare against. Per plan § 0a.4.4
these are fixtures, not docs — do not edit to "fix" a diff; investigate
the code change that produced the diff instead.

## Captured at branch point

| Artifact | File | Source command | Value |
|---|---|---|---|
| pytest collection total | `pytest_collect_only_tail.txt` | `pytest --collect-only -q | tail -n 5` | 14710 tests |
| `add_bronze_filtering` + `sps_notebook_overrides` usage counts | `rg_counts.txt` | `rg -c "add_bronze_filtering|sps_notebook_overrides" -- src docs` | 123 total across 24 files |

## Deferred (require user sign-off before running)

| Artifact | File | Why deferred |
|---|---|---|
| Generated SPS pipeline byte-level dump | `generated_sps_pipeline/` | Requires running NB00 → NB10 end-to-end against SPS data. Hours of compute. Capture before Phase 8 at the latest; Phase 1 golden-file parity tests use tutorial fixtures, not SPS. |
| `FeatureSpec.selected_features` list | `feature_spec_selected_features.txt` | Comes out of the NB10 run above. |
| `pytest --cov` tail | `pytest_cov_tail.txt` | Full coverage run on 14710 tests; not free. Capture when the user is ready to kick off a long run. |

Each deferred artifact has a zero-byte placeholder file beside this
README so the layout is visible; replace with real output when
captured. Do not backfill dates in this README — the plan's ground
truth is the baseline tag, not this doc.
