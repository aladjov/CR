# Adding a New Apply-Op

This is a contributor guide for adding a new framework function to the
parity-audit subsystem (`customer_retention.parity`). Full architectural
context is in [Parity-Audit](Parity-Audit.md); this page is the working
checklist when you are introducing a function whose primary effect is to
return a transformed DataFrame.

## When does a function need `@apply_op`?

Yes, decorate it, if **all** of these are true:

- The function's **primary return value** is a DataFrame (or a `(df, …)`
  tuple where the DataFrame is the leading element).
- It is **invoked from an exploration notebook** (NB00–NB09) or from a
  rendered production script (`landing_<x>.py`, `bronze_*.py`, …).
- It belongs to a **stage we audit**: landing, target_derive, bronze,
  silver, gold, training.

No, leave it alone, if any of these are true:

- It returns a `dict` / `list` / model / scalar / report. Sinks and
  analysers are not apply primitives.
- It is an **internal helper** that the public orchestrator delegates to.
  Decorate the orchestrator, not the helper (`enrich_lifecycle_dataset`,
  not `_enrich_pandas` / `_enrich_distributed`).
- It is a **dispatcher** that only routes to other decorated primitives
  (`TransformExecutor.apply_all` is one such case).
- It belongs to the **scoring** stage or the causal track — out of audit
  scope today.

## The four steps

### 1. Pick a kind

Look at `customer_retention/parity/kinds.py` and pick the closest existing
`ApplyOpKind`. If genuinely none fits, add a new enum value with a
`stage.operation_name` dotted value (e.g. `landing.new_thing`) and update
the test that asserts the count in `tests/parity/test_kinds.py`.

### 2. Decorate the function

```python
from customer_retention.parity import ApplyOpKind, apply_op

@apply_op(
    kind=ApplyOpKind.BRONZE_AGGREGATE,
    capture_kwargs={"windows", "value_columns", "agg_funcs"},  # optional
    dataset_kwarg="dataset_name",  # if a parameter carries the hint
    gate="my_intent.foo is not None",  # symbolic — optional
)
def aggregate(df, *, windows, value_columns, agg_funcs, dataset_name=None):
    ...
```

Rules of thumb:

- Decorate the **public orchestrator**, not its private helpers.
- `capture_kwargs=` is optional. Leave it out to capture every kwarg; pass
  a set to restrict the fingerprint to the kwargs whose values genuinely
  identify the operation (often a small subset). Avoid capturing
  `artifact_store` / large objects.
- `dataset_kwarg=` lets the runtime tracer pick up the dataset from a
  named argument when there is no enclosing `apply_context` block.

### 3. Add it to the completeness gate

Open `tests/parity/test_apply_op_completeness.py` and add your function's
fully qualified name (`module.path.Class.method`) to the `_EXPECTED` dict
with the same kind:

```python
_EXPECTED = {
    ...
    "customer_retention.stages.profiling.time_window_aggregator.TimeWindowAggregator.aggregate":
        ApplyOpKind.BRONZE_AGGREGATE,
    ...
}
```

Run the gate test — it should pass:

```bash
pytest tests/parity/test_apply_op_completeness.py
```

If the negative-list AST sweep flags your function (e.g. it lives outside
the scanned roots, or its name doesn't match `apply_*` / `derive_*` and
the static walker missed it), and you genuinely do not want to decorate
it, add an entry to `_NOT_APPLY_OP_ALLOWLIST` with a one-line rationale
the reviewer can read in the PR.

### 4. Wire any renderer-emitted name (optional)

If the renderer emits this operation as an **inline-generated function**
(rather than calling your decorated framework function directly), add the
generated function's name to `_TEMPLATE_EMITS_KIND` in
`customer_retention/parity/production_scan.py`:

```python
_TEMPLATE_EMITS_KIND: dict[str, ApplyOpKind] = {
    ...
    "apply_my_new_thing": ApplyOpKind.MY_NEW_KIND,
    ...
}
```

This is what lets the production scan recognise the semantic operation
even though the renderer hand-codes it inline (a common pattern — see
`apply_history_window`, `derive_feature_timestamp`, etc.).

### 5. Add a renderer-contract test (recommended)

If the operation is emitted by the renderer, add an assertion in
`tests/parity/test_renderer_contract.py`:

```python
def test_my_new_op_emitted(self, generated_sps_mini):
    manifest = scan_generated_pipeline(generated_sps_mini, scope=AuditScope.BRONZE)
    kinds = {e.kind for e in manifest.entries}
    assert ApplyOpKind.MY_NEW_KIND in kinds
```

This locks in the invariant that the renderer keeps emitting the op
after future refactors.

## What happens at runtime

When `CR_PARITY_TRACE=1` is set:

1. Each invocation of your decorated function appends a `TraceRecord` to
   a thread-local buffer with `(kind, dataset, kwargs_fingerprint,
   input_rows, output_rows)`.
2. At the end of the run, the trace is flushed to a YAML file in the run
   namespace.
3. The T3 audit (`audit_trace`) diffs exploration and production YAMLs
   and reports row-count drift beyond the per-kind tolerance as a
   `RUNTIME_DRIFT` parity gap.

With the env var unset the wrapper is a single `if trace_active()` branch
— zero overhead in normal production.

## Adding a tolerance override

The default tolerance is 0.5%. If your operation's output row count
inherently varies between runs (cancellation rate, sampling, etc.),
widen the tolerance in `customer_retention/parity/trace.py`:

```python
TOLERANCE_BY_KIND: dict[ApplyOpKind, float] = {
    ApplyOpKind.LIFECYCLE_ENRICH: 0.05,  # 5% — doubling rate varies
    ApplyOpKind.MY_NEW_KIND: 0.02,        # 2%
}
```

## Common pitfalls

- **Decorating helpers**: produces duplicate entries in the manifest
  because the orchestrator is also decorated. Decorate one or the other,
  not both.
- **Heavy `capture_kwargs`**: capturing a 10K-element list as a kwarg
  fingerprint blows up the YAML trace. Restrict to the parameters that
  actually identify the operation.
- **Forgetting the completeness gate**: CI will fail until either the
  decoration is in place or the function is allowlisted.

## References

- Closed enum of kinds: `src/customer_retention/parity/kinds.py`
- The decorator + registry: `src/customer_retention/parity/decorator.py`
- AST walkers: `src/customer_retention/parity/exploration_scan.py`,
  `production_scan.py`
- Audit orchestrators: `src/customer_retention/parity/audit.py`,
  `trace.py`
- Pre-flight notebook: `exploration_notebooks/-1_parity_contract.ipynb`
- Full architecture: [Parity-Audit](Parity-Audit.md)
