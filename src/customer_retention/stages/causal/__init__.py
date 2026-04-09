"""Causal modeling track.

Post-training stage that consumes the production churn model and produces the
Delta tables that feed the CSM-facing dashboard. Implements the data model
specified in ``docs/playbook_execution_data_model.md``:

- Definition layer (sub-layers 1, 2, 3, 4): ``playbook_catalog``,
  ``playbook_steps``, ``response_schemas``, ``vocabularies``,
  ``archetype_catalog``, ``eligibility_policy``, ``decision_policy``.
- Instance layer: ``eligibility_snapshot`` (per scoring run, per account, per
  playbook).
- Analytical-only DDL: ``assignments``, ``actions``, ``outcomes`` (defined for
  the writeback contract; not populated by this stage).

The package is organized as a thin library that the single generated notebook
``s12_playbook_assignment`` orchestrates via cell-level guards. See
``docs/playbook_execution_data_model.md`` for the full specification and
``/Users/Vital/.claude/plans/silly-drifting-blossom.md`` for the implementation
plan.
"""
