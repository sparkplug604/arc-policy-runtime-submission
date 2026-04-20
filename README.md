# Policy Runtime Submission Package

This folder is a cleaned submission-oriented package derived from a locked
working runtime reconstruction.

Design goals:
- keep the behaviorally important runtime logic
- remove snapshots, recordings, archaeology, and diagnostics clutter
- replace old internal branding with generic submission-facing names
- keep only the governance-state pieces actually needed by the bridge

## Included Components

- `agent_runner.py`
  - standalone runner for local verification
- `observation_adapter.py`
  - converts raw ARC observations into structured runtime state
- `policy_bridge.py`
  - goal/evidence bridge that scores actions and updates guidance
- `runtime_core/`
  - the full active ARC runtime package used by the working reconstruction
- `governance_state/`
  - reduced replacement for the original broader governance runtime package
  - only includes the pieces needed by `policy_bridge.py`

## What Was Intentionally Excluded

- all `*.snapshot_*`, `*.bak`, and reconstruction artifacts
- recordings and replay logs
- sidecars, analyzers, and trace-diff tools
- debug JSONL outputs
- historical notes and bug tracker documents
- the unused majority of the original internal governance runtime package

## Used Governance-State Pieces

`policy_bridge.py` only needs these public symbols:
- `MemoryCell`
- `EpistemicStatus`
- `Goal`
- `GoalPolicyClass`
- `GoalStatus`
- `PrecedenceClass`
- `SourceType`
- `compute_goal_weights`
- `create_initial_state`
- `create_runtime_support`

Those are implemented here via:
- `governance_state/cells.py`
- `governance_state/goals.py`
- `governance_state/state.py`
- `governance_state/bootstrap.py`
- plus small transitive support files:
  - `actions.py`
  - `branches.py`
  - `contradictions.py`
  - `persistence.py`

## Notes

- This is not yet a final Kaggle notebook package.
- It is the code-shaped core we can submit from.
- The next step is to wrap this logic in a notebook entrypoint that writes
  `submission.parquet` under Kaggle competition constraints.
