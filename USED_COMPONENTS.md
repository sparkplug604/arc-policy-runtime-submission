# Used Components

This manifest lists the files that are actually in the active import graph of
`agent_runner.py`.

## Root Modules

- `agent_runner.py`
- `observation_adapter.py`
- `policy_bridge.py`

## Runtime Core

- `runtime_core/__init__.py`
- `runtime_core/actions.py`
- `runtime_core/adapter.py`
- `runtime_core/bridge.py`
- `runtime_core/diff.py`
- `runtime_core/policy.py`
- `runtime_core/priors.py`
- `runtime_core/runtime.py`
- `runtime_core/state.py`
- `runtime_core/traces.py`
- `runtime_core/types.py`

## Governance State

- `governance_state/__init__.py`
- `governance_state/actions.py`
- `governance_state/bootstrap.py`
- `governance_state/branches.py`
- `governance_state/cells.py`
- `governance_state/contradictions.py`
- `governance_state/goals.py`
- `governance_state/persistence.py`
- `governance_state/state.py`

## Not Included

The following categories were intentionally left out of this folder:

- historical snapshots and backups
- recordings and replay artifacts
- debug traces and event dumps
- sidecars and analyzer scripts
- unused governance/runtime modules
- archived notes and reconstruction files
