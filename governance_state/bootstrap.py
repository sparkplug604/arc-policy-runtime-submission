from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .cells import MemoryCell
from .contradictions import Contradiction, ContradictionSeverity, ContradictionType
from .goals import Goal, GoalPolicyClass
from .persistence import FileCheckpointStore, FileEventStore
from .state import RuntimeState


DEFAULT_BUDGETS = {
    "token_budget": 512.0,
    "latency_budget_ms": 50.0,
    "energy_budget_j": 10.0,
    "memory_budget_bytes": 4096.0,
}


def create_initial_state(
    *,
    state_id: str = "runtime:initial",
    goals: Optional[Sequence[Goal]] = None,
    active_cells: Optional[Iterable[MemoryCell]] = None,
    candidate_cells: Optional[Iterable[MemoryCell]] = None,
    budgets: Optional[dict[str, float]] = None,
) -> RuntimeState:
    merged_budgets = dict(DEFAULT_BUDGETS)
    merged_budgets.update(budgets or {})
    state = RuntimeState(
        state_id=state_id,
        goals=list(
            goals
            or [
                Goal(
                    goal_id="goal:stability",
                    text="Maintain safe governed operation.",
                    hard=True,
                    policy_class=GoalPolicyClass.HARD,
                    required_bindings=["stable_runtime"],
                ),
                Goal(
                    goal_id="goal:throughput",
                    text="Activate useful governed memory without exceeding budgets.",
                    priority=1.2,
                    required_bindings=["useful_context"],
                ),
            ]
        ),
        active_cells={cell.cell_id: cell.clone() for cell in (active_cells or [])},
        candidate_cells={cell.cell_id: cell.clone() for cell in (candidate_cells or _default_candidate_cells())},
        token_budget=float(merged_budgets["token_budget"]),
        latency_budget_ms=float(merged_budgets["latency_budget_ms"]),
        energy_budget_j=float(merged_budgets["energy_budget_j"]),
        memory_budget_bytes=float(merged_budgets["memory_budget_bytes"]),
        state_vec=np.zeros(8, dtype=float),
        ref_vec=np.ones(8, dtype=float) * 0.1,
    )
    contradiction = Contradiction(
        contradiction_id="candidate:1",
        contradiction_type=ContradictionType.FACTUAL,
        severity=ContradictionSeverity.MEDIUM,
        mass=1.0,
        refs=["candidate:1"],
        summary="Candidate activation must be reconciled with existing context.",
    )
    state.contradiction_graph = {contradiction.contradiction_id: contradiction}
    state.contradictions = {contradiction.contradiction_id: contradiction.mass}
    state.telemetry_counters["max_authority_level"] = 5.0
    state.recompute_derived_state()
    state.telemetry_counters["previous_contradiction_mass"] = float(
        sum(item.penalty() for item in state.contradiction_objects().values())
    )
    return state


def create_runtime_support(root: Optional[Path] = None) -> dict[str, object]:
    root = root or Path("/tmp/governance_state_runtime")
    root.mkdir(parents=True, exist_ok=True)
    return {
        "event_store": FileEventStore(root / "events.jsonl"),
        "checkpoint_store": FileCheckpointStore(root / "checkpoints"),
    }


def _default_candidate_cells() -> list[MemoryCell]:
    cells: list[MemoryCell] = []
    for index in range(3):
        cell = MemoryCell(cell_id=f"candidate:{index + 1}")
        cell.content.text = f"Candidate runtime memory {index + 1}"
        cell.content.embedding = [0.1 * (index + 1)] * 8
        cell.metrics.goal_alignment = 0.3 + (0.2 * index)
        cell.spectral.goal_score = 0.4 + (0.1 * index)
        cell.bindings.provided_keys = ["useful_context"] if index < 2 else ["stable_runtime"]
        cell.bindings.goal_refs = ["goal:throughput"] if index < 2 else ["goal:stability"]
        cell.cost.token_cost = 8 + index
        cell.cost.estimated_latency_ms = 2.0 + index
        cell.cost.estimated_energy_j = 0.2 + (0.1 * index)
        cell.cost.memory_bytes = 128 + (64 * index)
        cells.append(cell)
    return cells
