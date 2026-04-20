from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional
import hashlib
import json
import numpy as np

from .cells import CellStatus, MemoryCell


@dataclass
class RuntimeState:
    state_id: str
    snapshot_version: str = "runtime_state.v3"
    active_cells: Dict[str, MemoryCell] = field(default_factory=dict)
    candidate_cells: Dict[str, MemoryCell] = field(default_factory=dict)
    archived_cells: Dict[str, MemoryCell] = field(default_factory=dict)
    goals: List[Any] = field(default_factory=list)
    required_bindings: List[str] = field(default_factory=list)
    satisfied_bindings: List[str] = field(default_factory=list)
    contradictions: Dict[str, float] = field(default_factory=dict)
    contradiction_graph: Dict[str, Any] = field(default_factory=dict)
    branch_probs: Dict[str, float] = field(default_factory=lambda: {"main": 1.0})
    branch_metadata: Dict[str, Any] = field(default_factory=dict)
    state_vec: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=float))
    ref_vec: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=float))
    policy_violation_mass: float = 0.0
    authority_violation_mass: float = 0.0
    temporal_staleness: float = 0.0
    token_cost: float = 0.0
    token_budget: float = 1.0
    latency_ms: float = 0.0
    latency_budget_ms: float = 1.0
    energy_j: float = 0.0
    energy_budget_j: float = 1.0
    memory_bytes: float = 0.0
    memory_budget_bytes: float = 1.0
    checkpoint_ref: Optional[str] = None
    checkpoint_snapshots: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)
    action_receipts: Dict[str, Any] = field(default_factory=dict, repr=False)
    telemetry_counters: Dict[str, float] = field(default_factory=dict)
    event_refs: List[str] = field(default_factory=list)
    phase: int = 0

    def __post_init__(self) -> None:
        self.goals = self._normalize_goals(self.goals)
        self.contradiction_graph = self._normalize_contradiction_graph(self.contradiction_graph)
        self.branch_metadata = self._normalize_branch_metadata(self.branch_metadata)
        self.action_receipts = self._normalize_action_receipts(self.action_receipts)
        if "main" not in self.branch_metadata:
            self.branch_metadata["main"] = self._default_branch_metadata("main")

    def operational_active_cells(self) -> Dict[str, MemoryCell]:
        return {k: v for k, v in self.active_cells.items() if v.lifecycle.status == CellStatus.ACTIVE}

    @staticmethod
    def _default_branch_metadata(branch_id: str) -> Any:
        from .branches import BranchMetadata

        return BranchMetadata(branch_id=branch_id)

    @staticmethod
    def _normalize_goals(goals: List[Any]) -> List[Any]:
        from .goals import Goal

        normalized: List[Goal] = []
        for index, goal in enumerate(goals):
            if isinstance(goal, Goal):
                normalized.append(goal.clone())
            elif isinstance(goal, str):
                normalized.append(Goal(goal_id=f"goal:{index + 1}", text=goal))
            elif isinstance(goal, dict):
                normalized.append(Goal.from_dict(goal))
            else:
                raise TypeError(f"Unsupported goal value: {goal!r}")
        return normalized

    @staticmethod
    def _normalize_contradiction_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
        from .contradictions import Contradiction

        normalized: Dict[str, Contradiction] = {}
        for key, value in graph.items():
            if isinstance(value, Contradiction):
                normalized[key] = value.clone()
            elif isinstance(value, dict):
                normalized[key] = Contradiction.from_dict(value)
            else:
                raise TypeError(f"Unsupported contradiction value for {key!r}: {value!r}")
        return normalized

    @staticmethod
    def _normalize_branch_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        from .branches import BranchMetadata

        normalized: Dict[str, BranchMetadata] = {}
        for key, value in metadata.items():
            if isinstance(value, BranchMetadata):
                normalized[key] = value.clone()
            elif isinstance(value, dict):
                normalized[key] = BranchMetadata.from_dict(value)
            else:
                raise TypeError(f"Unsupported branch metadata for {key!r}: {value!r}")
        return normalized

    @staticmethod
    def _normalize_action_receipts(receipts: Dict[str, Any]) -> Dict[str, Any]:
        from .actions import ActionReceipt

        normalized: Dict[str, ActionReceipt] = {}
        for key, value in receipts.items():
            if isinstance(value, ActionReceipt):
                normalized[key] = value.clone()
            elif isinstance(value, dict):
                normalized[key] = ActionReceipt.from_dict(value)
            else:
                raise TypeError(f"Unsupported action receipt for {key!r}: {value!r}")
        return normalized

    def goal_objects(self) -> List[Any]:
        return list(self.goals)

    def contradiction_objects(self) -> Dict[str, Any]:
        from .contradictions import Contradiction, ContradictionSeverity, ContradictionType

        merged = {k: v.clone() for k, v in self.contradiction_graph.items()}
        for key, mass in self.contradictions.items():
            if key not in merged:
                merged[key] = Contradiction(
                    contradiction_id=key,
                    contradiction_type=ContradictionType.FACTUAL,
                    severity=ContradictionSeverity.MEDIUM,
                    mass=float(mass),
                    refs=[key],
                    summary="Legacy contradiction mass",
                )
        return merged

    def branch_info(self, branch_id: str) -> Any:
        if branch_id not in self.branch_metadata:
            self.branch_metadata[branch_id] = self._default_branch_metadata(branch_id)
        return self.branch_metadata[branch_id]

    def normalize_branch_distribution(self) -> None:
        if not self.branch_probs:
            self.branch_probs = {"main": 1.0}
            return
        cleaned = {k: float(v) for k, v in self.branch_probs.items() if np.isfinite(v) and v >= 0.0}
        if not cleaned:
            self.branch_probs = {"main": 1.0}
            return
        total = sum(cleaned.values())
        if total <= 0:
            self.branch_probs = {"main": 1.0}
            return
        self.branch_probs = {k: v / total for k, v in cleaned.items()}

    def recompute_derived_state(self, *, normalize_branches: bool = True) -> None:
        cells = self.operational_active_cells()
        satisfied: set[str] = set()
        token_cost = 0.0
        latency_ms = 0.0
        energy_j = 0.0
        memory_bytes = 0.0
        vec = np.zeros_like(self.state_vec)
        for cell in sorted(cells.values(), key=lambda item: item.cell_id):
            satisfied.update(cell.bindings.provided_keys)
            token_cost += float(cell.cost.token_cost)
            latency_ms += float(cell.cost.estimated_latency_ms)
            energy_j += float(cell.cost.estimated_energy_j)
            memory_bytes += float(cell.cost.memory_bytes)
            if cell.content.embedding and len(cell.content.embedding) == len(vec):
                vec = vec + np.asarray(cell.content.embedding, dtype=float)
        for receipt in self.action_receipts.values():
            token_cost += float(receipt.resource_usage.token_cost)
            latency_ms += float(receipt.resource_usage.latency_ms)
            energy_j += float(receipt.resource_usage.energy_j)
            memory_bytes += float(receipt.resource_usage.memory_bytes)
        self.satisfied_bindings = sorted(satisfied)
        self.token_cost = token_cost
        self.latency_ms = latency_ms
        self.energy_j = energy_j
        self.memory_bytes = memory_bytes
        self.state_vec = vec
        if normalize_branches:
            self.normalize_branch_distribution()

    def capture_snapshot(self) -> Dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "state_id": self.state_id,
            "active_cells": {k: v.clone() for k, v in self.active_cells.items()},
            "candidate_cells": {k: v.clone() for k, v in self.candidate_cells.items()},
            "archived_cells": {k: v.clone() for k, v in self.archived_cells.items()},
            "goals": [goal.clone() for goal in self.goal_objects()],
            "required_bindings": list(self.required_bindings),
            "satisfied_bindings": list(self.satisfied_bindings),
            "contradictions": dict(self.contradictions),
            "contradiction_graph": {k: v.clone() for k, v in self.contradiction_graph.items()},
            "branch_probs": dict(self.branch_probs),
            "branch_metadata": {k: v.clone() for k, v in self.branch_metadata.items()},
            "state_vec": np.array(self.state_vec, copy=True),
            "ref_vec": np.array(self.ref_vec, copy=True),
            "policy_violation_mass": self.policy_violation_mass,
            "authority_violation_mass": self.authority_violation_mass,
            "temporal_staleness": self.temporal_staleness,
            "token_cost": self.token_cost,
            "token_budget": self.token_budget,
            "latency_ms": self.latency_ms,
            "latency_budget_ms": self.latency_budget_ms,
            "energy_j": self.energy_j,
            "energy_budget_j": self.energy_budget_j,
            "memory_bytes": self.memory_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "checkpoint_ref": self.checkpoint_ref,
            "action_receipts": {k: v.clone() for k, v in self.action_receipts.items()},
            "telemetry_counters": dict(self.telemetry_counters),
            "event_refs": list(self.event_refs),
            "phase": self.phase,
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Dict[str, Any],
        *,
        state_id: Optional[str] = None,
        checkpoint_snapshots: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "RuntimeState":
        state = cls(
            state_id=state_id or snapshot["state_id"],
            snapshot_version=snapshot.get("snapshot_version", "runtime_state.v2"),
            active_cells={k: v.clone() for k, v in snapshot["active_cells"].items()},
            candidate_cells={k: v.clone() for k, v in snapshot["candidate_cells"].items()},
            archived_cells={k: v.clone() for k, v in snapshot["archived_cells"].items()},
            goals=list(snapshot.get("goals", [])),
            required_bindings=list(snapshot["required_bindings"]),
            satisfied_bindings=list(snapshot["satisfied_bindings"]),
            contradictions=dict(snapshot["contradictions"]),
            contradiction_graph=dict(snapshot.get("contradiction_graph", {})),
            branch_probs=dict(snapshot["branch_probs"]),
            branch_metadata=dict(snapshot.get("branch_metadata", {})),
            state_vec=np.array(snapshot["state_vec"], copy=True),
            ref_vec=np.array(snapshot["ref_vec"], copy=True),
            policy_violation_mass=float(snapshot["policy_violation_mass"]),
            authority_violation_mass=float(snapshot["authority_violation_mass"]),
            temporal_staleness=float(snapshot["temporal_staleness"]),
            token_cost=float(snapshot["token_cost"]),
            token_budget=float(snapshot["token_budget"]),
            latency_ms=float(snapshot["latency_ms"]),
            latency_budget_ms=float(snapshot["latency_budget_ms"]),
            energy_j=float(snapshot["energy_j"]),
            energy_budget_j=float(snapshot["energy_budget_j"]),
            memory_bytes=float(snapshot["memory_bytes"]),
            memory_budget_bytes=float(snapshot["memory_budget_bytes"]),
            checkpoint_ref=snapshot["checkpoint_ref"],
            checkpoint_snapshots=deepcopy(checkpoint_snapshots or {}),
            action_receipts=dict(snapshot.get("action_receipts", {})),
            telemetry_counters=dict(snapshot.get("telemetry_counters", {})),
            event_refs=list(snapshot["event_refs"]),
            phase=int(snapshot["phase"]),
        )
        state.recompute_derived_state()
        return state

    def clone(self, *, state_id: Optional[str] = None) -> "RuntimeState":
        copied = replace(self)
        copied.active_cells = {k: v.clone() for k, v in self.active_cells.items()}
        copied.candidate_cells = {k: v.clone() for k, v in self.candidate_cells.items()}
        copied.archived_cells = {k: v.clone() for k, v in self.archived_cells.items()}
        copied.goals = [goal.clone() for goal in self.goal_objects()]
        copied.required_bindings = list(self.required_bindings)
        copied.satisfied_bindings = list(self.satisfied_bindings)
        copied.contradictions = dict(self.contradictions)
        copied.contradiction_graph = {k: v.clone() for k, v in self.contradiction_graph.items()}
        copied.branch_probs = dict(self.branch_probs)
        copied.branch_metadata = {k: v.clone() for k, v in self.branch_metadata.items()}
        copied.state_vec = np.array(self.state_vec, copy=True)
        copied.ref_vec = np.array(self.ref_vec, copy=True)
        copied.action_receipts = {k: v.clone() for k, v in self.action_receipts.items()}
        copied.telemetry_counters = dict(self.telemetry_counters)
        copied.event_refs = list(self.event_refs)
        copied.checkpoint_snapshots = deepcopy(self.checkpoint_snapshots)
        if state_id is not None:
            copied.state_id = state_id
        return copied

    def snapshot_digest(self) -> str:
        from .persistence import to_jsonable

        payload = to_jsonable(self.capture_snapshot())
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
