from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class CellStatus(str, Enum):
    ACTIVE = "active"
    COMPACT = "compact"
    ARCHIVED = "archived"
    DROPPED = "dropped"
    INVALID = "invalid"
    PROPOSED = "proposed"


class SelectedState(str, Enum):
    KEEP = "keep"
    COMPACT = "compact"
    DROP = "drop"
    DEFER = "defer"


class SourceType(str, Enum):
    SYSTEM = "system"
    USER = "user"
    EXTERNAL = "external"
    RUNTIME = "runtime"


class ValidationState(str, Enum):
    UNVALIDATED = "unvalidated"
    VALID = "valid"
    VIOLATED = "violated"
    QUARANTINED = "quarantined"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    DISPUTED = "disputed"


class PrecedenceClass(str, Enum):
    LAW = "law"
    POLICY = "policy"
    GOAL = "goal"
    EVIDENCE = "evidence"
    SPECULATION = "speculation"


class EpistemicStatus(str, Enum):
    ASSERTED = "asserted"
    OBSERVED = "observed"
    INFERRED = "inferred"
    PREDICTED = "predicted"
    SIMULATED = "simulated"


class BranchStatus(str, Enum):
    CANONICAL = "canonical"
    CANDIDATE = "candidate"
    SIMULATED = "simulated"
    REJECTED = "rejected"
    MERGED = "merged"


class ConflictResolutionStatus(str, Enum):
    UNRESOLVED = "unresolved"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(UTC)


def normalize_dt(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass
class CellContent:
    text: str = ""
    structured: Dict[str, Any] = field(default_factory=dict)
    embedding_ref: Optional[str] = None
    embedding: List[float] = field(default_factory=list)
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)


@dataclass
class CellBindings:
    required_keys: List[str] = field(default_factory=list)
    provided_keys: List[str] = field(default_factory=list)
    hard_constraints: Dict[str, Any] = field(default_factory=dict)
    policy_refs: List[str] = field(default_factory=list)
    spec_refs: List[str] = field(default_factory=list)
    goal_refs: List[str] = field(default_factory=list)
    human_invariant_refs: List[str] = field(default_factory=list)


@dataclass
class CellTrace:
    source_type: SourceType = SourceType.SYSTEM
    source_id: Optional[str] = None
    parent_cell_ids: List[str] = field(default_factory=list)
    decision_refs: List[str] = field(default_factory=list)
    artifact_refs: List[str] = field(default_factory=list)
    checkpoint_ref: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None


@dataclass
class CellMetrics:
    priority: float = 0.0
    persistence: float = 0.0
    confidence: float = 0.0
    freshness: float = 0.0
    relevance: float = 0.0
    goal_alignment: float = 0.0
    coherence: float = 0.0
    noise: float = 0.0
    drift: float = 0.0
    binding_criticality: float = 0.0
    retention_value: float = 0.0


@dataclass
class CellCost:
    token_cost: int = 0
    estimated_energy_j: float = 0.0
    estimated_latency_ms: float = 0.0
    memory_bytes: int = 0
    serialization_bytes: int = 0
    branch_cost: float = 0.0


@dataclass
class CellSpectral:
    m_local: float = 0.0
    goal_score: float = 0.0
    salience: float = 0.0
    cluster_id: Optional[str] = None
    mode_weights: List[float] = field(default_factory=list)
    coupling_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CellLifecycle:
    status: CellStatus = CellStatus.ACTIVE
    selected_state: SelectedState = SelectedState.DEFER
    ttl_phases: Optional[int] = None
    last_selected_at: Optional[datetime] = None
    selection_count: int = 0
    compacted_from: List[str] = field(default_factory=list)
    archive_reason: Optional[str] = None

    def __post_init__(self) -> None:
        self.last_selected_at = normalize_dt(self.last_selected_at)


@dataclass
class CellGovernance:
    authority_level: int = 0
    precedence_class: PrecedenceClass = PrecedenceClass.EVIDENCE
    validation_state: ValidationState = ValidationState.UNVALIDATED
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    approval_required: bool = False
    approved_by: List[str] = field(default_factory=list)
    violated_invariants: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    overrideable: bool = True
    supersedes: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None


@dataclass
class CellTemporal:
    observed_at: Optional[datetime] = None
    effective_from: Optional[datetime] = None
    effective_until: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    last_verified_at: Optional[datetime] = None
    time_sensitivity: float = 0.0

    def __post_init__(self) -> None:
        self.observed_at = normalize_dt(self.observed_at)
        self.effective_from = normalize_dt(self.effective_from)
        self.effective_until = normalize_dt(self.effective_until)
        self.expires_at = normalize_dt(self.expires_at)
        self.last_verified_at = normalize_dt(self.last_verified_at)


@dataclass
class CellEpistemic:
    epistemic_status: EpistemicStatus = EpistemicStatus.ASSERTED
    evidence_refs: List[str] = field(default_factory=list)
    falsifiable: bool = True
    uncertainty: float = 0.0
    source_reliability: float = 0.0


@dataclass
class CellBranch:
    world_state_id: str = "default"
    branch_id: str = "main"
    branch_parent_id: Optional[str] = None
    branch_status: BranchStatus = BranchStatus.CANONICAL
    merge_refs: List[str] = field(default_factory=list)


@dataclass
class CellConflict:
    conflicts_with: List[str] = field(default_factory=list)
    conflict_type: Optional[str] = None
    resolution_status: ConflictResolutionStatus = ConflictResolutionStatus.UNRESOLVED
    resolution_ref: Optional[str] = None


@dataclass
class CellHistoryEvent:
    event_type: str
    at: datetime
    actor: str
    reason: Optional[str] = None
    ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.at = normalize_dt(self.at) or utc_now()


@dataclass
class MemoryCell:
    cell_id: str
    schema_version: str = "context_cell.v3"
    cell_type: str = "document"
    role: str = "support"
    namespace: str = "default"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    content: CellContent = field(default_factory=CellContent)
    bindings: CellBindings = field(default_factory=CellBindings)
    trace: CellTrace = field(default_factory=CellTrace)
    metrics: CellMetrics = field(default_factory=CellMetrics)
    cost: CellCost = field(default_factory=CellCost)
    spectral: CellSpectral = field(default_factory=CellSpectral)
    lifecycle: CellLifecycle = field(default_factory=CellLifecycle)
    governance: CellGovernance = field(default_factory=CellGovernance)
    temporal: CellTemporal = field(default_factory=CellTemporal)
    epistemic: CellEpistemic = field(default_factory=CellEpistemic)
    branch: CellBranch = field(default_factory=CellBranch)
    conflict: CellConflict = field(default_factory=CellConflict)
    history: List[CellHistoryEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.created_at = normalize_dt(self.created_at) or utc_now()
        self.updated_at = normalize_dt(self.updated_at) or self.created_at

    def touch(self, at: Optional[datetime] = None) -> None:
        self.updated_at = normalize_dt(at) or utc_now()

    def clone(self) -> "MemoryCell":
        return deepcopy(self)
