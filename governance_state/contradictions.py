from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ContradictionType(str, Enum):
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    POLICY = "policy"
    AUTHORITY = "authority"
    BRANCH = "branch"


class ContradictionSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContradictionStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


_SEVERITY_WEIGHT = {
    ContradictionSeverity.LOW: 0.5,
    ContradictionSeverity.MEDIUM: 1.0,
    ContradictionSeverity.HIGH: 1.5,
    ContradictionSeverity.CRITICAL: 2.0,
}

_TYPE_WEIGHT = {
    ContradictionType.FACTUAL: 1.0,
    ContradictionType.TEMPORAL: 1.0,
    ContradictionType.POLICY: 1.5,
    ContradictionType.AUTHORITY: 1.5,
    ContradictionType.BRANCH: 1.2,
}


@dataclass
class Contradiction:
    contradiction_id: str
    contradiction_type: ContradictionType = ContradictionType.FACTUAL
    severity: ContradictionSeverity = ContradictionSeverity.MEDIUM
    status: ContradictionStatus = ContradictionStatus.OPEN
    mass: float = 1.0
    refs: List[str] = field(default_factory=list)
    peer_ids: List[str] = field(default_factory=list)
    summary: str = ""
    resolution_protocol: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def penalty(self) -> float:
        if self.status == ContradictionStatus.RESOLVED:
            return 0.0
        if self.status == ContradictionStatus.SUPPRESSED:
            return float(self.mass) * 0.25
        return float(self.mass) * _SEVERITY_WEIGHT[self.severity] * _TYPE_WEIGHT[self.contradiction_type]

    def clone(self) -> "Contradiction":
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "contradiction_type": self.contradiction_type,
            "severity": self.severity,
            "status": self.status,
            "mass": self.mass,
            "refs": list(self.refs),
            "peer_ids": list(self.peer_ids),
            "summary": self.summary,
            "resolution_protocol": self.resolution_protocol,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Contradiction":
        return cls(
            contradiction_id=payload["contradiction_id"],
            contradiction_type=ContradictionType(payload.get("contradiction_type", ContradictionType.FACTUAL)),
            severity=ContradictionSeverity(payload.get("severity", ContradictionSeverity.MEDIUM)),
            status=ContradictionStatus(payload.get("status", ContradictionStatus.OPEN)),
            mass=float(payload.get("mass", 1.0)),
            refs=list(payload.get("refs", [])),
            peer_ids=list(payload.get("peer_ids", [])),
            summary=payload.get("summary", ""),
            resolution_protocol=payload.get("resolution_protocol"),
            details=dict(payload.get("details", {})),
        )


@dataclass
class ContradictionGraph:
    contradictions: Dict[str, Contradiction] = field(default_factory=dict)

    def total_penalty(self) -> float:
        return float(sum(item.penalty() for item in self.contradictions.values()))

    def clone(self) -> "ContradictionGraph":
        return ContradictionGraph({k: v.clone() for k, v in self.contradictions.items()})
