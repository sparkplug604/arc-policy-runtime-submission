from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from .cells import normalize_dt, utc_now


class MergeStrategy(str, Enum):
    REPLACE = "replace"
    UNION = "union"
    AUTHORITATIVE = "authoritative"


class BranchLifecycleStatus(str, Enum):
    CANONICAL = "canonical"
    CANDIDATE = "candidate"
    MERGED = "merged"
    REJECTED = "rejected"


@dataclass
class BranchMetadata:
    branch_id: str
    parent_branch_id: Optional[str] = None
    authority_level: int = 0
    confidence: float = 1.0
    provenance: str = "runtime"
    status: BranchLifecycleStatus = BranchLifecycleStatus.CANONICAL
    merge_strategy: MergeStrategy = MergeStrategy.REPLACE
    canonical_after_merge: bool = True
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.created_at = normalize_dt(self.created_at) or utc_now()
        self.updated_at = normalize_dt(self.updated_at) or self.created_at

    def touch(self, at: Optional[datetime] = None) -> None:
        self.updated_at = normalize_dt(at) or utc_now()

    def clone(self) -> "BranchMetadata":
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "parent_branch_id": self.parent_branch_id,
            "authority_level": self.authority_level,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "status": self.status,
            "merge_strategy": self.merge_strategy,
            "canonical_after_merge": self.canonical_after_merge,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BranchMetadata":
        return cls(
            branch_id=payload["branch_id"],
            parent_branch_id=payload.get("parent_branch_id"),
            authority_level=int(payload.get("authority_level", 0)),
            confidence=float(payload.get("confidence", 1.0)),
            provenance=payload.get("provenance", "runtime"),
            status=BranchLifecycleStatus(payload.get("status", BranchLifecycleStatus.CANONICAL)),
            merge_strategy=MergeStrategy(payload.get("merge_strategy", MergeStrategy.REPLACE)),
            canonical_after_merge=bool(payload.get("canonical_after_merge", True)),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            notes=dict(payload.get("notes", {})),
        )
