from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from .cells import normalize_dt


class GoalStatus(str, Enum):
    ACTIVE = "active"
    SATISFIED = "satisfied"
    BLOCKED = "blocked"
    SUPERSEDED = "superseded"


class GoalPolicyClass(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    PREFERENCE = "preference"


@dataclass
class Goal:
    goal_id: str
    text: str
    description: Optional[str] = None
    weight: float = 1.0
    priority: float = 1.0
    priority_weight: float = 1.0
    authority_weight: float = 1.0
    urgency_weight: float = 1.0
    deadline: Optional[datetime] = None
    authority_level: int = 0
    hard: bool = False
    active: bool = True
    policy_class: GoalPolicyClass = GoalPolicyClass.SOFT
    status: GoalStatus = GoalStatus.ACTIVE
    satisfaction_score: float = 0.0
    required_bindings: List[str] = field(default_factory=list)
    policy_refs: List[str] = field(default_factory=list)
    compatibility_refs: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.deadline = normalize_dt(self.deadline)
        if self.description is None:
            self.description = self.text
        if not self.active and self.status == GoalStatus.ACTIVE:
            self.status = GoalStatus.BLOCKED

    def effective_weight(self, now: Optional[datetime] = None) -> float:
        weight = (
            max(float(self.weight), 0.0)
            * max(float(self.priority), 0.0)
            * max(float(self.priority_weight), 0.0)
            * max(float(self.authority_weight), 0.0)
            * max(float(self.urgency_weight), 0.0)
        )
        if self.hard or self.policy_class == GoalPolicyClass.HARD:
            weight *= 2.0
        if not self.active:
            return 0.0
        if self.status == GoalStatus.SUPERSEDED:
            return 0.0
        if self.status == GoalStatus.SATISFIED:
            return 0.0
        if self.deadline is not None:
            current = normalize_dt(now) or datetime.now(self.deadline.tzinfo)
            if current > self.deadline:
                return weight * 1.5
        return weight

    def clone(self) -> "Goal":
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "text": self.text,
            "description": self.description,
            "weight": self.weight,
            "priority": self.priority,
            "priority_weight": self.priority_weight,
            "authority_weight": self.authority_weight,
            "urgency_weight": self.urgency_weight,
            "deadline": self.deadline,
            "authority_level": self.authority_level,
            "hard": self.hard,
            "active": self.active,
            "policy_class": self.policy_class,
            "status": self.status,
            "satisfaction_score": self.satisfaction_score,
            "required_bindings": list(self.required_bindings),
            "policy_refs": list(self.policy_refs),
            "compatibility_refs": list(self.compatibility_refs),
            "superseded_by": self.superseded_by,
            "embedding": list(self.embedding),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Goal":
        return cls(
            goal_id=payload["goal_id"],
            text=payload["text"],
            description=payload.get("description"),
            weight=float(payload.get("weight", 1.0)),
            priority=float(payload.get("priority", 1.0)),
            priority_weight=float(payload.get("priority_weight", payload.get("priority", 1.0))),
            authority_weight=float(payload.get("authority_weight", 1.0)),
            urgency_weight=float(payload.get("urgency_weight", 1.0)),
            deadline=payload.get("deadline"),
            authority_level=int(payload.get("authority_level", 0)),
            hard=bool(payload.get("hard", False)),
            active=bool(payload.get("active", True)),
            policy_class=GoalPolicyClass(payload.get("policy_class", GoalPolicyClass.SOFT)),
            status=GoalStatus(payload.get("status", GoalStatus.ACTIVE)),
            satisfaction_score=float(payload.get("satisfaction_score", 0.0)),
            required_bindings=list(payload.get("required_bindings", [])),
            policy_refs=list(payload.get("policy_refs", [])),
            compatibility_refs=list(payload.get("compatibility_refs", [])),
            superseded_by=payload.get("superseded_by"),
            embedding=list(payload.get("embedding", [])),
            metadata=dict(payload.get("metadata", {})),
        )


def compute_goal_weights(goals: Iterable[Goal], *, now: Optional[datetime] = None) -> List[float]:
    current = normalize_dt(now)
    weights: List[float] = []
    for goal in goals:
        weight = goal.effective_weight(current)
        if goal.authority_level > 0:
            weight *= 1.0 + (0.1 * float(goal.authority_level))
        weights.append(weight)
    return weights
