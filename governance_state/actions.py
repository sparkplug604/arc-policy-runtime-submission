from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional

from .cells import normalize_dt, utc_now


class ActionStatus(str, Enum):
    REQUESTED = "requested"
    AUTHORIZED = "authorized"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ActionFailureClass(str, Enum):
    PERMISSION = "permission"
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    TIMEOUT = "timeout"
    INVALID = "invalid"


@dataclass
class ResourceUsage:
    token_cost: float = 0.0
    latency_ms: float = 0.0
    energy_j: float = 0.0
    memory_bytes: float = 0.0
    cpu_time_ms: float = 0.0
    gpu_time_ms: float = 0.0

    def clone(self) -> "ResourceUsage":
        return deepcopy(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ResourceUsage":
        return cls(
            token_cost=float(payload.get("token_cost", 0.0)),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            energy_j=float(payload.get("energy_j", 0.0)),
            memory_bytes=float(payload.get("memory_bytes", 0.0)),
            cpu_time_ms=float(payload.get("cpu_time_ms", 0.0)),
            gpu_time_ms=float(payload.get("gpu_time_ms", 0.0)),
        )


@dataclass
class ActionRequest:
    action_id: str
    action_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    target_cell_id: Optional[str] = None
    namespace: str = "default"
    authorization_ref: Optional[str] = None
    idempotency_key: Optional[str] = None
    allowed_side_effects: List[str] = field(default_factory=list)
    max_retries: int = 0
    timeout_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.idempotency_key is None:
            payload_hash = hashlib.sha256(
                json.dumps(self.payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            self.idempotency_key = f"{self.action_type}:{payload_hash}"

    def clone(self) -> "ActionRequest":
        return deepcopy(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ActionRequest":
        return cls(
            action_id=payload["action_id"],
            action_type=payload["action_type"],
            payload=dict(payload.get("payload", {})),
            target_cell_id=payload.get("target_cell_id"),
            namespace=payload.get("namespace", "default"),
            authorization_ref=payload.get("authorization_ref"),
            idempotency_key=payload.get("idempotency_key"),
            allowed_side_effects=list(payload.get("allowed_side_effects", [])),
            max_retries=int(payload.get("max_retries", 0)),
            timeout_ms=payload.get("timeout_ms"),
        )


@dataclass
class ActionReceipt:
    receipt_id: str
    action_id: str
    status: ActionStatus
    actor: str
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime = field(default_factory=utc_now)
    message: str = ""
    failure_class: Optional[ActionFailureClass] = None
    idempotency_key: Optional[str] = None
    payload_hash: Optional[str] = None
    authorization_ref: Optional[str] = None
    attempt_count: int = 1
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    output: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.started_at = normalize_dt(self.started_at) or utc_now()
        self.finished_at = normalize_dt(self.finished_at) or self.started_at

    def clone(self) -> "ActionReceipt":
        return deepcopy(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ActionReceipt":
        failure = payload.get("failure_class")
        return cls(
            receipt_id=payload["receipt_id"],
            action_id=payload["action_id"],
            status=ActionStatus(payload["status"]),
            actor=payload["actor"],
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            message=payload.get("message", ""),
            failure_class=ActionFailureClass(failure) if failure is not None else None,
            idempotency_key=payload.get("idempotency_key"),
            payload_hash=payload.get("payload_hash"),
            authorization_ref=payload.get("authorization_ref"),
            attempt_count=int(payload.get("attempt_count", 1)),
            resource_usage=ResourceUsage.from_dict(payload.get("resource_usage", {})),
            output=dict(payload.get("output", {})),
        )


@dataclass
class ActionResult:
    receipt: ActionReceipt
    state_updates: Dict[str, Any] = field(default_factory=dict)
    output_bindings: List[str] = field(default_factory=list)
