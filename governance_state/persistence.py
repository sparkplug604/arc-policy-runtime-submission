from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import importlib
import json
from pathlib import Path
import hashlib
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}
    if isinstance(value, Enum):
        return {"__type__": "enum", "class": f"{value.__class__.__module__}.{value.__class__.__name__}", "value": value.value}
    if isinstance(value, np.ndarray):
        return {"__type__": "ndarray", "value": value.tolist()}
    if isinstance(value, Path):
        return {"__type__": "path", "value": str(value)}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return {"__type__": "tuple", "value": [to_jsonable(item) for item in value]}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if is_dataclass(value):
        return {
            "__type__": "dataclass",
            "class": f"{value.__class__.__module__}.{value.__class__.__name__}",
            "fields": {item.name: to_jsonable(getattr(value, item.name)) for item in fields(value)},
        }
    return value


def from_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return [from_jsonable(item) for item in value]
    if isinstance(value, dict) and "__type__" in value:
        kind = value["__type__"]
        if kind == "datetime":
            return datetime.fromisoformat(value["value"])
        if kind == "enum":
            module_name, _, class_name = value["class"].rpartition(".")
            enum_cls = getattr(importlib.import_module(module_name), class_name)
            return enum_cls(value["value"])
        if kind == "ndarray":
            return np.asarray(value["value"], dtype=float)
        if kind == "tuple":
            return tuple(from_jsonable(item) for item in value["value"])
        if kind == "path":
            return Path(value["value"])
        if kind == "dataclass":
            module_name, _, class_name = value["class"].rpartition(".")
            cls = getattr(importlib.import_module(module_name), class_name)
            payload = {key: from_jsonable(item) for key, item in value["fields"].items()}
            return cls(**payload)
    if isinstance(value, dict):
        return {key: from_jsonable(item) for key, item in value.items()}
    return value


@dataclass
class EventRecord:
    event_id: str
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    kind: str = "transition"
    state_before: Optional[str] = None
    state_after: Optional[str] = None
    transition_type: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    prev_hash: Optional[str] = None
    record_hash: Optional[str] = None

    def compute_hash(self) -> str:
        payload = {
            "event_id": self.event_id,
            "at": self.at.isoformat(),
            "kind": self.kind,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "transition_type": self.transition_type,
            "payload": to_jsonable(self.payload),
            "prev_hash": self.prev_hash,
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CheckpointRecord:
    checkpoint_ref: str
    state_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    snapshot: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    version: str = "checkpoint.v1"

    def finalize(self) -> None:
        payload = json.dumps(to_jsonable(self.snapshot), sort_keys=True, separators=(",", ":"))
        self.checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EventStore(Protocol):
    def append(self, event: EventRecord) -> EventRecord:
        ...

    def list_events(self) -> List[EventRecord]:
        ...


class CheckpointStore(Protocol):
    def save(self, record: CheckpointRecord) -> CheckpointRecord:
        ...

    def load(self, checkpoint_ref: str) -> CheckpointRecord:
        ...

    def list_checkpoints(self) -> List[str]:
        ...


@dataclass
class FileEventStore:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, event: EventRecord) -> EventRecord:
        events = self.list_events()
        event.prev_hash = events[-1].record_hash if events else None
        event.record_hash = event.compute_hash()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(to_jsonable(asdict(event)), sort_keys=True) + "\n")
        return event

    def list_events(self) -> List[EventRecord]:
        events: List[EventRecord] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = from_jsonable(json.loads(line))
                events.append(EventRecord(**payload))
        return events


@dataclass
class FileCheckpointStore:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, checkpoint_ref: str) -> Path:
        return self.root / f"{checkpoint_ref}.json"

    def save(self, record: CheckpointRecord) -> CheckpointRecord:
        record.finalize()
        self._path(record.checkpoint_ref).write_text(
            json.dumps(to_jsonable(asdict(record)), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return record

    def load(self, checkpoint_ref: str) -> CheckpointRecord:
        payload = from_jsonable(json.loads(self._path(checkpoint_ref).read_text(encoding="utf-8")))
        record = CheckpointRecord(**payload)
        check = CheckpointRecord(
            checkpoint_ref=record.checkpoint_ref,
            state_id=record.state_id,
            created_at=record.created_at,
            snapshot=record.snapshot,
            version=record.version,
        )
        check.finalize()
        if check.checksum != record.checksum:
            raise ValueError(f"Checkpoint {checkpoint_ref!r} failed checksum validation.")
        return record

    def list_checkpoints(self) -> List[str]:
        return sorted(path.stem for path in self.root.glob("*.json"))
