from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

Point = tuple[int, int]


def normalize_point(value: Any) -> Optional[Point]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        stripped = value.strip().strip("()[]")
        if "," not in stripped:
            return None
        left, right = stripped.split(",", 1)
        try:
            return int(left.strip()), int(right.strip())
        except ValueError:
            return None
    if isinstance(value, dict):
        for x_key in ("x", "col", "column"):
            for y_key in ("y", "row"):
                if x_key in value and y_key in value:
                    try:
                        return int(value[x_key]), int(value[y_key])
                    except (TypeError, ValueError):
                        return None
    return None


def point_to_key(point: Point) -> str:
    return f"{point[0]},{point[1]}"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, tuple):
                safe[point_to_key(key)] = json_safe(item)
            else:
                safe[str(key)] = json_safe(item)
        return safe
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


@dataclass
class ArcState:
    level_id: str
    step_id: int

    player_pos: Optional[Point]
    current_tile: Optional[str]

    visible_tiles: dict[Point, str] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)
    reference_entities: list[dict[str, Any]] = field(default_factory=list)

    resources: dict[str, Any] = field(default_factory=dict)

    terminal: bool = False
    win: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "step_id": self.step_id,
            "player_pos": list(self.player_pos) if self.player_pos is not None else None,
            "current_tile": self.current_tile,
            "visible_tiles": {point_to_key(point): tile for point, tile in sorted(self.visible_tiles.items())},
            "entities": json_safe(self.entities),
            "reference_entities": json_safe(self.reference_entities),
            "resources": json_safe(self.resources),
            "terminal": self.terminal,
            "win": self.win,
        }

    def signature(self) -> str:
        entity_signature = tuple(
            sorted(
                (
                    entity.get("name", entity.get("type", "unknown")),
                    tuple(entity.get("position")) if isinstance(entity.get("position"), list) else entity.get("position"),
                )
                for entity in self.entities
            )
        )
        resource_signature = tuple(sorted((str(key), str(value)) for key, value in self.resources.items()))
        tile_signature = tuple(sorted((point_to_key(point), tile) for point, tile in self.visible_tiles.items()))
        return repr(
            (
                self.level_id,
                self.player_pos,
                self.current_tile,
                tile_signature,
                entity_signature,
                resource_signature,
                self.terminal,
                self.win,
            )
        )


@dataclass
class TransitionDelta:
    position_delta: Optional[Point]
    resource_deltas: dict[str, float] = field(default_factory=dict)

    entered_tile: Optional[str] = None
    crossed_marker: Optional[str] = None
    touched_entity: Optional[str] = None

    orientation_changed: bool = False
    terminal_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_delta": list(self.position_delta) if self.position_delta is not None else None,
            "resource_deltas": self.resource_deltas,
            "entered_tile": self.entered_tile,
            "crossed_marker": self.crossed_marker,
            "touched_entity": self.touched_entity,
            "orientation_changed": self.orientation_changed,
            "terminal_changed": self.terminal_changed,
        }


@dataclass
class TransitionReport:
    level_id: str
    step_id: int
    action: str

    prev_state: dict[str, Any]
    next_state: dict[str, Any]

    delta: TransitionDelta
    local_context: dict[str, Any] = field(default_factory=dict)

    terminal: bool = False
    win: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "step_id": self.step_id,
            "action": self.action,
            "prev_state": json_safe(self.prev_state),
            "next_state": json_safe(self.next_state),
            "delta": self.delta.to_dict(),
            "local_context": json_safe(self.local_context),
            "terminal": self.terminal,
            "win": self.win,
        }


@dataclass
class GuidancePacket:
    mode: str

    active_goal: dict[str, Any] = field(default_factory=dict)
    prioritized_targets: list[Any] = field(default_factory=list)
    avoid_targets: list[Any] = field(default_factory=list)

    learned_rules: list[Any] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "active_goal": json_safe(self.active_goal),
            "prioritized_targets": json_safe(self.prioritized_targets),
            "avoid_targets": json_safe(self.avoid_targets),
            "learned_rules": json_safe(self.learned_rules),
            "confidence": self.confidence,
        }


@dataclass
class RuntimeMemory:
    visited_states: set[str] = field(default_factory=set)
    state_action_pairs: dict[tuple[str, str], int] = field(default_factory=dict)
    transition_history: list[TransitionReport] = field(default_factory=list)
    tile_effect_evidence: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    marker_effect_evidence: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    visited_positions: set[Point] = field(default_factory=set)
    recent_positions: list[Point] = field(default_factory=list)
    recent_actions: list[str] = field(default_factory=list)
