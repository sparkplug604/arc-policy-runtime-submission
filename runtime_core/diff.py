from __future__ import annotations

from .types import ArcState, TransitionDelta, normalize_point

MARKER_HINTS = ("marker", "switch", "button", "portal", "goal", "door", "key")


def compute_delta(prev: ArcState, next: ArcState) -> TransitionDelta:
    position_delta = None
    if prev.player_pos is not None and next.player_pos is not None:
        position_delta = (
            next.player_pos[0] - prev.player_pos[0],
            next.player_pos[1] - prev.player_pos[1],
        )

    resource_deltas: dict[str, float] = {}
    for key in sorted(set(prev.resources) | set(next.resources)):
        prev_value = prev.resources.get(key)
        next_value = next.resources.get(key)
        if isinstance(prev_value, (int, float)) and isinstance(next_value, (int, float)):
            diff_value = float(next_value - prev_value)
            if diff_value != 0:
                resource_deltas[key] = diff_value

    entered_tile = None
    if next.player_pos is not None:
        entered_tile = next.visible_tiles.get(next.player_pos, next.current_tile)
    if position_delta == (0, 0) and prev.current_tile == entered_tile:
        entered_tile = None

    crossed_marker = None
    marker_entity = _marker_at_player(prev, next)
    if marker_entity is not None:
        crossed_marker = marker_entity
    else:
        candidate_marker = entered_tile or next.current_tile
        if candidate_marker is not None and any(hint in candidate_marker.lower() for hint in MARKER_HINTS):
            crossed_marker = candidate_marker

    touched_entity = _touched_entity(prev, next)
    prev_target_signature = _target_orientation_signature(prev)
    next_target_signature = _target_orientation_signature(next)
    orientation_changed = _orientation(prev) != _orientation(next)
    if prev_target_signature != next_target_signature:
        orientation_changed = True
    elif prev_target_signature is None and next_target_signature is None and prev.visible_tiles != next.visible_tiles:
        orientation_changed = True

    return TransitionDelta(
        position_delta=position_delta,
        resource_deltas=resource_deltas,
        entered_tile=entered_tile,
        crossed_marker=crossed_marker,
        touched_entity=touched_entity,
        orientation_changed=orientation_changed,
        terminal_changed=prev.terminal != next.terminal,
    )


def _touched_entity(prev: ArcState, next: ArcState) -> str | None:
    if next.player_pos is None:
        return None

    for entity in next.entities:
        position = normalize_point(entity.get("position"))
        label = str(entity.get("name", entity.get("type", "entity"))).lower()
        if position == next.player_pos and label not in {"player", "agent", "avatar", "self"}:
            return str(entity.get("name", entity.get("type", "entity")))

    prev_positions = {
        normalize_point(entity.get("position")): str(entity.get("name", entity.get("type", "entity")))
        for entity in prev.entities
        if normalize_point(entity.get("position")) is not None
    }
    if next.player_pos in prev_positions:
        return prev_positions[next.player_pos]
    return None


def _orientation(state: ArcState) -> str | None:
    for key in ("orientation", "facing", "direction", "heading"):
        value = state.resources.get(key)
        if value is not None:
            return str(value)
    return None


def _marker_at_player(prev: ArcState, next: ArcState) -> str | None:
    if next.player_pos is None:
        return None

    for state in (next, prev):
        for entity in state.entities:
            position = normalize_point(entity.get("position"))
            label = str(entity.get("name", entity.get("type", "entity"))).lower()
            shape_class = str(entity.get("shape_class", "")).lower()
            role = str(entity.get("role", "")).lower()
            if position == next.player_pos and (
                "cross" in label
                or "marker" in label
                or shape_class == "marker_cross"
                or role == "mechanic_marker"
            ):
                return str(entity.get("name", entity.get("type", "entity")))
    return None


def _target_orientation_signature(state: ArcState) -> str | None:
    active_match = state.resources.get("active_match_target")
    if isinstance(active_match, dict):
        match_key = str(active_match.get("match_key", ""))
        target_orientation = active_match.get("target_orientation")
        reference_orientation = active_match.get("reference_orientation")
        aligned = active_match.get("aligned")
        return f"{match_key}:{target_orientation}:{reference_orientation}:{aligned}"

    for entity in state.entities:
        if entity.get("role") == "match_target":
            return f"{entity.get('match_key', '')}:{entity.get('orientation_index', '')}:{entity.get('orientation_match', '')}"
    return None
