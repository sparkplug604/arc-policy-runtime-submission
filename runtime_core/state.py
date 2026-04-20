from __future__ import annotations

from typing import Any

from .types import ArcState, RuntimeMemory, json_safe, point_to_key


def state_hash(state: ArcState) -> str:
    return state.signature()


def build_local_context(
    prev_state: ArcState,
    next_state: ArcState,
    memory: RuntimeMemory,
    action_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    nearby_tiles: dict[str, str] = {}
    if next_state.player_pos is not None:
        px, py = next_state.player_pos
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)):
            point = (px + dx, py + dy)
            tile = next_state.visible_tiles.get(point)
            if tile is not None:
                nearby_tiles[point_to_key(point)] = tile

    overlapping_entities = [
        entity
        for entity in next_state.entities
        if next_state.player_pos is not None
        and (
            entity.get("position") == list(next_state.player_pos)
            or entity.get("position") == next_state.player_pos
        )
    ]
    top_action_scores = {}
    if action_scores:
        ranked_scores = sorted(action_scores.items(), key=lambda item: (-float(item[1]), item[0]))
        top_action_scores = {key: value for key, value in ranked_scores[:2]}

    active_match_target = next_state.resources.get("active_match_target")
    target_position = None
    if isinstance(active_match_target, dict):
        raw_target_position = active_match_target.get("target_position")
        if isinstance(raw_target_position, list) and len(raw_target_position) == 2:
            try:
                target_position = (int(raw_target_position[0]), int(raw_target_position[1]))
            except (TypeError, ValueError):
                target_position = None
        elif isinstance(raw_target_position, tuple) and len(raw_target_position) == 2:
            target_position = raw_target_position

    touching_target_entity = False
    target_name = str(active_match_target.get("target_name")) if isinstance(active_match_target, dict) else ""
    for entity in overlapping_entities:
        if not isinstance(entity, dict):
            continue
        if entity.get("role") == "match_target":
            touching_target_entity = True
            break
        if target_name and str(entity.get("name")) == target_name:
            touching_target_entity = True
            break

    on_target_cell = next_state.player_pos is not None and target_position is not None and next_state.player_pos == target_position
    completion_latch_inputs = {
        "on_target_cell": on_target_cell,
        "touching_target_entity": touching_target_entity,
        "pattern_match": bool(active_match_target.get("pattern_match")) if isinstance(active_match_target, dict) else False,
        "orientation_match": bool(active_match_target.get("orientation_match")) if isinstance(active_match_target, dict) else False,
        "aligned": bool(active_match_target.get("aligned")) if isinstance(active_match_target, dict) else False,
        "levels_completed": next_state.resources.get("levels_completed"),
    }
    completion_blockers = [
        name
        for name, active in (
            ("not_on_target_cell", completion_latch_inputs["on_target_cell"]),
            ("not_touching_target_entity", completion_latch_inputs["touching_target_entity"]),
            ("pattern_mismatch", completion_latch_inputs["pattern_match"]),
            ("orientation_mismatch", completion_latch_inputs["orientation_match"]),
            ("not_aligned", completion_latch_inputs["aligned"]),
        )
        if not active
    ]
    debug = next_state.resources.get("debug") if isinstance(next_state.resources.get("debug"), dict) else {}
    return {
        "prev_tile": prev_state.current_tile,
        "next_tile": next_state.current_tile,
        "current_tile": next_state.current_tile,
        "player_pos": list(next_state.player_pos) if next_state.player_pos is not None else None,
        "nearby_tiles": nearby_tiles,
        "overlapping_entities": json_safe(overlapping_entities),
        "active_match_target": json_safe(active_match_target),
        "player_pos_provenance": debug.get("player_pos_provenance"),
        "active_match_target_provenance": debug.get("active_match_target_provenance"),
        "entity_resource_divergence": debug.get("entity_resource_divergence"),
        "entity_match_target_snapshot": json_safe(debug.get("entity_match_target_snapshot")),
        "resource_match_target_snapshot": json_safe(debug.get("resource_match_target_snapshot")),
        "cell_size_source": debug.get("cell_size_source"),
        "cell_size_changed": debug.get("cell_size_changed"),
        "walkable_points_source": debug.get("walkable_points_source"),
        "completion_latch_inputs": completion_latch_inputs,
        "completion_blockers": completion_blockers,
        "mechanic_candidate_positions": json_safe(next_state.resources.get("mechanic_candidate_positions", [])),
        "mechanic_candidate_entities": json_safe(next_state.resources.get("mechanic_candidate_entities", [])),
        "yellow_interstitial": bool(next_state.resources.get("yellow_interstitial")),
        "yellow_interstitial_ratio": next_state.resources.get("yellow_interstitial_ratio"),
        "attempts_remaining": next_state.resources.get("attempts_remaining"),
        "attempts_spent": next_state.resources.get("attempts_spent"),
        "attempts_total": next_state.resources.get("attempts_total"),
        "health": next_state.resources.get("health"),
        "health_spent": next_state.resources.get("health_spent"),
        "health_total": next_state.resources.get("health_total"),
        "known_tile_rules": len(memory.tile_effect_evidence),
        "known_marker_rules": len(memory.marker_effect_evidence),
        "visited_states": len(memory.visited_states),
        "visited_positions": len(memory.visited_positions),
        "recent_actions": list(memory.recent_actions[-8:]),
        "recent_positions": json_safe(memory.recent_positions[-8:]),
        "top_action_scores": top_action_scores,
    }


def fresh_memory() -> RuntimeMemory:
    return RuntimeMemory()
