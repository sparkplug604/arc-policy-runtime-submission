from __future__ import annotations

from typing import Any

from .adapter import find_goal_positions
from .priors import PRIORS
from .types import ArcState, GuidancePacket, RuntimeMemory

ACTION_DELTAS = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def planning_bias(state, action, memory):
    if state.player_pos is None:
        return 0.0

    next_point = _predict_point(state, action)

    if next_point not in memory.visited_positions:
        return 1.0

    return -0.5


def is_dead_end(state, point):
    neighbors = [
        (point[0] + 1, point[1]),
        (point[0] - 1, point[1]),
        (point[0], point[1] + 1),
        (point[0], point[1] - 1),
    ]

    blocked = 0
    for n in neighbors:
        if n not in state.visible_tiles:
            blocked += 1

    return blocked >= 3


def score_actions(
    state: ArcState,
    candidates: list[str],
    guidance: GuidancePacket | None,
    memory: RuntimeMemory,
) -> dict[str, float]:
    mode = (guidance.mode if guidance is not None else "PROBE").upper()
    prioritized = {str(item).lower() for item in (guidance.prioritized_targets if guidance else [])}
    avoid = {str(item).lower() for item in (guidance.avoid_targets if guidance else [])}
    low_resources = {key for key, value in state.resources.items() if isinstance(value, (int, float)) and value <= 1}
    goal_positions = find_goal_positions(state)

    scores: dict[str, float] = {}
    state_sig = state.signature()

    for action in candidates:
        score = 0.0
        if PRIORS["movement_first"]:
            score += 1.0

        target_point = _predict_point(state, action)
        target_tile = state.visible_tiles.get(target_point) if target_point is not None else None
        target_entity = _entity_at(state, target_point)
        pair_count = memory.state_action_pairs.get((state_sig, action), 0)
        recent_repeat_penalty = memory.recent_actions[-2:].count(action) * 0.4
        score += planning_bias(state, action, memory)

        if target_point is not None and target_point not in memory.visited_positions:
            score += 1.5
        if target_tile is None:
            score += 1.0
        if target_point in memory.recent_positions:
            score -= 3.5
        if target_point and is_dead_end(state, target_point):
            score -= 2.5

        if state.player_pos and goal_positions:
            current_dist = min(manhattan(state.player_pos, g) for g in goal_positions)

            if target_point:
                next_dist = min(manhattan(target_point, g) for g in goal_positions)

                if next_dist < current_dist:
                    score += 3.0
                elif next_dist > current_dist:
                    score -= 2.0

        if PRIORS["unknown_object_priority"] and target_entity is not None:
            entity_name = target_entity.lower()
            if entity_name not in memory.tile_effect_evidence:
                score += 2.0

        if PRIORS["interaction_bias"] and target_entity is not None:
            score += 0.6

        if target_tile is not None:
            tile_name = target_tile.lower()
            if tile_name in prioritized:
                score += 4.0
            if tile_name in avoid:
                score -= 5.0
            score += _tile_rule_bias(tile_name, mode, memory)

        if target_entity is not None:
            entity_name = target_entity.lower()
            if entity_name in prioritized:
                score += 3.0
            if entity_name in avoid:
                score -= 4.0

        if mode == "PROBE":
            score += _probe_score(target_point, target_tile, target_entity, memory)
        elif mode == "EXECUTE":
            score += _execute_score(target_tile, target_entity, guidance, memory)
        elif mode == "RECOVER":
            score += _recover_score(target_tile, low_resources, memory)

        if PRIORS["resource_awareness"] and low_resources and target_tile is not None:
            score += _recover_score(target_tile, low_resources, memory) * 0.5

        score -= pair_count * 0.8
        score -= recent_repeat_penalty
        scores[action] = round(score, 4)

    return scores


def _predict_point(state: ArcState, action: str) -> tuple[int, int] | None:
    if state.player_pos is None:
        return None
    dx, dy = ACTION_DELTAS.get(action, (0, 0))
    return state.player_pos[0] + dx, state.player_pos[1] + dy


def _entity_at(state: ArcState, point: tuple[int, int] | None) -> str | None:
    if point is None:
        return None
    for entity in state.entities:
        position = entity.get("position")
        if position == point or position == list(point):
            return str(entity.get("name", entity.get("type", "entity")))
    return None


def _tile_rule_bias(tile_name: str, mode: str, memory: RuntimeMemory) -> float:
    evidence = memory.tile_effect_evidence.get(tile_name, [])
    if not evidence:
        return 0.0 if mode != "PROBE" else 1.0

    positive = 0
    negative = 0
    for sample in evidence:
        resource_gain = sum(value for value in sample.get("resource_deltas", {}).values() if value > 0)
        resource_loss = sum(abs(value) for value in sample.get("resource_deltas", {}).values() if value < 0)
        if sample.get("win"):
            positive += 2
        if sample.get("terminal") and not sample.get("win"):
            negative += 2
        if resource_gain > resource_loss:
            positive += 1
        elif resource_loss > resource_gain:
            negative += 1

    if mode == "EXECUTE":
        return (positive - negative) * 1.2
    if mode == "RECOVER":
        return positive * 1.5 - negative * 1.5
    return max(0.0, 1.0 - (positive + negative) * 0.1)


def _probe_score(
    target_point: tuple[int, int] | None,
    target_tile: str | None,
    target_entity: str | None,
    memory: RuntimeMemory,
) -> float:
    score = 0.0
    if target_point is not None and target_point not in memory.visited_positions:
        score += 1.5
    if target_tile is not None and target_tile.lower() not in memory.tile_effect_evidence:
        score += 1.0
    if target_entity is not None:
        score += 1.5
    return score


def _execute_score(
    target_tile: str | None,
    target_entity: str | None,
    guidance: GuidancePacket | None,
    memory: RuntimeMemory,
) -> float:
    score = 0.0
    if guidance and guidance.active_goal:
        goal_kind = guidance.active_goal.get("kind")

        if goal_kind == "match_reference_object":
            score += 2.0
            goal_targets = {
                str(target).lower()
                for target in guidance.active_goal.get("targets", [])
            }
            if target_entity is not None and target_entity.lower() in goal_targets:
                score += 2.5
            if target_tile is not None and target_tile.lower() in goal_targets:
                score += 1.5
    for rule in guidance.learned_rules if guidance else []:
        if not isinstance(rule, dict):
            continue
        trigger = str(rule.get("trigger", "")).lower()
        if target_tile is not None and trigger == target_tile.lower():
            score += float(rule.get("weight", 1.5))
        if target_entity is not None and trigger == target_entity.lower():
            score += float(rule.get("weight", 1.2))
    if target_tile is not None and target_tile.lower() in memory.tile_effect_evidence:
        score += 0.7
    return score


def _recover_score(target_tile: str | None, low_resources: set[str], memory: RuntimeMemory) -> float:
    if target_tile is None:
        return 0.0
    score = 0.0
    evidence = memory.tile_effect_evidence.get(target_tile.lower(), [])
    for sample in evidence:
        for resource_name, delta in sample.get("resource_deltas", {}).items():
            if resource_name in low_resources and delta > 0:
                score += delta * 2.0
            if delta < 0:
                score += delta
    if PRIORS["replenishment_search"] and not evidence:
        score += 0.3
    return score
