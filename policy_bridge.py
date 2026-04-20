from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from runtime_core import ArcState, GuidancePacket, RuntimeMemory, TransitionReport
from runtime_core.policy import score_actions as runtime_policy_scores
from governance_state import (
    MemoryCell,
    EpistemicStatus,
    Goal,
    GoalPolicyClass,
    GoalStatus,
    PrecedenceClass,
    SourceType,
    compute_goal_weights,
    create_initial_state,
    create_runtime_support,
)

REVERSE_ACTION = {
    "move_up": "move_down",
    "move_down": "move_up",
    "move_left": "move_right",
    "move_right": "move_left",
}

LOGGER = logging.getLogger(__name__)


class PolicyBridge:
    """
    Decision bridge that turns ARC transition facts into governed state,
    then derives the next GuidancePacket from goals and evidence cells.
    """

    def __init__(self, root: Path | None = None, *, state_id: str = "arc3:policy-brain") -> None:
        self.root = root or (Path.cwd() / "policy_memory_data")
        self.state_id = state_id
        self.support = create_runtime_support(self.root)
        self.state = create_initial_state(state_id=self.state_id)
        self.tile_effects: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.marker_effects: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.action_outcomes: dict[str, dict[str, float]] = defaultdict(lambda: {"success": 0.0, "stall": 0.0})
        self.edge_outcomes: dict[tuple[tuple[int, int], tuple[int, int], str], dict[str, float]] = defaultdict(
            lambda: {"success": 0.0, "stall": 0.0}
        )
        self.target_outcomes: dict[tuple[int, int], dict[str, float]] = defaultdict(lambda: {"success": 0.0, "stall": 0.0})
        self.mechanic_point_effects: dict[tuple[int, int], dict[str, float]] = defaultdict(
            lambda: {"state_change": 0.0, "no_effect": 0.0}
        )
        self.mechanic_label_effects: dict[str, dict[str, float]] = defaultdict(
            lambda: {"state_change": 0.0, "no_effect": 0.0}
        )
        self.mechanic_feature_effects: dict[str, dict[str, float]] = defaultdict(
            lambda: {"state_change": 0.0, "no_effect": 0.0}
        )
        self.claim_push_remaining = 0
        self.claim_push_match_key: str | None = None
        self.align_route_remaining = 0
        self.align_route_action: str | None = None
        self.align_route_match_key: str | None = None
        self.align_exploration_target: tuple[int, int] | None = None
        self.align_exploration_match_key: str | None = None
        self.mechanic_probe_target: tuple[int, int] | None = None
        self.mechanic_probe_match_key: str | None = None
        self.mechanic_probe_phase: str | None = None
        self.planned_match_key: str | None = None
        self.planned_target_position: tuple[int, int] | None = None
        self.planned_reference_position: tuple[int, int] | None = None
        self.planned_marker_positions: list[tuple[int, int]] = []
        self.planned_mechanic_positions: list[tuple[int, int]] = []
        self.planned_mechanic_targets: list[str] = []
        self.planned_mechanic_features: list[str] = []
        self.completed_levels = 0
        self.last_guidance = GuidancePacket(
            mode="PROBE",
            active_goal={"kind": "explore_mechanics"},
            confidence={"rules": 0.0, "goals": 0.0, "risk": 0.0},
        )

    def reset(self) -> None:
        self.support = create_runtime_support(self.root)
        self.state = create_initial_state(state_id=self.state_id)
        self.tile_effects.clear()
        self.marker_effects.clear()
        self.action_outcomes.clear()
        self.edge_outcomes.clear()
        self.target_outcomes.clear()
        self.mechanic_point_effects.clear()
        self.mechanic_label_effects.clear()
        self.mechanic_feature_effects.clear()
        self.claim_push_remaining = 0
        self.claim_push_match_key = None
        self.align_route_remaining = 0
        self.align_route_action = None
        self.align_route_match_key = None
        self.align_exploration_target = None
        self.align_exploration_match_key = None
        self.mechanic_probe_target = None
        self.mechanic_probe_match_key = None
        self.mechanic_probe_phase = None
        self.planned_match_key = None
        self.planned_target_position = None
        self.planned_reference_position = None
        self.planned_marker_positions = []
        self.planned_mechanic_positions = []
        self.planned_mechanic_targets = []
        self.planned_mechanic_features = []
        self.completed_levels = 0
        self.last_guidance = GuidancePacket(
            mode="PROBE",
            active_goal={"kind": "explore_mechanics"},
            confidence={"rules": 0.0, "goals": 0.0, "risk": 0.0},
        )

    def on_level_transition(self) -> None:
        self.action_outcomes.clear()
        self.edge_outcomes.clear()
        self.target_outcomes.clear()
        self.claim_push_remaining = 0
        self.claim_push_match_key = None
        self.align_route_remaining = 0
        self.align_route_action = None
        self.align_route_match_key = None
        self.align_exploration_target = None
        self.align_exploration_match_key = None
        self.mechanic_probe_target = None
        self.mechanic_probe_match_key = None
        self.mechanic_probe_phase = None
        self.planned_match_key = None
        self.planned_target_position = None
        self.planned_reference_position = None
        self.planned_marker_positions = []
        self.planned_mechanic_positions = []
        self.planned_mechanic_targets = []
        self.planned_mechanic_features = []
        self.state.phase = 0
        self.state.telemetry_counters.pop("stall_streak", None)
        self.state.telemetry_counters.pop("stalled_moves", None)
        self.state.telemetry_counters.pop("successful_moves", None)
        self.state.active_cells = {
            cell_id: cell
            for cell_id, cell in self.state.active_cells.items()
            if not (
                cell_id.startswith("arc-step:")
                or cell_id.startswith("candidate-action:")
                or cell_id.startswith("guidance:")
            )
        }
        self.last_guidance = GuidancePacket(
            mode="PROBE",
            active_goal={"kind": "explore_mechanics"},
            confidence={"rules": 0.0, "goals": 0.0, "risk": 0.0},
        )

    def on_attempt_reset(self) -> None:
        self.claim_push_remaining = 0
        self.claim_push_match_key = None
        self.align_route_remaining = 0
        self.align_route_action = None
        self.align_route_match_key = None
        self.align_exploration_target = None
        self.align_exploration_match_key = None
        self.mechanic_probe_target = None
        self.mechanic_probe_match_key = None
        self.mechanic_probe_phase = None
        self.state.phase = 0
        self.state.telemetry_counters.pop("stall_streak", None)
        self.state.telemetry_counters.pop("stalled_moves", None)
        self.state.telemetry_counters.pop("successful_moves", None)
        self.state.active_cells = {
            cell_id: cell
            for cell_id, cell in self.state.active_cells.items()
            if not (
                cell_id.startswith("arc-step:")
                or cell_id.startswith("candidate-action:")
                or cell_id.startswith("guidance:")
            )
        }
        if self.planned_match_key is not None:
            self.last_guidance = GuidancePacket(
                mode="EXECUTE",
                active_goal={
                    "kind": "align_target_orientation",
                    "match_key": self.planned_match_key,
                    "target_position": list(self.planned_target_position) if self.planned_target_position is not None else None,
                    "reference_position": list(self.planned_reference_position) if self.planned_reference_position is not None else None,
                    "marker_positions": [list(point) for point in self.planned_marker_positions],
                    "mechanic_positions": [list(point) for point in self.planned_mechanic_positions],
                    "mechanic_targets": list(self.planned_mechanic_targets),
                    "mechanic_features": list(self.planned_mechanic_features),
                    "aligned": False,
                    "plan_locked": True,
                },
                confidence={"rules": 0.5, "goals": 0.6, "risk": 0.2},
            )
        else:
            self.last_guidance = GuidancePacket(
                mode="PROBE",
                active_goal={"kind": "explore_mechanics"},
                confidence={"rules": 0.0, "goals": 0.0, "risk": 0.0},
            )

    def bootstrap_guidance(self) -> GuidancePacket:
        return self.last_guidance

    def update(self, report: TransitionReport) -> GuidancePacket:
        self._record_evidence(report)
        self._record_transition_cell(report)
        self._sync_goals(report)
        self.state.recompute_derived_state()

        learned_rules = self._infer_rules()
        active_goal = self._select_active_goal()
        prioritized_targets, avoid_targets = self._infer_targets(report, active_goal, learned_rules)
        mode = self._choose_mode(report, active_goal, prioritized_targets, avoid_targets)
        confidence = self._confidence(active_goal, learned_rules, avoid_targets)

        guidance = GuidancePacket(
            mode=mode,
            active_goal=active_goal,
            prioritized_targets=prioritized_targets,
            avoid_targets=avoid_targets,
            learned_rules=learned_rules,
            confidence=confidence,
        )
        self._record_guidance_cell(report.step_id, guidance)
        self.state.recompute_derived_state()
        self.last_guidance = guidance
        return guidance

    def score_actions(
        self,
        state: ArcState,
        candidates: list[str],
        guidance: GuidancePacket | None,
        memory: RuntimeMemory,
    ) -> dict[str, float]:
        current_guidance = guidance or self.last_guidance
        baseline = runtime_policy_scores(state, candidates, current_guidance, memory)

        active_goal = current_guidance.active_goal if current_guidance else {}
        goal_kind = str(active_goal.get("kind", ""))
        goal_targets = {
            str(target).lower()
            for target in active_goal.get("targets", [])
        }
        goal_positions = [
            tuple(position)
            for position in active_goal.get("positions", [])
            if isinstance(position, (list, tuple)) and len(position) == 2
        ]
        wall_points = _resource_points(state.resources.get("wall_points"))
        marker_positions = _resource_points(active_goal.get("marker_positions")) or _resource_points(state.resources.get("marker_positions"))
        mechanic_positions = (
            _resource_points(active_goal.get("mechanic_positions"))
            or _resource_points(state.resources.get("mechanic_candidate_positions"))
            or marker_positions
        )
        mechanic_targets = {
            str(target).lower()
            for target in active_goal.get("mechanic_targets", [])
        }
        mechanic_features = {
            str(feature).lower()
            for feature in active_goal.get("mechanic_features", [])
        }
        target_position = _normalize_point(active_goal.get("target_position"))
        target_alignment = active_goal.get("aligned")
        avoid_targets = {
            str(target).lower()
            for target in (current_guidance.avoid_targets if current_guidance else [])
        }
        match_key = str(active_goal.get("match_key", ""))
        exploration_target = None
        if goal_kind == "align_target_orientation":
            exploration_target = self._resolve_align_exploration_target(
                state=state,
                memory=memory,
                mechanic_positions=mechanic_positions,
                match_key=match_key,
            )

        route_target: tuple[int, int] | None = None
        if goal_kind == "align_target_orientation":
            if (
                self.mechanic_probe_target is not None
                and self.mechanic_probe_match_key == match_key
                and self.mechanic_probe_target in mechanic_positions
            ):
                route_target = self.mechanic_probe_target
            else:
                route_target = exploration_target
        elif goal_kind == "claim_aligned_target":
            route_target = None

        route_trajectory = _shortest_path(state, state.player_pos, route_target)
        next_waypoint = route_trajectory[0] if route_trajectory else None
        if current_guidance is not None and isinstance(current_guidance.active_goal, dict):
            current_guidance.active_goal["route_target"] = list(route_target) if route_target is not None else None
            current_guidance.active_goal["route_trajectory"] = [list(point) for point in route_trajectory]
            current_guidance.active_goal["next_waypoint"] = list(next_waypoint) if next_waypoint is not None else None

        scores: dict[str, float] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        candidate_step = int(self.state.phase) + 1
        stall_streak = float(self.state.telemetry_counters.get("stall_streak", 0.0))
        last_action = memory.recent_actions[-1] if memory.recent_actions else None
        sorted_targets = sorted(goal_positions)
        oscillation_axis = _oscillation_axis(memory.recent_actions)
        explored_ratio = _explored_ratio(state, memory)
        for action in candidates:
            score = baseline.get(action, 0.0) * 0.25
            target_point = _predict_point(state, action)
            target_tile = state.visible_tiles.get(target_point) if target_point is not None else None
            target_entity = _entity_at(state, target_point)
            target_entity_payload = _entity_record_at(state, target_point)
            current_entity = _entity_at(state, state.player_pos)
            reasons: list[str] = []
            edge_stats = self._edge_stats(state.player_pos, target_point, action)
            target_stats = self.target_outcomes.get(target_point, {"success": 0.0, "stall": 0.0}) if target_point is not None else {"success": 0.0, "stall": 0.0}

            if current_guidance is not None and current_guidance.mode == "EXECUTE":
                score += 1.2
                reasons.append("execute_mode")
            elif current_guidance is not None and current_guidance.mode == "RECOVER":
                score += 0.6
                reasons.append("recover_mode")

            if route_target is not None and state.player_pos is not None and target_point is not None:
                current_route_distance = _path_distance(state, state.player_pos, route_target)
                next_route_distance = _path_distance(state, target_point, route_target)
                if next_waypoint is not None and target_point == next_waypoint:
                    score += 18.0
                    reasons.append("follow_route_waypoint")
                elif (
                    current_route_distance is not None
                    and next_route_distance is not None
                    and next_route_distance < current_route_distance
                ):
                    score += 5.5
                    reasons.append("advance_along_route")
                elif (
                    current_route_distance is not None
                    and next_route_distance is not None
                    and next_route_distance > current_route_distance
                ):
                    score -= 4.5
                    reasons.append("drift_from_route")
                elif current_route_distance is not None and next_route_distance is None:
                    score -= 7.5
                    reasons.append("route_lost")

            if goal_kind == "align_target_orientation":
                probe_active = (
                    self.mechanic_probe_target is not None
                    and self.mechanic_probe_match_key == match_key
                    and self.mechanic_probe_target in mechanic_positions
                )
                probe_target = self.mechanic_probe_target if probe_active else None
                if probe_target is not None and state.player_pos is not None and target_point is not None:
                    current_probe_distance = _path_distance(state, state.player_pos, probe_target)
                    next_probe_distance = _path_distance(state, target_point, probe_target)
                    if state.player_pos == probe_target:
                        if target_point in wall_points:
                            score -= 20.0
                            reasons.append("probe_step_off_wall")
                        elif target_point not in mechanic_positions and _manhattan(target_point, probe_target) == 1:
                            score += 85.0
                            reasons.append("probe_step_off")
                        else:
                            score -= 12.0
                            reasons.append("probe_step_off_miss")
                    elif _manhattan(state.player_pos, probe_target) == 1:
                        if target_point == probe_target:
                            score += 130.0
                            reasons.append("probe_step_on")
                        else:
                            score -= 30.0
                            reasons.append("probe_adjacent_miss")
                    elif (
                        current_probe_distance is not None
                        and next_probe_distance is not None
                        and next_probe_distance < current_probe_distance
                    ):
                        score += 9.0
                        reasons.append("probe_approach")
                    elif (
                        current_probe_distance is not None
                        and next_probe_distance is not None
                        and next_probe_distance > current_probe_distance
                    ):
                        score -= 6.0
                        reasons.append("probe_drift")

                align_route_active = (
                    self.align_route_remaining > 0
                    and self.align_route_action is not None
                    and self.align_route_match_key is not None
                    and match_key == self.align_route_match_key
                )
                if align_route_active:
                    if action == self.align_route_action:
                        score += 15.0
                        reasons.append("align_route_commit")
                    elif action == REVERSE_ACTION.get(self.align_route_action):
                        score -= 9.0
                        reasons.append("align_route_reverse_block")

                untested_mechanic_positions = {
                    point
                    for point in mechanic_positions
                    if sum(self.mechanic_point_effects.get(point, {"state_change": 0.0, "no_effect": 0.0}).values()) == 0
                }
                if state.player_pos is not None and untested_mechanic_positions and target_point is not None:
                    current_unknown_distance = _nearest_path_distance(state, state.player_pos, untested_mechanic_positions)
                    next_unknown_distance = _nearest_path_distance(state, target_point, untested_mechanic_positions)
                    if (
                        current_unknown_distance is not None
                        and next_unknown_distance is not None
                        and next_unknown_distance < current_unknown_distance
                    ):
                        score += 5.5
                        reasons.append("closer_to_untested_mechanic")
                    elif (
                        current_unknown_distance is not None
                        and next_unknown_distance is not None
                        and next_unknown_distance > current_unknown_distance
                    ):
                        score -= 3.5
                        reasons.append("farther_from_untested_mechanic")

                if (
                    state.player_pos is not None
                    and target_point is not None
                    and state.player_pos not in mechanic_positions
                    and any(_manhattan(state.player_pos, position) == 1 for position in mechanic_positions)
                ):
                    if target_point in mechanic_positions:
                        score += 100.0
                        reasons.append("adjacent_mechanic_override")
                    else:
                        score -= 25.0
                        reasons.append("adjacent_mechanic_miss")

                if state.player_pos is not None and mechanic_positions and target_point is not None:
                    current_marker_distance = _nearest_path_distance(state, state.player_pos, mechanic_positions)
                    next_marker_distance = _nearest_path_distance(state, target_point, mechanic_positions)
                    if (
                        current_marker_distance is not None
                        and next_marker_distance is not None
                        and next_marker_distance < current_marker_distance
                    ):
                        score += 4.2
                        reasons.append("closer_to_mechanic")
                    elif (
                        current_marker_distance is not None
                        and next_marker_distance is not None
                        and next_marker_distance > current_marker_distance
                    ):
                        score -= 2.4
                        reasons.append("farther_from_mechanic")
                    elif current_marker_distance is None and next_marker_distance is not None:
                        score += 2.0
                        reasons.append("path_to_mechanic_found")
                    elif current_marker_distance is not None and next_marker_distance is None:
                        score -= 3.5
                        reasons.append("path_to_mechanic_lost")
                    if (
                        current_marker_distance is not None
                        and next_marker_distance is not None
                        and current_marker_distance <= 3
                    ):
                        if next_marker_distance == current_marker_distance - 1:
                            score += 7.5
                            reasons.append("commit_to_mechanic_route")
                        elif next_marker_distance >= current_marker_distance:
                            score -= 5.0
                            reasons.append("drift_from_mechanic_route")
                    if current_marker_distance is not None and current_marker_distance <= 3:
                        if target_point in wall_points:
                            score -= 10.0
                            reasons.append("wall_blocks_mechanic_route")
                        if next_marker_distance is None:
                            score -= 7.0
                            reasons.append("invalid_mechanic_route")
                if target_point is not None and target_point in mechanic_positions:
                    score += 4.8
                    reasons.append("step_on_mechanic")
                    if target_point in untested_mechanic_positions:
                        score += 6.0
                        reasons.append("step_on_untested_mechanic")
                if (
                    state.player_pos is not None
                    and any(_manhattan(state.player_pos, position) == 1 for position in mechanic_positions)
                    and target_point is not None
                    and target_point not in mechanic_positions
                ):
                    score -= 5.0
                    reasons.append("adjacent_to_mechanic_do_not_skip")
                point_effects = self.mechanic_point_effects.get(target_point, {"state_change": 0.0, "no_effect": 0.0}) if target_point is not None else {"state_change": 0.0, "no_effect": 0.0}
                if point_effects["state_change"] > 0:
                    score += min(6.0, 2.5 + point_effects["state_change"] * 1.5)
                    reasons.append("known_state_change_point")
                elif point_effects["no_effect"] > 0:
                    score -= min(4.0, point_effects["no_effect"] * 1.0)
                    reasons.append("known_no_effect_point")
                if target_entity is not None and (
                    "marker_cross" in target_entity.lower()
                    or target_entity.lower() in mechanic_targets
                ):
                    score += 3.5
                    reasons.append("mechanic_entity")
                    label_effects = self.mechanic_label_effects.get(target_entity.lower(), {"state_change": 0.0, "no_effect": 0.0})
                    if label_effects["state_change"] == 0 and label_effects["no_effect"] == 0:
                        score += 4.0
                        reasons.append("untested_mechanic_entity")
                    if label_effects["state_change"] > 0:
                        score += min(4.0, 1.5 + label_effects["state_change"] * 1.0)
                        reasons.append("known_state_change_entity")
                    elif label_effects["no_effect"] > 0:
                        score -= min(3.0, label_effects["no_effect"] * 0.75)
                        reasons.append("known_no_effect_entity")
                target_features = _mechanic_features_from_entity(target_entity_payload)
                positive_feature_hits = [
                    feature
                    for feature in target_features
                    if self.mechanic_feature_effects[feature]["state_change"] > 0
                ]
                negative_feature_hits = [
                    feature
                    for feature in target_features
                    if self.mechanic_feature_effects[feature]["no_effect"] > self.mechanic_feature_effects[feature]["state_change"]
                ]
                if positive_feature_hits:
                    score += min(5.5, 2.0 + len(positive_feature_hits) * 0.9)
                    reasons.append("known_state_change_feature")
                elif negative_feature_hits:
                    score -= min(3.5, 1.0 + len(negative_feature_hits) * 0.6)
                    reasons.append("known_no_effect_feature")
                elif target_features:
                    score += 1.8
                    reasons.append("untested_mechanic_feature")
                if mechanic_features and target_features.intersection(mechanic_features):
                    score += 2.4
                    reasons.append("mechanic_feature_match")
                if target_position is not None and target_point == target_position and not target_alignment:
                    score -= 6.0
                    reasons.append("premature_target_overlap")
                if (
                    target_point is not None
                    and target_point not in memory.visited_positions
                    and target_point not in wall_points
                ):
                    score += 1.6
                    reasons.append("novel_floor_explore")
                    if explored_ratio < 0.45:
                        score += 3.8
                        reasons.append("large_map_frontier")
                    elif explored_ratio < 0.7:
                        score += 1.8
                        reasons.append("mid_map_frontier")

                if (
                    state.player_pos is not None
                    and target_point is not None
                    and target_point not in wall_points
                ):
                    current_neighbors = _walkable_neighbor_count(state, state.player_pos)
                    next_neighbors = _walkable_neighbor_count(state, target_point)
                    current_frontier = _unvisited_walkable_neighbor_count(state, state.player_pos, memory)
                    next_frontier = _unvisited_walkable_neighbor_count(state, target_point, memory)
                    in_corridor = current_neighbors <= 2
                    toward_broader_frontier = next_frontier > current_frontier
                    deeper_into_tunnel = next_neighbors <= 2 and next_frontier > 0

                    if (
                        last_action == action
                        and stall_streak == 0
                        and edge_stats["stall"] == 0
                    ):
                        score += 2.6
                        reasons.append("continue_successful_heading")
                        if in_corridor:
                            score += 4.4
                            reasons.append("corridor_commit")
                        if deeper_into_tunnel or toward_broader_frontier:
                            score += 2.4
                            reasons.append("forward_into_frontier")

                    if (
                        last_action is not None
                        and REVERSE_ACTION.get(last_action) == action
                        and stall_streak == 0
                        and in_corridor
                    ):
                        score -= 5.2
                        reasons.append("avoid_corridor_reversal")

                    if toward_broader_frontier and target_point not in memory.visited_positions:
                        score += 2.2
                        reasons.append("broader_frontier")
                    elif next_frontier == 0 and next_neighbors <= 1 and target_point in memory.visited_positions:
                        score -= 2.0
                        reasons.append("spent_tunnel")

                if state.player_pos is not None and target_point is not None and exploration_target is not None:
                    current_explore_distance = _path_distance(state, state.player_pos, exploration_target)
                    next_explore_distance = _path_distance(state, target_point, exploration_target)
                    if (
                        current_explore_distance is not None
                        and next_explore_distance is not None
                        and next_explore_distance < current_explore_distance
                    ):
                        score += 7.0
                        reasons.append("closer_to_exploration_target")
                    elif (
                        current_explore_distance is not None
                        and next_explore_distance is not None
                        and next_explore_distance > current_explore_distance
                    ):
                        score -= 4.0
                        reasons.append("farther_from_exploration_target")
                    elif current_explore_distance is None and next_explore_distance is not None:
                        score += 4.0
                        reasons.append("path_to_exploration_target_found")
                    elif current_explore_distance is not None and next_explore_distance is None:
                        score -= 5.0
                        reasons.append("path_to_exploration_target_lost")

                    if (
                        last_action == action
                        and current_explore_distance is not None
                        and next_explore_distance is not None
                        and next_explore_distance < current_explore_distance
                    ):
                        score += 3.5
                        reasons.append("continue_toward_exploration_target")

            if goal_kind == "claim_aligned_target":
                claim_push_active = (
                    self.claim_push_remaining > 0
                    and self.claim_push_match_key is not None
                    and str(active_goal.get("match_key", "")) == self.claim_push_match_key
                )
                if claim_push_active:
                    if action == "move_up":
                        score += 240.0
                        reasons.append("claim_push_up_commit")
                    elif action == "move_down":
                        score -= 140.0
                        reasons.append("claim_push_down_block")
                    else:
                        score -= 40.0
                        reasons.append("claim_push_lateral_block")
                if (
                    target_position is not None
                    and state.player_pos is not None
                    and state.player_pos[0] == target_position[0]
                    and state.player_pos[1] > target_position[1]
                ):
                    if action == "move_up":
                        score += 150.0
                        reasons.append("hard_claim_up_override")
                    elif action == "move_down":
                        score -= 60.0
                        reasons.append("hard_claim_down_block")
                    else:
                        score -= 20.0
                        reasons.append("hard_claim_lateral_block")
                if (
                    target_position is not None
                    and state.player_pos is not None
                    and target_point is not None
                    and state.player_pos != target_position
                    and _manhattan(state.player_pos, target_position) == 1
                ):
                    if target_point == target_position:
                        score += 100.0
                        reasons.append("adjacent_target_override")
                    else:
                        score -= 25.0
                        reasons.append("adjacent_target_miss")
                if target_position is not None and state.player_pos is not None and target_point is not None:
                    current_claim_distance = _manhattan(state.player_pos, target_position)
                    next_claim_distance = _manhattan(target_point, target_position)
                    if current_claim_distance <= 3:
                        if next_claim_distance < current_claim_distance:
                            score += 18.0
                            reasons.append("commit_into_goal_chamber")
                        elif next_claim_distance > current_claim_distance:
                            score -= 12.0
                            reasons.append("retreat_from_goal_chamber")
                    if current_claim_distance <= 2 and target_point == target_position:
                        score += 12.0
                        reasons.append("center_on_target")
                    preferred_claim_action = _preferred_axis_action(state.player_pos, target_position)
                    if preferred_claim_action is not None and current_claim_distance <= 3:
                        if action == preferred_claim_action:
                            score += 24.0
                            reasons.append("continue_into_target")
                        elif action == REVERSE_ACTION.get(preferred_claim_action):
                            score -= 16.0
                            reasons.append("retreat_from_target_axis")
                if target_position is not None and state.player_pos is not None and target_point is not None:
                    current_target_distance = _path_distance(state, state.player_pos, target_position)
                    next_target_distance = _path_distance(state, target_point, target_position)
                    if (
                        current_target_distance is not None
                        and next_target_distance is not None
                        and next_target_distance < current_target_distance
                    ):
                        score += 5.0
                        reasons.append("closer_to_claim")
                    elif (
                        current_target_distance is not None
                        and next_target_distance is not None
                        and next_target_distance > current_target_distance
                    ):
                        score -= 3.0
                        reasons.append("farther_from_claim")
                    elif current_target_distance is None and next_target_distance is not None:
                        score += 2.5
                        reasons.append("path_to_target_found")
                    elif current_target_distance is not None and next_target_distance is None:
                        score -= 4.0
                        reasons.append("path_to_target_lost")
                if target_position is not None and target_point == target_position:
                    score += 7.0
                    reasons.append("claim_target")
                if target_entity is not None and _entity_matches_match_key(target_entity, active_goal.get("match_key")):
                    score += 4.5
                    reasons.append("target_signature_contact")
                if current_entity is not None and _entity_matches_match_key(current_entity, active_goal.get("match_key")):
                    score += 6.0
                    reasons.append("standing_on_target_signature")
                if target_point is not None and target_point in marker_positions:
                    score -= 1.5
                    reasons.append("marker_detour")

            if state.player_pos is not None and goal_positions and target_point is not None:
                current_distance = min(_manhattan(state.player_pos, position) for position in goal_positions)
                next_distance = min(_manhattan(target_point, position) for position in goal_positions)
                if next_distance < current_distance:
                    score += 4.5
                    reasons.append("closer_to_goal")
                elif next_distance > current_distance:
                    score -= 3.0
                    reasons.append("farther_from_goal")
                if len(sorted_targets) == 1:
                    goal_x, goal_y = sorted_targets[0]
                    if target_point[0] == goal_x or target_point[1] == goal_y:
                        score += 0.75
                        reasons.append("goal_axis_alignment")

            if target_entity is not None and target_entity.lower() in goal_targets:
                score += 4.0
                reasons.append("goal_entity")
            if target_tile is not None and target_tile.lower() in goal_targets:
                score += 3.0
                reasons.append("goal_tile")

            if target_point in wall_points:
                target_stats = self.target_outcomes.get(target_point, {"success": 0.0, "stall": 0.0})
                if target_stats["success"] == 0:
                    score -= 6.5
                    reasons.append("visual_wall_candidate")
                else:
                    score -= 1.0
                    reasons.append("wall_guess_overridden")

            if target_tile is not None and target_tile.lower() in avoid_targets:
                score -= 4.5
                reasons.append("avoid_tile")
            if target_entity is not None and target_entity.lower() in avoid_targets:
                score -= 4.0
                reasons.append("avoid_entity")

            moved_success = self.action_outcomes[action]["success"]
            moved_stall = self.action_outcomes[action]["stall"]
            if moved_success > 0:
                score += min(1.5, moved_success * 0.18)
                reasons.append("historical_success")
            if moved_stall > 0:
                score -= min(2.0, moved_stall * 0.3)
                reasons.append("historical_stall")

            if edge_stats["stall"] > 0 and edge_stats["success"] == 0:
                score -= min(7.5, 3.5 + edge_stats["stall"] * 1.25)
                reasons.append("known_blocked_edge")
            elif edge_stats["success"] > 0:
                score += min(3.0, 1.2 + edge_stats["success"] * 0.4)
                reasons.append("known_working_edge")
            else:
                score += 1.25
                reasons.append("untested_edge")

            if target_stats["stall"] > 0 and target_stats["success"] == 0:
                score -= min(4.0, target_stats["stall"] * 1.25)
                reasons.append("blocked_target_point")
            elif target_stats["success"] > 0:
                score += min(1.5, target_stats["success"] * 0.3)
                reasons.append("reachable_target_point")

            if target_point in memory.recent_positions:
                score -= 2.0
                reasons.append("recent_loop")
            if target_point is None or target_tile is None:
                score -= 1.5
                reasons.append("unknown_target")

            if last_action is not None and action == last_action:
                score -= 0.8
                reasons.append("repeat_last_action")
            if last_action is not None and REVERSE_ACTION.get(last_action) == action:
                score -= 1.4
                reasons.append("reverse_oscillation")

            if (
                goal_kind == "align_target_orientation"
                and last_action is not None
                and action == last_action
                and stall_streak == 0
                and edge_stats["stall"] == 0
            ):
                score += 1.4
                reasons.append("override_repeat_for_progress")

            if stall_streak >= 2:
                if edge_stats["stall"] == 0:
                    score += 1.8
                    reasons.append("stall_escape")
                if action != last_action:
                    score += 0.9
                    reasons.append("action_change_under_stall")
                if target_point is not None and target_point not in memory.visited_positions:
                    score += 1.2
                    reasons.append("novel_under_stall")

            if goal_kind == "align_target_orientation" and stall_streak >= 2 and oscillation_axis is not None:
                if oscillation_axis == "vertical":
                    if action in {"move_up", "move_down"}:
                        score -= 5.0
                        reasons.append("break_vertical_oscillation")
                    else:
                        score += 3.0
                        reasons.append("escape_vertical_oscillation")
                elif oscillation_axis == "horizontal":
                    if action in {"move_left", "move_right"}:
                        score -= 5.0
                        reasons.append("break_horizontal_oscillation")
                    else:
                        score += 3.0
                        reasons.append("escape_horizontal_oscillation")

            if stall_streak >= 4 and edge_stats["stall"] > 0:
                score -= 2.5
                reasons.append("persistent_stall_penalty")

            if target_tile is not None:
                tile_score = self._effect_score(self.tile_effects.get(target_tile.lower(), []))
                if tile_score > 0:
                    score += min(2.5, tile_score * 0.5)
                    reasons.append("positive_tile_history")
                elif tile_score < 0:
                    score += max(-3.0, tile_score * 0.6)
                    reasons.append("negative_tile_history")

            score = round(score, 4)
            scores[action] = score
            diagnostics[action] = {
                "score": score,
                "target_point": list(target_point) if target_point is not None else None,
                "target_tile": target_tile,
                "target_entity": target_entity,
                "reasons": list(reasons),
            }
            self._record_candidate_cell(
                step_id=candidate_step,
                action=action,
                score=score,
                target_point=target_point,
                target_tile=target_tile,
                target_entity=target_entity,
                reasons=reasons,
                active_goal=active_goal,
            )

        if (
            goal_kind == "claim_aligned_target"
            and state.player_pos is not None
            and target_position is not None
            and _manhattan(state.player_pos, target_position) <= 3
        ):
            LOGGER.info(
                "Claim diagnostics step=%s player=%s target=%s current_entity=%s diagnostics=%s",
                candidate_step,
                list(state.player_pos),
                list(target_position),
                current_entity,
                json.dumps(diagnostics, sort_keys=True),
            )
        if (
            goal_kind == "align_target_orientation"
            and state.player_pos is not None
            and stall_streak >= 4
        ):
            LOGGER.info(
                "Align diagnostics step=%s player=%s mechanics=%s oscillation=%s diagnostics=%s",
                candidate_step,
                list(state.player_pos),
                [list(point) for point in sorted(mechanic_positions)],
                oscillation_axis,
                json.dumps(diagnostics, sort_keys=True),
            )
        if goal_kind == "align_target_orientation" and exploration_target is not None:
            LOGGER.info(
                "Align target step=%s player=%s exploration_target=%s explored_ratio=%.3f",
                candidate_step,
                list(state.player_pos) if state.player_pos is not None else None,
                list(exploration_target),
                explored_ratio,
            )

        self.state.phase = candidate_step
        self.state.recompute_derived_state()
        return scores
    def _resolve_align_exploration_target(
        self,
        *,
        state: ArcState,
        memory: RuntimeMemory,
        mechanic_positions: set[tuple[int, int]],
        match_key: str,
    ) -> tuple[int, int] | None:
        if (
            self.mechanic_probe_target is not None
            and self.mechanic_probe_match_key == match_key
            and self.mechanic_probe_target in mechanic_positions
            and _path_distance(state, state.player_pos, self.mechanic_probe_target) is not None
        ):
            return self.mechanic_probe_target

        current = self.align_exploration_target
        if (
            self.align_exploration_match_key != match_key
            or current is None
            or current == state.player_pos
            or current in memory.visited_positions
            or _path_distance(state, state.player_pos, current) is None
        ):
            current = _select_align_exploration_target(
                state=state,
                memory=memory,
                mechanic_positions=mechanic_positions,
                mechanic_point_effects=self.mechanic_point_effects,
            )
            self.align_exploration_target = current
            self.align_exploration_match_key = match_key if current is not None else None
            if current is not None and current in mechanic_positions:
                self.mechanic_probe_target = current
                self.mechanic_probe_match_key = match_key
                self.mechanic_probe_phase = "approach"
        return current

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_cells": sorted(self.state.active_cells.keys()),
            "goals": [goal.to_dict() for goal in self.state.goal_objects()],
            "telemetry_counters": dict(self.state.telemetry_counters),
        }

    def _record_evidence(self, report: TransitionReport) -> None:
        delta = report.delta
        if self.claim_push_remaining > 0:
            self.claim_push_remaining = max(0, self.claim_push_remaining - 1)
        if self.align_route_remaining > 0:
            self.align_route_remaining = max(0, self.align_route_remaining - 1)
        tile = (delta.entered_tile or "").lower()
        marker = (delta.crossed_marker or "").lower()
        target_state_changed = _target_state_changed(report)
        resource_state_changed = _meaningful_resource_change(delta.resource_deltas)
        effectful_interaction = target_state_changed or resource_state_changed
        if target_state_changed:
            self.align_exploration_target = None
            self.align_exploration_match_key = None
            self.mechanic_probe_target = None
            self.mechanic_probe_match_key = None
            self.mechanic_probe_phase = None
        interaction_points = _interaction_points(report)
        interaction_labels = _interaction_labels(report)
        sample = {
            "resource_deltas": dict(delta.resource_deltas),
            "terminal": report.terminal,
            "win": report.win,
            "action": report.action,
            "target_state_changed": target_state_changed,
            "resource_state_changed": resource_state_changed,
        }
        if tile:
            self.tile_effects[tile].append(sample)
        if marker:
            self.marker_effects[marker].append(sample)
        for label in interaction_labels:
            self.marker_effects[label].append(sample)
            if effectful_interaction:
                self.mechanic_label_effects[label]["state_change"] += 1.0
            else:
                self.mechanic_label_effects[label]["no_effect"] += 1.0
        for feature in _interaction_features(report):
            if effectful_interaction:
                self.mechanic_feature_effects[feature]["state_change"] += 1.0
            else:
                self.mechanic_feature_effects[feature]["no_effect"] += 1.0
        for point in interaction_points:
            if effectful_interaction:
                self.mechanic_point_effects[point]["state_change"] += 1.0
            else:
                self.mechanic_point_effects[point]["no_effect"] += 1.0

        moved = delta.position_delta not in {None, (0, 0)}
        action_key = str(report.action)
        prev_pos = _normalize_point(report.prev_state.get("player_pos"))
        next_pos = _normalize_point(report.next_state.get("player_pos"))
        attempted_target = _predict_from_point(prev_pos, action_key)
        if moved:
            self.action_outcomes[action_key]["success"] += 1.0
        else:
            self.action_outcomes[action_key]["stall"] += 1.0
        if prev_pos is not None and attempted_target is not None:
            edge_key = (prev_pos, attempted_target, action_key)
            if moved and next_pos is not None:
                self.edge_outcomes[edge_key]["success"] += 1.0
                self.target_outcomes[next_pos]["success"] += 1.0
            else:
                self.edge_outcomes[edge_key]["stall"] += 1.0
                self.target_outcomes[attempted_target]["stall"] += 1.0

        if moved:
            self.state.telemetry_counters["successful_moves"] = float(
                self.state.telemetry_counters.get("successful_moves", 0.0) + 1.0
            )
            self.state.telemetry_counters["stall_streak"] = 0.0
        else:
            self.state.telemetry_counters["stalled_moves"] = float(
                self.state.telemetry_counters.get("stalled_moves", 0.0) + 1.0
            )
            self.state.telemetry_counters["stall_streak"] = float(
                self.state.telemetry_counters.get("stall_streak", 0.0) + 1.0
            )

        active_match = report.next_state.get("resources", {}).get("active_match_target")
        if isinstance(active_match, dict):
            match_key = str(active_match.get("match_key", ""))
            aligned = bool(active_match.get("aligned"))
            touched_target = _entity_matches_match_key(delta.touched_entity, match_key)
            upward_target_contact = aligned and delta.position_delta == (0, -1) and touched_target
            prev_pos = _normalize_point(report.prev_state.get("player_pos"))
            next_pos = _normalize_point(report.next_state.get("player_pos"))
            mechanic_points = _resource_points(report.next_state.get("resources", {}).get("mechanic_candidate_positions"))
            if upward_target_contact and match_key:
                self.claim_push_remaining = 2
                self.claim_push_match_key = match_key
            elif report.win or not aligned:
                self.claim_push_remaining = 0
                self.claim_push_match_key = None

            if self.mechanic_probe_match_key not in {None, match_key}:
                self.mechanic_probe_target = None
                self.mechanic_probe_match_key = None
                self.mechanic_probe_phase = None
            elif self.mechanic_probe_target is not None and self.mechanic_probe_match_key == match_key:
                if effectful_interaction:
                    self.mechanic_probe_target = None
                    self.mechanic_probe_match_key = None
                    self.mechanic_probe_phase = None
                    self.align_exploration_target = None
                    self.align_exploration_match_key = None
                elif next_pos == self.mechanic_probe_target:
                    self.mechanic_probe_phase = "step_off"
                elif prev_pos == self.mechanic_probe_target and next_pos != self.mechanic_probe_target:
                    self.mechanic_probe_target = None
                    self.mechanic_probe_match_key = None
                    self.mechanic_probe_phase = None
                    self.align_exploration_target = None
                    self.align_exploration_match_key = None
                elif next_pos is not None:
                    self.mechanic_probe_phase = "approach"

            if (
                not aligned
                and match_key
                and prev_pos is not None
                and next_pos is not None
                and delta.position_delta not in {None, (0, 0)}
                and mechanic_points
            ):
                prev_distance = _nearest_manhattan(prev_pos, mechanic_points)
                next_distance = _nearest_manhattan(next_pos, mechanic_points)
                if prev_distance is not None and next_distance is not None and next_distance < prev_distance:
                    self.align_route_remaining = 2
                    self.align_route_action = report.action
                    self.align_route_match_key = match_key
                elif target_state_changed or next_pos in mechanic_points:
                    self.align_route_remaining = 0
                    self.align_route_action = None
                    self.align_route_match_key = None
            elif not aligned and delta.position_delta in {None, (0, 0)} and report.action == self.align_route_action:
                self.align_route_remaining = 0
                self.align_route_action = None
                self.align_route_match_key = None

            exploration_target = (
                self.align_exploration_target
                if self.align_exploration_match_key == match_key
                else None
            )
            if (
                not aligned
                and match_key
                and prev_pos is not None
                and next_pos is not None
                and delta.position_delta not in {None, (0, 0)}
                and exploration_target is not None
            ):
                prev_explore_distance = _manhattan(prev_pos, exploration_target)
                next_explore_distance = _manhattan(next_pos, exploration_target)
                if next_explore_distance < prev_explore_distance:
                    self.align_route_remaining = max(self.align_route_remaining, 2)
                    self.align_route_action = report.action
                    self.align_route_match_key = match_key
                elif report.action == self.align_route_action and next_explore_distance >= prev_explore_distance:
                    self.align_route_remaining = 0
                    self.align_route_action = None
                    self.align_route_match_key = None
            elif (
                not aligned
                and match_key
                and delta.position_delta in {None, (0, 0)}
                and self.align_exploration_match_key == match_key
            ):
                attempted_exploration_target = self.align_exploration_target
                attempted_point = _predict_from_point(prev_pos, action_key) if prev_pos is not None else None
                if attempted_exploration_target is not None and attempted_point is not None:
                    route = _shortest_path_from_points(
                        report.next_state,
                        prev_pos,
                        attempted_exploration_target,
                    )
                    next_waypoint = route[0] if route else None
                    if next_waypoint is not None and attempted_point == next_waypoint:
                        self.align_exploration_target = None
                        self.align_exploration_match_key = None
                        self.align_route_remaining = 0
                        self.align_route_action = None
                        self.align_route_match_key = None
            if aligned:
                self.align_exploration_target = None
                self.align_exploration_match_key = None
        else:
            self.claim_push_remaining = 0
            self.claim_push_match_key = None
            self.align_route_remaining = 0
            self.align_route_action = None
            self.align_route_match_key = None
            self.align_exploration_target = None
            self.align_exploration_match_key = None
            self.mechanic_probe_target = None
            self.mechanic_probe_match_key = None
            self.mechanic_probe_phase = None

    def _record_transition_cell(self, report: TransitionReport) -> None:
        cell = MemoryCell(cell_id=f"arc-step:{report.step_id}", cell_type="transition", role="evidence", namespace="arc")
        cell.content.text = json.dumps(
            {
                "action": report.action,
                "delta": report.delta.to_dict(),
                "terminal": report.terminal,
                "win": report.win,
                "local_context": report.local_context,
            },
            sort_keys=True,
        )
        cell.content.structured = {
            "action": report.action,
            "delta": report.delta.to_dict(),
            "terminal": report.terminal,
            "win": report.win,
            "prev_state": report.prev_state,
            "next_state": report.next_state,
        }
        cell.content.summary = f"ARC step {report.step_id}: {report.action}"
        cell.content.keywords = [
            "arc",
            str(report.action),
            str(report.delta.entered_tile or "no_tile"),
            str(report.delta.touched_entity or "no_touch"),
        ]
        cell.trace.source_type = SourceType.RUNTIME
        cell.trace.source_id = "arc_runtime"
        cell.metrics.confidence = 0.95
        cell.metrics.goal_alignment = float(self.last_guidance.confidence.get("goals", 0.0))
        cell.metrics.relevance = 0.9
        cell.bindings.provided_keys = ["useful_context", "arc_transition"]
        cell.governance.precedence_class = PrecedenceClass.EVIDENCE
        cell.epistemic.epistemic_status = EpistemicStatus.OBSERVED
        self.state.active_cells[cell.cell_id] = cell

    def _record_candidate_cell(
        self,
        *,
        step_id: int,
        action: str,
        score: float,
        target_point: tuple[int, int] | None,
        target_tile: str | None,
        target_entity: str | None,
        reasons: list[str],
        active_goal: dict[str, Any],
    ) -> None:
        cell = MemoryCell(
            cell_id=f"candidate-action:{step_id}:{action}",
            cell_type="candidate_action",
            role="controller",
            namespace="arc",
        )
        cell.content.structured = {
            "action": action,
            "score": score,
            "target_point": list(target_point) if target_point is not None else None,
            "target_tile": target_tile,
            "target_entity": target_entity,
            "reasons": list(reasons),
            "active_goal": active_goal,
        }
        cell.content.text = json.dumps(cell.content.structured, sort_keys=True)
        cell.content.summary = f"Action candidate {action}: score={score}"
        cell.content.keywords = ["arc", "candidate_action", action] + list(reasons[:4])
        cell.trace.source_type = SourceType.RUNTIME
        cell.trace.source_id = "policy_bridge"
        cell.metrics.priority = float(score)
        cell.metrics.goal_alignment = float(self.last_guidance.confidence.get("goals", 0.0))
        cell.metrics.confidence = 0.75
        cell.bindings.provided_keys = ["guidance", "arc_action_candidate"]
        cell.governance.precedence_class = PrecedenceClass.GOAL
        cell.epistemic.epistemic_status = EpistemicStatus.INFERRED
        self.state.active_cells[cell.cell_id] = cell

    def _record_guidance_cell(self, step_id: int, guidance: GuidancePacket) -> None:
        cell = MemoryCell(cell_id=f"guidance:{step_id}", cell_type="guidance", role="controller", namespace="arc")
        cell.content.text = json.dumps(guidance.to_dict(), sort_keys=True)
        cell.content.structured = guidance.to_dict()
        cell.content.summary = f"Guidance step {step_id}: mode={guidance.mode}"
        cell.content.keywords = [
            "arc",
            "guidance",
            guidance.mode.lower(),
            str(guidance.active_goal.get("kind", "none")),
        ]
        cell.trace.source_type = SourceType.RUNTIME
        cell.trace.source_id = "policy_bridge"
        cell.metrics.confidence = float(guidance.confidence.get("goals", 0.0))
        cell.metrics.goal_alignment = float(guidance.confidence.get("goals", 0.0))
        cell.bindings.provided_keys = ["stable_runtime", "guidance"]
        cell.governance.precedence_class = PrecedenceClass.GOAL
        cell.epistemic.epistemic_status = EpistemicStatus.INFERRED
        self.state.active_cells[cell.cell_id] = cell

    def _capture_level_plan(
        self,
        *,
        active_match: dict[str, Any],
        marker_positions: list[list[int] | tuple[int, int]],
        mechanic_positions: list[list[int] | tuple[int, int]],
        mechanic_entities: list[dict[str, Any]],
    ) -> None:
        match_key = str(active_match.get("match_key", ""))
        if not match_key:
            return
        if self.planned_match_key == match_key and self.planned_mechanic_positions:
            return

        normalized_markers = [
            point
            for point in (_normalize_point(position) for position in marker_positions)
            if point is not None
        ]
        normalized_mechanics = [
            point
            for point in (_normalize_point(position) for position in mechanic_positions)
            if point is not None
        ]
        mechanic_targets = sorted(
            {
                str(entity.get("name", entity.get("type", "mechanic"))).lower()
                for entity in mechanic_entities
                if isinstance(entity, dict)
            }
        )
        mechanic_features = sorted(
            {
                feature
                for entity in mechanic_entities
                if isinstance(entity, dict)
                for feature in _mechanic_features_from_entity(entity)
            }
        )

        self.planned_match_key = match_key
        self.planned_target_position = _normalize_point(active_match.get("target_position"))
        self.planned_reference_position = _normalize_point(active_match.get("reference_position"))
        self.planned_marker_positions = sorted(set(normalized_markers))
        self.planned_mechanic_positions = sorted(set(normalized_mechanics) | set(self.planned_marker_positions))
        self.planned_mechanic_targets = mechanic_targets
        self.planned_mechanic_features = mechanic_features
        LOGGER.info(
            "Locked level plan match=%s target=%s mechanics=%s",
            self.planned_match_key,
            list(self.planned_target_position) if self.planned_target_position is not None else None,
            [list(point) for point in self.planned_mechanic_positions],
        )

    def _sync_goals(self, report: TransitionReport) -> None:
        preserved_goals = [
            goal.clone()
            for goal in self.state.goal_objects()
            if not str(goal.goal_id).startswith("goal:arc:")
        ]
        dynamic_goals: list[Goal] = []
        resources = report.next_state.get("resources", {})
        active_match = resources.get("active_match_target")
        marker_positions = [position for position in resources.get("marker_positions", []) if _normalize_point(position) is not None]
        mechanic_positions = [position for position in resources.get("mechanic_candidate_positions", []) if _normalize_point(position) is not None]
        mechanic_entities = [
            entity
            for entity in resources.get("mechanic_candidate_entities", [])
            if isinstance(entity, dict)
        ]
        mechanic_features = sorted(
            {
                feature
                for entity in mechanic_entities
                for feature in _mechanic_features_from_entity(entity)
            }
        )
        known_mechanic_positions = [
            list(point)
            for point, stats in sorted(self.mechanic_point_effects.items())
            if stats["state_change"] > 0
        ]
        known_mechanic_labels = [
            label
            for label, stats in sorted(self.mechanic_label_effects.items())
            if stats["state_change"] > 0
        ]
        known_mechanic_features = [
            feature
            for feature, stats in sorted(self.mechanic_feature_effects.items())
            if stats["state_change"] > 0
        ]

        if isinstance(active_match, dict):
            self._capture_level_plan(
                active_match=active_match,
                marker_positions=marker_positions,
                mechanic_positions=mechanic_positions,
                mechanic_entities=mechanic_entities,
            )
            match_key = str(active_match.get("match_key", "match"))
            target_name = str(active_match.get("target_name", "target"))
            target_position = self.planned_target_position or _normalize_point(active_match.get("target_position"))
            reference_position = self.planned_reference_position or _normalize_point(active_match.get("reference_position"))
            planned_marker_positions = [list(point) for point in self.planned_marker_positions]
            planned_mechanic_positions = [list(point) for point in self.planned_mechanic_positions]
            planned_mechanic_targets = list(self.planned_mechanic_targets)
            planned_mechanic_features = list(self.planned_mechanic_features)
            aligned = bool(active_match.get("aligned"))

            if not aligned:
                dynamic_goals.append(
                    Goal(
                        goal_id=f"goal:arc:align:{match_key}",
                        text=f"Investigate unusual walkable tiles or objects to align {target_name} with its reference pattern.",
                        priority=3.6,
                        priority_weight=1.9,
                        authority_level=1,
                        policy_class=GoalPolicyClass.SOFT,
                        required_bindings=["arc_transition"],
                        status=GoalStatus.ACTIVE,
                        satisfaction_score=0.25,
                        metadata={
                            "domain": "arc",
                            "kind": "align_target_orientation",
                            "targets": [target_name] + (known_mechanic_labels or planned_mechanic_targets),
                            "positions": known_mechanic_positions or planned_mechanic_positions or marker_positions,
                            "marker_positions": planned_marker_positions or marker_positions,
                            "mechanic_positions": known_mechanic_positions or planned_mechanic_positions or marker_positions,
                            "mechanic_targets": known_mechanic_labels or planned_mechanic_targets,
                            "mechanic_features": known_mechanic_features or planned_mechanic_features,
                            "target_position": target_position,
                            "reference_position": reference_position,
                            "match_key": match_key,
                            "aligned": False,
                            "plan_locked": True,
                        },
                    )
                )
            else:
                dynamic_goals.append(
                    Goal(
                        goal_id=f"goal:arc:claim:{match_key}",
                        text=f"Move onto aligned target {target_name} to complete the pattern.",
                        priority=3.9,
                        priority_weight=2.0,
                        authority_level=1,
                        policy_class=GoalPolicyClass.SOFT,
                        required_bindings=["arc_transition"],
                        status=GoalStatus.SATISFIED if report.win else GoalStatus.ACTIVE,
                        satisfaction_score=1.0 if report.win else 0.65,
                        metadata={
                            "domain": "arc",
                            "kind": "claim_aligned_target",
                            "targets": [target_name],
                            "positions": [target_position] if target_position is not None else [],
                            "marker_positions": planned_marker_positions or marker_positions,
                            "target_position": target_position,
                            "reference_position": reference_position,
                            "match_key": match_key,
                            "aligned": True,
                            "plan_locked": True,
                        },
                    )
                )

        references = [
            entity
            for entity in report.next_state.get("reference_entities", [])
            if isinstance(entity, dict)
        ]
        if references and not dynamic_goals:
            by_name: dict[str, list[Any]] = defaultdict(list)
            for entity in references:
                name = str(entity.get("name", entity.get("type", ""))).lower()
                if not name:
                    continue
                by_name[name].append(entity.get("position"))
            for name, positions in sorted(by_name.items()):
                dynamic_goals.append(
                    Goal(
                        goal_id=f"goal:arc:match:{name}",
                        text=f"Match or reach reference object {name}.",
                        priority=2.5,
                        priority_weight=1.5,
                        authority_level=1,
                        policy_class=GoalPolicyClass.SOFT,
                        required_bindings=["arc_transition"],
                        status=GoalStatus.SATISFIED if report.win else GoalStatus.ACTIVE,
                        satisfaction_score=1.0 if report.win else 0.35,
                        metadata={
                            "domain": "arc",
                            "kind": "match_reference_object",
                            "targets": [name],
                            "positions": [position for position in positions if position is not None],
                        },
                    )
                )
        else:
            helpful_tiles = [tile for tile, samples in self.tile_effects.items() if self._effect_score(samples) > 0]
            dynamic_goals.append(
                Goal(
                    goal_id="goal:arc:explore",
                    text="Explore ARC mechanics and discover useful transitions.",
                    priority=1.5,
                    priority_weight=1.2,
                    authority_level=1,
                    policy_class=GoalPolicyClass.SOFT,
                    required_bindings=["arc_transition"],
                    status=GoalStatus.ACTIVE,
                    satisfaction_score=0.2 if helpful_tiles else 0.0,
                    metadata={
                        "domain": "arc",
                        "kind": "seek_positive_transition",
                        "targets": helpful_tiles[:3],
                    },
                )
            )

        if report.terminal and not report.win:
            dynamic_goals.append(
                Goal(
                    goal_id="goal:arc:recover",
                    text="Recover from failed or blocked trajectories.",
                    priority=2.2,
                    priority_weight=1.4,
                    authority_level=1,
                    policy_class=GoalPolicyClass.SOFT,
                    required_bindings=["arc_transition"],
                    status=GoalStatus.ACTIVE,
                    satisfaction_score=0.0,
                    metadata={"domain": "arc", "kind": "recover_resources"},
                )
            )

        self.state.goals = preserved_goals + dynamic_goals

    def _select_active_goal(self) -> dict[str, Any]:
        arc_goals = [
            goal
            for goal in self.state.goal_objects()
            if getattr(goal, "metadata", {}).get("domain") == "arc"
        ]
        if not arc_goals:
            return {"kind": "explore_mechanics"}

        weights = compute_goal_weights(arc_goals)
        ranked = sorted(
            zip(arc_goals, weights),
            key=lambda item: (item[1], float(item[0].satisfaction_score), item[0].goal_id),
            reverse=True,
        )
        goal, weight = ranked[0]
        return {
            "goal_id": goal.goal_id,
            "kind": goal.metadata.get("kind", "explore_mechanics"),
            "targets": list(goal.metadata.get("targets", [])),
            "positions": list(goal.metadata.get("positions", [])),
            "marker_positions": list(goal.metadata.get("marker_positions", [])),
            "mechanic_positions": list(goal.metadata.get("mechanic_positions", [])),
            "mechanic_targets": list(goal.metadata.get("mechanic_targets", [])),
            "mechanic_features": list(goal.metadata.get("mechanic_features", [])),
            "target_position": goal.metadata.get("target_position"),
            "reference_position": goal.metadata.get("reference_position"),
            "match_key": goal.metadata.get("match_key"),
            "aligned": goal.metadata.get("aligned"),
            "weight": round(float(weight), 3),
            "satisfaction": round(float(goal.satisfaction_score), 3),
            "text": goal.text,
        }

    def _infer_rules(self) -> list[dict[str, Any]]:
        learned_rules: list[dict[str, Any]] = []
        for tile, samples in sorted(self.tile_effects.items()):
            total = len(samples)
            if total == 0:
                continue

            avg_effects: dict[str, float] = defaultdict(float)
            helpful = 0
            harmful = 0
            state_changes = 0
            for sample in samples:
                for resource, delta in sample["resource_deltas"].items():
                    avg_effects[resource] += delta
                if sample["terminal"] and not sample["win"]:
                    harmful += 1
                if sample.get("target_state_changed"):
                    state_changes += 1
                if sample["win"] or sum(delta for delta in sample["resource_deltas"].values() if delta > 0) > 0:
                    helpful += 1
                elif sample.get("target_state_changed"):
                    helpful += 1

            normalized_effects = {
                key: round(value / total, 3)
                for key, value in avg_effects.items()
                if value != 0
            }
            if state_changes:
                normalized_effects["target_state_change"] = round(state_changes / total, 3)
            confidence = round(max(helpful, harmful, 1) / total, 3)
            score = self._effect_score(samples)
            learned_rules.append(
                {
                    "trigger": tile,
                    "effects": normalized_effects,
                    "mechanic_type": _infer_mechanic_type(tile, normalized_effects),
                    "classification": "helpful" if score >= 0 else "harmful",
                    "weight": 2.0 if score >= 0 else -2.0,
                    "confidence": confidence,
                    "source": "policy_bridge",
                }
            )
        for label, stats in sorted(self.mechanic_label_effects.items()):
            total = stats["state_change"] + stats["no_effect"]
            if total <= 0:
                continue
            learned_rules.append(
                {
                    "trigger": label,
                    "effects": {"target_state_change": round(stats["state_change"] / total, 3)},
                    "mechanic_type": "state_change",
                    "classification": "helpful" if stats["state_change"] >= stats["no_effect"] else "unknown",
                    "weight": round(stats["state_change"] - stats["no_effect"], 3),
                    "confidence": round(max(stats["state_change"], stats["no_effect"]) / total, 3),
                    "source": "policy_bridge",
                }
            )
        for feature, stats in sorted(self.mechanic_feature_effects.items()):
            total = stats["state_change"] + stats["no_effect"]
            if total <= 0:
                continue
            learned_rules.append(
                {
                    "trigger": feature,
                    "effects": {"target_state_change": round(stats["state_change"] / total, 3)},
                    "mechanic_type": "state_change",
                    "classification": "helpful" if stats["state_change"] >= stats["no_effect"] else "unknown",
                    "weight": round(stats["state_change"] - stats["no_effect"], 3),
                    "confidence": round(max(stats["state_change"], stats["no_effect"]) / total, 3),
                    "source": "policy_bridge",
                }
            )
        return learned_rules

    def _infer_targets(
        self,
        report: TransitionReport,
        active_goal: dict[str, Any],
        learned_rules: list[dict[str, Any]],
    ) -> tuple[list[str], list[str]]:
        prioritized: list[str] = []
        avoid: list[str] = []

        prioritized.extend(str(target).lower() for target in active_goal.get("targets", []))
        if active_goal.get("kind") == "align_target_orientation":
            avoid.extend(
                str(value).lower()
                for value in active_goal.get("targets", [])
                if str(value).lower() not in {
                    "marker_cross",
                    *{str(target).lower() for target in active_goal.get("mechanic_targets", [])},
                }
            )
            prioritized.extend(str(target).lower() for target in active_goal.get("mechanic_targets", []))

        for rule in learned_rules:
            trigger = str(rule.get("trigger", "")).lower()
            if not trigger:
                continue
            if rule.get("classification") == "helpful":
                prioritized.append(trigger)
            else:
                avoid.append(trigger)

        current_entities = [
            str(entity.get("name", entity.get("type", ""))).lower()
            for entity in report.next_state.get("entities", [])
            if isinstance(entity, dict)
        ]
        for name in current_entities:
            if name in prioritized:
                prioritized.append(name)

        return _dedupe(prioritized), _dedupe(avoid)

    def _choose_mode(
        self,
        report: TransitionReport,
        active_goal: dict[str, Any],
        prioritized_targets: list[str],
        avoid_targets: list[str],
    ) -> str:
        resources = report.next_state.get("resources", {})
        low_resource_keys = {"health", "energy", "fuel", "ammo", "resource", "stamina", "mana"}
        low_resources = any(
            isinstance(value, (int, float))
            and value <= 1
            and any(token in str(key).lower() for token in low_resource_keys)
            for key, value in resources.items()
        )
        stall_streak = float(self.state.telemetry_counters.get("stall_streak", 0.0))
        if report.terminal and not report.win:
            return "RECOVER"
        if active_goal.get("kind") in {"align_target_orientation", "claim_aligned_target", "match_reference_object"}:
            return "EXECUTE"
        if low_resources and avoid_targets:
            return "RECOVER"
        if stall_streak >= 4 and not prioritized_targets:
            return "PROBE"
        if prioritized_targets:
            return "EXECUTE"
        return "PROBE"

    def _confidence(
        self,
        active_goal: dict[str, Any],
        learned_rules: list[dict[str, Any]],
        avoid_targets: list[str],
    ) -> dict[str, float]:
        arc_goals = [
            goal
            for goal in self.state.goal_objects()
            if getattr(goal, "metadata", {}).get("domain") == "arc"
        ]
        weights = compute_goal_weights(arc_goals) if arc_goals else []
        rule_confidences = [float(rule.get("confidence", 0.0)) for rule in learned_rules]
        goal_conf = 0.0
        if weights:
            total = sum(weights)
            active_weight = float(active_goal.get("weight", 0.0))
            goal_conf = round(active_weight / total, 3) if total > 0 else 0.0
        return {
            "rules": round(sum(rule_confidences) / len(rule_confidences), 3) if rule_confidences else 0.0,
            "goals": goal_conf,
            "risk": round(min(1.0, len(avoid_targets) * 0.25), 3),
        }

    @staticmethod
    def _effect_score(samples: list[dict[str, Any]]) -> float:
        score = 0.0
        for sample in samples:
            score += sum(sample["resource_deltas"].values())
            if sample["win"]:
                score += 3.0
            if sample["terminal"] and not sample["win"]:
                score -= 3.0
        return score

    def _edge_stats(
        self,
        player_pos: tuple[int, int] | None,
        target_point: tuple[int, int] | None,
        action: str,
    ) -> dict[str, float]:
        if player_pos is None or target_point is None:
            return {"success": 0.0, "stall": 0.0}
        return self.edge_outcomes.get((player_pos, target_point, action), {"success": 0.0, "stall": 0.0})


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _infer_mechanic_type(tile: str, effects: dict[str, float]) -> str:
    if "orientation" in tile or "cross" in tile or effects.get("target_state_change", 0.0) > 0:
        return "state_change"
    if any(value > 0 for value in effects.values()):
        return "resource_gain"
    if any(value < 0 for value in effects.values()):
        return "resource_loss"
    return "unknown"


def _meaningful_resource_change(resource_deltas: dict[str, float]) -> bool:
    if not resource_deltas:
        return False
    interesting_tokens = {"health", "energy", "fuel", "ammo", "stamina", "mana", "life", "resource"}
    for key, delta in resource_deltas.items():
        if not isinstance(delta, (int, float)) or delta == 0:
            continue
        lowered = str(key).lower()
        if any(token in lowered for token in interesting_tokens):
            return True
    return False


def _predict_point(state: ArcState, action: str) -> tuple[int, int] | None:
    return _predict_from_point(state.player_pos, action)


def _entity_at(state: ArcState, point: tuple[int, int] | None) -> str | None:
    if point is None:
        return None
    for entity in state.entities:
        position = entity.get("position")
        if position == point or position == list(point):
            return str(entity.get("name", entity.get("type", "entity")))
    return None


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _nearest_manhattan(point: tuple[int, int] | None, goals: set[tuple[int, int]] | list[tuple[int, int]]) -> int | None:
    if point is None:
        return None
    distances = [_manhattan(point, goal) for goal in goals]
    return min(distances) if distances else None


def _oscillation_axis(actions: list[str]) -> str | None:
    if len(actions) < 4:
        return None
    window = actions[-4:]
    if window in (
        ["move_up", "move_down", "move_up", "move_down"],
        ["move_down", "move_up", "move_down", "move_up"],
    ):
        return "vertical"
    if window in (
        ["move_left", "move_right", "move_left", "move_right"],
        ["move_right", "move_left", "move_right", "move_left"],
    ):
        return "horizontal"
    return None


def _walkable_points(state: ArcState) -> set[tuple[int, int]]:
    points = set(state.visible_tiles)
    points -= _resource_points(state.resources.get("wall_points"))
    playfield_bounds = state.resources.get("playfield_bounds")
    if isinstance(playfield_bounds, list) and len(playfield_bounds) == 4:
        left, top, right, bottom = playfield_bounds
        points = {
            point
            for point in points
            if left <= point[0] <= right and top <= point[1] <= bottom
        }
    return points


def _state_dict_walkable_points(state_dict: dict[str, Any]) -> set[tuple[int, int]]:
    visible_tiles = state_dict.get("visible_tiles", {})
    if not isinstance(visible_tiles, dict):
        return set()
    points = {
        _normalize_point(point)
        for point in visible_tiles
    }
    points = {point for point in points if point is not None}
    resources = state_dict.get("resources", {})
    if isinstance(resources, dict):
        points -= _resource_points(resources.get("wall_points"))
        playfield_bounds = resources.get("playfield_bounds")
        if isinstance(playfield_bounds, list) and len(playfield_bounds) == 4:
            left, top, right, bottom = playfield_bounds
            points = {
                point
                for point in points
                if left <= point[0] <= right and top <= point[1] <= bottom
            }
    return points


def _walkable_neighbor_count(state: ArcState, point: tuple[int, int] | None) -> int:
    if point is None:
        return 0
    walkable = _walkable_points(state)
    return sum(1 for neighbor in _neighbors(point) if neighbor in walkable)


def _unvisited_walkable_neighbor_count(
    state: ArcState,
    point: tuple[int, int] | None,
    memory: RuntimeMemory,
) -> int:
    if point is None:
        return 0
    walkable = _walkable_points(state)
    return sum(
        1
        for neighbor in _neighbors(point)
        if neighbor in walkable and neighbor not in memory.visited_positions
    )


def _explored_ratio(state: ArcState, memory: RuntimeMemory) -> float:
    walkable = _walkable_points(state)
    if not walkable:
        return 1.0
    explored = sum(1 for point in walkable if point in memory.visited_positions)
    return explored / len(walkable)


def _select_align_exploration_target(
    *,
    state: ArcState,
    memory: RuntimeMemory,
    mechanic_positions: set[tuple[int, int]],
    mechanic_point_effects: dict[tuple[int, int], dict[str, float]],
) -> tuple[int, int] | None:
    player_pos = state.player_pos
    if player_pos is None:
        return None

    untested_mechanics = [
        point
        for point in sorted(mechanic_positions)
        if sum(mechanic_point_effects.get(point, {"state_change": 0.0, "no_effect": 0.0}).values()) == 0
    ]
    reachable_mechanics = [
        (point, _path_distance(state, player_pos, point))
        for point in untested_mechanics
    ]
    reachable_mechanics = [(point, distance) for point, distance in reachable_mechanics if distance is not None]
    if reachable_mechanics:
        reachable_mechanics.sort(key=lambda item: (item[1], item[0][1], item[0][0]))
        return reachable_mechanics[0][0]

    edge_points = _resource_points(state.resources.get("walkable_edge_points"))
    junction_points = _resource_points(state.resources.get("junction_points"))
    landmark_points = (edge_points | junction_points) - memory.visited_positions
    frontier_points = [
        point
        for point in (_walkable_points(state) | landmark_points)
        if point not in memory.visited_positions and _unvisited_walkable_neighbor_count(state, point, memory) > 0
    ]
    reachable_frontiers = [
        (point, _path_distance(state, player_pos, point))
        for point in frontier_points
    ]
    reachable_frontiers = [(point, distance) for point, distance in reachable_frontiers if distance is not None]
    if not reachable_frontiers:
        return None

    # Prefer deeper frontier points on large maps so the agent keeps pushing outward.
    reachable_frontiers.sort(
        key=lambda item: (
            -item[1],
            -_unvisited_walkable_neighbor_count(state, item[0], memory),
            item[0][1],
            item[0][0],
        )
    )
    return reachable_frontiers[0][0]


def _path_distance(
    state: ArcState,
    start: tuple[int, int] | None,
    goal: tuple[int, int] | None,
) -> int | None:
    if start is None or goal is None:
        return None
    walkable = _walkable_points(state)
    if start not in walkable or goal not in walkable:
        return None
    if start == goal:
        return 0

    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        point, distance = queue.popleft()
        for neighbor in _neighbors(point):
            if neighbor in seen or neighbor not in walkable:
                continue
            if neighbor == goal:
                return distance + 1
            seen.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def _shortest_path(
    state: ArcState,
    start: tuple[int, int] | None,
    goal: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    if start is None or goal is None:
        return []
    walkable = _walkable_points(state)
    if start not in walkable or goal not in walkable:
        return []
    if start == goal:
        return []

    queue: deque[tuple[int, int]] = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while queue:
        point = queue.popleft()
        for neighbor in _neighbors(point):
            if neighbor in parents or neighbor not in walkable:
                continue
            parents[neighbor] = point
            if neighbor == goal:
                path: list[tuple[int, int]] = [goal]
                cursor = point
                while cursor is not None and cursor != start:
                    path.append(cursor)
                    cursor = parents.get(cursor)
                path.reverse()
                return path
            queue.append(neighbor)
    return []


def _shortest_path_from_points(
    state_dict: dict[str, Any],
    start: tuple[int, int] | None,
    goal: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    if start is None or goal is None:
        return []
    walkable = _state_dict_walkable_points(state_dict)
    if start not in walkable or goal not in walkable:
        return []
    if start == goal:
        return []

    queue: deque[tuple[int, int]] = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while queue:
        point = queue.popleft()
        for neighbor in _neighbors(point):
            if neighbor in parents or neighbor not in walkable:
                continue
            parents[neighbor] = point
            if neighbor == goal:
                path: list[tuple[int, int]] = [goal]
                cursor = point
                while cursor is not None and cursor != start:
                    path.append(cursor)
                    cursor = parents.get(cursor)
                path.reverse()
                return path
            queue.append(neighbor)
    return []


def _nearest_path_distance(
    state: ArcState,
    start: tuple[int, int] | None,
    goals: set[tuple[int, int]] | list[tuple[int, int]],
) -> int | None:
    distances = [
        _path_distance(state, start, goal)
        for goal in goals
    ]
    distances = [distance for distance in distances if distance is not None]
    return min(distances) if distances else None


def _predict_from_point(point: tuple[int, int] | None, action: str) -> tuple[int, int] | None:
    if point is None:
        return None
    deltas = {
        "move_up": (0, -1),
        "move_down": (0, 1),
        "move_left": (-1, 0),
        "move_right": (1, 0),
    }
    dx, dy = deltas.get(action, (0, 0))
    return point[0] + dx, point[1] + dy


def _neighbors(point: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _normalize_point(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _entity_matches_match_key(entity_name: str | None, match_key: Any) -> bool:
    if entity_name is None or match_key is None:
        return False
    name = str(entity_name).lower()
    required_tokens = [token for token in str(match_key).lower().split("|") if token]
    if not required_tokens:
        return False
    return all(token in name for token in required_tokens)


def _preferred_axis_action(
    current: tuple[int, int] | None,
    target: tuple[int, int] | None,
) -> str | None:
    if current is None or target is None or current == target:
        return None
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    if abs(dy) >= abs(dx) and dy != 0:
        return "move_up" if dy < 0 else "move_down"
    if dx != 0:
        return "move_left" if dx < 0 else "move_right"
    return None


def _resource_points(value: Any) -> set[tuple[int, int]]:
    if not isinstance(value, list):
        return set()
    points: set[tuple[int, int]] = set()
    for item in value:
        point = _normalize_point(item)
        if point is not None:
            points.add(point)
    return points


def _active_match_from_state(state_dict: dict[str, Any]) -> dict[str, Any] | None:
    resources = state_dict.get("resources", {})
    active_match = resources.get("active_match_target")
    return active_match if isinstance(active_match, dict) else None


def _target_state_changed(report: TransitionReport) -> bool:
    prev_match = _active_match_from_state(report.prev_state)
    next_match = _active_match_from_state(report.next_state)
    if prev_match is None or next_match is None:
        return False
    prev_sig = (
        prev_match.get("match_key"),
        prev_match.get("target_orientation"),
        prev_match.get("reference_orientation"),
        prev_match.get("aligned"),
    )
    next_sig = (
        next_match.get("match_key"),
        next_match.get("target_orientation"),
        next_match.get("reference_orientation"),
        next_match.get("aligned"),
    )
    return prev_sig != next_sig


def _interaction_points(report: TransitionReport) -> set[tuple[int, int]]:
    resources = report.next_state.get("resources", {})
    mechanic_points = _resource_points(resources.get("mechanic_candidate_positions"))
    points: set[tuple[int, int]] = set()
    next_pos = _normalize_point(report.next_state.get("player_pos"))
    prev_pos = _normalize_point(report.prev_state.get("player_pos"))
    if next_pos is not None and next_pos in mechanic_points:
        points.add(next_pos)
    attempted = _predict_from_point(prev_pos, str(report.action))
    if attempted is not None and attempted in mechanic_points:
        points.add(attempted)
    return points


def _interaction_labels(report: TransitionReport) -> set[str]:
    labels: set[str] = set()
    next_pos = _normalize_point(report.next_state.get("player_pos"))
    if next_pos is not None:
        for entity in report.next_state.get("entities", []):
            if not isinstance(entity, dict):
                continue
            position = _normalize_point(entity.get("position"))
            if position != next_pos:
                continue
            label = str(entity.get("name", entity.get("type", ""))).lower()
            if label and label not in {"player", "agent", "avatar", "self"}:
                labels.add(label)
    touched = report.delta.touched_entity
    if touched:
        labels.add(str(touched).lower())
    crossed = report.delta.crossed_marker
    if crossed:
        labels.add(str(crossed).lower())
    return labels


def _interaction_features(report: TransitionReport) -> set[str]:
    features: set[str] = set()
    next_pos = _normalize_point(report.next_state.get("player_pos"))
    for entity in report.next_state.get("entities", []):
        if not isinstance(entity, dict):
            continue
        position = _normalize_point(entity.get("position"))
        label = str(entity.get("name", entity.get("type", ""))).lower()
        touched = str(report.delta.touched_entity or "").lower()
        if position == next_pos or (touched and label == touched):
            features.update(_mechanic_features_from_entity(entity))
    return features


def _mechanic_features_from_entity(entity: dict[str, Any] | None) -> set[str]:
    if not isinstance(entity, dict):
        return set()
    features: set[str] = set()
    shape_class = str(entity.get("shape_class", "")).lower()
    if shape_class:
        features.add(f"shape:{shape_class}")
    colors_key = str(entity.get("colors_key", "")).lower()
    if colors_key:
        features.add(f"colors:{colors_key}")
    role = str(entity.get("role", "")).lower()
    if role:
        features.add(f"role:{role}")
    entity_type = str(entity.get("type", entity.get("name", ""))).lower()
    if entity_type:
        features.add(f"type:{entity_type}")
    return features


def _entity_record_at(state: ArcState, point: tuple[int, int] | None) -> dict[str, Any] | None:
    if point is None:
        return None
    for entity in state.entities:
        position = _normalize_point(entity.get("position"))
        if position == point:
            return entity
    return None
