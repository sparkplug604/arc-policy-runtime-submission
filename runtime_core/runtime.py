from __future__ import annotations

from .actions import generate_actions
from .adapter import parse_observation
from .bridge import GuidanceBridge
from .diff import compute_delta
from .policy import score_actions
from .state import build_local_context, fresh_memory, state_hash
from .traces import TraceLogger
from .types import ArcState, GuidancePacket, RuntimeMemory, TransitionReport


class ARCRuntime:
    def __init__(
        self,
        guidance_bridge: GuidanceBridge | None = None,
        operation_mode: str = "offline",
        trace_logger: TraceLogger | None = None,
    ) -> None:
        self.guidance_bridge = guidance_bridge
        self.operation_mode = operation_mode
        self.trace_logger = trace_logger or TraceLogger()

        self.level_id = "unknown"
        self.memory: RuntimeMemory = fresh_memory()
        self.guidance = GuidancePacket(mode="PROBE")

        self.current_state: ArcState | None = None
        self.previous_state: ArcState | None = None
        self.last_action: str | None = None
        self.last_action_scores: dict[str, float] = {}
        self.last_score_source: str = "runtime_policy"
        self._step_counter = 0

    def reset(self, level_id: str) -> None:
        self.level_id = level_id
        self.memory = fresh_memory()
        self.guidance = GuidancePacket(mode="PROBE")
        self.current_state = None
        self.previous_state = None
        self.last_action = None
        self.last_action_scores = {}
        self.last_score_source = "runtime_policy"
        self._step_counter = 0
        if self.guidance_bridge is not None and hasattr(self.guidance_bridge, "reset"):
            self.guidance_bridge.reset()

    def begin_new_level(self, level_id: str | None = None) -> None:
        preserved_tile_evidence = dict(self.memory.tile_effect_evidence)
        preserved_marker_evidence = dict(self.memory.marker_effect_evidence)
        if level_id is not None:
            self.level_id = level_id
        self.memory = fresh_memory()
        self.memory.tile_effect_evidence = preserved_tile_evidence
        self.memory.marker_effect_evidence = preserved_marker_evidence
        self.guidance = GuidancePacket(mode="PROBE")
        self.current_state = None
        self.previous_state = None
        self.last_action = None
        self.last_action_scores = {}
        self.last_score_source = "runtime_policy"
        self._step_counter = 0
        if self.guidance_bridge is not None and hasattr(self.guidance_bridge, "on_level_transition"):
            self.guidance_bridge.on_level_transition()

    def begin_new_attempt(self, level_id: str | None = None) -> None:
        preserved_tile_evidence = dict(self.memory.tile_effect_evidence)
        preserved_marker_evidence = dict(self.memory.marker_effect_evidence)
        if level_id is not None:
            self.level_id = level_id
        self.memory = fresh_memory()
        self.memory.tile_effect_evidence = preserved_tile_evidence
        self.memory.marker_effect_evidence = preserved_marker_evidence
        self.guidance = GuidancePacket(mode="PROBE")
        self.current_state = None
        self.previous_state = None
        self.last_action = None
        self.last_action_scores = {}
        self.last_score_source = "runtime_policy"
        self._step_counter = 0
        if self.guidance_bridge is not None and hasattr(self.guidance_bridge, "on_attempt_reset"):
            self.guidance_bridge.on_attempt_reset()

    def step(self, observation) -> str:
        state = self._normalize_state(parse_observation(observation), is_outcome=False)
        self.current_state = state

        state_sig = state_hash(state)
        self.memory.visited_states.add(state_sig)
        if state.player_pos is not None:
            self.memory.visited_positions.add(state.player_pos)
            self.memory.recent_positions.append(state.player_pos)
            self.memory.recent_positions = self.memory.recent_positions[-10:]

        candidates = generate_actions(state)
        if self.guidance_bridge is not None and hasattr(self.guidance_bridge, "score_actions"):
            scores = self.guidance_bridge.score_actions(state, candidates, self.guidance, self.memory)
            self.last_score_source = "policy_bridge"
        else:
            scores = score_actions(state, candidates, self.guidance, self.memory)
            self.last_score_source = "runtime_policy"
        action = min(
            candidates,
            key=lambda item: (
                -scores.get(item, float("-inf")),
                self.memory.state_action_pairs.get((state_sig, item), 0),
                item,
            ),
        )

        self.previous_state = state
        self.last_action = action
        self.last_action_scores = scores
        self.memory.state_action_pairs[(state_sig, action)] = self.memory.state_action_pairs.get((state_sig, action), 0) + 1
        self.memory.recent_actions.append(action)
        self.memory.recent_actions = self.memory.recent_actions[-12:]
        return action

    def observe_outcome(self, action, next_observation) -> TransitionReport:
        if self.previous_state is None:
            raise RuntimeError("observe_outcome() requires step() to be called first.")

        next_state = self._normalize_state(parse_observation(next_observation), is_outcome=True)
        delta = compute_delta(self.previous_state, next_state)
        local_context = build_local_context(
            self.previous_state,
            next_state,
            self.memory,
            action_scores=self.last_action_scores,
        )
        if isinstance(self.guidance.active_goal, dict):
            active_plan = self.guidance.active_goal.get("plan")
            if active_plan is not None:
                local_context["active_plan"] = active_plan
            if self.guidance.active_goal.get("plan_phase") is not None:
                local_context["plan_phase"] = self.guidance.active_goal.get("plan_phase")
            if self.guidance.active_goal.get("current_objective_point") is not None:
                local_context["current_objective_point"] = self.guidance.active_goal.get("current_objective_point")

        report = TransitionReport(
            level_id=next_state.level_id,
            step_id=next_state.step_id,
            action=str(action),
            prev_state=self.previous_state.to_dict(),
            next_state=next_state.to_dict(),
            delta=delta,
            local_context=local_context,
            terminal=next_state.terminal,
            win=next_state.win,
        )

        self.memory.transition_history.append(report)
        self._record_evidence(report)
        self.trace_logger.log_step(report, self.guidance)

        self.current_state = next_state
        self.memory.visited_states.add(state_hash(next_state))
        if next_state.player_pos is not None:
            self.memory.visited_positions.add(next_state.player_pos)
        self._step_counter = next_state.step_id
        return report

    def apply_guidance(self, guidance: GuidancePacket) -> None:
        self.guidance = guidance

    def _normalize_state(self, state: ArcState, is_outcome: bool) -> ArcState:
        if state.level_id == "unknown" and self.level_id:
            state.level_id = self.level_id
        elif state.level_id != "unknown":
            self.level_id = state.level_id

        if state.step_id == 0 and is_outcome:
            state.step_id = self._step_counter + 1
        elif state.step_id == 0 and self.current_state is not None and state.signature() == self.current_state.signature():
            state.step_id = self.current_state.step_id

        return state

    def _record_evidence(self, report: TransitionReport) -> None:
        delta = report.delta
        tile = (delta.entered_tile or "").lower()
        sample = {
            "resource_deltas": dict(delta.resource_deltas),
            "terminal": report.terminal,
            "win": report.win,
            "action": report.action,
        }
        if tile:
            self.memory.tile_effect_evidence.setdefault(tile, []).append(sample)

        marker = (delta.crossed_marker or "").lower()
        if marker:
            self.memory.marker_effect_evidence.setdefault(marker, []).append(sample)
