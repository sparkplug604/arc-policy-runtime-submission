from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False

PROJECT_ROOT = Path(__file__).resolve().parent

# Use only the bundled package layout so this folder stays self-contained.
for root in (PROJECT_ROOT,):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

from observation_adapter import describe_toolkit_observation, toolkit_obs_to_arc_state
from policy_bridge import PolicyBridge
from runtime_core import ARCRuntime

ACTION_MAP = {
    "move_up": GameAction.ACTION1,
    "move_down": GameAction.ACTION2,
    "move_left": GameAction.ACTION3,
    "move_right": GameAction.ACTION4,
}


def to_toolkit_action(action_str: str) -> GameAction:
    try:
        return ACTION_MAP[action_str]
    except KeyError as exc:
        raise ValueError(f"Unsupported runtime action: {action_str}") from exc


def run(
    game_id: str = "ls20",
    *,
    max_steps: int = 100,
    render_mode: Optional[str] = None,
    inspect_first_observation: bool = True,
    operation_mode: Optional[str] = None,
) -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    arcade_kwargs: dict[str, Any] = {}
    if operation_mode:
        arcade_kwargs["operation_mode"] = OperationMode(operation_mode.lower())

    brain = PolicyBridge(root=PROJECT_ROOT / "policy_memory_data")
    arc = arc_agi.Arcade(**arcade_kwargs)
    tags = ["agent", "policy_runtime_submission", game_id]
    card_id = arc.open_scorecard(tags=tags)
    print(f"Opened scorecard: {card_id}")
    try:
        env = arc.make(game_id, scorecard_id=card_id, render_mode=render_mode)
        if env is None:
            raise RuntimeError(f"Unable to create ARC environment for game '{game_id}'.")

        runtime = ARCRuntime(guidance_bridge=brain)
        runtime.reset(game_id)
        runtime.apply_guidance(brain.bootstrap_guidance())

        obs = env.reset()
        if obs is None:
            raise RuntimeError("Environment reset() returned no observation.")

        if inspect_first_observation:
            print("Initial observation summary:")
            print(json.dumps(describe_toolkit_observation(obs), indent=2, sort_keys=True))

        last_player_pos: Optional[tuple[int, int]] = None
        last_action: Optional[str] = None
        last_mechanic_positions: list[tuple[int, int]] = []
        last_state_dict: Optional[dict[str, Any]] = None
        levels_completed = 0

        for step_index in range(max_steps):
            if bool(getattr(obs, "full_reset", False)) and step_index > 0:
                print("Attempt reset detected via full_reset. Preserving same-level knowledge and clearing attempt-local state.")
                runtime.begin_new_attempt(game_id)
                runtime.apply_guidance(brain.bootstrap_guidance())
                last_action = None
                last_player_pos = None
                last_mechanic_positions = []
                last_state_dict = None
                continue

            arc_state_dict = toolkit_obs_to_arc_state(
                obs,
                level_id=game_id,
                step_id=step_index,
                previous_player_pos=last_player_pos,
                last_action=last_action,
                previous_mechanic_positions=last_mechanic_positions,
                previous_state=last_state_dict,
            )
            action_str = runtime.step(arc_state_dict)
            toolkit_action = to_toolkit_action(action_str)
            next_obs = env.step(toolkit_action)
            if next_obs is None:
                raise RuntimeError("Environment step() returned no observation.")

            next_state_dict = toolkit_obs_to_arc_state(
                next_obs,
                level_id=game_id,
                step_id=step_index + 1,
                previous_player_pos=arc_state_dict.get("player_pos"),
                last_action=action_str,
                previous_mechanic_positions=last_mechanic_positions,
                previous_state=arc_state_dict,
            )
            report = runtime.observe_outcome(action_str, next_state_dict)
            guidance = brain.update(report)
            runtime.apply_guidance(guidance)

            print(
                f"step={step_index} action={action_str} toolkit_action={toolkit_action.name} "
                f"scores={runtime.last_action_scores} delta={report.delta.to_dict()} "
                f"mode={runtime.guidance.mode} goal={runtime.guidance.active_goal.get('kind')} "
                f"score_source={runtime.last_score_source}"
            )

            obs = next_obs
            next_levels_completed = int(next_state_dict.get("resources", {}).get("levels_completed", levels_completed) or 0)
            if next_levels_completed > levels_completed:
                print(
                    f"Level transition detected: completed={levels_completed} -> {next_levels_completed}. "
                    "Clearing transient route memory and keeping mechanic knowledge."
                )
                runtime.begin_new_level(game_id)
                runtime.apply_guidance(brain.bootstrap_guidance())
                last_action = None
                last_player_pos = None
                last_mechanic_positions = []
                last_state_dict = None
                levels_completed = next_levels_completed
            else:
                levels_completed = next_levels_completed
                last_action = action_str
                last_player_pos = next_state_dict.get("player_pos")
                last_mechanic_positions = [
                    tuple(point)
                    for point in next_state_dict.get("resources", {}).get("mechanic_candidate_positions", [])
                    if isinstance(point, list) and len(point) == 2
                ]
                last_state_dict = next_state_dict

            if report.terminal:
                print("Episode finished")
                break
    finally:
        final_scorecard = arc.close_scorecard(card_id)
        print("Final scorecard:", final_scorecard)
        scheme = os.environ.get("SCHEME", "https")
        host = os.environ.get("HOST", "three.arcprize.org")
        port = os.environ.get("PORT", "443")
        if (scheme == "http" and str(port) == "80") or (scheme == "https" and str(port) == "443"):
            root_url = f"{scheme}://{host}"
        else:
            root_url = f"{scheme}://{host}:{port}"
        if arc.operation_mode == OperationMode.ONLINE:
            print(f"Scorecard URL: {root_url}/scorecards/{card_id}")
        print("Policy bridge snapshot:", json.dumps(brain.snapshot(), indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
