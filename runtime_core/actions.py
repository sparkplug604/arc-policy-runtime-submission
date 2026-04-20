from __future__ import annotations

from .types import ArcState


def generate_actions(state: ArcState) -> list[str]:
    del state
    return [
        "move_up",
        "move_down",
        "move_left",
        "move_right",
    ]

