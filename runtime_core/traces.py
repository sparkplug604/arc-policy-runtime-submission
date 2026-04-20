from __future__ import annotations

import json
from pathlib import Path

from .types import GuidancePacket, TransitionReport, json_safe


class TraceLogger:
    def __init__(self, path: str | Path = "arc_runtime_trace.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_step(self, report: TransitionReport, guidance: GuidancePacket | None) -> None:
        resources = report.next_state.get("resources", {}) if isinstance(report.next_state, dict) else {}
        payload = {
            "state": report.next_state,
            "action": report.action,
            "delta": report.delta.to_dict(),
            "local_context": report.local_context,
            "adapter_debug": resources.get("debug"),
            "state_divergence": {
                "entity_resource_divergence": report.local_context.get("entity_resource_divergence"),
                "entity_match_target_snapshot": report.local_context.get("entity_match_target_snapshot"),
                "resource_match_target_snapshot": report.local_context.get("resource_match_target_snapshot"),
            },
            "goal_latch_state": {
                "completion_latch_inputs": report.local_context.get("completion_latch_inputs"),
                "completion_blockers": report.local_context.get("completion_blockers"),
                "route_failure_reason": report.local_context.get("route_failure_reason"),
                "route_path_length": report.local_context.get("route_path_length"),
                "route_diagnostics": report.local_context.get("route_diagnostics"),
            },
            "mode": guidance.mode if guidance else None,
            "goal": guidance.active_goal if guidance else None,
            "guidance": guidance.to_dict() if guidance is not None else None,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")
