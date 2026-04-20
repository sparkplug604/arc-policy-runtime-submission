from __future__ import annotations

from collections import defaultdict
from typing import Any

from .types import GuidancePacket, TransitionReport


class GuidanceBridge:
    def __init__(self) -> None:
        self.tile_effects: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.marker_effects: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def reset(self) -> None:
        self.tile_effects.clear()
        self.marker_effects.clear()

    def update(self, report: TransitionReport) -> GuidancePacket:
        delta = report.delta
        tile = (delta.entered_tile or "").lower()
        if tile:
            self.tile_effects[tile].append(
                {
                    "resource_deltas": dict(delta.resource_deltas),
                    "terminal": report.terminal,
                    "win": report.win,
                }
            )

        marker = (delta.crossed_marker or "").lower()
        if marker:
            self.marker_effects[marker].append(
                {
                    "resource_deltas": dict(delta.resource_deltas),
                    "terminal": report.terminal,
                    "win": report.win,
                }
            )

        learned_rules = self._infer_rules()
        prioritized_targets, avoid_targets = self._infer_targets(report)
        mode = self._choose_mode(report, prioritized_targets, avoid_targets)
        active_goal = self._infer_goal(report, prioritized_targets)
        confidence = self._confidence(learned_rules, prioritized_targets, avoid_targets)

        return GuidancePacket(
            mode=mode,
            active_goal=active_goal,
            prioritized_targets=prioritized_targets,
            avoid_targets=avoid_targets,
            learned_rules=learned_rules,
            confidence=confidence,
        )

    def _infer_rules(self) -> list[dict[str, Any]]:
        learned_rules: list[dict[str, Any]] = []
        for tile, samples in sorted(self.tile_effects.items()):
            total = len(samples)
            if total == 0:
                continue

            avg_effects: dict[str, float] = defaultdict(float)
            harmful = 0
            helpful = 0

            for sample in samples:
                for resource, delta in sample["resource_deltas"].items():
                    avg_effects[resource] += delta
                if sample["terminal"] and not sample["win"]:
                    harmful += 1
                if sample["win"] or sum(delta for delta in sample["resource_deltas"].values() if delta > 0) > 0:
                    helpful += 1

            normalized_effects = {
                key: round(value / total, 3)
                for key, value in avg_effects.items()
                if value != 0
            }
            mean_effect = sum(normalized_effects.values())
            if helpful == harmful == 0:
                if mean_effect > 0:
                    helpful = total
                elif mean_effect < 0:
                    harmful = total
            confidence = round(max(helpful, harmful) / total, 3)
            if not normalized_effects and helpful == harmful == 0:
                continue

            classification = "helpful" if helpful >= harmful else "harmful"
            learned_rules.append(
                {
                    "trigger": tile,
                    "effects": normalized_effects,
                    "mechanic_type": _infer_mechanic_type(tile, normalized_effects),
                    "classification": classification,
                    "weight": 2.0 if classification == "helpful" else -2.0,
                    "confidence": confidence,
                }
            )
        return learned_rules

    def _infer_targets(self, report: TransitionReport) -> tuple[list[str], list[str]]:
        prioritized: list[str] = []
        avoid: list[str] = []

        for tile, samples in sorted(self.tile_effects.items()):
            score = 0.0
            for sample in samples:
                score += sum(sample["resource_deltas"].values())
                if sample["win"]:
                    score += 3.0
                if sample["terminal"] and not sample["win"]:
                    score -= 3.0
            if score > 0:
                prioritized.append(tile)
            if score < 0:
                avoid.append(tile)

        reference_names = [
            str(entity.get("name", entity.get("type", ""))).lower()
            for entity in report.next_state.get("reference_entities", [])
            if isinstance(entity, dict)
        ]
        if reference_names:
            prioritized.extend(name for name in reference_names if name)

        return _dedupe(prioritized), _dedupe(avoid)

    def _infer_goal(self, report: TransitionReport, prioritized_targets: list[str]) -> dict[str, Any]:
        reference_names = [
            str(entity.get("name", entity.get("type", ""))).lower()
            for entity in report.next_state.get("reference_entities", [])
            if isinstance(entity, dict)
        ]
        if reference_names:
            return {"kind": "match_reference_object", "targets": _dedupe(reference_names)}
        if prioritized_targets:
            return {"kind": "seek_positive_transition", "targets": prioritized_targets[:3]}
        if report.win:
            return {"kind": "preserve_success"}
        return {"kind": "explore_mechanics"}

    def _choose_mode(
        self,
        report: TransitionReport,
        prioritized_targets: list[str],
        avoid_targets: list[str],
    ) -> str:
        resources = report.next_state.get("resources", {})
        low_resources = any(
            isinstance(value, (int, float)) and value <= 1
            for value in resources.values()
        )
        if report.terminal and not report.win:
            return "RECOVER"
        if low_resources and avoid_targets:
            return "RECOVER"
        if prioritized_targets:
            return "EXECUTE"
        return "PROBE"

    def _confidence(
        self,
        learned_rules: list[dict[str, Any]],
        prioritized_targets: list[str],
        avoid_targets: list[str],
    ) -> dict[str, float]:
        rule_confidences = [float(rule.get("confidence", 0.0)) for rule in learned_rules]
        return {
            "rules": round(sum(rule_confidences) / len(rule_confidences), 3) if rule_confidences else 0.0,
            "goals": round(min(1.0, len(prioritized_targets) * 0.25), 3),
            "risk": round(min(1.0, len(avoid_targets) * 0.25), 3),
        }


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _infer_mechanic_type(tile, effects):
    if "orientation" in tile or "cross" in tile:
        return "state_change"
    if any(v > 0 for v in effects.values()):
        return "resource_gain"
    if any(v < 0 for v in effects.values()):
        return "resource_loss"
    return "unknown"
