from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import Any

from .types import ArcState, normalize_point

PLAYER_TOKENS = {"player", "agent", "avatar", "@", "p", "self"}
GRID_KEYS = ("grid", "board", "cells", "map", "observation")
TILE_KEYS = ("visible_tiles", "tiles")
ENTITY_KEYS = ("entities", "objects", "sprites", "agents")
REFERENCE_KEYS = ("reference_entities", "references", "legend", "goal_objects", "targets")
RESOURCE_KEYS = ("resources", "resource_bars", "hud", "status")


def parse_observation(obs: Any) -> ArcState:
    if isinstance(obs, ArcState):
        return ArcState(
            level_id=obs.level_id,
            step_id=obs.step_id,
            player_pos=obs.player_pos,
            current_tile=obs.current_tile,
            visible_tiles=dict(obs.visible_tiles),
            entities=[dict(entity) for entity in obs.entities],
            reference_entities=[dict(entity) for entity in obs.reference_entities],
            resources=dict(obs.resources),
            terminal=obs.terminal,
            win=obs.win,
        )

    mapping = _to_mapping(obs)
    inferred_entities: list[dict[str, Any]] = []
    visible_tiles = _parse_visible_tiles(mapping, inferred_entities)
    entities = _parse_entities(mapping, inferred_entities)
    player_pos = _parse_player_pos(mapping, entities, visible_tiles)
    current_tile = _pick_first(mapping, "current_tile", "tile_under_player")
    if current_tile is None and player_pos is not None:
        current_tile = visible_tiles.get(player_pos)

    return ArcState(
        level_id=str(_pick_first(mapping, "level_id", "task_id", "id", default="unknown")),
        step_id=int(_pick_first(mapping, "step_id", "step", "timestep", default=0) or 0),
        player_pos=player_pos,
        current_tile=str(current_tile) if current_tile is not None else None,
        visible_tiles=visible_tiles,
        entities=entities,
        reference_entities=_parse_reference_entities(mapping),
        resources=_parse_resources(mapping),
        terminal=bool(_pick_first(mapping, "terminal", "done", "is_terminal", default=False)),
        win=bool(_pick_first(mapping, "win", "success", "solved", default=False)),
    )


def _to_mapping(obs: Any) -> dict[str, Any]:
    if isinstance(obs, dict):
        return dict(obs)
    if is_dataclass(obs):
        return asdict(obs)
    if hasattr(obs, "__dict__"):
        return {
            key: value
            for key, value in vars(obs).items()
            if not key.startswith("_")
        }
    return {"observation": obs}


def _pick_first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _parse_visible_tiles(mapping: dict[str, Any], inferred_entities: list[dict[str, Any]]) -> dict[tuple[int, int], str]:
    tiles: dict[tuple[int, int], str] = {}
    for key in TILE_KEYS:
        candidate = mapping.get(key)
        if isinstance(candidate, dict):
            for raw_point, raw_tile in candidate.items():
                point = normalize_point(raw_point)
                if point is None and isinstance(raw_tile, dict):
                    point = normalize_point(raw_tile.get("position"))
                if point is None:
                    continue
                tiles[point] = _coerce_tile_name(raw_tile)
            if tiles:
                return tiles

    for key in GRID_KEYS:
        grid = mapping.get(key)
        if _is_grid(grid):
            for y, row in enumerate(grid):
                for x, cell in enumerate(row):
                    tile_name, maybe_entity, is_player = _classify_cell(cell)
                    tiles[(x, y)] = tile_name
                    if maybe_entity is not None:
                        maybe_entity = dict(maybe_entity)
                        maybe_entity["position"] = (x, y)
                        inferred_entities.append(maybe_entity)
                    if is_player:
                        inferred_entities.append({"name": "player", "type": "player", "position": (x, y)})
            if tiles:
                return tiles

    return tiles


def _parse_entities(mapping: dict[str, Any], inferred_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for key in ENTITY_KEYS:
        raw_entities = mapping.get(key)
        if isinstance(raw_entities, Iterable) and not isinstance(raw_entities, (str, bytes, dict)):
            for raw in raw_entities:
                entity = _normalize_entity(raw)
                if entity is not None:
                    entities.append(entity)
    if entities:
        return entities

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int] | None]] = set()
    for entity in inferred_entities:
        normalized = _normalize_entity(entity)
        if normalized is None:
            continue
        key = (
            normalized.get("name", normalized.get("type", "unknown")),
            tuple(normalized["position"]) if isinstance(normalized.get("position"), list) else normalized.get("position"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _parse_player_pos(
    mapping: dict[str, Any],
    entities: list[dict[str, Any]],
    visible_tiles: dict[tuple[int, int], str],
) -> tuple[int, int] | None:
    explicit = _pick_first(mapping, "player_pos", "agent_pos", "avatar_pos")
    point = normalize_point(explicit)
    if point is not None:
        return point

    raw_player = mapping.get("player")
    if raw_player is not None:
        point = normalize_point(raw_player)
        if point is not None:
            return point
        if isinstance(raw_player, dict):
            point = normalize_point(raw_player.get("position"))
            if point is not None:
                return point

    for entity in entities:
        label = str(entity.get("name", entity.get("type", ""))).lower()
        if entity.get("is_player") or label in PLAYER_TOKENS:
            point = normalize_point(entity.get("position"))
            if point is not None:
                return point

    for point, tile in visible_tiles.items():
        if str(tile).lower() in PLAYER_TOKENS:
            return point

    return None


def _parse_reference_entities(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for key in REFERENCE_KEYS:
        raw = mapping.get(key)
        if isinstance(raw, dict):
            for ref_key, ref_value in raw.items():
                point = normalize_point(ref_value)
                if point is not None:
                    entity = {"name": str(ref_key), "type": str(ref_key), "position": point}
                else:
                    entity = _normalize_entity(ref_value)
                    if entity is None:
                        entity = {"name": str(ref_key), "value": ref_value}
                    elif "name" not in entity or entity["name"] in {str(ref_value), "entity"}:
                        entity["name"] = str(ref_key)
                        entity["type"] = str(entity.get("type", ref_key))
                references.append(entity)
        elif isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
            for item in raw:
                entity = _normalize_entity(item)
                references.append(entity or {"name": str(item)})
        if references:
            return references
    return references


def find_goal_positions(state: ArcState) -> list[tuple[int, int]]:
    goals = []
    for entity in state.reference_entities:
        pos = entity.get("position")
        if pos:
            goals.append(tuple(pos))
    return goals


def _parse_resources(mapping: dict[str, Any]) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    for key in RESOURCE_KEYS:
        raw = mapping.get(key)
        if isinstance(raw, dict):
            for resource_key, resource_value in raw.items():
                coerced = _coerce_resource_value(resource_value)
                if coerced is not None:
                    resources[str(resource_key)] = coerced
        if resources:
            return resources

    for key, value in mapping.items():
        if key in {"level_id", "task_id", "step_id", "step", "player_pos", "entities"}:
            continue
        coerced = _coerce_resource_value(value)
        if coerced is not None and not isinstance(value, bool):
            resources[str(key)] = coerced
    return resources


def _coerce_resource_value(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_resource_value(item) for item in value]
    if isinstance(value, dict):
        coerced: dict[str, Any] = {}
        for key, item in value.items():
            converted = _coerce_resource_value(item)
            if converted is not None:
                coerced[str(key)] = converted
        return coerced
    return None


def _normalize_entity(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        entity = dict(raw)
        point = normalize_point(entity.get("position") or entity.get("pos"))
        if point is not None:
            entity["position"] = point
        if entity.get("is_player") is None:
            label = str(entity.get("name", entity.get("type", entity.get("kind", "")))).lower()
            if label in PLAYER_TOKENS:
                entity["is_player"] = True
        if "name" not in entity:
            entity["name"] = str(entity.get("type", entity.get("kind", "entity")))
        if "type" not in entity:
            entity["type"] = str(entity.get("name", "entity"))
        return entity
    if isinstance(raw, (tuple, list)) and len(raw) == 2 and normalize_point(raw[1]) is not None:
        return {"name": str(raw[0]), "type": str(raw[0]), "position": normalize_point(raw[1])}
    return {"name": str(raw), "type": str(raw)}


def _classify_cell(cell: Any) -> tuple[str, dict[str, Any] | None, bool]:
    if isinstance(cell, dict):
        tile_name = _coerce_tile_name(cell.get("tile", cell.get("background", cell.get("kind", "unknown"))))
        entity_name = cell.get("entity") or cell.get("object") or cell.get("name")
        role = str(cell.get("role", "")).lower()
        is_player = bool(cell.get("is_player")) or role == "player"
        if entity_name is not None:
            entity = {"name": str(entity_name), "type": str(cell.get("type", entity_name))}
        elif is_player:
            entity = {"name": "player", "type": "player", "is_player": True}
        else:
            entity = None
        return tile_name, entity, is_player

    if isinstance(cell, str):
        lowered = cell.lower()
        if lowered in PLAYER_TOKENS:
            return "empty", {"name": "player", "type": "player", "is_player": True}, True
        return cell, None, False

    if isinstance(cell, (int, float, bool)):
        return str(cell), None, False

    return "unknown", None, False


def _coerce_tile_name(raw_tile: Any) -> str:
    if isinstance(raw_tile, dict):
        for key in ("tile", "name", "type", "kind", "value"):
            if key in raw_tile and raw_tile[key] is not None:
                return str(raw_tile[key])
        return "unknown"
    if raw_tile is None:
        return "unknown"
    return str(raw_tile)


def _is_grid(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict))
