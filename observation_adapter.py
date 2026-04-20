from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent

MOVE_DELTAS = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}


@dataclass
class PixelComponent:
    value: str
    points: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    touches_border: bool

    @property
    def count(self) -> int:
        return len(self.points)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def center(self) -> tuple[float, float]:
        left, top, right, bottom = self.bbox
        return ((left + right) / 2.0, (top + bottom) / 2.0)


@dataclass
class ObjectCluster:
    components: list[PixelComponent]
    bbox: tuple[int, int, int, int]
    touches_border: bool
    inside_playfield: bool
    colors: tuple[str, ...]
    count: int
    grid_position: tuple[int, int]
    pixel_center: tuple[float, float]


@dataclass
class SceneAnalysis:
    cell_size: int
    playfield_bbox: tuple[int, int, int, int] | None
    visible_tiles: dict[tuple[int, int], str]
    player_pos: tuple[int, int] | None
    entities: list[dict[str, Any]]
    reference_entities: list[dict[str, Any]]
    background_tile: str
    floor_tile: str | None
    wall_points: set[tuple[int, int]]
    marker_positions: list[tuple[int, int]]
    match_targets: list[dict[str, Any]]
    mechanic_candidate_positions: list[tuple[int, int]]
    mechanic_candidate_entities: list[dict[str, Any]]


def toolkit_obs_to_arc_state(
    obs: Any,
    level_id: str = "arc3",
    *,
    step_id: int = 0,
    previous_player_pos: Optional[tuple[int, int]] = None,
    last_action: Optional[str] = None,
    previous_mechanic_positions: Optional[list[tuple[int, int]]] = None,
    previous_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Convert an ARC toolkit observation into the ArcState-compatible dict
    expected by the ARC Runtime parser.
    """
    previous_resources = previous_state.get("resources", {}) if isinstance(previous_state, dict) else {}
    previous_player_colors = _player_colors_from_state(previous_state)
    forced_cell_size = _coerce_positive_int(previous_resources.get("cell_size"))
    forced_playfield_bbox = _bounds_to_bbox(previous_resources.get("playfield_bounds"), forced_cell_size)
    grid = _extract_grid(obs)
    scene = _analyze_scene(
        grid,
        previous_player_pos=previous_player_pos,
        last_action=last_action,
        previous_mechanic_positions=previous_mechanic_positions,
        forced_cell_size=forced_cell_size,
        forced_playfield_bbox=forced_playfield_bbox,
    )

    player_pos = _extract_player_pos(obs) or scene.player_pos
    predicted_player_pos = None
    if previous_player_pos is not None and last_action in MOVE_DELTAS:
        dx, dy = MOVE_DELTAS[last_action]
        predicted_player_pos = (previous_player_pos[0] + dx, previous_player_pos[1] + dy)

    if player_pos is None and predicted_player_pos is not None:
        if predicted_player_pos in scene.visible_tiles and predicted_player_pos not in scene.wall_points:
            # Only fall back to prediction when the player sprite was not found at all.
            player_pos = predicted_player_pos

    if player_pos is None and predicted_player_pos is not None and previous_player_colors:
        merged_player_pos = _recover_player_from_merged_entity(
            entities=scene.entities,
            previous_player_pos=previous_player_pos,
            predicted_player_pos=predicted_player_pos,
            previous_player_colors=previous_player_colors,
            cell_size=scene.cell_size,
            visible_tiles=scene.visible_tiles,
            wall_points=scene.wall_points,
        )
        if merged_player_pos is not None:
            player_pos = merged_player_pos
    elif player_pos is None and previous_player_pos is not None and previous_player_colors:
        merged_player_pos = _recover_player_from_merged_entity(
            entities=scene.entities,
            previous_player_pos=previous_player_pos,
            predicted_player_pos=None,
            previous_player_colors=previous_player_colors,
            cell_size=scene.cell_size,
            visible_tiles=scene.visible_tiles,
            wall_points=scene.wall_points,
        )
        if merged_player_pos is not None:
            player_pos = merged_player_pos

    player_pos = _stabilize_player_position(
        detected_player_pos=player_pos,
        previous_player_pos=previous_player_pos,
        predicted_player_pos=predicted_player_pos,
        visible_tiles=scene.visible_tiles,
        wall_points=scene.wall_points,
    )

    match_targets = _stabilize_match_targets(
        current_targets=scene.match_targets,
        previous_targets=previous_resources.get("match_targets"),
    )
    entities = list(scene.entities)
    if player_pos is not None and not any(entity.get("is_player") for entity in entities):
        entities.insert(
            0,
            {
                "name": "player",
                "type": "player",
                "position": player_pos,
                "is_player": True,
                "colors": list(previous_player_colors),
            },
        )

    current_tile = scene.visible_tiles.get(player_pos) if player_pos is not None else None
    resources = _extract_resources(obs)
    resources.setdefault("cell_size", forced_cell_size or scene.cell_size)
    resources["background_tile"] = scene.background_tile
    resources["floor_tile"] = scene.floor_tile
    resources["wall_points"] = sorted(scene.wall_points)
    resources["marker_positions"] = [list(point) for point in scene.marker_positions]
    resources["match_targets"] = match_targets
    resources["mechanic_candidate_positions"] = [list(point) for point in scene.mechanic_candidate_positions]
    resources["mechanic_candidate_entities"] = scene.mechanic_candidate_entities
    if match_targets:
        resources["active_match_target"] = match_targets[0]
        resources["target_object_aligned"] = bool(match_targets[0].get("aligned"))
    if scene.playfield_bbox is not None:
        left, top, right, bottom = scene.playfield_bbox
        resources["playfield_bounds"] = [
            left // scene.cell_size,
            top // scene.cell_size,
            right // scene.cell_size,
            bottom // scene.cell_size,
        ]

    return {
        "level_id": level_id,
        "step_id": step_id,
        "player_pos": player_pos,
        "current_tile": current_tile,
        "visible_tiles": scene.visible_tiles,
        "entities": entities,
        "reference_entities": scene.reference_entities,
        "resources": resources,
        "terminal": _is_terminal(obs),
        "win": _is_win(obs),
    }


def describe_toolkit_observation(obs: Any) -> dict[str, Any]:
    raw_grid = _extract_grid(obs)
    scene = _analyze_scene(raw_grid)
    return {
        "observation_type": type(obs).__name__,
        "keys": sorted(obs.keys()) if isinstance(obs, dict) else [],
        "attributes": sorted(name for name in dir(obs) if not name.startswith("_"))[:30],
        "raw_grid_height": len(raw_grid),
        "raw_grid_width": len(raw_grid[0]) if raw_grid else 0,
        "coarse_tiles": len(scene.visible_tiles),
        "cell_size": scene.cell_size,
        "background_tile": scene.background_tile,
        "playfield_bbox": list(scene.playfield_bbox) if scene.playfield_bbox is not None else None,
        "player_pos": _extract_player_pos(obs),
        "inferred_player_pos": scene.player_pos,
        "entity_count": len(scene.entities),
        "reference_count": len(scene.reference_entities),
        "wall_count": len(scene.wall_points),
        "floor_tile": scene.floor_tile,
        "marker_positions": [list(point) for point in scene.marker_positions],
        "mechanic_candidate_positions": [list(point) for point in scene.mechanic_candidate_positions],
        "active_match_target": scene.match_targets[0] if scene.match_targets else None,
        "state": str(_get_field(obs, "state")),
        "available_actions": _get_field(obs, "available_actions"),
        "terminal": _is_terminal(obs),
        "win": _is_win(obs),
    }


def _analyze_scene(
    raw_grid: list[list[Any]],
    *,
    previous_player_pos: Optional[tuple[int, int]] = None,
    last_action: Optional[str] = None,
    previous_mechanic_positions: Optional[list[tuple[int, int]]] = None,
    forced_cell_size: Optional[int] = None,
    forced_playfield_bbox: Optional[tuple[int, int, int, int]] = None,
) -> SceneAnalysis:
    if not raw_grid:
        return SceneAnalysis(
            cell_size=1,
            playfield_bbox=None,
            visible_tiles={},
            player_pos=previous_player_pos,
            entities=[],
            reference_entities=[],
            background_tile="empty",
            floor_tile=None,
            wall_points=set(),
            marker_positions=[],
            match_targets=[],
            mechanic_candidate_positions=[],
            mechanic_candidate_entities=[],
        )

    normalized_grid = [[_normalize_tile_value(value) for value in row] for row in raw_grid]
    background_tile = _most_common_tile(normalized_grid)
    cell_size = forced_cell_size or _infer_cell_size(normalized_grid, background_tile)
    visible_tiles = _downsample_grid(normalized_grid, cell_size)

    playfield = _find_playfield_component(normalized_grid, background_tile)
    playfield_bbox = forced_playfield_bbox or (playfield.bbox if playfield is not None else None)

    compact_components = [
        component
        for component in _find_color_components(normalized_grid, background_tile)
        if _is_compact_component(component, cell_size)
    ]
    clusters = _merge_components(compact_components, playfield_bbox, cell_size)
    player_cluster = _choose_player_cluster(
        clusters,
        playfield_bbox=playfield_bbox,
        previous_player_pos=previous_player_pos,
        last_action=last_action,
    )

    entities: list[dict[str, Any]] = []
    reference_entities: list[dict[str, Any]] = []

    for cluster in clusters:
        entity = _cluster_to_entity(cluster)
        if entity is None:
            continue
        if player_cluster is not None and cluster is player_cluster:
            continue
        if cluster.inside_playfield:
            entities.append(entity)
        else:
            if cluster.count < 8:
                continue
            reference_entities.append(entity)

    player_pos = player_cluster.grid_position if player_cluster is not None else None
    if player_pos is not None:
        entities.insert(
            0,
            {
                "name": "player",
                "type": "player",
                "position": player_pos,
                "is_player": True,
                "colors": list(player_cluster.colors),
            },
        )

    dynamic_points = {
        entity["position"]
        for entity in entities
        if isinstance(entity.get("position"), tuple)
    }
    marker_positions = sorted(
        entity["position"]
        for entity in entities
        if entity.get("shape_class") == "marker_cross" and isinstance(entity.get("position"), tuple)
    )
    match_targets = _annotate_match_relationships(entities, reference_entities)
    floor_tile = _infer_floor_tile(visible_tiles, playfield_bbox, cell_size, dynamic_points)
    wall_points = _infer_wall_points(
        visible_tiles,
        playfield_bbox=playfield_bbox,
        cell_size=cell_size,
        background_tile=background_tile,
        floor_tile=floor_tile,
        dynamic_points=dynamic_points,
    )
    mechanic_candidate_positions, mechanic_candidate_entities = _infer_mechanic_candidates(
        visible_tiles,
        entities=entities,
        playfield_bbox=playfield_bbox,
        cell_size=cell_size,
        floor_tile=floor_tile,
        wall_points=wall_points,
    )
    marker_positions, mechanic_candidate_positions = _stabilize_mechanic_positions(
        marker_positions=marker_positions,
        mechanic_candidate_positions=mechanic_candidate_positions,
        previous_mechanic_positions=previous_mechanic_positions or [],
        player_pos=player_pos,
        wall_points=wall_points,
        visible_tiles=visible_tiles,
    )
    mechanic_candidate_entities = _refresh_mechanic_entities(
        mechanic_candidate_entities,
        mechanic_candidate_positions,
    )

    return SceneAnalysis(
        cell_size=cell_size,
        playfield_bbox=playfield_bbox,
        visible_tiles=visible_tiles,
        player_pos=player_pos,
        entities=entities,
        reference_entities=reference_entities,
        background_tile=background_tile,
        floor_tile=floor_tile,
        wall_points=wall_points,
        marker_positions=marker_positions,
        match_targets=match_targets,
        mechanic_candidate_positions=mechanic_candidate_positions,
        mechanic_candidate_entities=mechanic_candidate_entities,
    )


def _extract_grid(obs: Any) -> list[list[Any]]:
    grid = _get_field(obs, "grid") or _get_field(obs, "board")
    if grid is not None:
        return _ensure_2d(grid)

    frame = _get_field(obs, "frame")
    if frame is None:
        return []

    frame_2d = _frame_to_2d(frame)
    return _ensure_2d(frame_2d)


def _frame_to_2d(frame: Any) -> list[list[Any]]:
    if not isinstance(frame, list) or not frame:
        return []

    if len(frame) == 1 and hasattr(frame[0], "tolist"):
        single_layer = frame[0].tolist()
        return _ensure_2d(single_layer)

    if _looks_like_channel_stack(frame):
        height = len(frame[0])
        width = len(frame[0][0]) if height else 0
        collapsed: list[list[Any]] = []
        for y in range(height):
            row: list[Any] = []
            for x in range(width):
                pixel = tuple(_scalarize(channel[y][x]) for channel in frame)
                row.append(pixel)
            collapsed.append(row)
        return collapsed

    return frame


def _looks_like_channel_stack(frame: list[Any]) -> bool:
    if not frame:
        return False
    if hasattr(frame[0], "tolist"):
        sample = frame[0].tolist()
    else:
        sample = frame[0]
    if not isinstance(sample, list):
        return False
    if not sample or not isinstance(sample[0], list):
        return False
    return isinstance(sample[0][0], (int, float, bool))


def _ensure_2d(grid: Any) -> list[list[Any]]:
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if not isinstance(grid, list):
        return []
    out: list[list[Any]] = []
    for row in grid:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if isinstance(row, list):
            out.append(row)
        else:
            out.append([row])
    return out


def _normalize_tile_value(value: Any) -> str:
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(_scalarize(value))


def _scalarize(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _most_common_tile(grid: list[list[str]]) -> str:
    counts = Counter(tile for row in grid for tile in row)
    return counts.most_common(1)[0][0] if counts else "empty"


def _infer_cell_size(grid: list[list[str]], background_tile: str) -> int:
    components = _find_color_components(grid, background_tile)
    playfield = _find_playfield_component(grid, background_tile)
    playfield_bbox = playfield.bbox if playfield is not None else None
    compact_components: list[PixelComponent] = []
    candidate_dims: list[int] = []

    for component in components:
        if component.touches_border:
            continue
        if component.count <= 1:
            continue
        if component.width > 20 or component.height > 20:
            continue
        compact_components.append(component)
        for dim in (component.width, component.height):
            if 3 <= dim <= 20:
                candidate_dims.append(dim)

    preliminary_clusters = _merge_components(compact_components, playfield_bbox, 3)
    multi_color_sizes = sorted(
        min(cluster.bbox[2] - cluster.bbox[0] + 1, cluster.bbox[3] - cluster.bbox[1] + 1)
        for cluster in preliminary_clusters
        if cluster.inside_playfield
        and not cluster.touches_border
        and len(cluster.colors) > 1
        and cluster.count >= 8
    )
    if multi_color_sizes:
        inferred = multi_color_sizes[0]
        if 2 <= inferred <= 8:
            return inferred

    if not candidate_dims:
        return 1

    best_unit = 1
    best_score = float("-inf")
    for unit in range(2, 9):
        score = 0.0
        for dim in candidate_dims:
            if dim == unit:
                score += 2.0
            if dim == unit * 2:
                score += 1.5
            if abs(dim - unit) == 1:
                score += 0.8
            if abs(dim - (unit * 2)) == 1:
                score += 0.6
            if dim % unit == 0:
                score += 0.3
        score += unit * 0.05
        if score > best_score:
            best_score = score
            best_unit = unit
    return best_unit


def _downsample_grid(grid: list[list[str]], cell_size: int) -> dict[tuple[int, int], str]:
    if cell_size <= 1:
        return {
            (x, y): tile
            for y, row in enumerate(grid)
            for x, tile in enumerate(row)
        }

    coarse_tiles: dict[tuple[int, int], str] = {}
    height = len(grid)
    width = len(grid[0]) if height else 0
    for y0 in range(0, height, cell_size):
        for x0 in range(0, width, cell_size):
            block = [
                grid[y][x]
                for y in range(y0, min(height, y0 + cell_size))
                for x in range(x0, min(width, x0 + cell_size))
            ]
            if not block:
                continue
            coarse_tiles[(x0 // cell_size, y0 // cell_size)] = Counter(block).most_common(1)[0][0]
    return coarse_tiles


def _infer_floor_tile(
    visible_tiles: dict[tuple[int, int], str],
    playfield_bbox: tuple[int, int, int, int] | None,
    cell_size: int,
    dynamic_points: set[tuple[int, int]],
) -> str | None:
    playfield_points = _playfield_points(visible_tiles, playfield_bbox, cell_size)
    if not playfield_points:
        return None

    interior_points = [point for point in playfield_points if not _is_playfield_border(point, playfield_points)]
    for points in (interior_points, playfield_points):
        counts: Counter[str] = Counter()
        for point in points:
            if point in dynamic_points:
                continue
            tile = visible_tiles.get(point)
            if tile is None:
                continue
            counts[tile] += 1
        if counts:
            return counts.most_common(1)[0][0]
    return None


def _infer_wall_points(
    visible_tiles: dict[tuple[int, int], str],
    *,
    playfield_bbox: tuple[int, int, int, int] | None,
    cell_size: int,
    background_tile: str,
    floor_tile: str | None,
    dynamic_points: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    wall_points: set[tuple[int, int]] = set()
    playfield_points = _playfield_points(visible_tiles, playfield_bbox, cell_size)
    playfield_set = set(playfield_points)

    for point in visible_tiles:
        if point not in playfield_set:
            wall_points.add(point)

    if floor_tile is None:
        return wall_points

    tile_groups: dict[str, set[tuple[int, int]]] = {}
    for point in playfield_points:
        if point in dynamic_points:
            continue
        tile = visible_tiles.get(point)
        if tile is None or tile == floor_tile:
            continue
        tile_groups.setdefault(tile, set()).add(point)

    for tile, points in tile_groups.items():
        component_sets = _coarse_components(points)
        for component in component_sets:
            border_touch = any(_is_playfield_border(point, playfield_points) for point in component)
            same_tile_neighbors = max(_same_tile_neighbor_count(point, component) for point in component)
            strong_structural = (
                tile == background_tile
                or (border_touch and len(component) >= 2)
                or len(component) >= 4
                or same_tile_neighbors >= 2
            )
            if strong_structural:
                wall_points.update(component)

    return wall_points


def _find_playfield_component(grid: list[list[str]], background_tile: str) -> PixelComponent | None:
    components = _find_mask_components(grid, background_tile)
    if not components:
        return None

    non_border = [component for component in components if not component.touches_border]
    if non_border:
        return max(non_border, key=lambda component: component.count)
    return max(components, key=lambda component: component.count)


def _find_mask_components(grid: list[list[str]], background_tile: str) -> list[PixelComponent]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    seen: set[tuple[int, int]] = set()
    components: list[PixelComponent] = []

    for y in range(height):
        for x in range(width):
            if grid[y][x] == background_tile or (x, y) in seen:
                continue
            points: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(x, y)])
            seen.add((x, y))
            touches_border = False

            while queue:
                cx, cy = queue.popleft()
                points.append((cx, cy))
                if cx in {0, width - 1} or cy in {0, height - 1}:
                    touches_border = True
                for nx, ny in _neighbors(cx, cy):
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if grid[ny][nx] == background_tile or (nx, ny) in seen:
                        continue
                    seen.add((nx, ny))
                    queue.append((nx, ny))

            components.append(
                PixelComponent(
                    value="mask",
                    points=points,
                    bbox=_points_bbox(points),
                    touches_border=touches_border,
                )
            )
    return components


def _find_color_components(grid: list[list[str]], background_tile: str) -> list[PixelComponent]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    seen: set[tuple[int, int]] = set()
    components: list[PixelComponent] = []

    for y in range(height):
        for x in range(width):
            value = grid[y][x]
            if value == background_tile or (x, y) in seen:
                continue
            points: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(x, y)])
            seen.add((x, y))
            touches_border = False

            while queue:
                cx, cy = queue.popleft()
                points.append((cx, cy))
                if cx in {0, width - 1} or cy in {0, height - 1}:
                    touches_border = True
                for nx, ny in _neighbors(cx, cy):
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if grid[ny][nx] != value or (nx, ny) in seen:
                        continue
                    seen.add((nx, ny))
                    queue.append((nx, ny))

            components.append(
                PixelComponent(
                    value=value,
                    points=points,
                    bbox=_points_bbox(points),
                    touches_border=touches_border,
                )
            )
    return components


def _is_compact_component(component: PixelComponent, cell_size: int) -> bool:
    max_dimension = max(cell_size * 2 + 2, 12)
    max_area = max(cell_size * cell_size * 3, 120)
    return (
        component.count <= max_area
        and component.width <= max_dimension
        and component.height <= max_dimension
    )


def _merge_components(
    components: list[PixelComponent],
    playfield_bbox: tuple[int, int, int, int] | None,
    cell_size: int,
) -> list[ObjectCluster]:
    clusters: list[list[PixelComponent]] = []
    assigned: set[int] = set()

    for index, component in enumerate(components):
        if index in assigned:
            continue
        stack = [index]
        group: list[PixelComponent] = []
        assigned.add(index)

        while stack:
            current_index = stack.pop()
            current = components[current_index]
            group.append(current)
            for other_index, other in enumerate(components):
                if other_index in assigned:
                    continue
                if _bboxes_close(current.bbox, other.bbox, gap=max(1, cell_size // 3)):
                    assigned.add(other_index)
                    stack.append(other_index)

        clusters.append(group)

    merged: list[ObjectCluster] = []
    for group in clusters:
        points = [point for component in group for point in component.points]
        bbox = _points_bbox(points)
        pixel_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        grid_position = (int(pixel_center[0] // cell_size), int(pixel_center[1] // cell_size))
        inside_playfield = _bbox_center_in_bbox(bbox, playfield_bbox, margin=cell_size) if playfield_bbox else True
        merged.append(
            ObjectCluster(
                components=group,
                bbox=bbox,
                touches_border=any(component.touches_border for component in group),
                inside_playfield=inside_playfield,
                colors=tuple(sorted({component.value for component in group})),
                count=sum(component.count for component in group),
                grid_position=grid_position,
                pixel_center=pixel_center,
            )
        )
    return sorted(merged, key=lambda cluster: (cluster.inside_playfield, cluster.count), reverse=True)


def _choose_player_cluster(
    clusters: list[ObjectCluster],
    *,
    playfield_bbox: tuple[int, int, int, int] | None,
    previous_player_pos: tuple[int, int] | None,
    last_action: str | None,
) -> ObjectCluster | None:
    candidates = [
        cluster
        for cluster in clusters
        if cluster.inside_playfield
        and not cluster.touches_border
        and 8 <= cluster.count <= 80
        and cluster.bbox[2] - cluster.bbox[0] + 1 >= 4
        and cluster.bbox[3] - cluster.bbox[1] + 1 >= 4
    ]
    if not candidates:
        return None

    predicted = previous_player_pos
    if previous_player_pos is not None and last_action in MOVE_DELTAS:
        dx, dy = MOVE_DELTAS[last_action]
        predicted = (previous_player_pos[0] + dx, previous_player_pos[1] + dy)

    playfield_center_y = None
    playfield_center_x = None
    if playfield_bbox is not None:
        playfield_center_x = (playfield_bbox[0] + playfield_bbox[2]) / 2.0
        playfield_center_y = (playfield_bbox[1] + playfield_bbox[3]) / 2.0

    best_cluster: ObjectCluster | None = None
    best_score = float("-inf")

    for cluster in candidates:
        left, top, right, bottom = cluster.bbox
        width = right - left + 1
        height = bottom - top + 1
        score = 0.0

        if len(cluster.colors) > 1:
            score += 3.5
        if 4 <= cluster.count <= 36:
            score += 2.5
        if width <= 8 and height <= 8:
            score += 1.5
        if playfield_center_y is not None and cluster.pixel_center[1] >= playfield_center_y:
            score += 1.0
        if playfield_center_x is not None:
            score += max(0.0, 1.5 - abs(cluster.pixel_center[0] - playfield_center_x) / 12.0)
        score += min(2.0, len(cluster.colors) * 0.5)

        if predicted is not None:
            distance = abs(cluster.grid_position[0] - predicted[0]) + abs(cluster.grid_position[1] - predicted[1])
            score += max(0.0, 6.0 - distance * 1.8)

        if score > best_score:
            best_score = score
            best_cluster = cluster

    return best_cluster


def _cluster_to_entity(cluster: ObjectCluster) -> dict[str, Any] | None:
    if cluster.count <= 1:
        return None
    pattern = _cluster_pattern(cluster, grid_size=5)
    mask_pattern = _cluster_mask_pattern(cluster, grid_size=5)
    canonical_pattern, _ = _canonicalize_pattern(pattern)
    canonical_mask, _ = _canonicalize_pattern(mask_pattern)
    layout_signature, orientation_index = _component_layout_signature(cluster)
    colors_key = "|".join(cluster.colors)
    shape_class = _classify_shape(cluster, canonical_mask)
    name = "marker_cross" if shape_class == "marker_cross" else "obj_" + "_".join(cluster.colors)
    return {
        "name": name,
        "type": name,
        "position": cluster.grid_position,
        "colors": list(cluster.colors),
        "bbox": list(cluster.bbox),
        "size": cluster.count,
        "shape_class": shape_class,
        "colors_key": colors_key,
        "visual_signature": f"{colors_key}::{layout_signature}",
        "pattern_signature": pattern,
        "mask_pattern_signature": mask_pattern,
        "canonical_signature": canonical_pattern,
        "mask_signature": canonical_mask,
        "layout_signature": layout_signature,
        "orientation_index": orientation_index,
        "orientation_label": f"rot_{orientation_index}",
    }


def _cluster_pattern(cluster: ObjectCluster, *, grid_size: int) -> str:
    left, top, right, bottom = cluster.bbox
    width = max(1, right - left + 1)
    height = max(1, bottom - top + 1)
    bins: list[list[Counter[str]]] = [
        [Counter() for _ in range(grid_size)]
        for _ in range(grid_size)
    ]

    for component in cluster.components:
        for x, y in component.points:
            bx = min(grid_size - 1, int((x - left) * grid_size / width))
            by = min(grid_size - 1, int((y - top) * grid_size / height))
            bins[by][bx][component.value] += 1

    rows: list[str] = []
    for row in bins:
        tokens: list[str] = []
        for counter in row:
            tokens.append(counter.most_common(1)[0][0] if counter else ".")
        rows.append("".join(tokens))
    return "/".join(rows)


def _cluster_mask_pattern(cluster: ObjectCluster, *, grid_size: int) -> str:
    pattern = _cluster_pattern(cluster, grid_size=grid_size)
    rows = pattern.split("/")
    return "/".join("".join("1" if token != "." else "." for token in row) for row in rows)


def _rotate_pattern(pattern: str) -> str:
    rows = pattern.split("/")
    size = len(rows)
    rotated = [
        "".join(rows[size - 1 - y][x] for y in range(size))
        for x in range(size)
    ]
    return "/".join(rotated)


def _canonicalize_pattern(pattern: str) -> tuple[str, int]:
    rotations = [pattern]
    for _ in range(3):
        rotations.append(_rotate_pattern(rotations[-1]))
    canonical = min(rotations)
    orientation_index = rotations.index(pattern)
    canonical_rotations = [canonical]
    for _ in range(3):
        canonical_rotations.append(_rotate_pattern(canonical_rotations[-1]))
    if pattern in canonical_rotations:
        orientation_index = canonical_rotations.index(pattern)
    return canonical, orientation_index


def _classify_shape(cluster: ObjectCluster, canonical_mask: str) -> str:
    if cluster.count <= 9 and "1" in cluster.colors:
        return "marker_cross"
    if len(cluster.colors) >= 2:
        return "pattern_object"
    return "object"


def _dominant_color(cluster: ObjectCluster) -> str:
    color_counts = Counter({component.value: component.count for component in cluster.components})
    return color_counts.most_common(1)[0][0]


def _component_layout_signature(cluster: ObjectCluster) -> tuple[str, int]:
    left, top, right, bottom = cluster.bbox
    width = max(1, right - left + 1)
    height = max(1, bottom - top + 1)
    dominant = _dominant_color(cluster)

    items: list[tuple[str, float, float, float]] = []
    for component in cluster.components:
        if component.value == dominant:
            continue
        cx, cy = component.center
        items.append(
            (
                component.value,
                round((cx - left) / width, 1),
                round((cy - top) / height, 1),
                round(component.count / float(width * height), 1),
            )
        )

    if not items:
        return dominant, 0

    rotations: list[tuple[tuple[str, float, float, float], ...]] = []
    current = items
    for _ in range(4):
        signature = tuple(sorted(current))
        rotations.append(signature)
        current = [
            (color, round(1.0 - y, 1), round(x, 1), area)
            for color, x, y, area in current
        ]
    canonical = min(rotations)
    orientation_index = rotations.index(canonical)
    canonical_rotations = [canonical]
    current_canonical = list(canonical)
    for _ in range(3):
        current_canonical = [
            (color, round(1.0 - y, 1), round(x, 1), area)
            for color, x, y, area in current_canonical
        ]
        canonical_rotations.append(tuple(sorted(current_canonical)))
    if tuple(sorted(items)) in canonical_rotations:
        orientation_index = canonical_rotations.index(tuple(sorted(items)))
    encoded = ";".join(f"{color}@{x:.1f},{y:.1f}:{area:.1f}" for color, x, y, area in canonical)
    return encoded, orientation_index


def _annotate_match_relationships(
    entities: list[dict[str, Any]],
    reference_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference_by_signature: dict[str, list[dict[str, Any]]] = {}
    for reference in reference_entities:
        signature = str(reference.get("colors_key") or reference.get("visual_signature", ""))
        if not signature:
            continue
        reference_by_signature.setdefault(signature, []).append(reference)

    match_targets: list[dict[str, Any]] = []
    for entity in entities:
        if entity.get("is_player"):
            continue
        if entity.get("shape_class") == "marker_cross":
            entity["role"] = "mechanic_marker"
            continue

        signature = str(entity.get("colors_key") or entity.get("visual_signature", ""))
        if not signature or signature not in reference_by_signature:
            continue

        reference = reference_by_signature[signature][0]
        match_key = signature
        entity["role"] = "match_target"
        entity["match_key"] = match_key
        reference["role"] = "reference_target"
        reference["match_key"] = match_key

        entity_orientation = int(entity.get("orientation_index", 0))
        reference_orientation = int(reference.get("orientation_index", 0))
        orientation_match = entity_orientation == reference_orientation
        entity_pattern = str(entity.get("pattern_signature", ""))
        reference_pattern = str(reference.get("pattern_signature", ""))
        entity_mask_pattern = str(entity.get("mask_pattern_signature", ""))
        reference_mask_pattern = str(reference.get("mask_pattern_signature", ""))
        pattern_match = bool(entity_pattern) and entity_pattern == reference_pattern
        mask_pattern_match = bool(entity_mask_pattern) and entity_mask_pattern == reference_mask_pattern
        aligned = orientation_match and (pattern_match or mask_pattern_match)
        entity["orientation_match"] = orientation_match
        reference["orientation_match"] = orientation_match
        entity["pattern_match"] = pattern_match or mask_pattern_match
        reference["pattern_match"] = pattern_match or mask_pattern_match

        match_targets.append(
            {
                "match_key": match_key,
                "target_name": entity.get("name"),
                "reference_name": reference.get("name"),
                "target_position": list(entity["position"]) if isinstance(entity.get("position"), tuple) else entity.get("position"),
                "reference_position": list(reference["position"]) if isinstance(reference.get("position"), tuple) else reference.get("position"),
                "target_orientation": entity_orientation,
                "reference_orientation": reference_orientation,
                "orientation_match": orientation_match,
                "target_pattern_signature": entity_pattern,
                "reference_pattern_signature": reference_pattern,
                "target_mask_pattern_signature": entity_mask_pattern,
                "reference_mask_pattern_signature": reference_mask_pattern,
                "pattern_match": pattern_match,
                "mask_pattern_match": mask_pattern_match,
                "aligned": aligned,
            }
        )

    return sorted(
        match_targets,
        key=lambda item: (
            bool(item.get("aligned")),
            str(item.get("target_name", "")),
        ),
    )


def _infer_mechanic_candidates(
    visible_tiles: dict[tuple[int, int], str],
    *,
    entities: list[dict[str, Any]],
    playfield_bbox: tuple[int, int, int, int] | None,
    cell_size: int,
    floor_tile: str | None,
    wall_points: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    playfield_points = _playfield_points(visible_tiles, playfield_bbox, cell_size)
    candidate_points: set[tuple[int, int]] = set()
    candidate_entities: list[dict[str, Any]] = []
    occupied_points = {
        point
        for entity in entities
        for point in [_normalize_point(entity.get("position"))]
        if point is not None
    }

    for entity in entities:
        point = entity.get("position")
        if not isinstance(point, tuple):
            continue
        if point not in playfield_points or point in wall_points:
            continue
        if entity.get("is_player") or entity.get("role") == "match_target":
            continue
        candidate_points.add(point)
        candidate_entities.append(
            {
                "name": entity.get("name"),
                "type": entity.get("type"),
                "position": list(point),
                "role": entity.get("role"),
                "shape_class": entity.get("shape_class"),
            }
        )

    if floor_tile is not None:
        for point in playfield_points:
            if point in wall_points:
                continue
            if point in occupied_points:
                continue
            tile = visible_tiles.get(point)
            if tile is None or tile == floor_tile:
                continue
            candidate_points.add(point)

    unique_entities: list[dict[str, Any]] = []
    seen_entity_keys: set[tuple[str, tuple[int, int] | None]] = set()
    for entity in candidate_entities:
        point = _normalize_point(entity.get("position"))
        key = (str(entity.get("name", entity.get("type", "entity"))), point)
        if key in seen_entity_keys:
            continue
        seen_entity_keys.add(key)
        unique_entities.append(entity)

    return sorted(candidate_points), unique_entities


def _stabilize_mechanic_positions(
    *,
    marker_positions: list[tuple[int, int]],
    mechanic_candidate_positions: list[tuple[int, int]],
    previous_mechanic_positions: list[tuple[int, int]],
    player_pos: tuple[int, int] | None,
    wall_points: set[tuple[int, int]],
    visible_tiles: dict[tuple[int, int], str],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    previous = [tuple(point) for point in previous_mechanic_positions if tuple(point) in visible_tiles and tuple(point) not in wall_points]
    if marker_positions:
        stable = sorted(set(marker_positions))
        return stable, sorted(set(mechanic_candidate_positions) | set(stable))

    if previous:
        # When the player stands on the mechanic, the white marker can disappear under the sprite.
        if player_pos is not None and player_pos in previous:
            return previous, previous

        inferred = {tuple(point) for point in mechanic_candidate_positions}
        if not inferred or all(_manhattan(point, prev) <= 1 for point in inferred for prev in previous):
            return previous, previous

    return marker_positions, mechanic_candidate_positions


def _refresh_mechanic_entities(
    mechanic_candidate_entities: list[dict[str, Any]],
    mechanic_candidate_positions: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    by_point = {
        _normalize_point(entity.get("position")): dict(entity)
        for entity in mechanic_candidate_entities
        if _normalize_point(entity.get("position")) is not None
    }
    refreshed: list[dict[str, Any]] = []
    for point in mechanic_candidate_positions:
        entity = by_point.get(point)
        if entity is None:
            entity = {
                "name": "mechanic_candidate",
                "type": "mechanic_candidate",
                "position": list(point),
                "role": "mechanic_candidate",
                "shape_class": "mechanic_candidate",
            }
        refreshed.append(entity)
    return refreshed


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _player_colors_from_state(state: Optional[dict[str, Any]]) -> list[str]:
    if not isinstance(state, dict):
        return []
    for entity in state.get("entities", []):
        if not isinstance(entity, dict):
            continue
        if entity.get("is_player") or str(entity.get("name")) == "player":
            colors = entity.get("colors")
            if isinstance(colors, list):
                return [str(color) for color in colors]
    return []


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def _bounds_to_bbox(value: Any, cell_size: Optional[int]) -> Optional[tuple[int, int, int, int]]:
    if cell_size is None or cell_size <= 0:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (int(part) for part in value)
    except (TypeError, ValueError):
        return None
    return (x0 * cell_size, y0 * cell_size, x1 * cell_size, y1 * cell_size)


def _stabilize_player_position(
    *,
    detected_player_pos: Optional[tuple[int, int]],
    previous_player_pos: Optional[tuple[int, int]],
    predicted_player_pos: Optional[tuple[int, int]],
    visible_tiles: dict[tuple[int, int], str],
    wall_points: set[tuple[int, int]],
) -> Optional[tuple[int, int]]:
    if previous_player_pos is None:
        return detected_player_pos or predicted_player_pos

    if detected_player_pos is not None and _manhattan(detected_player_pos, previous_player_pos) <= 1:
        return detected_player_pos

    if predicted_player_pos is not None and _is_walkable_point(predicted_player_pos, visible_tiles, wall_points):
        return predicted_player_pos

    if _is_walkable_point(previous_player_pos, visible_tiles, wall_points):
        return previous_player_pos

    return detected_player_pos


def _recover_player_from_merged_entity(
    *,
    entities: list[dict[str, Any]],
    previous_player_pos: tuple[int, int] | None,
    predicted_player_pos: tuple[int, int] | None,
    previous_player_colors: list[str],
    cell_size: int,
    visible_tiles: dict[tuple[int, int], str],
    wall_points: set[tuple[int, int]],
) -> Optional[tuple[int, int]]:
    if not previous_player_colors:
        return None

    required_colors = {str(color) for color in previous_player_colors}
    for entity in entities:
        colors = {str(color) for color in entity.get("colors", [])}
        if not required_colors.issubset(colors):
            continue
        bbox = entity.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        left, top, right, bottom = bbox
        x0 = left // cell_size
        y0 = top // cell_size
        x1 = right // cell_size
        y1 = bottom // cell_size
        candidates: list[tuple[int, int]] = []
        if predicted_player_pos is not None:
            candidates.append(predicted_player_pos)
        if previous_player_pos is not None:
            candidates.append(previous_player_pos)
            candidates.extend(_neighbors(*previous_player_pos))

        valid_candidates = [
            point
            for point in candidates
            if x0 <= point[0] <= x1
            and y0 <= point[1] <= y1
            and point in visible_tiles
            and point not in wall_points
        ]
        if valid_candidates:
            anchor = predicted_player_pos or previous_player_pos or valid_candidates[0]
            return min(valid_candidates, key=lambda point: _manhattan(point, anchor))
    return None


def _is_walkable_point(
    point: tuple[int, int],
    visible_tiles: dict[tuple[int, int], str],
    wall_points: set[tuple[int, int]],
) -> bool:
    return point in visible_tiles and point not in wall_points


def _stabilize_match_targets(
    *,
    current_targets: list[dict[str, Any]],
    previous_targets: Any,
) -> list[dict[str, Any]]:
    previous_list = previous_targets if isinstance(previous_targets, list) else []
    if not current_targets and previous_list:
        return [dict(target) for target in previous_list if isinstance(target, dict)]

    previous_by_key = {
        str(target.get("match_key")): target
        for target in previous_list
        if isinstance(target, dict) and target.get("match_key") is not None
    }

    stabilized: list[dict[str, Any]] = []
    sole_previous = previous_list[0] if len(previous_list) == 1 and isinstance(previous_list[0], dict) else None
    for current in current_targets:
        stable_target = dict(current)
        previous = previous_by_key.get(str(current.get("match_key")))
        if previous is None and sole_previous is not None:
            previous = sole_previous
        if previous is not None:
            for field_name in (
                "match_key",
                "target_name",
                "reference_name",
                "target_position",
                "reference_position",
                "reference_orientation",
                "target_pattern_signature",
                "reference_pattern_signature",
                "target_mask_pattern_signature",
                "reference_mask_pattern_signature",
            ):
                if previous.get(field_name) is not None:
                    stable_target[field_name] = previous[field_name]
        stabilized.append(stable_target)

    if not stabilized and previous_list:
        stabilized = [dict(target) for target in previous_list if isinstance(target, dict)]

    return sorted(
        stabilized,
        key=lambda item: (
            bool(item.get("aligned")),
            str(item.get("target_name", "")),
        ),
    )


def _extract_player_pos(obs: Any) -> Optional[tuple[int, int]]:
    for field_name in ("agent_pos", "player_pos"):
        pos = _get_field(obs, field_name)
        normalized = _normalize_point(pos)
        if normalized is not None:
            return normalized
    return None


def _extract_resources(obs: Any) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    for field_name in ("levels_completed", "win_levels", "score", "reward"):
        value = _get_field(obs, field_name)
        if value is not None:
            resources[field_name] = _scalarize(value)
    return resources


def _is_terminal(obs: Any) -> bool:
    done = _get_field(obs, "done")
    if done is not None:
        return bool(done)

    state = _get_field(obs, "state")
    state_name = getattr(state, "name", str(state)).upper()
    return state_name in {"WIN", "GAME_OVER", "DONE"}


def _is_win(obs: Any) -> bool:
    reward = _get_field(obs, "reward")
    if isinstance(reward, (int, float)) and reward > 0:
        return True

    state = _get_field(obs, "state")
    state_name = getattr(state, "name", str(state)).upper()
    return state_name == "WIN"


def _get_field(obs: Any, field_name: str) -> Any:
    if isinstance(obs, dict):
        return obs.get(field_name)
    return getattr(obs, field_name, None)


def _normalize_point(value: Any) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict) and {"x", "y"} <= set(value):
        try:
            return int(value["x"]), int(value["y"])
        except (TypeError, ValueError):
            return None
    return None


def _points_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _neighbors(x: int, y: int) -> tuple[tuple[int, int], ...]:
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _bboxes_close(
    left_bbox: tuple[int, int, int, int],
    right_bbox: tuple[int, int, int, int],
    *,
    gap: int = 1,
) -> bool:
    left_x0, left_y0, left_x1, left_y1 = left_bbox
    right_x0, right_y0, right_x1, right_y1 = right_bbox
    return not (
        left_x1 + gap < right_x0
        or right_x1 + gap < left_x0
        or left_y1 + gap < right_y0
        or right_y1 + gap < left_y0
    )


def _bbox_center_in_bbox(
    inner_bbox: tuple[int, int, int, int],
    outer_bbox: tuple[int, int, int, int] | None,
    *,
    margin: int = 0,
) -> bool:
    if outer_bbox is None:
        return True
    center_x = (inner_bbox[0] + inner_bbox[2]) / 2.0
    center_y = (inner_bbox[1] + inner_bbox[3]) / 2.0
    left, top, right, bottom = outer_bbox
    return (
        left - margin <= center_x <= right + margin
        and top - margin <= center_y <= bottom + margin
    )


def _playfield_points(
    visible_tiles: dict[tuple[int, int], str],
    playfield_bbox: tuple[int, int, int, int] | None,
    cell_size: int,
) -> list[tuple[int, int]]:
    if playfield_bbox is None:
        return sorted(visible_tiles)
    left, top, right, bottom = playfield_bbox
    x0 = left // cell_size
    y0 = top // cell_size
    x1 = right // cell_size
    y1 = bottom // cell_size
    return [
        point
        for point in sorted(visible_tiles)
        if x0 <= point[0] <= x1 and y0 <= point[1] <= y1
    ]


def _coarse_components(points: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(points)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for nx, ny in _neighbors(cx, cy):
                if (nx, ny) in remaining:
                    remaining.remove((nx, ny))
                    component.add((nx, ny))
                    queue.append((nx, ny))
        components.append(component)
    return components


def _same_tile_neighbor_count(point: tuple[int, int], component: set[tuple[int, int]]) -> int:
    return sum(1 for neighbor in _neighbors(*point) if neighbor in component)


def _is_playfield_border(point: tuple[int, int], playfield_points: list[tuple[int, int]]) -> bool:
    if not playfield_points:
        return False
    xs = [p[0] for p in playfield_points]
    ys = [p[1] for p in playfield_points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return point[0] in {min_x, max_x} or point[1] in {min_y, max_y}
