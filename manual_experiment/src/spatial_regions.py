"""Continuous circular task regions and their grid over-approximations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


OBSERVATION_X_BOUNDS = (-1.0, 1.0)
OBSERVATION_Y_BOUNDS = (0.0, 1.5)


@dataclass(frozen=True)
class CircularRegion:
    """A circular proposition region in LunarLander observation coordinates."""

    center_x: float
    center_y: float
    radius: float

    def __post_init__(self) -> None:
        values = (self.center_x, self.center_y, self.radius)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("Region center coordinates and radius must be numbers")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Region center coordinates and radius must be finite")
        if self.radius <= 0:
            raise ValueError("Region radius must be greater than zero")
        if not OBSERVATION_X_BOUNDS[0] <= self.center_x <= OBSERVATION_X_BOUNDS[1]:
            raise ValueError(f"Region center x={self.center_x} is outside {OBSERVATION_X_BOUNDS}")
        if not OBSERVATION_Y_BOUNDS[0] <= self.center_y <= OBSERVATION_Y_BOUNDS[1]:
            raise ValueError(f"Region center y={self.center_y} is outside {OBSERVATION_Y_BOUNDS}")

    def contains(self, x: float, y: float) -> bool:
        dx = float(x) - self.center_x
        dy = float(y) - self.center_y
        return dx * dx + dy * dy <= self.radius * self.radius

    @classmethod
    def from_dict(cls, data: Mapping[str, object], name: str = "region") -> "CircularRegion":
        if not isinstance(data, Mapping):
            raise ValueError(f"{name!r} must be a JSON object")
        center = data.get("center")
        if isinstance(center, (str, bytes)) or not isinstance(center, Sequence) or len(center) != 2:
            raise ValueError(f"{name!r}.center must contain exactly [x, y]")
        if "radius" not in data:
            raise ValueError(f"{name!r} must define radius")
        return cls(center_x=center[0], center_y=center[1], radius=data["radius"])

    def as_dict(self) -> dict[str, object]:
        return {"center": [self.center_x, self.center_y], "radius": self.radius}


def load_regions(raw_regions: Mapping[str, object]) -> dict[str, CircularRegion]:
    if not isinstance(raw_regions, Mapping) or not raw_regions:
        raise ValueError("trajectory.json must define a non-empty 'regions' object")
    regions = {}
    for name, data in raw_regions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Region proposition names must be non-empty strings")
        regions[name] = CircularRegion.from_dict(data, name=name)
    return regions


def truth_assignment_from_observation(
    regions: Mapping[str, CircularRegion], observation: Sequence[float]
) -> dict[str, bool]:
    if len(observation) < 2:
        raise ValueError("An environment observation must contain x and y")
    x, y = float(observation[0]), float(observation[1])
    return {name: region.contains(x, y) for name, region in regions.items()}


def grid_cell_bounds(x: int, y: int, width: int, height: int) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("Grid dimensions must be positive")
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError(f"Cell ({x}, {y}) is outside the {width}x{height} grid")
    x_step = (OBSERVATION_X_BOUNDS[1] - OBSERVATION_X_BOUNDS[0]) / width
    y_step = (OBSERVATION_Y_BOUNDS[1] - OBSERVATION_Y_BOUNDS[0]) / height
    x_min = OBSERVATION_X_BOUNDS[0] + x * x_step
    y_min = OBSERVATION_Y_BOUNDS[0] + y * y_step
    return x_min, x_min + x_step, y_min, y_min + y_step


def circle_intersects_cell(region: CircularRegion, bounds: Sequence[float]) -> bool:
    if len(bounds) != 4:
        raise ValueError("Cell bounds must be (x_min, x_max, y_min, y_max)")
    x_min, x_max, y_min, y_max = bounds
    closest_x = min(max(region.center_x, x_min), x_max)
    closest_y = min(max(region.center_y, y_min), y_max)
    dx = closest_x - region.center_x
    dy = closest_y - region.center_y
    return dx * dx + dy * dy <= region.radius * region.radius


def rasterize_regions(
    regions: Mapping[str, CircularRegion], width: int, height: int
) -> dict[str, frozenset[tuple[int, int]]]:
    rasterized = {}
    for name, region in regions.items():
        cells = frozenset(
            (x, y)
            for x in range(width)
            for y in range(height)
            if circle_intersects_cell(region, grid_cell_bounds(x, y, width, height))
        )
        if not cells:
            raise ValueError(f"Region {name!r} does not intersect the {width}x{height} grid")
        rasterized[name] = cells
    return rasterized


def truth_assignment_from_cell(
    region_cells: Mapping[str, frozenset[tuple[int, int]]], x: int, y: int
) -> dict[str, bool]:
    return {name: (x, y) in cells for name, cells in region_cells.items()}
