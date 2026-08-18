"""Configuration and mappings for a hierarchy of rectangular grid abstractions.

All mappings operate in the normalised square ``[0, 1] x [0, 1]``.  This makes
them independent of LunarLander's observation bounds and, importantly, allows
both the source and destination grids to have arbitrary dimensions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GridLevel:
    """One configured abstraction level."""

    width: int
    height: int
    name: str

    def __post_init__(self):
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise ValueError(f"{self.name}.grid_w must be a positive integer")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            raise ValueError(f"{self.name}.grid_h must be a positive integer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Every abstraction level must have a non-empty name")

    @property
    def shape(self):
        """Return ``(width, height)`` for mapping helpers."""
        return self.width, self.height


@dataclass(frozen=True)
class AbstractionConfig:
    """Validated ordered collection of abstraction levels.

    ``levels[0]`` is always the abstraction used by the automaton and training.
    The remaining levels are ordered dependencies: states at level *i* are
    mapped online to level *i + 1*, whose V-function is used as potential.
    """

    levels: tuple[GridLevel, ...]
    inter_level_shaping_scale: float = 1.0

    def __post_init__(self):
        if not self.levels:
            raise ValueError("abstraction.json must define at least one level")
        names = [level.name for level in self.levels]
        if len(names) != len(set(names)):
            raise ValueError("Abstraction level names must be unique")
        if (
            isinstance(self.inter_level_shaping_scale, bool)
            or not isinstance(self.inter_level_shaping_scale, (int, float))
            or not math.isfinite(self.inter_level_shaping_scale)
        ):
            raise ValueError("inter_level_shaping_scale must be a finite number")
        object.__setattr__(
            self,
            "inter_level_shaping_scale",
            float(self.inter_level_shaping_scale),
        )

    @property
    def primary(self):
        """Return level 1, whose coordinates define the automaton semantics."""
        return self.levels[0]

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("The abstraction configuration must be a JSON object")
        raw_levels = data.get("levels")
        if not isinstance(raw_levels, list) or not raw_levels:
            raise ValueError("abstraction.json must contain a non-empty 'levels' array")

        levels = []
        for index, raw_level in enumerate(raw_levels, start=1):
            if not isinstance(raw_level, dict):
                raise ValueError(f"levels[{index - 1}] must be a JSON object")
            name = raw_level.get("name", f"level{index}")
            width = raw_level.get("grid_w", raw_level.get("width"))
            height = raw_level.get("grid_h", raw_level.get("height"))
            if width is None or height is None:
                raise ValueError(
                    f"{name} must define grid_w/grid_h (or width/height)"
                )
            levels.append(GridLevel(width=width, height=height, name=name))
        return cls(
            tuple(levels),
            inter_level_shaping_scale=data.get(
                "inter_level_shaping_scale",
                1.0,
            ),
        )

    @classmethod
    def load(cls, path):
        """Load and validate an ``abstraction.json`` file."""
        path = Path(path)
        with path.open(encoding="utf-8") as config_file:
            return cls.from_dict(json.load(config_file))


def _validate_dimensions(width, height, label):
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"{label} dimensions must be positive integers")


def _validate_cell(cell, width, height, label="source"):
    if not isinstance(cell, (tuple, list)) or len(cell) != 2:
        raise ValueError("A grid cell must contain exactly two coordinates")
    x, y = cell
    if isinstance(x, bool) or not isinstance(x, int):
        raise ValueError("Grid x-coordinate must be an integer")
    if isinstance(y, bool) or not isinstance(y, int):
        raise ValueError("Grid y-coordinate must be an integer")
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError(
            f"Cell ({x}, {y}) is outside the {label} grid {width}x{height}"
        )
    return x, y


def map_cell(
    cell,
    source_width,
    source_height,
    target_width,
    target_height,
):
    """Map one cell centre between arbitrary rectangular grids.

    The same function is used in either direction by swapping source and target
    dimensions.  A centre-based mapping is deterministic even when a coarse
    cell overlaps multiple finer cells.
    """
    _validate_dimensions(source_width, source_height, "Source")
    _validate_dimensions(target_width, target_height, "Target")
    x, y = _validate_cell(cell, source_width, source_height)
    target_x = min(
        int(((x + 0.5) / source_width) * target_width),
        target_width - 1,
    )
    target_y = min(
        int(((y + 0.5) / source_height) * target_height),
        target_height - 1,
    )
    return target_x, target_y


def map_state(
    state,
    source_width,
    source_height,
    target_width,
    target_height,
):
    """Map the spatial part of ``(x, y, q)`` while preserving DFA state ``q``."""
    if not isinstance(state, (tuple, list)) or len(state) != 3:
        raise ValueError("An abstract state must be (x, y, q)")
    x, y = map_cell(
        state[:2],
        source_width,
        source_height,
        target_width,
        target_height,
    )
    return x, y, state[2]


def overlapping_cells(
    cell,
    source_width,
    source_height,
    target_width,
    target_height,
):
    """Return every destination cell with positive area overlap.

    This is the set-valued counterpart of :func:`map_cell`, useful for exact
    coarse-to-fine and fine-to-coarse relationships.
    """
    _validate_dimensions(source_width, source_height, "Source")
    _validate_dimensions(target_width, target_height, "Target")
    x, y = _validate_cell(cell, source_width, source_height)

    min_x = int(math.floor(x * target_width / source_width))
    max_x = int(math.ceil((x + 1) * target_width / source_width) - 1)
    min_y = int(math.floor(y * target_height / source_height))
    max_y = int(math.ceil((y + 1) * target_height / source_height) - 1)
    return [
        (target_x, target_y)
        for target_x in range(max(0, min_x), min(target_width - 1, max_x) + 1)
        for target_y in range(max(0, min_y), min(target_height - 1, max_y) + 1)
    ]


def map_waypoints(
    waypoints,
    source_width,
    source_height,
    target_width,
    target_height,
):
    """Map a proposition-to-cell dictionary between arbitrary grids."""
    return {
        name: map_cell(
            tuple(coordinates),
            source_width,
            source_height,
            target_width,
            target_height,
        )
        for name, coordinates in waypoints.items()
    }
