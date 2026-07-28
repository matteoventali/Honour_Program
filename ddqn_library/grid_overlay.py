"""Visualise the abstract grid on top of the LunarLander environment.

The conversion used here is the inverse of ``utils.phi_mapping_grid``.  Grid
cells in the resulting image therefore represent exactly the abstract states
used by the trainer, rather than an evenly spaced, screen-only decoration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from utils import phi_mapping_grid, spatial_grid_boundaries


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "trajectory.json"


@dataclass(frozen=True)
class LunarLanderGeometry:
    """Screen/world constants required to project observations onto a frame."""

    viewport_width: int
    viewport_height: int
    scale: float
    helipad_y: float
    leg_down: float


def geometry_from_env(env: gym.Env) -> LunarLanderGeometry:
    """Read projection constants from a reset LunarLander environment."""
    from gymnasium.envs.box2d import lunar_lander

    base_env = env.unwrapped
    if not hasattr(base_env, "helipad_y"):
        raise TypeError("The supplied environment is not a LunarLander environment")
    return LunarLanderGeometry(
        viewport_width=lunar_lander.VIEWPORT_W,
        viewport_height=lunar_lander.VIEWPORT_H,
        scale=lunar_lander.SCALE,
        helipad_y=float(base_env.helipad_y),
        leg_down=lunar_lander.LEG_DOWN,
    )


def observation_to_pixel(
    observation: Sequence[float], geometry: LunarLanderGeometry
) -> tuple[float, float]:
    """Project LunarLander's normalised (x, y) observation onto RGB pixels."""
    half_world_width = geometry.viewport_width / geometry.scale / 2.0
    half_world_height = geometry.viewport_height / geometry.scale / 2.0
    world_x = (float(observation[0]) + 1.0) * half_world_width
    world_y = (
        float(observation[1]) * half_world_height
        + geometry.helipad_y
        + geometry.leg_down / geometry.scale
    )
    pixel_x = world_x * geometry.scale
    pixel_y = geometry.viewport_height - world_y * geometry.scale
    return pixel_x, pixel_y


def _grid_boundaries(
    grid_w: int,
    grid_h: int,
    geometry: LunarLanderGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the boundaries implied by the active spatial discretizer."""
    x_normalised, y_normalised = spatial_grid_boundaries(grid_w, grid_h)
    x_pixels = np.array(
        [observation_to_pixel((x, 0.0), geometry)[0] for x in x_normalised]
    )
    y_pixels = np.array(
        [observation_to_pixel((0.0, y), geometry)[1] for y in y_normalised]
    )
    return x_pixels, y_pixels


def abstract_cell_to_pixel(
    grid_x: int,
    grid_y: int,
    grid_w: int,
    grid_h: int,
    geometry: LunarLanderGeometry,
) -> tuple[float, float]:
    """Return the pixel coordinates of an abstract cell's centre."""
    if not (0 <= grid_x < grid_w and 0 <= grid_y < grid_h):
        raise ValueError(f"Abstract cell ({grid_x}, {grid_y}) is outside the grid")
    x_lines, y_lines = _grid_boundaries(grid_w, grid_h, geometry)
    return (
        float((x_lines[grid_x] + x_lines[grid_x + 1]) / 2.0),
        float((y_lines[grid_y] + y_lines[grid_y + 1]) / 2.0),
    )


def draw_abstract_grid(
    frame: np.ndarray,
    geometry: LunarLanderGeometry,
    grid_w: int,
    grid_h: int,
    waypoints: Mapping[str, Sequence[int]] | None = None,
    observation: Sequence[float] | None = None,
    title: str = "LunarLander with Abstract Grid",
):
    """Create a Matplotlib figure containing frame, grid, waypoints and state."""
    if grid_w < 2 or grid_h < 2:
        raise ValueError("grid_w and grid_h must both be at least 2")

    figure, axis = plt.subplots(figsize=(12, 8))
    axis.imshow(frame)
    x_lines, y_lines = _grid_boundaries(grid_w, grid_h, geometry)
    grid_color = "#ff1744"

    for x_pixel in x_lines:
        axis.axvline(x_pixel, color=grid_color, linewidth=1.6, alpha=0.95)
    for y_pixel in y_lines:
        axis.axhline(y_pixel, color=grid_color, linewidth=1.6, alpha=0.95)

    # The mapping clips everything outside its stated observation domain into
    # an edge cell; tint the currently occupied abstract cell when requested.
    if observation is not None:
        abstract_x, abstract_y = phi_mapping_grid(observation, grid_w, grid_h)
        x0, x1 = sorted((x_lines[abstract_x], x_lines[abstract_x + 1]))
        y0, y1 = sorted((y_lines[abstract_y], y_lines[abstract_y + 1]))
        x0, x1 = np.clip((x0, x1), 0, geometry.viewport_width)
        y0, y1 = np.clip((y0, y1), 0, geometry.viewport_height)
        axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="#00e5ff",
                edgecolor="#00e5ff",
                linewidth=2.5,
                alpha=0.25,
                label=f"Current cell ({abstract_x}, {abstract_y})",
            )
        )

    for name, coordinates in (waypoints or {}).items():
        if len(coordinates) != 2:
            raise ValueError(f"Waypoint {name!r} must contain [x, y]")
        grid_x, grid_y = int(coordinates[0]), int(coordinates[1])
        if not (0 <= grid_x < grid_w and 0 <= grid_y < grid_h):
            raise ValueError(f"Waypoint {name!r} is outside the abstract grid")
        pixel_x, pixel_y = abstract_cell_to_pixel(
            grid_x, grid_y, grid_w, grid_h, geometry
        )
        axis.scatter(pixel_x, pixel_y, s=150, marker="o", color="#ffca28",
                     edgecolor="black", linewidth=1.3, zorder=5)
        axis.annotate(
            f"{name} ({grid_x}, {grid_y})",
            (pixel_x, pixel_y),
            xytext=(7, -10),
            textcoords="offset points",
            color="black",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "fc": "#ffca28", "alpha": 0.9},
            zorder=6,
        )

    axis.set_xlim(0, geometry.viewport_width)
    # A small part of the configured y-domain can lie above the RGB viewport.
    # Keep it in view so that no abstract row or coordinate label disappears.
    visible_top = min(0.0, float(np.min(y_lines)))
    axis.set_ylim(geometry.viewport_height, visible_top)
    axis.set_title(title)

    # Put the abstract coordinates at cell centres. Since image coordinates
    # grow downwards while abstract y grows upwards, y_centres is descending:
    # label 0 consequently appears at the bottom and grid_h - 1 at the top.
    x_centres = (x_lines[:-1] + x_lines[1:]) / 2.0
    y_centres = (y_lines[:-1] + y_lines[1:]) / 2.0
    axis.set_xticks(x_centres, labels=range(grid_w))
    axis.set_yticks(y_centres, labels=range(grid_h))
    axis.set_xlabel("Abstract x-coordinate")
    axis.set_ylabel("Abstract y-coordinate")
    axis.tick_params(
        axis="both",
        which="major",
        color=grid_color,
        labelcolor=grid_color,
        labelsize=10,
        width=1.5,
        length=5,
    )
    for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        label.set_fontweight("bold")

    if observation is not None:
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
        figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    else:
        figure.tight_layout()
    return figure


def generate_overlay(
    output_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    seed: int | None = 0,
) -> Path:
    """Reset LunarLander and save one annotated RGB frame as a PNG."""
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLander-v3", continuous=False, render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=seed)
        frame = env.render()
        geometry = geometry_from_env(env)
        figure = draw_abstract_grid(
            frame=frame,
            geometry=geometry,
            grid_w=int(config.get("grid_w", 12)),
            grid_h=int(config.get("grid_h", 12)),
            waypoints=config.get("waypoints_dict", {}),
            observation=observation,
        )
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    finally:
        env.close()
    return output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a LunarLander frame with the abstract grid overlaid."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output_path = args.output or SCRIPT_DIR / "img" / "abstract_grid_overlay.png"
    saved_path = generate_overlay(output_path, args.config, args.seed)
    print(f"Image saved to: {saved_path}")


if __name__ == "__main__":
    main()
