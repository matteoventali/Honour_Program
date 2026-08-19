"""Continuous interpolation of abstract grid value functions."""

import math

from spatial_regions import OBSERVATION_X_BOUNDS, OBSERVATION_Y_BOUNDS


def _center_coordinate(value, lower, upper, size):
    """Map a physical coordinate to a clipped fractional cell-centre index."""
    coordinate = (float(value) - lower) / (upper - lower) * size - 0.5
    return min(max(coordinate, 0.0), float(size - 1))


def bilinear_grid_potential(observation, q, abstract_mdp):
    """Interpolate ``V*(x, y, q)`` at a continuous environment position."""
    if len(observation) < 2:
        raise ValueError("An environment observation must contain x and y")
    u = _center_coordinate(
        observation[0], *OBSERVATION_X_BOUNDS, abstract_mdp.width
    )
    v = _center_coordinate(
        observation[1], *OBSERVATION_Y_BOUNDS, abstract_mdp.height
    )
    x0, y0 = int(math.floor(u)), int(math.floor(v))
    x1 = min(x0 + 1, abstract_mdp.width - 1)
    y1 = min(y0 + 1, abstract_mdp.height - 1)
    tx, ty = u - x0, v - y0

    value_00 = float(abstract_mdp.v_star.get((x0, y0, q), 0.0))
    value_10 = float(abstract_mdp.v_star.get((x1, y0, q), 0.0))
    value_01 = float(abstract_mdp.v_star.get((x0, y1, q), 0.0))
    value_11 = float(abstract_mdp.v_star.get((x1, y1, q), 0.0))
    lower_value = (1.0 - tx) * value_00 + tx * value_10
    upper_value = (1.0 - tx) * value_01 + tx * value_11
    return (1.0 - ty) * lower_value + ty * upper_value
