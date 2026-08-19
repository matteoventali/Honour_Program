"""Tests for continuous interpolation of abstract potentials."""

import unittest
from types import SimpleNamespace

from continuous_potential import bilinear_grid_potential


class BilinearPotentialTests(unittest.TestCase):
    def setUp(self):
        self.mdp = SimpleNamespace(
            width=2,
            height=2,
            v_star={
                (0, 0, 0): 0.0,
                (1, 0, 0): 10.0,
                (0, 1, 0): 20.0,
                (1, 1, 0): 30.0,
            },
        )

    def test_cell_centres_reproduce_grid_values(self):
        self.assertAlmostEqual(
            bilinear_grid_potential((-0.5, 0.375), 0, self.mdp),
            0.0,
        )
        self.assertAlmostEqual(
            bilinear_grid_potential((0.5, 1.125), 0, self.mdp),
            30.0,
        )

    def test_midpoint_averages_four_neighbours(self):
        self.assertAlmostEqual(
            bilinear_grid_potential((0.0, 0.75), 0, self.mdp),
            15.0,
        )

    def test_out_of_bounds_coordinates_are_clamped(self):
        self.assertAlmostEqual(
            bilinear_grid_potential((-5.0, -5.0), 0, self.mdp),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
