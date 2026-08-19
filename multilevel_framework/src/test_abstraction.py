"""Tests for dependency-free multilevel configuration and grid mappings."""

import unittest

from abstraction import (
    AbstractionConfig,
    map_cell,
    map_state,
    map_waypoints,
    overlapping_cells,
)
from abstract_mdps import LTLfWaypointMDP, MultiLevelWaypointMDP
from spatial_regions import CircularRegion


def _goal_region(x=0.75, y=0.75, radius=0.01):
    return {"goal": CircularRegion(x, y, radius)}


class AbstractionConfigTests(unittest.TestCase):
    def test_level1_is_primary_and_names_are_optional(self):
        config = AbstractionConfig.from_dict(
            {
                "levels": [
                    {"grid_w": 12, "grid_h": 8},
                    {"name": "coarse", "width": 5, "height": 3},
                ]
            }
        )
        self.assertEqual(config.primary.shape, (12, 8))
        self.assertEqual(config.levels[0].name, "level1")
        self.assertEqual(config.levels[1].name, "coarse")

    def test_empty_hierarchy_is_rejected(self):
        with self.assertRaises(ValueError):
            AbstractionConfig.from_dict({"levels": []})

    def test_invalid_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            AbstractionConfig.from_dict(
                {"levels": [{"grid_w": 0, "grid_h": 4}]}
            )


class GridMappingTests(unittest.TestCase):
    def test_arbitrary_dimensions_work_in_both_directions(self):
        fine_cell = (7, 6)
        coarse_cell = map_cell(fine_cell, 12, 8, 5, 3)
        self.assertEqual(coarse_cell, (3, 2))
        mapped_back = map_cell(coarse_cell, 5, 3, 12, 8)
        self.assertIn(fine_cell, overlapping_cells(coarse_cell, 5, 3, 12, 8))
        self.assertIn(mapped_back, overlapping_cells(coarse_cell, 5, 3, 12, 8))

    def test_state_mapping_preserves_automaton_state(self):
        self.assertEqual(map_state((11, 7, 42), 12, 8, 5, 3), (4, 2, 42))

    def test_set_valued_mapping_covers_all_overlapping_cells(self):
        self.assertEqual(
            overlapping_cells((0, 0), 2, 1, 5, 1),
            [(0, 0), (1, 0), (2, 0)],
        )

    def test_waypoints_can_be_projected_and_mapped_back(self):
        projected = map_waypoints({"goal": (11, 7)}, 12, 8, 5, 3)
        self.assertEqual(projected, {"goal": (4, 2)})
        mapped_back = map_waypoints(projected, 5, 3, 12, 8)
        self.assertIn(mapped_back["goal"], overlapping_cells((4, 2), 5, 3, 12, 8))


class _TinyAutomaton:
    states = [0, 1]
    accepting_states = {1}
    num_phases = 2

    @staticmethod
    def is_goal_reached(q):
        return q == 1

    @staticmethod
    def get_next_q(q, truth_assignment):
        return 1 if truth_assignment.get("goal", False) else q


class TerminalRewardTests(unittest.TestCase):
    def test_abstract_transitions_never_emit_reward(self):
        mdp = LTLfWaypointMDP(
            regions=_goal_region(0.5),
            ltlf_automaton=_TinyAutomaton(),
            width=2,
            height=1,
            gamma=0.9,
            goal_reward=10.0,
        )

        next_state, reward = mdp.get_transitions((0, 0, 0), 3)
        self.assertEqual(next_state, (1, 0, 1))
        self.assertEqual(reward, 0.0)

        _, repeated_reward = mdp.get_transitions((1, 0, 1), 3)
        self.assertEqual(repeated_reward, 0.0)

    def test_accepting_states_hold_the_goal_reward(self):
        mdp = LTLfWaypointMDP(
            regions=_goal_region(0.5),
            ltlf_automaton=_TinyAutomaton(),
            width=2,
            height=1,
            gamma=0.9,
            goal_reward=10.0,
        )
        mdp.value_iteration(theta=0.0001, print_policy=False)

        self.assertEqual(mdp.v_star[(0, 0, 1)], 10.0)
        self.assertEqual(mdp.v_star[(1, 0, 1)], 10.0)
        self.assertEqual(mdp.v_star[(0, 0, 0)], 9.0)


class HierarchyTests(unittest.TestCase):
    def test_level_i_uses_level_i_plus_one_as_shaping_potential(self):
        config = AbstractionConfig.from_dict(
            {
                "levels": [
                    {"grid_w": 4, "grid_h": 3},
                    {"grid_w": 3, "grid_h": 2},
                    {"grid_w": 2, "grid_h": 1},
                ]
            }
        )
        hierarchy = MultiLevelWaypointMDP(
            regions=_goal_region(0.75, 1.25),
            ltlf_automaton=_TinyAutomaton(),
            abstraction_config=config,
            gamma=0.9,
            goal_reward=10.0,
        )
        hierarchy.compute_value_functions(theta=0.0001)

        self.assertEqual(hierarchy.primary_mdp.region_cells["goal"], {(3, 2)})
        self.assertEqual(hierarchy.levels[1].region_cells["goal"], {(2, 1)})
        self.assertIsNone(hierarchy.levels[-1].upper_level_mdp)
        self.assertIs(
            hierarchy.primary_mdp.upper_level_mdp,
            hierarchy.levels[1],
        )
        state = (0, 0, 0)
        next_state, _ = hierarchy.primary_mdp.get_transitions(state, 3)
        mapped_state = hierarchy.primary_mdp.map_state_to_upper_level(state)
        mapped_next_state = (
            hierarchy.primary_mdp.map_state_to_upper_level(next_state)
        )
        expected_state_potential = hierarchy.levels[1].v_star[mapped_state]
        expected_next_potential = hierarchy.levels[1].v_star[mapped_next_state]
        self.assertEqual(
            hierarchy.primary_mdp.get_upper_level_potential(state),
            expected_state_potential,
        )
        expected_shaping = 0.9 * expected_next_potential - expected_state_potential
        self.assertAlmostEqual(
            hierarchy.primary_mdp.get_inter_level_shaping_reward(
                state,
                next_state,
            ),
            expected_shaping,
        )
        self.assertAlmostEqual(hierarchy.primary_mdp.v_star[(3, 2, 1)], 10.0)

    def test_upper_value_is_read_online_without_a_precomputed_table(self):
        def build_hierarchy():
            mdp_config = AbstractionConfig.from_dict(
                {
                    "levels": [
                        {"grid_w": 4, "grid_h": 1},
                        {"grid_w": 2, "grid_h": 1},
                    ],
                }
            )
            hierarchy = MultiLevelWaypointMDP(
                regions=_goal_region(),
                ltlf_automaton=_TinyAutomaton(),
                abstraction_config=mdp_config,
                gamma=0.9,
                goal_reward=10.0,
            )
            hierarchy.compute_value_functions(theta=0.0001)
            return hierarchy

        shaped = build_hierarchy().primary_mdp

        self.assertIsNotNone(shaped.upper_level_mdp)
        self.assertFalse(hasattr(shaped, "inter_level_potential"))
        self.assertFalse(hasattr(shaped, "warm_start_v_star"))
        state = (0, 0, 0)
        mapped_state = shaped.map_state_to_upper_level(state)
        shaped.upper_level_mdp.v_star[mapped_state] = 123.0
        self.assertEqual(shaped.get_upper_level_potential(state), 123.0)

    def test_boundary_region_labels_every_intersected_cell_and_mapping_preserves_q(self):
        config = AbstractionConfig.from_dict(
            {
                "levels": [
                    {"grid_w": 4, "grid_h": 1},
                    {"grid_w": 2, "grid_h": 1},
                ]
            }
        )
        hierarchy = MultiLevelWaypointMDP(
            regions=_goal_region(0.0, 0.75, 0.05),
            ltlf_automaton=_TinyAutomaton(),
            abstraction_config=config,
            gamma=0.9,
            goal_reward=10.0,
        )
        hierarchy.compute_value_functions(theta=0.0001)
        fine = hierarchy.primary_mdp

        self.assertEqual(fine.region_cells["goal"], {(1, 0), (2, 0)})
        self.assertEqual(hierarchy.levels[1].region_cells["goal"], {(0, 0), (1, 0)})
        self.assertEqual(
            fine.map_state_to_upper_level((2, 0, 0)),
            (1, 0, 0),
        )


if __name__ == "__main__":
    unittest.main()
