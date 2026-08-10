"""Unit tests for the configurable cyclic-waypoint automaton."""

import unittest

from abstract_mdps import ManualWaypointMDP
from manual_automaton import AlternatingGoalsAutomaton, CyclicWaypointsAutomaton


class CyclicWaypointsAutomatonTests(unittest.TestCase):
    def test_three_waypoints_must_be_reached_in_order(self):
        automaton = CyclicWaypointsAutomaton(["start", "middle", "finish"])

        step = automaton.advance("q1", {"middle": True})
        self.assertEqual(step.next_state, "q1")
        self.assertFalse(step.completed_cycle)

        step = automaton.advance("q1", {"start": True})
        self.assertEqual(step.next_state, "q2")
        self.assertEqual(step.reached_waypoint, "start")

        step = automaton.advance("q2", {"middle": True})
        self.assertEqual(step.next_state, "q3")

        step = automaton.advance("q3", {"finish": True})
        self.assertEqual(step.next_state, "q1")
        self.assertTrue(step.completed_cycle)
        self.assertEqual(step.reached_waypoint, "finish")

    def test_one_waypoint_cycle_is_supported(self):
        automaton = CyclicWaypointsAutomaton(["goal"])

        step = automaton.advance("q1", {"goal": True})

        self.assertEqual(automaton.active_states, ("q1",))
        self.assertEqual(step.next_state, "q1")
        self.assertTrue(step.completed_cycle)

    def test_mdp_rewards_only_the_last_waypoint(self):
        automaton = CyclicWaypointsAutomaton(["a", "b", "c"])
        mdp = ManualWaypointMDP(
            waypoints_dict={"a": (1, 0), "b": (2, 0), "c": (3, 0)},
            automaton=automaton,
            width=4,
            height=1,
            goal_reward=10,
        )

        state, reward = mdp.get_transitions((0, 0, "q1"), 3)
        self.assertEqual((state, reward), ((1, 0, "q2"), 0.0))
        state, reward = mdp.get_transitions(state, 3)
        self.assertEqual((state, reward), ((2, 0, "q3"), 0.0))
        state, reward = mdp.get_transitions(state, 3)
        self.assertEqual((state, reward), ((3, 0, "q1"), 10.0))

    def test_invalid_cycles_and_missing_waypoints_are_rejected(self):
        with self.assertRaises(ValueError):
            CyclicWaypointsAutomaton([])
        with self.assertRaises(ValueError):
            CyclicWaypointsAutomaton(["a", "a"])

        automaton = CyclicWaypointsAutomaton(["a", "b"])
        with self.assertRaises(ValueError):
            automaton.validate_waypoints({"a": (0, 0)}, width=2, height=2)

    def test_old_two_goal_api_keeps_the_same_states(self):
        automaton = AlternatingGoalsAutomaton()

        self.assertEqual(automaton.active_states, ("q1", "q2"))
        self.assertEqual(automaton.accepting_state, "q3")
        self.assertEqual(automaton.first_goal, "g1")
        self.assertEqual(automaton.second_goal, "g2")


if __name__ == "__main__":
    unittest.main()
