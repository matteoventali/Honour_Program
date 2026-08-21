import unittest
from collections import deque
from unittest.mock import patch

import numpy as np

import trainer
from agent import ReplayBuffer


class _Memory:
    def __init__(self):
        self.transitions = []
        self.position = 0

    def push(self, *transition):
        self.transitions.append(transition)
        self.position = len(self.transitions)

    def q_fraction_onehot(self, _index, _num_states):
        return 0.0

    def sample_indices(self, batch_size, rng):
        return rng.sample(range(len(self.transitions)), batch_size)

    def __len__(self):
        return len(self.transitions)


class _Learner:
    def __init__(self):
        self.memory = _Memory()
        self.batch_size = 1
        self.gamma = 0.9
        self.eps = 0.5
        self.eps_min = 0.01
        self.eps_decay = 0.9
        self.action_calls = 0
        self.optimization_calls = 0
        self.optimization_batches = []

    def select_action(self, _state):
        self.action_calls += 1
        return 2

    def optimize_model(self, batch_indices=None):
        self.optimization_calls += 1
        self.optimization_batches.append(tuple(batch_indices) if batch_indices is not None else None)


class _Automaton:
    states = [0, 1]
    accepting_states = {1}
    formula_str = "F(goal)"

    def get_initial_q(self):
        return 0

    def get_next_q(self, current_q, truth_assignment):
        return 1 if truth_assignment["goal"] else current_q

    def is_goal_reached(self, q):
        return q in self.accepting_states


class _AbstractMDP:
    width = 2
    height = 1
    gamma = 0.9
    upper_level_mdp = None
    regions = {}

    def __init__(self):
        self.automaton = _Automaton()
        self.v_star = {(0, 0, 0): 2.0, (1, 0, 1): 5.0}

    def _get_truth_assignment(self, x, _y):
        return {"goal": x == 1}

    def get_environment_truth_assignment(self, observation):
        return {"goal": observation[0] == 1}


class _Environment:
    def reset(self, seed=None):
        return np.asarray([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        if action != 2:
            raise AssertionError("The action must come from the biased learner")
        return np.asarray([1.0, 0.0], dtype=np.float32), 999.0, False, False, {}


class _UnchangedAbstractStateEnvironment:
    def reset(self, seed=None):
        return np.asarray([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        return np.asarray([0.0, 0.0], dtype=np.float32), 0.0, True, False, {}


class DualLearnerTrainingTest(unittest.TestCase):
    def test_shared_transition_uses_distinct_rewards(self):
        biased = _Learner()
        unbiased = _Learner()

        evaluation_result = {
            "success_rate": 0.5,
            "mean_task_reward": 5.0,
            "mean_episode_length": 12.0,
        }
        with patch.object(
            trainer,
            "_abstract_position",
            side_effect=lambda observation, _mdp: (int(observation[0]), 0),
        ), patch.object(
            trainer,
            "_evaluate_agent_greedily",
            return_value=evaluation_result,
        ) as evaluate:
            metrics = trainer.run_sequential_training(
                env=_Environment(),
                biased_agent=biased,
                unbiased_agent=unbiased,
                abstract_mdp=_AbstractMDP(),
                episodes=1,
                goal_reward=10.0,
                save_policy=False,
                log_interval=1,
            )

        self.assertEqual(biased.action_calls, 1)
        self.assertEqual(unbiased.action_calls, 0)
        self.assertEqual(biased.optimization_calls, 1)
        self.assertEqual(unbiased.optimization_calls, 1)
        self.assertEqual(biased.optimization_batches, [(0,)])
        self.assertEqual(unbiased.optimization_batches, [(0,)])

        biased_transition = biased.memory.transitions[0]
        unbiased_transition = unbiased.memory.transitions[0]
        self.assertEqual(biased_transition[1], unbiased_transition[1])
        self.assertAlmostEqual(biased_transition[2], 12.5)
        self.assertAlmostEqual(unbiased_transition[2], 10.0)
        self.assertTrue(biased_transition[4])
        self.assertTrue(unbiased_transition[4])
        self.assertAlmostEqual(biased_transition[5], 0.9)
        self.assertAlmostEqual(unbiased_transition[5], 0.9)
        self.assertEqual(metrics["biased_learning_rewards"], [12.5])
        self.assertEqual(metrics["unbiased_learning_rewards"], [10.0])
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(metrics["evaluation_steps"], [1])
        self.assertEqual(metrics["unbiased_eval_success_rates"], [0.5])

    def test_learners_can_use_distinct_gammas(self):
        biased = _Learner()
        unbiased = _Learner()
        biased.gamma = 0.8
        unbiased.gamma = 0.95
        evaluation_result = {
            "success_rate": 0.5,
            "mean_task_reward": 5.0,
            "mean_episode_length": 12.0,
        }
        with patch.object(
            trainer,
            "_abstract_position",
            side_effect=lambda observation, _mdp: (int(observation[0]), 0),
        ), patch.object(
            trainer,
            "_evaluate_agent_greedily",
            return_value=evaluation_result,
        ):
            metrics = trainer.run_sequential_training(
                env=_Environment(), biased_agent=biased,
                unbiased_agent=unbiased, abstract_mdp=_AbstractMDP(),
                episodes=1, goal_reward=10.0, save_policy=False,
                log_interval=1,
            )

        self.assertAlmostEqual(biased.memory.transitions[0][2], 12.0)
        self.assertAlmostEqual(biased.memory.transitions[0][5], 0.8)
        self.assertAlmostEqual(unbiased.memory.transitions[0][5], 0.95)
        self.assertAlmostEqual(metrics["biased_gamma"], 0.8)
        self.assertAlmostEqual(metrics["unbiased_gamma"], 0.95)

    def test_gamma_shaping_one_matches_cell_change_heuristic(self):
        biased = _Learner()
        unbiased = _Learner()
        evaluation_result = {
            "success_rate": 0.0,
            "mean_task_reward": 0.0,
            "mean_episode_length": 1.0,
        }
        with patch.object(
            trainer,
            "_abstract_position",
            return_value=(0, 0),
        ), patch.object(
            trainer,
            "_evaluate_agent_greedily",
            return_value=evaluation_result,
        ):
            metrics = trainer.run_sequential_training(
                env=_UnchangedAbstractStateEnvironment(),
                biased_agent=biased,
                unbiased_agent=unbiased,
                abstract_mdp=_AbstractMDP(),
                episodes=1,
                save_policy=False,
                gamma_shaping=1.0,
                log_interval=1,
            )

        self.assertEqual(metrics["shaping_rewards"], [0.0])
        self.assertEqual(biased.memory.transitions[0][2], 0.0)
        self.assertEqual(metrics["gamma_shaping"], 1.0)


class ReplayBufferSamplingTest(unittest.TestCase):
    def test_prescribed_indices_define_the_minibatch_for_both_learners(self):
        buffer = ReplayBuffer(capacity=10, num_phases=0)
        for reward in (10.0, 20.0, 30.0):
            state = np.asarray([reward], dtype=np.float32)
            buffer.push(state, 0, reward, state, False, 0.9)

        _states, _actions, rewards, _next_states, _dones, discounts = buffer.sample(
            2,
            [2, 0],
        )
        np.testing.assert_array_equal(rewards, np.asarray([30.0, 10.0]))
        np.testing.assert_array_equal(discounts, np.asarray([0.9, 0.9]))


class NStepReturnTest(unittest.TestCase):
    def test_five_step_return_keeps_rewards_separate(self):
        transitions = []
        for index in range(5):
            transitions.append(
                (
                    np.asarray([index], dtype=np.float32),
                    index,
                    float(index + 1),
                    10.0 if index == 4 else 0.0,
                    np.asarray([index + 1], dtype=np.float32),
                    index == 4,
                )
            )

        biased_return, unbiased_return, shared = trainer._aggregate_n_step_window(
            transitions,
            gamma=0.9,
        )

        expected_biased = sum(0.9**index * (index + 1) for index in range(5))
        self.assertAlmostEqual(biased_return, expected_biased)
        self.assertAlmostEqual(unbiased_return, 0.9**4 * 10.0)
        self.assertTrue(shared[3])
        self.assertAlmostEqual(shared[4], 0.9**5)

    def test_short_truncated_prefix_bootstraps_with_actual_length(self):
        transitions = [
            ("s0", 0, 1.0, 0.0, "s1", False),
            ("s1", 1, 2.0, 0.0, "s2", False),
        ]
        biased_return, unbiased_return, shared = trainer._aggregate_n_step_window(
            transitions,
            gamma=0.9,
        )

        self.assertAlmostEqual(biased_return, 1.0 + 0.9 * 2.0)
        self.assertEqual(unbiased_return, 0.0)
        self.assertEqual(shared[2], "s2")
        self.assertFalse(shared[3])
        self.assertAlmostEqual(shared[4], 0.9**2)

    def test_distinct_gammas_discount_each_n_step_return_and_bootstrap(self):
        transitions = [
            ("s0", 0, 1.0, 1.0, "s1", False),
            ("s1", 1, 2.0, 2.0, "s2", False),
        ]
        biased_return, unbiased_return, shared = trainer._aggregate_n_step_window(
            transitions, gamma=0.8, unbiased_gamma=0.95,
        )

        self.assertAlmostEqual(biased_return, 1.0 + 0.8 * 2.0)
        self.assertAlmostEqual(unbiased_return, 1.0 + 0.95 * 2.0)
        self.assertAlmostEqual(shared[4], 0.8**2)
        self.assertAlmostEqual(shared[5], 0.95**2)

    def test_terminal_flush_propagates_reward_to_five_prefixes(self):
        pending = deque(
            (
                f"s{index}",
                index,
                10.0 if index == 4 else 0.0,
                10.0 if index == 4 else 0.0,
                f"s{index + 1}",
                index == 4,
            )
            for index in range(5)
        )
        biased_memory = _Memory()
        unbiased_memory = _Memory()

        while pending:
            trainer._emit_n_step_prefix(
                pending,
                n_step=5,
                gamma=0.9,
                biased_memory=biased_memory,
                unbiased_memory=unbiased_memory,
            )

        expected_rewards = [0.9**power * 10.0 for power in range(4, -1, -1)]
        self.assertEqual(len(unbiased_memory.transitions), 5)
        for transition, expected_reward in zip(
            unbiased_memory.transitions,
            expected_rewards,
        ):
            self.assertAlmostEqual(transition[2], expected_reward)
            self.assertTrue(transition[4])


if __name__ == "__main__":
    unittest.main()
