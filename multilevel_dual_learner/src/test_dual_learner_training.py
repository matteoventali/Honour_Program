import unittest
from unittest.mock import patch

import numpy as np
import torch

import trainer
from agent import DualReplayBuffer, HierarchicalDQNLearner, ReplayBuffer


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
        self.optimization_rewards = []
        self.device = torch.device("cpu")

    def select_action(self, _state):
        self.action_calls += 1
        return 2

    def optimize_tensor_batch(
        self,
        _states,
        _actions,
        rewards,
        _next_states,
        _dones,
        _positive_count,
    ):
        self.optimization_calls += 1
        self.optimization_rewards.append(rewards.cpu().numpy().reshape(-1).tolist())


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
        self.assertEqual(biased.optimization_rewards, [[12.5]])
        self.assertEqual(unbiased.optimization_rewards, [[10.0]])

        self.assertIs(biased.memory, unbiased.memory)
        transition = biased.memory.buffer[0]
        self.assertAlmostEqual(transition[2], 12.5)
        self.assertAlmostEqual(transition[3], 10.0)
        self.assertTrue(transition[5])
        self.assertEqual(len(transition), 6)
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

        self.assertAlmostEqual(biased.memory.buffer[0][2], 12.0)
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
        self.assertEqual(biased.memory.buffer[0][2], 0.0)
        self.assertEqual(metrics["gamma_shaping"], 1.0)

    def test_unbiased_reward_scale_does_not_change_task_or_biased_reward(self):
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
                unbiased_reward_scale=0.1, log_interval=1,
            )

        transition = biased.memory.buffer[0]
        self.assertAlmostEqual(transition[2], 12.5)
        self.assertAlmostEqual(transition[3], 1.0)
        self.assertEqual(metrics["task_rewards"], [10.0])
        self.assertEqual(metrics["biased_learning_rewards"], [12.5])
        self.assertEqual(metrics["unbiased_learning_rewards"], [1.0])
        self.assertEqual(metrics["unbiased_reward_scale"], 0.1)


class ReplayBufferSamplingTest(unittest.TestCase):
    def test_prescribed_indices_define_the_minibatch_for_both_learners(self):
        buffer = ReplayBuffer(capacity=10, num_phases=0)
        for reward in (10.0, 20.0, 30.0):
            state = np.asarray([reward], dtype=np.float32)
            buffer.push(state, 0, reward, state, False)

        _states, _actions, rewards, _next_states, _dones = buffer.sample(
            2,
            [2, 0],
        )
        np.testing.assert_array_equal(rewards, np.asarray([30.0, 10.0]))

    def test_dual_buffer_stores_shared_transition_once_with_two_rewards(self):
        buffer = DualReplayBuffer(capacity=10, num_phases=0)
        state = np.asarray([1.0], dtype=np.float32)
        buffer.push(state, 2, 3.0, 0.25, state + 1.0, False)

        sampled = buffer.sample(1, [0])
        self.assertEqual(len(buffer), 1)
        self.assertEqual(len(buffer.buffer[0]), 6)
        self.assertEqual(sampled[2].tolist(), [3.0])
        self.assertEqual(sampled[3].tolist(), [0.25])


class _Space:
    def __init__(self, shape=None, n=None):
        self.shape = shape
        self.n = n


class _NetworkEnvironment:
    observation_space = _Space(shape=(3,))
    action_space = _Space(n=4)


class ZeroOutputInitializationTest(unittest.TestCase):
    def _build_learner(self, network_type):
        return HierarchicalDQNLearner(
            env=_NetworkEnvironment(),
            extra_state_dims=2,
            network_type=network_type,
        )

    def test_standard_output_and_target_are_zero_initialized_on_request(self):
        learner = self._build_learner("standard")
        learner.zero_initialize_output_layer()
        states = torch.randn(7, 5, device=learner.device)

        self.assertTrue(learner.output_layer_zero_initialized)
        self.assertTrue(torch.equal(learner.policy_net(states), torch.zeros(7, 4, device=learner.device)))
        self.assertTrue(torch.equal(learner.target_net(states), torch.zeros(7, 4, device=learner.device)))

    def test_dueling_outputs_and_target_are_zero_initialized_on_request(self):
        learner = self._build_learner("dueling")
        learner.zero_initialize_output_layer()
        states = torch.randn(7, 5, device=learner.device)

        self.assertTrue(learner.output_layer_zero_initialized)
        self.assertTrue(torch.equal(learner.policy_net(states), torch.zeros(7, 4, device=learner.device)))
        self.assertTrue(torch.equal(learner.target_net(states), torch.zeros(7, 4, device=learner.device)))


if __name__ == "__main__":
    unittest.main()
