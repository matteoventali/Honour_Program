"""Dependency-light tests for the CleanRL-style SAC-Discrete components."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    import torch

    from agent import Actor, DiscreteSACAgent, ReplayBuffer, SoftQNetwork
except ModuleNotFoundError as dependency_error:
    np = torch = None
    Actor = DiscreteSACAgent = ReplayBuffer = SoftQNetwork = None
    IMPORT_ERROR = dependency_error
else:
    IMPORT_ERROR = None


class _ActionSpace:
    n = 4

    @staticmethod
    def sample():
        return 0


def _environment():
    return SimpleNamespace(
        observation_space=SimpleNamespace(shape=(8,)),
        action_space=_ActionSpace(),
    )


@unittest.skipIf(IMPORT_ERROR is not None, f"optional ML dependency unavailable: {IMPORT_ERROR}")
class NetworkTests(unittest.TestCase):
    def test_actor_and_critics_return_one_value_per_action(self):
        observations = torch.zeros((3, 11))
        actor = Actor(11, 4)
        critic = SoftQNetwork(11, 4)

        actions, log_probabilities, probabilities = actor.get_action(observations)
        self.assertEqual(actions.shape, (3,))
        self.assertEqual(log_probabilities.shape, (3, 4))
        self.assertEqual(critic(observations).shape, (3, 4))
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(3))


@unittest.skipIf(IMPORT_ERROR is not None, f"optional ML dependency unavailable: {IMPORT_ERROR}")
class ReplayBufferTests(unittest.TestCase):
    def test_ring_replacement_updates_dfa_counts(self):
        buffer = ReplayBuffer(capacity=2, state_dim=5, num_phases=2)
        q0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        q1 = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        buffer.push(q0, 0, 0.0, q1, False)
        buffer.push(q1, 1, 1.0, q0, False)
        self.assertEqual(buffer.q_fraction_onehot(0, 2), 0.5)
        self.assertEqual(buffer.q_fraction_onehot(1, 2), 0.5)

        buffer.push(q1, 2, 2.0, q1, True)
        self.assertEqual(buffer.q_fraction_onehot(0, 2), 0.0)
        self.assertEqual(buffer.q_fraction_onehot(1, 2), 1.0)


@unittest.skipIf(IMPORT_ERROR is not None, f"optional ML dependency unavailable: {IMPORT_ERROR}")
class AgentTests(unittest.TestCase):
    def test_agent_updates_and_writes_a_complete_checkpoint(self):
        torch.manual_seed(7)
        np.random.seed(7)
        agent = DiscreteSACAgent(
            _environment(),
            extra_state_dims=2,
            hidden_dim=16,
            buffer_size=16,
            batch_size=2,
            learning_starts=0,
            update_frequency=1,
            target_network_frequency=1,
            device="cpu",
        )
        state = np.zeros(10, dtype=np.float32)
        state[-2] = 1.0
        next_state = state.copy()
        next_state[-2:] = (0.0, 1.0)
        agent.store_transition(state, 0, 1.0, next_state, False)
        agent.store_transition(next_state, 1, 2.0, state, True)

        losses = agent.optimize_model()
        self.assertIsNotNone(losses)
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        self.assertEqual(agent.optimization_steps, 1)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "policy.pth"
            agent.save(checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        self.assertEqual(checkpoint["algorithm"], "cleanrl_sac_discrete")
        self.assertIn("actor_state_dict", checkpoint)
        self.assertIn("qf1_state_dict", checkpoint)
        self.assertIn("qf2_state_dict", checkpoint)
        self.assertEqual(checkpoint["state_dim"], 10)
        self.assertEqual(checkpoint["action_dim"], 4)


if __name__ == "__main__":
    unittest.main()
