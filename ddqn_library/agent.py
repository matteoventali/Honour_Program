"""Tianshou-based Double DQN agent used by the LTLf LunarLander trainer."""

# ==============================
# Standard library imports
# ==============================

import logging
import os
import random as ran

# ==============================
# External imports
# ==============================

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.algorithm import DQN
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Batch, ReplayBuffer
from tianshou.utils.torch_utils import policy_within_training_step

# Silence library diagnostics: the framework owns all user-facing logging.
logging.getLogger("tianshou").setLevel(logging.CRITICAL)

# ==============================
# Neural network
# ==============================

class QNetwork(nn.Module):
    """Q-network matching the manual implementation exactly: 128-128 MLP."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, observation, state=None, info=None):
        """Return action values and the unchanged recurrent state expected by Tianshou."""
        del info
        if not isinstance(observation, torch.Tensor):
            observation = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=self.fc1.weight.device,
            )
        else:
            observation = observation.to(
                device=self.fc1.weight.device,
                dtype=torch.float32,
            )
        hidden = F.relu(self.fc1(observation))
        hidden = F.relu(self.fc2(hidden))
        return self.fc3(hidden), state


# ==============================
# Replay-buffer adapter
# ==============================

class PhaseAwareReplayBuffer(ReplayBuffer):
    """Extend Tianshou's buffer with the original DFA-composition diagnostics."""

    def __init__(self, capacity, num_phases):
        if num_phases <= 0:
            raise ValueError("num_phases must be greater than zero")
        super().__init__(size=capacity)
        self.num_phases = num_phases
        self.phase_indices = np.full(capacity, -1, dtype=np.int64)
        self.phase_counts = np.zeros(num_phases, dtype=np.int64)

    def add(self, batch, buffer_ids=None):
        """Insert transitions through Tianshou and update DFA-state counts."""
        pointer, episode_return, episode_length, episode_start = super().add(
            batch,
            buffer_ids=buffer_ids,
        )
        slot = int(np.asarray(pointer).reshape(-1)[0])
        observation = np.asarray(batch.obs)
        if observation.ndim > 1:
            observation = observation[0]
        phase_index = int(np.argmax(observation[-self.num_phases:]))
        replaced_phase = int(self.phase_indices[slot])
        if replaced_phase >= 0:
            self.phase_counts[replaced_phase] -= 1
        self.phase_indices[slot] = phase_index
        self.phase_counts[phase_index] += 1
        return pointer, episode_return, episode_length, episode_start

    def sample_indices(self, batch_size):
        """Match the original uniform sampling without replacement."""
        if batch_size is None:
            batch_size = len(self)
        if batch_size <= 0:
            return super().sample_indices(batch_size)
        return np.asarray(ran.sample(range(len(self)), batch_size), dtype=np.int64)

    def q_fraction_onehot(self, q_index, num_phases):
        """Return the fraction of transitions collected in one DFA state."""
        if num_phases != self.num_phases:
            raise ValueError(
                f"Expected {self.num_phases} DFA states, received {num_phases}"
            )
        if not 0 <= q_index < self.num_phases:
            raise IndexError(f"DFA state index {q_index} is out of range")
        if len(self) == 0:
            return 0.0
        return float(self.phase_counts[q_index] / len(self))


# ==============================
# Tianshou DDQN learner
# ==============================

class HierarchicalDQNLearner:
    """Preserve the framework interface while delegating DDQN to Tianshou."""

    def __init__(
        self,
        env,
        abstract_mdp=None,
        max_episodes=1000,
        eps_decay=0.995,
        gamma=0.99,
        policy_name="policy",
        extra_state_dims=0,
        target_update_freq=1,
    ):
        if target_update_freq <= 0:
            raise ValueError("target_update_freq must be greater than zero for DDQN")

        self.env = env
        self.abstract_mdp = abstract_mdp
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.policy_name = policy_name
        self.algo_name = "DDQN"

        self.batch_size = 64
        self.lr = 1e-3
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = eps_decay

        # Append one feature for every DFA state, exactly as in the manual agent.
        state_dim = self.env.observation_space.shape[0] + extra_state_dims
        action_dim = self.env.action_space.n
        augmented_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        print(f"Using device:{self.device}")

        policy = DiscreteQLearningPolicy(
            model=self.policy_net,
            action_space=self.env.action_space,
            observation_space=augmented_observation_space,
            eps_training=self.eps,
            eps_inference=0.0,
        )
        self.algorithm = DQN(
            policy=policy,
            optim=AdamOptimizerFactory(lr=self.lr),
            gamma=self.gamma,
            n_step_return_horizon=1,
            target_update_freq=target_update_freq,
            is_double=True,
            huber_loss_delta=None,
        )
        self.memory = PhaseAwareReplayBuffer(
            capacity=300000,
            num_phases=extra_state_dims,
        )

    def select_action(self, state):
        """Select an epsilon-greedy action using Tianshou's DDQN policy."""
        self.algorithm.policy.set_eps_training(self.eps)
        batch = Batch(
            obs=np.asarray(state, dtype=np.float32)[None, :],
            info=np.asarray([{}], dtype=object),
        )
        with policy_within_training_step(self.algorithm.policy):
            result = self.algorithm.policy(batch)
            action = self.algorithm.policy.add_exploration_noise(result.act, batch)
        return int(np.asarray(action).reshape(-1)[0])

    def optimize_model(self):
        """Run one library DDQN update after the replay warm-up."""
        if len(self.memory) < self.batch_size:
            return None
        with policy_within_training_step(self.algorithm.policy):
            return self.algorithm.update(
            buffer=self.memory,
                sample_size=self.batch_size,
            )

    def _save_policy(self):
        """Save only the online Q-network for compatibility with evaluation."""
        os.makedirs("./policy", exist_ok=True)
        torch.save(
            self.policy_net.state_dict(),
            f"./policy/{self.policy_name}",
        )
