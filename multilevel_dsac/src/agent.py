"""Discrete Soft Actor-Critic components for vector observations.

The update equations are adapted from CleanRL's ``sac_atari.py`` implementation
of SAC-Discrete.  Atari-specific convolutional networks and environment wrappers
are intentionally replaced by MLPs so the algorithm can operate directly on
LunarLander's vector observation and native discrete action space.

CleanRL source: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py
Copyright (c) 2019 CleanRL developers. Permission is hereby granted, free of
charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software,
subject to inclusion of this copyright and permission notice. THE SOFTWARE IS
PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical


def _mlp(input_dim, output_dim, hidden_dim=128):
    """Build the two-hidden-layer network used by actor and critics."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class SoftQNetwork(nn.Module):
    """Estimate one soft Q-value for every discrete action."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = _mlp(state_dim, action_dim, hidden_dim)

    def forward(self, observations):
        return self.network(observations)


class Actor(nn.Module):
    """Categorical policy over the environment's native discrete actions."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = _mlp(state_dim, action_dim, hidden_dim)

    def forward(self, observations):
        return self.network(observations)

    def get_action(self, observations):
        """Sample actions and return their full log-probability vectors."""
        logits = self(observations)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        log_probabilities = F.log_softmax(logits, dim=1)
        probabilities = distribution.probs
        return actions, log_probabilities, probabilities


class ReplayBuffer:
    """Fixed-size ring buffer with constant-time DFA-state accounting."""

    def __init__(self, capacity, state_dim, num_phases):
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        if state_dim <= 0:
            raise ValueError("state_dim must be greater than zero")
        if num_phases < 0:
            raise ValueError("num_phases cannot be negative")

        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.num_phases = int(num_phases)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.phase_indices = np.full(capacity, -1, dtype=np.int64)
        self.phase_counts = np.zeros(num_phases, dtype=np.int64)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """Insert one transition, replacing the oldest one when full."""
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if state.shape != (self.state_dim,) or next_state.shape != (self.state_dim,):
            raise ValueError(f"states must have shape ({self.state_dim},)")

        if self.size == self.capacity:
            replaced_phase = self.phase_indices[self.position]
            if replaced_phase >= 0:
                self.phase_counts[replaced_phase] -= 1

        phase_index = (
            int(np.argmax(state[-self.num_phases:]))
            if self.num_phases > 0
            else -1
        )
        self.states[self.position] = state
        self.next_states[self.position] = next_state
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.dones[self.position] = float(done)
        self.phase_indices[self.position] = phase_index
        if phase_index >= 0:
            self.phase_counts[phase_index] += 1

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        """Sample a uniformly random tensor batch."""
        if batch_size > self.size:
            raise ValueError("batch_size cannot exceed the replay-buffer size")
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": torch.as_tensor(self.states[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_states": torch.as_tensor(self.next_states[indices], device=device),
            "dones": torch.as_tensor(self.dones[indices], device=device),
        }

    def __len__(self):
        return self.size

    def q_fraction_onehot(self, q_index, num_phases):
        """Return the fraction of stored observations in one DFA state."""
        if num_phases != self.num_phases:
            raise ValueError(f"Expected {self.num_phases} DFA states, received {num_phases}")
        if not 0 <= q_index < self.num_phases:
            raise IndexError(f"DFA state index {q_index} is out of range")
        if self.size == 0:
            return 0.0
        return float(self.phase_counts[q_index] / self.size)


class DiscreteSACAgent:
    """CleanRL-style SAC-Discrete agent for low-dimensional observations."""

    def __init__(
        self,
        env,
        gamma=0.99,
        extra_state_dims=0,
        hidden_dim=128,
        buffer_size=300000,
        batch_size=64,
        learning_starts=20000,
        policy_lr=3e-4,
        q_lr=3e-4,
        update_frequency=4,
        target_network_frequency=8000,
        tau=1.0,
        alpha=0.2,
        autotune=True,
        target_entropy_scale=0.89,
        policy_dir="policy",
        device="auto",
    ):
        if not hasattr(env.action_space, "n"):
            raise TypeError("DiscreteSACAgent requires a discrete action space")
        if extra_state_dims < 0:
            raise ValueError("extra_state_dims cannot be negative")
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in the interval (0, 1]")
        if batch_size <= 0 or learning_starts < 0:
            raise ValueError("batch_size must be positive and learning_starts non-negative")
        if update_frequency <= 0 or target_network_frequency <= 0:
            raise ValueError("update frequencies must be greater than zero")
        if alpha <= 0.0:
            raise ValueError("alpha must be greater than zero")
        if target_entropy_scale <= 0.0:
            raise ValueError("target_entropy_scale must be greater than zero")

        self.env = env
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.update_frequency = int(update_frequency)
        self.target_network_frequency = int(target_network_frequency)
        self.tau = float(tau)
        self.autotune = bool(autotune)
        self.policy_dir = os.fspath(policy_dir)
        self.algo_name = "CleanRL SAC-Discrete"
        self.environment_steps = 0
        self.optimization_steps = 0
        self.last_losses = None

        state_dim = int(np.prod(env.observation_space.shape)) + extra_state_dims
        action_dim = int(env.action_space.n)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.qf1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.qf2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.qf1_target = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.qf2_target = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.qf1_target.eval()
        self.qf2_target.eval()

        # CleanRL uses eps=1e-4 for numerical stability in SAC-Discrete.
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=q_lr,
            eps=1e-4,
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=policy_lr, eps=1e-4
        )

        self.target_entropy = target_entropy_scale * float(np.log(action_dim))
        if self.autotune:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=q_lr, eps=1e-4)
            self.alpha = float(self.log_alpha.exp().item())
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.alpha = float(alpha)

        self.memory = ReplayBuffer(buffer_size, state_dim, extra_state_dims)
        print(f"Using device: {self.device}")

    def select_action(self, state, deterministic=False):
        """Use random warm-up, then sample from or greedily evaluate the actor."""
        if not deterministic and self.environment_steps < self.learning_starts:
            return self.env.action_space.sample()
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            logits = self.actor(state_tensor)
            if deterministic:
                return int(logits.argmax(dim=1).item())
            return int(Categorical(logits=logits).sample().item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.environment_steps += 1

    def optimize_model(self):
        """Run one CleanRL SAC-Discrete update when its schedule is due."""
        if (
            self.environment_steps <= self.learning_starts
            or len(self.memory) < self.batch_size
            or self.environment_steps % self.update_frequency != 0
        ):
            return None

        batch = self.memory.sample(self.batch_size, self.device)
        with torch.no_grad():
            _, next_log_pi, next_action_probs = self.actor.get_action(
                batch["next_states"]
            )
            next_q = torch.min(
                self.qf1_target(batch["next_states"]),
                self.qf2_target(batch["next_states"]),
            )
            soft_next_value = (
                next_action_probs * (next_q - self.alpha * next_log_pi)
            ).sum(dim=1)
            target_q = batch["rewards"] + (
                1.0 - batch["dones"]
            ) * self.gamma * soft_next_value

        qf1_values = self.qf1(batch["states"])
        qf2_values = self.qf2(batch["states"])
        action_indices = batch["actions"].long().unsqueeze(1)
        qf1_action_values = qf1_values.gather(1, action_indices).squeeze(1)
        qf2_action_values = qf2_values.gather(1, action_indices).squeeze(1)
        qf1_loss = F.mse_loss(qf1_action_values, target_q)
        qf2_loss = F.mse_loss(qf2_action_values, target_q)
        q_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        _, log_pi, action_probs = self.actor.get_action(batch["states"])
        with torch.no_grad():
            min_q = torch.min(self.qf1(batch["states"]), self.qf2(batch["states"]))
        actor_loss = (action_probs * (self.alpha * log_pi - min_q)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss_value = np.nan
        if self.autotune:
            alpha_loss = (
                action_probs.detach()
                * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = float(self.log_alpha.exp().item())
            alpha_loss_value = float(alpha_loss.item())

        self.optimization_steps += 1
        if self.environment_steps % self.target_network_frequency == 0:
            with torch.no_grad():
                for source, target in (
                    (self.qf1, self.qf1_target),
                    (self.qf2, self.qf2_target),
                ):
                    for parameter, target_parameter in zip(
                        source.parameters(), target.parameters()
                    ):
                        target_parameter.data.copy_(
                            self.tau * parameter.data
                            + (1.0 - self.tau) * target_parameter.data
                        )

        self.last_losses = {
            "qf1_loss": float(qf1_loss.item()),
            "qf2_loss": float(qf2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_value,
        }
        return dict(self.last_losses)

    def save(self, path):
        """Save the full training state and architecture metadata."""
        path = os.fspath(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "algorithm": "cleanrl_sac_discrete",
            "actor_state_dict": self.actor.state_dict(),
            "qf1_state_dict": self.qf1.state_dict(),
            "qf2_state_dict": self.qf2.state_dict(),
            "qf1_target_state_dict": self.qf1_target.state_dict(),
            "qf2_target_state_dict": self.qf2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "alpha": self.alpha,
            "autotune": self.autotune,
            "target_entropy": self.target_entropy,
            "environment_steps": self.environment_steps,
            "optimization_steps": self.optimization_steps,
            "state_dim": self.actor.network[0].in_features,
            "action_dim": self.actor.network[-1].out_features,
            "hidden_dim": self.actor.network[0].out_features,
        }
        if self.autotune:
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            checkpoint["alpha_optimizer_state_dict"] = (
                self.alpha_optimizer.state_dict()
            )
        torch.save(checkpoint, path)

    def _save_policy(self):
        """Compatibility helper used by the shared framework checkpoint logic."""
        self.save(os.path.join(self.policy_dir, self.policy_name))
