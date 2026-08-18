import numpy as np
import random as ran
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingQNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_dim)

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return value + advantages - advantages.mean(dim=1, keepdim=True)

class ReplayBuffer:
    def __init__(self, capacity, num_phases):
        if num_phases < 0:
            raise ValueError("num_phases cannot be negative")
        self.capacity = capacity
        self.num_phases = num_phases
        self.buffer = []
        self.phase_indices = []
        self.phase_counts = np.zeros(num_phases, dtype=np.int64)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Insert a transition and update DFA-state counts in constant time."""
        transition = (state, action, reward, next_state, done)
        phase_index = (
            int(np.argmax(state[-self.num_phases:]))
            if self.num_phases > 0
            else None
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.phase_indices.append(phase_index)
        else:
            replaced_phase_index = self.phase_indices[self.position]
            if replaced_phase_index is not None:
                self.phase_counts[replaced_phase_index] -= 1
            self.buffer[self.position] = transition
            self.phase_indices[self.position] = phase_index

        if phase_index is not None:
            self.phase_counts[phase_index] += 1
        self.position = (self.position + 1) % self.capacity

    def sample_indices(self, batch_size, rng):
        """Draw stable buffer indices using an explicit random stream."""
        if batch_size > len(self.buffer):
            raise ValueError("batch_size cannot exceed the replay-buffer size")
        return rng.sample(range(len(self.buffer)), batch_size)

    def sample(self, batch_size, indices):
        """Read a prescribed minibatch from the indexable ring buffer."""
        if len(indices) != batch_size:
            raise ValueError("indices must contain exactly batch_size entries")
        if len(set(indices)) != len(indices):
            raise ValueError("minibatch indices must be unique")
        if any(index < 0 or index >= len(self.buffer) for index in indices):
            raise IndexError("minibatch index is outside the replay buffer")
        batch = [self.buffer[index] for index in indices]
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def q_fraction_onehot(self, q_index, num_phases):
        """Return a DFA-state fraction using incrementally maintained counts."""
        if num_phases != self.num_phases:
            raise ValueError(f"Expected {self.num_phases} DFA states, received {num_phases}")
        if not 0 <= q_index < self.num_phases:
            raise IndexError(f"DFA state index {q_index} is out of range")
        if len(self.buffer) == 0:
            return 0.0
        return float(self.phase_counts[q_index] / len(self.buffer))

class HierarchicalDQNLearner:
    def __init__(
        self,
        env,
        abstract_mdp=None,
        max_episodes=1000,
        eps_decay=0.995,
        gamma=0.99,
        policy_name="policy",
        extra_state_dims=0,
        use_polyak=True,
        tau=0.005,
        target_update_freq=1000,
        network_type="standard",
        policy_dir="policy",
        random_seed=None,
    ):
        if extra_state_dims < 0:
            raise ValueError("extra_state_dims cannot be negative")
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in the interval (0, 1]")
        if target_update_freq <= 0:
            raise ValueError("target_update_freq must be greater than zero")
        if network_type not in {"standard", "dueling"}:
            raise ValueError("network_type must be one of: standard, dueling")

        self.env = env
        self.abstract_mdp = abstract_mdp
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.policy_name = policy_name
        self.policy_dir = os.fspath(policy_dir)
        self.network_type = network_type
        self.algo_name = "Dueling DDQN" if network_type == "dueling" else "DDQN"
        self.exploration_rng = ran.Random(random_seed)
        replay_seed = None if random_seed is None else random_seed + 1
        self.replay_rng = ran.Random(replay_seed)
        
        self.batch_size = 64
        self.lr = 1e-3
        self.use_polyak = use_polyak
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.optimization_steps = 0
        self.reset_diagnostics()
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = eps_decay
        
        # Account for dynamic one-hot phases appended to state
        state_dim = self.env.observation_space.shape[0] + extra_state_dims
        action_dim = self.env.action_space.n
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network_cls = DuelingQNetwork if network_type == "dueling" else QNetwork
        self.policy_net = network_cls(state_dim, action_dim).to(self.device)
        print(f"Using device:{self.device}")
        
        self.target_net = network_cls(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(capacity=300000, num_phases=extra_state_dims)

    def select_action(self, state):
        if self.exploration_rng.random() < self.eps:
            return self.exploration_rng.randrange(self.env.action_space.n)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def optimize_model(self, batch_indices=None):
        if len(self.memory) < self.batch_size: return
        if batch_indices is None:
            batch_indices = self.memory.sample_indices(
                self.batch_size,
                self.replay_rng,
            )

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size,
            batch_indices,
        )
        positive_count = int(np.count_nonzero(rewards > 0))
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1).unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, best_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # Accumulate read-only diagnostics over the active logging window.  These
        # measurements deliberately do not alter sampling or optimisation. GPU
        # statistics are sampled every 100 updates to avoid synchronising the
        # device on every environment step.
        diagnostics = self._diagnostics
        diagnostics["updates"] += 1
        diagnostics["positive_samples"] += positive_count
        diagnostics["samples"] += rewards.numel()
        diagnostics["positive_batches"] += int(positive_count > 0)
        if self.optimization_steps % 100 == 0:
            gradient_squared_norm = sum(
                parameter.grad.detach().pow(2).sum()
                for parameter in self.policy_net.parameters()
                if parameter.grad is not None
            )
            loss_value = loss.detach().item()
            gradient_norm = gradient_squared_norm.sqrt().item()
            diagnostics["stats_updates"] += 1
            diagnostics["loss_sum"] += loss_value
            diagnostics["loss_max"] = max(diagnostics["loss_max"], loss_value)
            diagnostics["q_abs_sum"] += q_values.detach().abs().mean().item()
            diagnostics["q_abs_max"] = max(
                diagnostics["q_abs_max"], q_values.detach().abs().max().item()
            )
            diagnostics["target_abs_sum"] += target_q_values.detach().abs().mean().item()
            diagnostics["target_abs_max"] = max(
                diagnostics["target_abs_max"], target_q_values.detach().abs().max().item()
            )
            diagnostics["gradient_norm_sum"] += gradient_norm
            diagnostics["gradient_norm_max"] = max(
                diagnostics["gradient_norm_max"], gradient_norm
            )

        self.optimizer.step()
        
        self.optimization_steps += 1
        if self.use_polyak:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        elif self.optimization_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset_diagnostics(self):
        """Reset optimisation statistics accumulated for training logs."""
        self._diagnostics = {
            "updates": 0,
            "stats_updates": 0,
            "loss_sum": 0.0,
            "loss_max": 0.0,
            "q_abs_sum": 0.0,
            "q_abs_max": 0.0,
            "target_abs_sum": 0.0,
            "target_abs_max": 0.0,
            "gradient_norm_sum": 0.0,
            "gradient_norm_max": 0.0,
            "positive_samples": 0,
            "samples": 0,
            "positive_batches": 0,
        }

    def consume_diagnostics(self):
        """Return window averages and start a fresh diagnostic window."""
        diagnostics = self._diagnostics
        updates = diagnostics["updates"]
        stats_updates = diagnostics["stats_updates"]
        samples = diagnostics["samples"]
        result = {
            "updates": updates,
            "stats_updates": stats_updates,
            "mean_loss": diagnostics["loss_sum"] / stats_updates if stats_updates else 0.0,
            "max_loss": diagnostics["loss_max"],
            "mean_abs_q": diagnostics["q_abs_sum"] / stats_updates if stats_updates else 0.0,
            "max_abs_q": diagnostics["q_abs_max"],
            "mean_abs_target": diagnostics["target_abs_sum"] / stats_updates if stats_updates else 0.0,
            "max_abs_target": diagnostics["target_abs_max"],
            "mean_gradient_norm": diagnostics["gradient_norm_sum"] / stats_updates if stats_updates else 0.0,
            "max_gradient_norm": diagnostics["gradient_norm_max"],
            "positive_sample_fraction": diagnostics["positive_samples"] / samples if samples else 0.0,
            "positive_batch_fraction": diagnostics["positive_batches"] / updates if updates else 0.0,
        }
        self.reset_diagnostics()
        return result

    def _save_policy(self):
        os.makedirs(self.policy_dir, exist_ok=True)
        torch.save(
            self.policy_net.state_dict(),
            os.path.join(self.policy_dir, self.policy_name),
        )
