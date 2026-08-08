import numpy as np
import random as ran
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Classic shared Q-network receiving the physical state and DFA one-hot."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """Classic dueling Q-network receiving the augmented state."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
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


def _select_head_outputs(heads, features, head_indices):
    """Evaluate all heads and select one output row per batch element."""
    if head_indices.ndim != 1:
        raise ValueError("head_indices must be a one-dimensional tensor")
    if features.shape[0] != head_indices.shape[0]:
        raise ValueError("one head index is required for each batch element")
    if torch.any(head_indices < 0) or torch.any(head_indices >= len(heads)):
        raise IndexError("multi-head index is out of range")

    all_outputs = torch.stack(
        [head(features) for head in heads],
        dim=1,
    )
    batch_indices = torch.arange(features.shape[0], device=features.device)
    return all_outputs[batch_indices, head_indices]


class MultiHeadQNetwork(nn.Module):
    """Shared physical-state encoder with one Q-value head per DFA state."""

    def __init__(self, state_dim, action_dim, num_heads):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be greater than zero")
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            nn.Linear(128, action_dim) for _ in range(num_heads)
        )

    def forward(self, x, head_indices):
        features = self.feature(x)
        return _select_head_outputs(self.heads, features, head_indices)


class MultiHeadDuelingQNetwork(nn.Module):
    """Dueling DDQN variant with separate value/advantage heads per DFA state."""

    def __init__(self, state_dim, action_dim, num_heads):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be greater than zero")
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_heads = nn.ModuleList(
            nn.Linear(128, 1) for _ in range(num_heads)
        )
        self.advantage_heads = nn.ModuleList(
            nn.Linear(128, action_dim) for _ in range(num_heads)
        )

    def forward(self, x, head_indices):
        features = self.feature(x)
        value = _select_head_outputs(
            self.value_heads,
            features,
            head_indices,
        )
        advantages = _select_head_outputs(
            self.advantage_heads,
            features,
            head_indices,
        )
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

    def sample(self, batch_size):
        """Sample transitions efficiently from the indexable ring buffer."""
        batch = ran.sample(self.buffer, batch_size)
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
        network_architecture="multi-head",
        network_type="standard",
    ):
        if extra_state_dims <= 0:
            raise ValueError("DFA-guided DDQN requires at least one DFA state")
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in the interval (0, 1]")
        if target_update_freq <= 0:
            raise ValueError("target_update_freq must be greater than zero")
        if network_architecture not in {"classic", "multi-head"}:
            raise ValueError("network_architecture must be one of: classic, multi-head")
        if network_type not in {"standard", "dueling"}:
            raise ValueError("network_type must be one of: standard, dueling")

        self.env = env
        self.abstract_mdp = abstract_mdp
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.policy_name = policy_name
        self.network_architecture = network_architecture
        self.network_type = network_type
        base_name = "Dueling DDQN" if network_type == "dueling" else "DDQN"
        self.algo_name = (
            f"Multi-head {base_name}"
            if network_architecture == "multi-head"
            else f"Classic {base_name}"
        )
        self.num_heads = extra_state_dims
        self.observation_dim = self.env.observation_space.shape[0]
        
        self.batch_size = 64
        self.lr = 1e-3
        self.use_polyak = use_polyak
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.optimization_steps = 0
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = eps_decay
        
        # The classic network receives the augmented state. In the multi-head
        # architecture the trunk sees only the physical observation and the
        # one-hot is used exclusively to route samples to their heads.
        state_dim = (
            self.observation_dim + self.num_heads
            if network_architecture == "classic"
            else self.observation_dim
        )
        action_dim = self.env.action_space.n
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if network_architecture == "multi-head":
            network_cls = MultiHeadDuelingQNetwork if network_type == "dueling" else MultiHeadQNetwork
            network_args = (state_dim, action_dim, self.num_heads)
        else:
            network_cls = DuelingQNetwork if network_type == "dueling" else QNetwork
            network_args = (state_dim, action_dim)
        self.policy_net = network_cls(*network_args).to(self.device)
        architecture_details = (
            f"{self.num_heads} heads"
            if network_architecture == "multi-head"
            else "one shared output"
        )
        print(f"Using device:{self.device} | Architecture: {self.algo_name} ({architecture_details})")
        
        self.target_net = network_cls(*network_args).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(capacity=300000, num_phases=extra_state_dims)

    def _split_augmented_states(self, augmented_states):
        """Separate physical observations from their DFA head indices."""
        expected_width = self.observation_dim + self.num_heads
        if augmented_states.ndim != 2 or augmented_states.shape[1] != expected_width:
            raise ValueError(
                f"Expected augmented states with shape (batch, {expected_width})"
            )
        physical_states = augmented_states[:, :self.observation_dim]
        phase_one_hot = augmented_states[:, self.observation_dim:]
        head_indices = phase_one_hot.argmax(dim=1).long()
        return physical_states, head_indices

    def _network_values(self, network, augmented_states):
        """Evaluate either architecture through one common learner interface."""
        if self.network_architecture == "classic":
            return network(augmented_states)
        physical_states, head_indices = self._split_augmented_states(
            augmented_states
        )
        return network(physical_states, head_indices)

    def select_action(self, state):
        if ran.random() < self.eps:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                augmented_state = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                q_values = self._network_values(
                    self.policy_net,
                    augmented_state,
                )
                return q_values.argmax(dim=1).item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size: return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        augmented_states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        augmented_next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self._network_values(
            self.policy_net,
            augmented_states,
        ).gather(1, actions)
        
        with torch.no_grad():
            best_actions = self._network_values(
                self.policy_net,
                augmented_next_states,
            ).argmax(dim=1).unsqueeze(1)
            next_q_values = self._network_values(
                self.target_net,
                augmented_next_states,
            ).gather(1, best_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.optimization_steps += 1
        if self.use_polyak:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        elif self.optimization_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _save_policy(self):
        os.makedirs("./policy", exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"./policy/{self.policy_name}")
