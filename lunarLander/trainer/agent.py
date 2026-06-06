import numpy as np
import random as ran
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = ran.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class HierarchicalDQNLearner:
    def __init__(self, env, abstract_mdp, mapping_fn, max_episodes=1000, gamma=0.99, policy_name="policy", use_ddqn=False):
        self.env = env
        self.abstract_mdp = abstract_mdp
        self.mapping_fn = mapping_fn
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.policy_name = policy_name
        self.use_ddqn = use_ddqn
        self.algo_name = "DDQN" if use_ddqn else "TRUE_SINGLE_DQN"
        
        self.batch_size = 64
        self.lr = 1e-3
        self.tau = 0.005 
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995 
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        
        if self.use_ddqn:
            self.target_net = QNetwork(state_dim, action_dim).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(capacity=100000)

    def select_action(self, state):
        if ran.random() < self.eps:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size: return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            if self.use_ddqn:
                best_actions = self.policy_net(next_states).argmax(dim=1).unsqueeze(1)
                next_q_values = self.target_net(next_states).gather(1, best_actions)
            else:
                next_q_values = self.policy_net(next_states).max(1)[0].unsqueeze(1)
                
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.use_ddqn:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def train_normal(self):
        print(f"[{self.policy_name}] Starting Normal Training...")
        total_rewards = []
        for n_episode in range(self.max_episodes):
            s_raw, _ = self.env.reset()
            terminated = truncated = False
            episode_reward = 0
            while not (terminated or truncated):
                a = self.select_action(s_raw)
                ns_raw, reward, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                self.memory.push(s_raw, a, reward, ns_raw, done)
                self.optimize_model()
                s_raw = ns_raw
                episode_reward += reward
            self.eps = max(self.eps_min, self.eps * self.eps_decay)
            total_rewards.append(episode_reward)
            if (n_episode + 1) % 500 == 0:
                self._save_policy()
                print(f"[{self.policy_name}] Episode {n_episode+1}/{self.max_episodes} | Avg Reward (last 100): {np.mean(total_rewards[-100:]):.2f}")
        return np.array(total_rewards)

    def train_shaping(self):
        print(f"[{self.policy_name}] Starting Training with Reward Shaping...")
        self.abstract_mdp.value_iteration()
        K = 100
        total_rewards = []
        for n_episode in range(self.max_episodes):
            s_raw, _ = self.env.reset()
            terminated = truncated = False
            episode_reward = 0
            while not (terminated or truncated):
                a = self.select_action(s_raw)
                ns_raw, reward, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                
                phi_s = self.abstract_mdp.v_star[self.mapping_fn(s_raw)]
                
                # MODIFICA PAPER: Non Return-Invariant RS. 
                # Nessun azzeramento del potenziale agli stati terminali.
                phi_ns = self.abstract_mdp.v_star[self.mapping_fn(ns_raw)]
                
                shaping_signal = K * (self.gamma * phi_ns - phi_s)
                
                self.memory.push(s_raw, a, reward + shaping_signal, ns_raw, done)
                self.optimize_model()
                s_raw = ns_raw
                episode_reward += reward
            self.eps = max(self.eps_min, self.eps * self.eps_decay)
            total_rewards.append(episode_reward)
            if (n_episode + 1) % 500 == 0:
                self._save_policy()
                print(f"[{self.policy_name}] Episode {n_episode+1}/{self.max_episodes} | Avg Reward (last 100): {np.mean(total_rewards[-100:]):.2f}")
        return np.array(total_rewards)

    def train_goal_mdp(self):
        print(f"[{self.policy_name}] Starting Goal-MDP Training with Reward Shaping...")
        self.abstract_mdp.value_iteration()
        K = 100
        true_episode_rewards, goal_episode_rewards = [], []
        for n_episode in range(self.max_episodes):
            s_raw, _ = self.env.reset()
            terminated = truncated = False
            episode_true_reward = episode_goal_reward = 0
            while not (terminated or truncated):
                a = self.select_action(s_raw)
                ns_raw, original_reward, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                episode_true_reward += original_reward
                
                env_goal_reward = original_reward if terminated else 0.0
                
                if terminated and self.mapping_fn(ns_raw) == self.abstract_mdp.goal_state:
                    env_goal_reward += 100.0 # Dagli un vero motivo per restare a sinistra!

                episode_goal_reward += env_goal_reward

                phi_s = self.abstract_mdp.v_star[self.mapping_fn(s_raw)]
                phi_ns = self.abstract_mdp.v_star[self.mapping_fn(ns_raw)]

                shaping_signal = K * (self.gamma * phi_ns - phi_s)
                
                self.memory.push(s_raw, a, env_goal_reward + shaping_signal, ns_raw, done)
                self.optimize_model()
                s_raw = ns_raw
                
            self.eps = max(self.eps_min, self.eps * self.eps_decay)
            true_episode_rewards.append(episode_true_reward)
            goal_episode_rewards.append(episode_goal_reward)
            if (n_episode + 1) % 500 == 0:
                self._save_policy()
                print(f"[{self.policy_name}] Episode {n_episode+1}/{self.max_episodes} | Avg True Reward (last 100): {np.mean(true_episode_rewards[-100:]):.2f}")
        return np.array(true_episode_rewards), np.array(goal_episode_rewards)

    def _save_policy(self):
        os.makedirs("./policy", exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"./policy/{self.policy_name}")