import numpy as np
import gymnasium as gym
from collections import defaultdict, deque
import random as ran
import os
import matplotlib.pyplot as plt
import argparse
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# =====================================================================
# PLOTTING FUNCTION
# =====================================================================

def plot_training_results(rewards, window_size=100, shaping=False, algo="DQN", filename=None):
    """
    Plots the raw rewards and a moving average to show the learning trend.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot raw rewards with low opacity to see the variance
    plt.plot(rewards, color='lightgray', alpha=0.6, label='Raw Episode Reward')
    
    # Calculate and plot moving average to see the actual trend
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(rewards)), moving_avg, color='red', linewidth=2.5, label=f'Moving Average ({window_size} eps)')
    
    if shaping:
        plt.title(f'Hierarchical {algo} Training Results (With Reward Shaping)')
    else:
        plt.title(f'Hierarchical {algo} Training Results (No Reward Shaping)')
    
    plt.xlabel('Episode #')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    # Save or show the plot
    if filename:
        plt.savefig(filename)
        print(f"Plot saved to: {filename}")
    else:
        plt.show()
    plt.close()

# =====================================================================
# ABSTRACT MDP: GRID WORLD REPRESENTATION
# =====================================================================

class AbstractGridMDP:
    """
    Represents a simplified 2D grid of the LunarLander environment.
    The landing pad is always at coordinates (0,0) in the continuous space,
    which we map to the bottom-center of our grid.
    """
    def __init__(self, width=10, height=10, gamma=0.99):
        self.width = width
        self.height = height
        self.gamma = gamma
        self.states = [(x, y) for x in range(width) for y in range(height)]
        self.actions = [0, 1, 2, 3]  # Up, Down, Left, Right
        
        # Goal: Bottom center (landing pad area)
        self.goal_state = (width // 2, 0)
        self.v_star = defaultdict(float)

    def get_transitions(self, state, action):
        """Simple deterministic transitions for the grid."""
        x, y = state
        if action == 0: y = min(y + 1, self.height - 1) # Up
        elif action == 1: y = max(y - 1, 0)             # Down
        elif action == 2: x = max(x - 1, 0)             # Left
        elif action == 3: x = min(x + 1, self.width - 1) # Right
        
        next_state = (x, y)
        reward = 1.0 if next_state == self.goal_state else 0.0
        return next_state, reward

    def value_iteration(self, theta=0.001):
        print("Solving Abstract MDP with Value Iteration...")
        while True:
            delta = 0
            new_v = self.v_star.copy()
            for s in self.states:
                if s == self.goal_state: continue
                
                v_actions = []
                for a in self.actions:
                    ns, r = self.get_transitions(s, a)
                    v_actions.append(r + self.gamma * self.v_star[ns])
                
                best_v = max(v_actions)
                delta = max(delta, abs(best_v - self.v_star[s]))
                new_v[s] = best_v
            self.v_star = new_v
            if delta < theta: break
        print("Abstract Value Function computed.")

class DiagonalAbstractGridMDP(AbstractGridMDP):
    """
    Extension of AbstractGridMDP for diagonal movements.
    """
    def __init__(self, width=10, height=10, gamma=0.99):
        super().__init__(width, height, gamma)
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7]

    def get_transitions(self, state, action):
        x, y = state
        
        # Vertical movements
        if action in [0, 4, 5]:    # Up, Up-Left, Up-Right
            y = min(y + 1, self.height - 1)
        elif action in [1, 6, 7]:  # Down, Down-Left, Down-Right
            y = max(y - 1, 0)
            
        # Horizontal movements
        if action in [2, 4, 6]:    # Left, Up-Left, Down-Left
            x = max(x - 1, 0)
        elif action in [3, 5, 7]:  # Right, Up-Right, Down-Right
            x = min(x + 1, self.width - 1)
            
        next_state = (x, y)
        reward = 1.0 if next_state == self.goal_state else 0.0
        return next_state, reward

# =====================================================================
# STATE MAPPING (phi function)
# =====================================================================

def phi_mapping(obs, grid_w=10, grid_h=10):
    """
    Maps continuous LunarLander state to Abstract Grid State.
    obs[0]: x position (-1 to 1)
    obs[1]: y position (-0.5 to 1.5)
    """
    x, y = obs[0], obs[1]
    
    # Normalize and bin x from [-1, 1] to [0, grid_w-1]
    abstract_x = int(np.clip((x + 1) / 2 * (grid_w - 1), 0, grid_w - 1))
    # Normalize and bin y from [0, 1.5] to [0, grid_h-1]
    abstract_y = int(np.clip(y / 1.5 * (grid_h - 1), 0, grid_h - 1))
    
    return (abstract_x, abstract_y)

# =====================================================================
# DQN COMPONENTS
# =====================================================================

class QNetwork(nn.Module):
    """Neural Network Architecture for the DQN"""
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
    """Buffer to store past experiences"""
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

# =====================================================================
# HIERARCHICAL DQN/DDQN LEARNER
# =====================================================================

class HierarchicalDQNLearner:
    def __init__(self, env, abstract_mdp, max_episodes=1000, gamma=0.99, policy_name="policy", use_ddqn=False):
        self.env = env
        self.abstract_mdp = abstract_mdp
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.policy_name = policy_name
        self.use_ddqn = use_ddqn
        self.algo_name = "DDQN" if use_ddqn else "TRUE_SINGLE_DQN"
        
        # Hyperparameters
        self.batch_size = 64
        self.lr = 1e-3
        self.tau = 0.005 # For the target network soft update (only used in DDQN now)
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995 
        
        # Neural Networks Initialization
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize ONE network
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        
        # Initialize Target Network ONLY if using DDQN
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
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)
        
        # Compute Target Q
        with torch.no_grad():
            if self.use_ddqn:
                # DDQN: Select best action using policy_net, evaluate using target_net
                best_actions = self.policy_net(next_states).argmax(dim=1).unsqueeze(1)
                next_q_values = self.target_net(next_states).gather(1, best_actions)
            else:
                # TRUE SINGLE DQN: Compute max Q over next states directly from policy_net
                # (No target network used!)
                next_q_values = self.policy_net(next_states).max(1)[0].unsqueeze(1)
                
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Compute Loss and Backpropagation
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update for the Target Network (ONLY for DDQN)
        if self.use_ddqn:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def train_with_shaping(self):
        print(f"[{self.policy_name}] Starting HRL Training with Reward Shaping ({self.algo_name})...")
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
                
                # REWARD SHAPING CALCULATION
                phi_s = self.abstract_mdp.v_star[phi_mapping(s_raw)]
                phi_ns = self.abstract_mdp.v_star[phi_mapping(ns_raw)]
                
                shaping_signal = K * (self.gamma * phi_ns - phi_s)
                shaped_reward = reward + shaping_signal 
                
                # Save the experience in the replay buffer using SHAPED REWARD
                self.memory.push(s_raw, a, shaped_reward, ns_raw, done)
                
                # Network training step
                self.optimize_model()

                s_raw = ns_raw
                episode_reward += reward

            self.eps = max(self.eps_min, self.eps * self.eps_decay)
            total_rewards.append(episode_reward)

            if (n_episode + 1) % 500 == 0:
                self._save_policy()
                print(f"[{self.policy_name}] Episode {n_episode+1}/{self.max_episodes} | Avg Reward (last 100): {np.mean(total_rewards[-100:]):.2f} | Eps: {self.eps:.2f}")

        return np.array(total_rewards)

    def train(self):
        print(f"[{self.policy_name}] Starting Normal Training ({self.algo_name})...")
        total_rewards = []
        
        for n_episode in range(self.max_episodes):
            s_raw, _ = self.env.reset()
            terminated = truncated = False
            episode_reward = 0
            
            while not (terminated or truncated):
                a = self.select_action(s_raw)
                ns_raw, reward, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                
                # Save the experience using the original reward
                self.memory.push(s_raw, a, reward, ns_raw, done)
                
                self.optimize_model()

                s_raw = ns_raw
                episode_reward += reward

            self.eps = max(self.eps_min, self.eps * self.eps_decay)
            total_rewards.append(episode_reward)

            if (n_episode + 1) % 500 == 0:
                self._save_policy()
                print(f"[{self.policy_name}] Episode {n_episode+1}/{self.max_episodes} | Avg Reward (last 100): {np.mean(total_rewards[-100:]):.2f} | Eps: {self.eps:.2f}")
        
        return np.array(total_rewards)

    def _save_policy(self):
        os.makedirs("./policy", exist_ok=True)
        # Save PyTorch model weights (.pth) instead of the pickle dict
        torch.save(self.policy_net.state_dict(), f"./policy/{self.policy_name}")

# =====================================================================
# MAIN
# =====================================================================

def run_training(mode, episodes, use_ddqn):
    env = gym.make("LunarLander-v3", continuous=False)
    
    algo_str = "ddqn" if use_ddqn else "single_dqn"
    algo_display = "DDQN" if use_ddqn else "True Single DQN"
    
    if mode == "normal":
        abstract = AbstractGridMDP(width=12, height=12)
        policy_name = f"{algo_str}_normally.pth"
        agent = HierarchicalDQNLearner(env, abstract, max_episodes=episodes, policy_name=policy_name, use_ddqn=use_ddqn)
        rewards = agent.train()
        plot_training_results(rewards, window_size=100, shaping=False, algo=algo_display, filename=f"./img/{algo_str}_training_normally.png")
        
    elif mode == "shaping":
        abstract = AbstractGridMDP(width=12, height=12)
        policy_name = f"{algo_str}_shaping.pth"
        agent = HierarchicalDQNLearner(env, abstract, max_episodes=episodes, policy_name=policy_name, use_ddqn=use_ddqn)
        rewards = agent.train_with_shaping()
        plot_training_results(rewards, window_size=100, shaping=True, algo=algo_display, filename=f"./img/{algo_str}_training_with_shaping.png")
        
    elif mode == "extended":
        abstractExtended = DiagonalAbstractGridMDP(width=12, height=12)
        policy_name = f"{algo_str}_shaping_extended.pth"
        agent = HierarchicalDQNLearner(env, abstractExtended, max_episodes=episodes, policy_name=policy_name, use_ddqn=use_ddqn)
        rewards = agent.train_with_shaping()
        plot_training_results(rewards, window_size=100, shaping=True, algo=algo_display, filename=f"./img/{algo_str}_training_with_shaping_extended.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch HRL training for LunarLander using True Single DQN or DDQN.")
    parser.add_argument("--mode", type=str, choices=["normal", "shaping", "extended", "all"], default="normal", help="Choose which training mode to run")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of episodes for training")
    parser.add_argument("--parallel", action="store_true", help="Run all 3 models in parallel (ignores --mode)")
    parser.add_argument("--ddqn", action="store_true", help="Enable Double DQN (DDQN) architecture (Uses Target Network)")
    
    args = parser.parse_args()
    
    os.makedirs("./img", exist_ok=True)
    os.makedirs("./policy", exist_ok=True)

    algo = "DDQN" if args.ddqn else "True Single DQN"

    if args.parallel:
        print(f"--- Starting parallel {algo} training for {args.episodes} episodes ---")
        modes = ["normal", "shaping", "extended"]
        processes = []
        
        for m in modes:
            p = multiprocessing.Process(target=run_training, args=(m, args.episodes, args.ddqn))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        if args.mode == "all":
            print(f"--- Starting sequential {algo} training for {args.episodes} episodes ---")
            run_training("normal", args.episodes, args.ddqn)
            run_training("shaping", args.episodes, args.ddqn)
            run_training("extended", args.episodes, args.ddqn)
        else:
            print(f"--- Starting single {algo} training: {args.mode.upper()} for {args.episodes} episodes ---")
            run_training(args.mode, args.episodes, args.ddqn)