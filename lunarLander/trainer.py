import numpy as np
import gymnasium as gym
from collections import defaultdict
import random as ran
import pickle
import os
import matplotlib.pyplot as plt

# =====================================================================
# PLOTTING FUNCTION
# =====================================================================

def plot_training_results(rewards, window_size=100, shaping=False):
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
        plt.title('Hierarchical Q-Learning Training Results (With Reward Shaping)')
    else:
        plt.title('Hierarchical Q-Learning Training Results (No Reward Shaping)')
    
    plt.xlabel('Episode #')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

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
    Maps continuous LunarLander state to Abstract Grid State[cite: 145].
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
# HIERARCHICAL Q-LEARNER
# =====================================================================

class HierarchicalQLearner:
    def __init__(self, env, abstract_mdp, max_episodes=5000, alpha=0.1, gamma=0.99, policy_name="policy"):
        self.env = env
        self.abstract_mdp = abstract_mdp
        self.max_episodes = max_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1.0
        self.eps_decay = 0.9995
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.policy_name = policy_name

    def _discretize(self, obs):
        x_intervals     = [-0.5, 0.5]
        y_intervals     = [-0.1, 0.1, 1.5]
        vx_intervals    = [-7.5, -5, -0.3, -0.1, 0.1, 0.3, 5, 7.5]
        vy_intervals    = [-7.5, -5, -0.3, -0.1, 0.1, 0.3, 5, 7.5]
        theta_intervals = [-1.25663706,  -0.1, 0.1, 1.25663706]
        omega_intervals = [-7.5, -5, -0.1, 0.1, 5, 7.5]
        
        res = [
            np.digitize(obs[0], x_intervals), np.digitize(obs[1], y_intervals),
            np.digitize(obs[2], vx_intervals), np.digitize(obs[3], vy_intervals),
            np.digitize(obs[4], theta_intervals), np.digitize(obs[5], omega_intervals),
            int(obs[6]), int(obs[7])
        ]
        return tuple(res)

    def train_with_shaping(self):
        # Step 1: Solve Abstraction [cite: 197]
        print(f"Starting HRL Training with Reward Shaping...")
        self.abstract_mdp.value_iteration()
        
        K = 100
        total_rewards = []
        
        for n_episode in range(self.max_episodes):
            s_raw, _ = self.env.reset()
            s_disc = self._discretize(s_raw)
            
            terminated = truncated = False
            episode_reward = 0
            
            while not (terminated or truncated):
                # Epsilon-greedy action selection
                if ran.random() < self.eps:
                    a = self.env.action_space.sample()
                else:
                    a = np.argmax(self.q_table[s_disc])
                
                ns_raw, reward, terminated, truncated, _ = self.env.step(a)
                ns_disc = self._discretize(ns_raw)
                
                # REWARD SHAPING CALCULATION [cite: 123]
                # Phi(s) = V_star(phi(s))
                phi_s = self.abstract_mdp.v_star[phi_mapping(s_raw)]
                phi_ns = self.abstract_mdp.v_star[phi_mapping(ns_raw)]
                
                # F(s, a, s') = gamma * Phi(s') - Phi(s)
                shaping_signal = K * (self.gamma * phi_ns - phi_s)
                # Total shaped reward
                shaped_reward = reward + shaping_signal 
                
                # Q-Table Update
                best_next_q = np.max(self.q_table[ns_disc])
                self.q_table[s_disc][a] += self.alpha * (
                    shaped_reward + self.gamma * best_next_q - self.q_table[s_disc][a]
                )

                s_raw, s_disc = ns_raw, ns_disc
                episode_reward += reward

            self.eps = max(0.01, self.eps * self.eps_decay)
            total_rewards.append(episode_reward)

            # Saving the policy obtained
            if (n_episode + 1) % 1000 == 0:
                self._save_policy()

            if (n_episode + 1) % 500 == 0:
                print(f"Episode {n_episode+1} | Avg Reward: {np.mean(total_rewards[-100:]):.2f}")

        return np.array(total_rewards)

    def train(self):
        print(f"Starting Normal Training")
        
        total_rewards = []
        
        for n_episode in range(self.max_episodes):
            s_raw, _ = self.env.reset()
            s_disc = self._discretize(s_raw)
            
            terminated = truncated = False
            episode_reward = 0
            
            while not (terminated or truncated):
                # Epsilon-greedy action selection
                if ran.random() < self.eps:
                    a = self.env.action_space.sample()
                else:
                    a = np.argmax(self.q_table[s_disc])
                
                ns_raw, reward, terminated, truncated, _ = self.env.step(a)
                ns_disc = self._discretize(ns_raw)
                
                # Q-Table Update
                best_next_q = np.max(self.q_table[ns_disc])
                self.q_table[s_disc][a] += self.alpha * (
                    reward + self.gamma * best_next_q - self.q_table[s_disc][a]
                )

                s_raw, s_disc = ns_raw, ns_disc
                episode_reward += reward

            self.eps = max(0.01, self.eps * self.eps_decay)
            total_rewards.append(episode_reward)

            # Saving the policy obtained
            if (n_episode + 1) % 1000 == 0:
                self._save_policy()
            
            if (n_episode + 1) % 500 == 0:
                print(f"Episode {n_episode+1} | Avg Reward: {np.mean(total_rewards[-100:]):.2f}")
        
        return np.array(total_rewards)

    def _save_policy(self):
        os.makedirs("./policy", exist_ok=True)
        with open("./policy/" + self.policy_name, "wb") as f:
            pickle.dump(dict(self.q_table), f)
    


if __name__ == "__main__":
    env = gym.make("LunarLander-v3", continuous=False)
    #env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    
    # Create the abstract level
    abstract = AbstractGridMDP(width=12, height=12)
    abstractExtended = DiagonalAbstractGridMDP(width=12, height=12)
    
    # Training normally
    agent = HierarchicalQLearner(env, abstract, max_episodes=15000, policy_name="training_normally")
    rewards = agent.train()
    plot_training_results(rewards, window_size=600, shaping=False)

    # Training with shaping
    agent_shaping = HierarchicalQLearner(env, abstract, max_episodes=15000, policy_name="training_with_shaping")
    rewards_shaping = agent_shaping.train_with_shaping()
    plot_training_results(rewards_shaping, window_size=600, shaping=True)

    # Training with shaping and action space extended
    agent_shaping = HierarchicalQLearner(env, abstractExtended, max_episodes=15000, policy_name="training_with_shaping_extended")
    rewards_shaping = agent_shaping.train_with_shaping()
    plot_training_results(rewards_shaping, window_size=600, shaping=True)