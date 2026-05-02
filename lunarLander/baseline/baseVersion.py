# Authors: Matteo Ventali
# Baseline Version: Pure Tabular Q-Learning (No Hierarchical RL / Abstraction)

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from collections import defaultdict
import random as ran
import pickle
import os

# =====================================================================
# PLOTTING FUNCTION
# =====================================================================

def plot_training_results(rewards, window_size=100):
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
    
    plt.title('Baseline Q-Learning Training Results (No Abstraction)')
    plt.xlabel('Episode #')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# =====================================================================
# STATE DISCRETIZATION
# =====================================================================

def discretize(obs):
    """
    Discretizes the 8D continuous observation space into a finite set of bins
    to allow Tabular Q-Learning to work.
    """
    result = []
    
    x_intervals     = [-0.5, 0.5]
    y_intervals     = [-0.1, 0.1, 1.5]
    vx_intervals    = [-7.5, -5, -0.3, -0.1, 0.1, 0.3, 5, 7.5]
    vy_intervals    = [-7.5, -5, -0.3, -0.1, 0.1, 0.3, 5, 7.5]
    theta_intervals = [-1.25663706,  -0.1, 0.1, 1.25663706]
    omega_intervals = [-7.5, -5, -0.1, 0.1, 5, 7.5]
    
    result.append(np.digitize(obs[0], x_intervals))
    result.append(np.digitize(obs[1], y_intervals))
    result.append(np.digitize(obs[2], vx_intervals))
    result.append(np.digitize(obs[3], vy_intervals))
    result.append(np.digitize(obs[4], theta_intervals))
    result.append(np.digitize(obs[5], omega_intervals))

    # Booleans are already discrete
    result.append(int(obs[6]))
    result.append(int(obs[7]))
    return tuple(result)

# =====================================================================
# Q-LEARNER CLASS (BASELINE)
# =====================================================================

class QLearner():
    def __init__(self, env:gym.Env, max_episodes=5000, gamma=0.99, alpha=0.1, end_eps=0.01, start_eps=1.0, eps_decay=0.9995):
        self.env = env
        self.max_episodes = max_episodes        
        self.gamma = gamma
        self.alpha = alpha
        self.end_eps = end_eps
        self.eps = start_eps
        self.eps_decay = eps_decay
        
        # Initialize the Q-table
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def _epsilon_update(self):
        self.eps = max(self.eps_decay * self.eps, self.end_eps)

    def _next_action(self, state):
        if ran.random() < self.eps: 
            return self.env.action_space.sample() # Exploration
        else: 
            return np.argmax(self.q_table[state]) # Exploitation

    def train(self):
        print(f"Starting Baseline Q-Learning training for {self.max_episodes} episodes...")
        total_rewards = []
        
        for n_episode in range(self.max_episodes):
            s_raw, _ = self.env.reset()
            s_disc = discretize(s_raw)

            terminated = truncated = False
            episode_reward = 0
            
            while not (terminated or truncated): 
                # Action selection based purely on the physical discrete state
                a = self._next_action(s_disc)

                ns_raw, reward, terminated, truncated, _ = self.env.step(a)
                ns_disc = discretize(ns_raw)
                
                episode_reward += reward

                # Standard Q-table update (No Reward Shaping here!)
                self.q_table[s_disc][a] += self.alpha * (
                    reward + self.gamma * np.max(self.q_table[ns_disc]) - self.q_table[s_disc][a]
                )

                # Prepare for next iteration
                if not (terminated or truncated):
                    s_disc = ns_disc
            
            self._epsilon_update()
            total_rewards.append(episode_reward)
            
            # Print progress
            if (n_episode + 1) % 100 == 0:
                avg_reward = np.mean(total_rewards[-100:])
                print(f"Episode {n_episode + 1}/{self.max_episodes} | Epsilon: {self.eps:.3f} | Avg Reward (last 100): {avg_reward:.2f}")

        # Save the baseline policy
        os.makedirs("./policy", exist_ok=True)
        with open("./policy/policy_lunar_lander_baseline", "wb") as f:
            pickle.dump(dict(self.q_table), f)
        print("Training completed. Baseline policy saved in './policy/basePolicy'.")
        
        return np.array(total_rewards)

# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    
    # Initialize the baseline learner
    ql = QLearner(env, max_episodes=15000)
    
    # Train the agent
    rewards = ql.train()
    
    # Plot the results
    plot_training_results(rewards, window_size=100)