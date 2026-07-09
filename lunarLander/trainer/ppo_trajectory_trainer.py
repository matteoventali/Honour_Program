import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Make sure these imports match your local modules
from abstract_mdps import SequentialWaypointMDP
from utils import (phi_mapping_sequential, save_sequential_heatmaps)

# =====================================================================
# PLOTTING UTILITY ADATTATE PER PPO
# =====================================================================

def plot_shaping_reward_breakdown(true_rewards, total_rewards, window_size=100, filename="img/shaping_reward_breakdown_ppo.png"):
    """
    Plots the true environment reward vs the total reward (env + shaping) for PPO.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    if len(true_rewards) >= window_size:
        true_ma = pd.Series(true_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
        total_ma = pd.Series(total_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    else:
        true_ma = true_rewards
        total_ma = total_rewards
    x_axis = np.arange(len(true_rewards))
        
    ax1.plot(x_axis, true_ma, color='green', linestyle='-', linewidth=2, label='True Environment Reward')
    ax1.plot(x_axis, total_ma, color='purple', linestyle='-', linewidth=2.5, label='Total Reward (Env + Shaping)')
    
    ax1.set_title("PPO Shaping Agent Reward Analysis", fontsize=15, fontweight='bold')
    ax1.set_xlabel(f"Episode # (Moving Average Window = {window_size})", fontsize=12)
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax1.legend(loc="lower right", fontsize=11)
    
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n>>> PPO Shaping reward breakdown plot saved to: {filename}")
    plt.close(fig)

def plot_episode_phase_fractions(q0_history, q10_history, window_size=100, filename="img/phase_fractions_ppo.png"):
    """
    Shows the evolution of time spent in different phases (q=0 vs q=10) per episode for PPO.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    q0_ma = pd.Series(q0_history).rolling(window=window_size, min_periods=1, center=True).mean()
    q10_ma = pd.Series(q10_history).rolling(window=window_size, min_periods=1, center=True).mean()
    x_axis = np.arange(len(q0_history))
    
    ax.plot(x_axis, q0_ma, color='blue', linewidth=2.5, label='Time in Phase q=0 (Waypoint)')
    ax.plot(x_axis, q10_ma, color='green', linewidth=2.5, label='Time in Phase q=10 (Goal)')
    
    ax.set_title(f"Fraction of time per phase in each episode (Window = {window_size})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Fraction of Steps", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Ideal Equilibrium (50%)')
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="best", fontsize=11)
    
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n>>> Phase fractions plot saved to: {filename}")
    plt.close(fig)

# =====================================================================
# CUSTOM GYM ENVIRONMENT WRAPPER FOR PPO
# =====================================================================

class SequentialTaskWrapper(gym.Wrapper):
    """
    This Wrapper encapsulates the phase logic (q=0, q=10) and shaping
    inside the environment to make it compatible out-of-the-box with Stable Baselines3's PPO.
    """
    def __init__(self, env, abstract_mdp, use_shaping=True, K=1.0):
        super().__init__(env)
        self.abstract_mdp = abstract_mdp
        self.use_shaping = use_shaping
        self.K = K
        
        # We augment the observation space to include the one-hot q-state (2 dimensions)
        obs_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.append(obs_space.low, [0.0, 0.0]).astype(np.float32),
            high=np.append(obs_space.high, [1.0, 1.0]).astype(np.float32),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        s_raw, info = self.env.reset(**kwargs)
        self.current_s_raw = s_raw
        self.q = 0
        self.passed_through_waypoint = False
        
        # Episode metrics
        self.episode_true_reward = 0.0
        self.episode_total_reward = 0.0
        self.q0_steps = 0
        self.q10_steps = 0
        self.episode_waypoint_hits = 0
        self.episode_goal_hits = 0
        self.goal_reached = False
        
        return self._augment_state(s_raw), info

    def _augment_state(self, s_raw):
        q_one_hot = np.array([1.0, 0.0]) if self.q == 0 else np.array([0.0, 1.0])
        return np.concatenate((s_raw, q_one_hot)).astype(np.float32)

    def step(self, action):
        ns_raw, _, terminated, truncated, info = self.env.step(action)
        
        env_goal_reward = 0.0
        abstract_x_ns, abstract_y_ns, _ = phi_mapping_sequential(ns_raw, self.q)
        next_q = self.q

        # STATE TRANSITION LOGIC: Check for waypoint
        if abstract_x_ns == 1 and abstract_y_ns == 8 and self.q == 0:
            self.passed_through_waypoint = True
            self.episode_waypoint_hits += 1
            next_q = 10
            env_goal_reward = 0.0 
            
        abstract_ns = (abstract_x_ns, abstract_y_ns, next_q)

        # Check for final goal
        if abstract_ns == self.abstract_mdp.goal_state and next_q == 10 and self.passed_through_waypoint:
            env_goal_reward = 10000.0
            self.episode_goal_hits += 1
            self.goal_reached = True
            terminated = True
        
        # Shaping Signal Calculation (without gamma as in your most recent snippet)
        shaping_signal = 0.0
        if self.use_shaping:
            abstract_s = phi_mapping_sequential(self.current_s_raw, self.q)
            if abstract_s != abstract_ns:
                if (terminated or truncated) and not self.goal_reached:
                    phi_ns = 0.0
                else:
                    phi_ns = self.abstract_mdp.v_star.get(abstract_ns, 0.0)

                phi_s = self.abstract_mdp.v_star.get(abstract_s, 0.0)
                #phi_ns = self.abstract_mdp.v_star.get(abstract_ns, 0.0)
                shaping_signal = self.K * (self.abstract_mdp.gamma * phi_ns - phi_s)
        
        total_step_reward = env_goal_reward + shaping_signal
        
        # Update counters
        self.episode_true_reward += env_goal_reward
        self.episode_total_reward += total_step_reward
        if self.q == 0:
            self.q0_steps += 1
        else:
            self.q10_steps += 1
            
        # Prepare info for the callback
        if terminated or truncated:
            info['episode_true_reward'] = self.episode_true_reward
            info['episode_total_reward'] = self.episode_total_reward
            info['q0_steps'] = self.q0_steps
            info['q10_steps'] = self.q10_steps
            info['waypoint_hits'] = self.episode_waypoint_hits
            info['goal_hits'] = self.episode_goal_hits

        # Update for the next step
        self.current_s_raw = ns_raw
        self.q = next_q
        
        return self._augment_state(ns_raw), total_step_reward, terminated, truncated, info

# =====================================================================
# CALLBACK FOR LOGGING AND METRICS
# =====================================================================

class RewardLoggingCallback(BaseCallback):
    def __init__(self, use_shaping, verbose=0):
        super().__init__(verbose)
        self.true_episode_rewards = []
        self.total_episode_rewards = []
        self.q0_fractions = []
        self.q10_fractions = []
        self.waypoint_hits_history = []
        self.goal_hits_history = []
        self.episodes_count = 0
        self.use_shaping = use_shaping
        self.max_episodes = 15000 # A default large number

    def _on_step(self) -> bool:
        # PPO runs envs vectorized internally (even if it's just one)
        for idx, done in enumerate(self.locals.get("dones", [])): # self.locals["dones"]
            if done:
                info = self.locals["infos"][idx]
                if 'episode_true_reward' in info:
                    self.true_episode_rewards.append(info['episode_true_reward'])
                    self.total_episode_rewards.append(info['episode_total_reward'])
                    
                    tot_steps = info['q0_steps'] + info['q10_steps']
                    q0_frac = info['q0_steps'] / tot_steps if tot_steps > 0 else 0
                    q10_frac = info['q10_steps'] / tot_steps if tot_steps > 0 else 0
                    
                    self.q0_fractions.append(q0_frac)
                    self.q10_fractions.append(q10_frac)
                    self.waypoint_hits_history.append(info.get('waypoint_hits', 0))
                    self.goal_hits_history.append(info.get('goal_hits', 0))
                    
                    self.episodes_count += 1
                    
                    # Print every 100 episodes
                    if self.episodes_count % 100 == 0:
                        recent_avg = np.mean(self.true_episode_rewards[-100:])
                        recent_avg_shaping = np.mean(self.total_episode_rewards[-100:])
                        mode_str = "SHAPING" if self.use_shaping else "BASELINE"
                        
                        print(f"[{mode_str} PPO] Episode {self.episodes_count}")
                        print(f"  Avg Reward              : {recent_avg:.6f}")
                        print(f"  Avg With Shaping Reward : {recent_avg_shaping:.6f}")
                        print(f"  Avg Time in q0 % / q10% : {np.mean(self.q0_fractions[-100:]):.4f}, {np.mean(self.q10_fractions[-100:]):.4f}")
                        print(f"  Waypoint Hits (last 100): {np.sum(self.waypoint_hits_history[-100:])}")
                        print(f"  Goal Hits (last 100)    : {np.sum(self.goal_hits_history[-100:])}")
                        print("-" * 40)

                    # Save intermediate model every 500 episodes
                    if self.episodes_count > 0 and self.episodes_count % 2000 == 0:
                        os.makedirs("models", exist_ok=True)
                        save_path = f"models/ppo_shaping_sequential_policy_episode_{self.episodes_count}.zip"
                        self.model.save(save_path)
                        print(f"--- Intermediate model saved to {save_path} ---")
        
        # Stop training if the desired number of episodes is reached
        if self.episodes_count >= self.max_episodes:
            return False
        return True

# =====================================================================
# MAIN EXPERIMENT ORCHESTRATOR (PPO)
# =====================================================================

def main():
    print("=== STARTING SEQUENTIAL TASK EXPERIMENT: SHAPING WITH PPO ===")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("img", exist_ok=True)
    
    # HYPERPARAMETERS
    episodes = 50000
    gamma = 0.99
    K_scaling = 1
    
    print("\n1. Initializing Environment and Abstract MDP...")
    env = gym.make("LunarLander-v3", continuous=False)
    
    abstract_mdp = SequentialWaypointMDP(width=12, height=12, gamma=gamma)
    abstract_mdp.value_iteration()

    print("   -> Plotting Value Functions (V*) Heatmaps...")
    save_sequential_heatmaps(abstract_mdp, filename_prefix="seq_experiment")

    print("\n=======================================================")
    print("TRAINING: SHAPING AGENT (PPO)")
    print("=======================================================")
    
    # Wrap the environment in the Custom Wrapper to handle Q and Shaping
    wrapped_env = SequentialTaskWrapper(env, abstract_mdp, use_shaping=True, K=K_scaling)
    
    # Initialize PPO
    model = PPO(
        "MlpPolicy", 
        wrapped_env, 
        gamma=gamma, 
        verbose=0,
        learning_rate=3e-4, 
        n_steps=2048, # Number of steps to run for each environment per update
        batch_size=64,
        n_epochs=10
    )
    
    # Initialize the Callback for tracking
    logging_callback = RewardLoggingCallback(use_shaping=True)
    logging_callback.max_episodes = episodes
    
    # Run the Training
    model.learn(total_timesteps=float('inf'), callback=logging_callback)
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_shaping_sequential_policy_final.zip")

    # -----------------------------------------------------------------
    # PLOTTING RESULTS
    # -----------------------------------------------------------------
    print("\n3. Generating plots...")
    plot_shaping_reward_breakdown(
        logging_callback.true_episode_rewards, 
        logging_callback.total_episode_rewards, 
        window_size=100, 
        filename="img/shaping_reward_breakdown_ppo.png"
    )
    
    plot_episode_phase_fractions(
        logging_callback.q0_fractions, 
        logging_callback.q10_fractions, 
        window_size=100, 
        filename="img/phase_fractions_ppo.png"
    )
    
    env.close()

if __name__ == "__main__":
    main()