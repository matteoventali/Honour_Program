import os
import re
import itertools
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from abstract_mdps import ConfigurableDiagonalMDP
from utils import phi_mapping_grid

# =====================================================================
# PART 1: STATISTICAL UTILS & PLOTTING
# =====================================================================

def moving_average(data, window_size):
    """
    Computes a centered moving average of the same length as the input[cite: 1].
    """
    return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()

def get_smoothed_mean_and_std(runs_data, window_size):
    """
    Computes the mean and standard deviation across multiple runs after the moving average[cite: 1].
    """
    runs_data = np.asarray(runs_data)
    if runs_data.ndim == 1:
        runs_data = runs_data[np.newaxis, :]
    smoothed = np.array([moving_average(run, window_size) for run in runs_data])
    return np.mean(smoothed, axis=0), np.std(smoothed, axis=0)

def save_discrete_value_function_heatmap(abstract_mdp, filename, width=12, height=12, title="Discrete Potential Map V*"):
    print(f"   -> Generating discrete V* map: {filename}")
    Z = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            Z[y, x] = abstract_mdp.v_star.get((x, y), 0.0)
            
    plt.figure(figsize=(10, 9))
    im = plt.imshow(Z, cmap='viridis', origin='lower', extent=[0, width, 0, height], interpolation='none')
    plt.colorbar(im, label="Discrete Potential Value (V*)")
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("X (Horizontal Position)", fontsize=13)
    plt.ylabel("Y (Altitude)", fontsize=13)
    plt.xticks(np.arange(0, width + 1, 1))
    plt.yticks(np.arange(0, height + 1, 1))
    plt.grid(color='white', linestyle='-', linewidth=2, alpha=0.5)
    
    for y in range(height):
        for x in range(width):
            val = Z[y, x]
            if val > 0.01:
                text_color = 'white' if val < np.max(Z) * 0.7 else 'black'
                plt.text(x + 0.5, y + 0.5, f"{val:.1f}", ha='center', va='center', color=text_color, fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_shaded_comparisons(results_dict, window_size=100, base_dir="img/shaded_plots"):
    """
    Generates the final plots[cite: 1].
    NOTE: The secondary axis logic (Epsilon) has been removed, as PPO does not use epsilon.
    """
    os.makedirs(base_dir, exist_ok=True)
    goals = sorted(set(re.search(r"Goal:(.*?)\s\|", k).group(1) for k in results_dict if "Goal:" in k))
    gammas = sorted(set(re.search(r"Gamma:(.*?)\s\|", k).group(1) for k in results_dict if "Gamma:" in k))
    cmap = plt.get_cmap("Set1")

    for goal in goals:
        for gamma in gammas:
            plot_results = {k: v for k, v in results_dict.items() if f"Goal:{goal} |" in k and f"Gamma:{gamma} |" in k}
            if len(plot_results) == 0: continue

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_title(f"Learning Curves ({goal}, γ={gamma}) - PPO", fontsize=15, fontweight="bold")
            ax1.set_xlabel("Training Episode", fontsize=12)
            ax1.set_ylabel("Episode Return", fontsize=12)
            ax1.grid(True, linestyle="--", alpha=0.3)
            ax1.axhline(100, color="black", linestyle=":", linewidth=1.4, alpha=0.7, label="Goal reward")

            for idx, (config_name, runs_data) in enumerate(plot_results.items()):
                if "Baseline" in config_name:
                    color, label, zorder = "black", "Baseline", 10
                else:
                    reward = re.search(r"Rew:(.*?)$", config_name).group(1)
                    color, label, zorder = cmap(idx % 9), f"PBRS (Goal={reward})", 5

                mean, std = get_smoothed_mean_and_std(runs_data, window_size)
                x = np.arange(len(mean))
                ax1.fill_between(x, mean - std, mean + std, alpha=0.18, color=color, zorder=zorder - 1)
                ax1.plot(x, mean, color=color, linewidth=2.2, label=label, zorder=zorder)

            ax1.legend(loc="lower right", framealpha=0.95, fontsize=10)

            fig.tight_layout()
            filename = os.path.join(base_dir, f"ppo_comparison_{goal}_gamma_{str(gamma).replace('.', '_')}.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

# =====================================================================
# PART 2: WRAPPER & CALLBACK PER PPO
# =====================================================================

class AbstractShapingWrapper(gym.Wrapper):
    def __init__(self, env, abstract_mdp, use_shaping=True, gamma=0.99):
        super().__init__(env)
        self.abstract_mdp = abstract_mdp
        self.use_shaping = use_shaping
        self.gamma = gamma
        self.K = 100.0 / abstract_mdp.goal_reward if abstract_mdp.goal_reward > 0 else 1.0
        
        self.last_s_raw = None
        self.ep_true_reward = 0.0
        self.ep_total_reward = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_s_raw = obs
        self.ep_true_reward = 0.0
        self.ep_total_reward = 0.0
        return obs, info

    def step(self, action):
        ns_raw, _, terminated, truncated, info = self.env.step(action)
        
        abstract_s = phi_mapping_grid(self.last_s_raw, self.abstract_mdp.width, self.abstract_mdp.height)
        abstract_ns = phi_mapping_grid(ns_raw, self.abstract_mdp.width, self.abstract_mdp.height)
        
        # Goal Reward Override
        env_goal_reward = 0.0
        if abstract_ns in self.abstract_mdp.goal_states:
            env_goal_reward = 100.0
            terminated = True
            
        # Shaping Signal[cite: 2]
        shaping_signal = 0.0
        if self.use_shaping and abstract_ns != abstract_s:
            phi_ns = self.abstract_mdp.v_star.get(abstract_ns, 0.0)
            phi_s = self.abstract_mdp.v_star.get(abstract_s, 0.0)
            shaping_signal = self.K * (self.gamma * phi_ns - phi_s)
                
        total_reward = env_goal_reward + shaping_signal
        
        self.ep_true_reward += env_goal_reward
        self.ep_total_reward += total_reward
        self.last_s_raw = ns_raw
        
        # Save episode results for the Callback
        if terminated or truncated:
            info['episode_true_return'] = self.ep_true_reward
            info['episode_total_return'] = self.ep_total_reward
            
        return ns_raw, total_reward, terminated, truncated, info

class PPOCallback(BaseCallback):
    def __init__(self, max_episodes, config_name, policy_name, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.config_name = config_name
        self.true_episode_rewards = []
        self.policy_name = policy_name
        self.total_episode_rewards = []
        self.episodes_done = 0

    def _on_step(self) -> bool:
        done = self.locals.get("dones")[0] 
        infos = self.locals.get("infos")[0]
        
        if done:
            self.episodes_done += 1
            
            # Log stats
            self.true_episode_rewards.append(infos.get('episode_true_return', 0.0))
            self.total_episode_rewards.append(infos.get('episode_total_return', 0.0))
            
            if self.episodes_done % 100 == 0:
                recent_avg = np.mean(self.true_episode_rewards[-100:])
                print(f"[{self.config_name}] Episode {self.episodes_done}/{self.max_episodes} | Avg Reward (100): {recent_avg:.2f}")

            if self.episodes_done % 300 == 0:
                self.model.save(f"models/{self.policy_name}")
            
            if self.episodes_done >= self.max_episodes:
                return False 
        
        return True

# =====================================================================
# PART 3: PIPELINE ORCHESTRATOR (MAIN)
# =====================================================================

def main():
    print("=== STARTING PPO EXPERIMENT PIPELINE ===")
    os.makedirs("img/heatmaps", exist_ok=True)
    os.makedirs("img/shaded_plots", exist_ok=True)
    os.makedirs("models", exist_ok=True) 

    NUM_SEEDS = 5
    EPISODES = 1
    MAX_TIMESTEPS = EPISODES * 1500
    
    goal_configs = {
        "1x1_Strict": [(1,8)]
    }
    gammas = [0.99]
    goal_rewards = [100.0]
    
    results = {}
    combinations = list(itertools.product(goal_configs.items(), gammas, goal_rewards))

    # 1. RUN BASELINE EXPERIMENTS
    print("\n--- PHASE 1: TRAINING BASELINES (PPO) ---")
    for (goal_name, goal_states), gamma in list(itertools.product(goal_configs.items(), gammas)):
        config_name = f"Goal:{goal_name} | Gamma:{gamma} | Baseline"
        policy_name = f"ppo_baseline_{goal_name}_g{str(gamma).replace('.','')}"
        print(f"\n[Config] {config_name}")
        
        runs_data = []
        for seed in range(NUM_SEEDS):
            print(f"   -> Seed {seed+1}/{NUM_SEEDS}")
            env = gym.make("LunarLander-v3", continuous=False)
            
            abstract_mdp = ConfigurableDiagonalMDP(gamma=gamma, goal_states=goal_states, goal_reward=1.0)
            abstract_mdp.value_iteration()
            
            wrapped_env = AbstractShapingWrapper(env, abstract_mdp, use_shaping=False, gamma=gamma)
            
            model = PPO(
                "MlpPolicy",
                wrapped_env,
                gamma=gamma,
                seed=seed,
                learning_rate=3e-4, 
                n_steps=1024,
                batch_size=64,
                n_epochs=4,
                ent_coef=0.01,
                policy_kwargs=dict(net_arch=[128, 128]),
                verbose=0
            )
            
            callback = PPOCallback(max_episodes=EPISODES, config_name="PPO_BASELINE", policy_name=policy_name)
            model.learn(total_timesteps=MAX_TIMESTEPS, callback=callback)
            
            model.save(f"models/{policy_name}")
            runs_data.append(np.array(callback.true_episode_rewards))
            env.close()
            
        results[config_name] = np.array(runs_data)

    # 2. RUN SHAPING EXPERIMENTS
    print("\n--- PHASE 2: TRAINING WITH PBRS (PPO) ---")
    for idx, ((goal_name, goal_states), gamma, g_rew) in enumerate(combinations):
        config_name = f"Goal:{goal_name} | Gamma:{gamma} | Rew:{g_rew}"
        discrete_heatmap_file = f"img/heatmaps/discrete_v_{goal_name.split('_')[0]}_g{str(gamma).replace('.','')}_r{g_rew}.png"
        policy_name = f"ppo_shaping_{goal_name}_g{str(gamma).replace('.','')}_r{g_rew}"
        print(f"\n[{idx+1}/{len(combinations)}] {config_name}")
        
        abstract_mdp_temp = ConfigurableDiagonalMDP(gamma=gamma, goal_states=goal_states, goal_reward=g_rew)
        abstract_mdp_temp.value_iteration()
        save_discrete_value_function_heatmap(abstract_mdp_temp, filename=discrete_heatmap_file, title=f"Discrete V* | {goal_name} | G:{gamma}")
        
        runs_data = []
        for seed in range(NUM_SEEDS):
            print(f"   -> Seed {seed+1}/{NUM_SEEDS}")
            env = gym.make("LunarLander-v3", continuous=False)
            
            abstract_mdp = ConfigurableDiagonalMDP(gamma=gamma, goal_states=goal_states, goal_reward=g_rew)
            abstract_mdp.value_iteration()
            
            wrapped_env = AbstractShapingWrapper(env, abstract_mdp, use_shaping=True, gamma=gamma)
            
            model = PPO(
                "MlpPolicy",
                wrapped_env,
                gamma=gamma,
                seed=seed,
                learning_rate=3e-4, 
                n_steps=1024,
                batch_size=64,
                n_epochs=4,
                ent_coef=0.01,
                policy_kwargs=dict(net_arch=[128, 128]),
                verbose=0
            )
            
            callback = PPOCallback(max_episodes=EPISODES, config_name="PPO_SHAPING", policy_name=policy_name)
            model.learn(total_timesteps=MAX_TIMESTEPS, callback=callback)
            
            model.save(f"models/{policy_name}")
            runs_data.append(np.array(callback.true_episode_rewards))
            env.close()
        
        results[config_name] = np.array(runs_data)

    # 3. PLOTTING
    plot_shaded_comparisons(results, window_size=150)
    print(">>> ALL DONE! Check the images in the 'img/' folder.")

if __name__ == "__main__":
    main()