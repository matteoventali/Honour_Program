import os
import re
import itertools
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from abstract_mdps import ConfigurableDiagonalMDP
from agent import HierarchicalDQNLearner
from utils import phi_mapping_grid

# =====================================================================
# PART 1: STATISTICAL UTILS & PLOTTING
# =====================================================================

def moving_average_old(data, window_size):
    """
    Computes a centered moving average with the same length as the input.
    """
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="same")

def moving_average(data, window_size):
    """
    Calcola una media mobile centrata della stessa lunghezza dell'input,
    usando Pandas per gestire dinamicamente i bordi.
    """
    return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()

def get_smoothed_mean_and_std(runs_data, window_size):
    """
    Computes the mean and standard deviation across multiple runs after
    applying a moving average.
    """

    runs_data = np.asarray(runs_data)

    if runs_data.ndim == 1:
        runs_data = runs_data[np.newaxis, :]

    smoothed = np.array(
        [moving_average(run, window_size) for run in runs_data]
    )

    return (
        np.mean(smoothed, axis=0),
        np.std(smoothed, axis=0),
    )

def save_discrete_value_function_heatmap(abstract_mdp, filename, width=12, height=12, title="Discrete Potential Map V*"):
    print(f"   -> Generating discrete V* map: {filename}")
    Z = np.zeros((height, width))
    
    # Build the discrete matrix from the exact values of the V* dictionary
    for y in range(height):
        for x in range(width):
            Z[y, x] = abstract_mdp.v_star.get((x, y), 0.0)
            
    plt.figure(figsize=(10, 9))
    im = plt.imshow(Z, cmap='viridis', origin='lower', extent=[0, width, 0, height], interpolation='none')
    plt.colorbar(im, label="Potential Value (V*) Discreto")
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("X (Horizontal Position)", fontsize=13)
    plt.ylabel("Y (Altitude)", fontsize=13)
    plt.xticks(np.arange(0, width + 1, 1))
    plt.yticks(np.arange(0, height + 1, 1))
    plt.grid(color='white', linestyle='-', linewidth=2, alpha=0.5)
    
    # Write numerical values in the center of each cell (if greater than 0)
    for y in range(height):
        for x in range(width):
            val = Z[y, x]
            if val > 0.01:
                # Use black text for light backgrounds and white for dark backgrounds
                text_color = 'white' if val < np.max(Z) * 0.7 else 'black'
                plt.text(x + 0.5, y + 0.5, f"{val:.1f}", ha='center', va='center', color=text_color, fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_shaded_comparisons(results_dict, epsilon_dict, window_size=100, base_dir="img/shaded_plots"):

    os.makedirs(base_dir, exist_ok=True)

    goals = sorted(
        set(
            re.search(r"Goal:(.*?)\s\|", k).group(1)
            for k in results_dict
            if "Goal:" in k
        )
    )

    gammas = sorted(
        set(
            re.search(r"Gamma:(.*?)\s\|", k).group(1)
            for k in results_dict
            if "Gamma:" in k
        )
    )

    cmap = plt.get_cmap("Set1")

    for goal in goals:
        for gamma in gammas:

            plot_results = {
                k: v
                for k, v in results_dict.items()
                if f"Goal:{goal} |" in k
                and f"Gamma:{gamma} |" in k
            }

            if len(plot_results) == 0:
                continue

            fig, ax1 = plt.subplots(figsize=(10, 6))

            ax1.set_title(
                f"Learning Curves ({goal}, γ={gamma})",
                fontsize=15,
                fontweight="bold",
            )

            ax1.set_xlabel("Training Episode", fontsize=12)
            ax1.set_ylabel("Episode Return", fontsize=12)

            ax1.grid(True, linestyle="--", alpha=0.3)

            ax1.axhline(
                100,
                color="black",
                linestyle=":",
                linewidth=1.4,
                alpha=0.7,
                label="Goal reward",
            )

            for idx, (config_name, runs_data) in enumerate(plot_results.items()):

                if "Baseline" in config_name:
                    color = "black"
                    label = "Baseline"
                    zorder = 10
                else:
                    reward = re.search(r"Rew:(.*?)$", config_name).group(1)
                    color = cmap(idx % 9)
                    label = f"PBRS (Goal={reward})"
                    zorder = 5

                mean, std = get_smoothed_mean_and_std(
                    runs_data,
                    window_size,
                )

                x = np.arange(len(mean))

                ax1.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    alpha=0.18,
                    color=color,
                    zorder=zorder - 1,
                )

                ax1.plot(
                    x,
                    mean,
                    color=color,
                    linewidth=2.2,
                    label=label,
                    zorder=zorder,
                )

            # -------------------------------
            # Secondary axis (epsilon)
            # -------------------------------

            ax2 = ax1.twinx()

            ax2.set_ylabel(
                "Exploration rate (ε)",
                color="orange",
                fontsize=12,
            )

            first_config = next(iter(plot_results))

            eps = np.asarray(epsilon_dict[first_config])

            ax2.plot(
                np.arange(len(eps)),
                eps,
                linestyle="--",
                linewidth=1.6,
                color="orange",
                label="ε",
            )

            ax2.set_ylim(0, 1.05)
            ax2.tick_params(axis="y", labelcolor="orange")

            # -------------------------------
            # Merge legends
            # -------------------------------

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()

            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="lower right",
                framealpha=0.95,
                fontsize=10,
            )

            fig.tight_layout()

            filename = os.path.join(
                base_dir,
                f"comparison_{goal}_gamma_{str(gamma).replace('.', '_')}.png",
            )

            plt.savefig(
                filename,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

# =====================================================================
# PART 2: CORE TRAINING LOOP
# =====================================================================

def run_grid_search_training(env, agent, abstract_mdp, episodes, K=1, use_shaping=True):
    true_episode_rewards = []
    total_episode_rewards = []
    eps_history = []
    
    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        terminated = truncated = False
        episode_true_reward = 0.0 # Environment reward
        episode_total_reward = 0.0 # Environment reward + shaping reward
        
        # Loop for a single episode
        while not (terminated or truncated):
            a = agent.select_action(s_raw)
            ns_raw, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            
            # Computing the abstractions
            abstract_s = phi_mapping_grid(s_raw, abstract_mdp.width, abstract_mdp.height)
            abstract_ns = phi_mapping_grid(ns_raw, abstract_mdp.width, abstract_mdp.height)
            
            # Environment reward
            env_goal_reward = 0.0

            # If the agent has reached the goal state
            if abstract_ns in abstract_mdp.goal_states:
                env_goal_reward = abstract_mdp.goal_reward
                terminated = True # End the current episode
            done = terminated or truncated

            episode_true_reward += env_goal_reward

            shaping_signal = 0

            if use_shaping:
                #px_s, py_s = get_continuous_grid_coords(s_raw, abstract_mdp.width, abstract_mdp.height)
                #px_ns, py_ns = get_continuous_grid_coords(ns_raw, abstract_mdp.width, abstract_mdp.height)
                #
                #phi_s = get_bilinear_potential(px_s, py_s, abstract_mdp.v_star, abstract_mdp.width, abstract_mdp.height)
                #phi_ns = get_bilinear_potential(px_ns, py_ns, abstract_mdp.v_star, abstract_mdp.width, abstract_mdp.height)
                
                # Giving shaping only when the 2 abstract cells are different
                if abstract_ns != abstract_s:
                    phi_ns = abstract_mdp.v_star.get(abstract_ns, 0.0)
                    phi_s = abstract_mdp.v_star.get(abstract_s, 0.0)
                    shaping_signal = K * (agent.gamma * phi_ns - phi_s) # F(s,a,s') = gamma * phi(s') - phi(s)
                    
            total_reward = env_goal_reward + shaping_signal
            episode_total_reward += total_reward
            agent.memory.push(s_raw, a, total_reward, ns_raw, done)
            agent.optimize_model()
            s_raw = ns_raw
        
        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            mode_str = "SHAPING" if use_shaping else "BASELINE"
            print(
                f"[{mode_str}] Episode {n_episode + 1}/{episodes} | "
                f"Avg Reward : {recent_avg:.6f} | "
                f"Epsilon: {agent.eps:.6f}\n"
            )
            agent._save_policy()

        eps_history.append(agent.eps)
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)
        total_episode_rewards.append(episode_total_reward)
        
    return np.array(true_episode_rewards), np.array(total_episode_rewards), np.array(eps_history)

# =====================================================================
# PART 3: PIPELINE ORCHESTRATOR (MAIN)
# =====================================================================

def main():
    print("=== STARTING UNIFIED EXPERIMENT PIPELINE ===")
    os.makedirs("img/heatmaps", exist_ok=True)
    os.makedirs("img/shaded_plots", exist_ok=True)

    # --- GLOBAL HYPERPARAMETERS ---
    NUM_SEEDS = 1
    EPISODES = 2000
    
    goal_configs = {
        "1x1_Strict": [(1,8)]
    }
    gammas = [0.99]
    goal_rewards = [10000.0]
    
    results = {}
    epsilon_values = {}
    combinations = list(itertools.product(goal_configs.items(), gammas, goal_rewards))

    # 1. RUN BASELINE EXPERIMENTS (No Shaping)
    #print("\n--- PHASE 1: TRAINING BASELINES ---")
    #for (goal_name, goal_states), gamma in list(itertools.product(goal_configs.items(), gammas)):
    #    config_name = f"Goal:{goal_name} | Gamma:{gamma} | Baseline"
    #    policy_name = f"baseline_{goal_name}_g{str(gamma).replace('.','')}"
    #    print(f"\n[Config] {config_name}")
    #    
    #    runs_data = []
    #    for seed in range(NUM_SEEDS):
    #        print(f"   -> Seed {seed+1}/{NUM_SEEDS}")
    #        env = gym.make("LunarLander-v3", continuous=False)
    #        np.random.seed(seed); torch.manual_seed(seed); env.reset(seed=seed)
    #        
    #        abstract_mdp = ConfigurableDiagonalMDP(gamma=gamma, goal_states=goal_states, goal_reward=1.0)
    #        abstract_mdp.value_iteration()
    #        
    #        agent = HierarchicalDQNLearner(env, abstract_mdp, phi_mapping_grid, max_episodes=EPISODES, use_ddqn=True, policy_name=policy_name, gamma=gamma)
    #        curve, _, eps_history = run_grid_search_training(env, agent, abstract_mdp, EPISODES, use_shaping=False)
    #        runs_data.append(curve)
    #        env.close()
    #    epsilon_values[config_name] = eps_history
    #    results[config_name] = np.array(runs_data)

    # 2. RUN SHAPING EXPERIMENTS
    print("\n--- PHASE 2: TRAINING WITH CONTINUOUS SHAPING ---")
    for idx, ((goal_name, goal_states), gamma, g_rew) in enumerate(combinations):
        config_name = f"Goal:{goal_name} | Gamma:{gamma} | Rew:{g_rew}"
        discrete_heatmap_file = f"img/heatmaps/discrete_v_{goal_name.split('_')[0]}_g{str(gamma).replace('.','')}_r{g_rew}.png"
        policy_name = f"shaping_{goal_name}_g{str(gamma).replace('.','')}_r{g_rew}"
        print(f"\n[{idx+1}/{len(combinations)}] {config_name}")
        
        # Save the heatmap
        abstract_mdp_temp = ConfigurableDiagonalMDP(gamma=gamma, goal_states=goal_states, goal_reward=g_rew)
        abstract_mdp_temp.value_iteration()
        save_discrete_value_function_heatmap(abstract_mdp_temp, filename=discrete_heatmap_file, title=f"Discrete V* | {goal_name} | G:{gamma}")
        
        runs_data = []
        for seed in range(NUM_SEEDS):
            print(f"   -> Seed {seed+1}/{NUM_SEEDS}")
            env = gym.make("LunarLander-v3", continuous=False)
            np.random.seed(seed); torch.manual_seed(seed); env.reset(seed=seed)
            
            abstract_mdp = ConfigurableDiagonalMDP(gamma=gamma, goal_states=goal_states, goal_reward=g_rew)
            abstract_mdp.value_iteration()
            
            agent = HierarchicalDQNLearner(env, abstract_mdp, phi_mapping_grid, max_episodes=EPISODES, use_ddqn=True, policy_name=policy_name, gamma=gamma)
            curve, _, eps_history = run_grid_search_training(env, agent, abstract_mdp, EPISODES, use_shaping=True)
            runs_data.append(curve)
            env.close()
        
        epsilon_values[config_name] = eps_history
        results[config_name] = np.array(runs_data)

    # 3. PLOTTING
    plot_shaded_comparisons(results, epsilon_values, window_size=150)
    print(">>> ALL DONE! Check the images in the 'img/' folder.")

if __name__ == "__main__":
    main()