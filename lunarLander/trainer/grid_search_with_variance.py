import os
import re
import pickle
import itertools
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from abstract_mdps import ConfigurableDiagonalMDP
from agent import HierarchicalDQNLearner
from utils import phi_mapping_grid, get_continuous_grid_coords, get_bilinear_potential

# =====================================================================
# PARTE 1: UTILS & PLOTTING STATISTICO
# =====================================================================

def moving_average(data, window_size):
    if len(data) < window_size: return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def get_smoothed_mean_and_std(runs_data, window_size):
    if isinstance(runs_data, list) or len(runs_data.shape) == 1:
        runs_data = np.array([runs_data])
    smoothed = np.array([moving_average(run, window_size) for run in runs_data])
    return np.mean(smoothed, axis=0), np.std(smoothed, axis=0)

def save_value_function_heatmap(abstract_mdp, filename, width=12, height=12, title="Potential Map V*"):
    print(f"   -> Generazione della mappa V* interpolata: {filename}")
    resolution = 20 
    x_c = np.linspace(0, width, width * resolution)
    y_c = np.linspace(0, height, height * resolution)
    Z = np.zeros((len(y_c), len(x_c)))
    
    for i, py in enumerate(y_c):
        for j, px in enumerate(x_c):
            Z[i, j] = get_bilinear_potential(px, py, abstract_mdp.v_star, width, height)
            
    plt.figure(figsize=(10, 9))
    im = plt.imshow(Z, cmap='viridis', origin='lower', extent=[0, width, 0, height], interpolation='none')
    plt.colorbar(im, label="Potential Value (V*) Interpolato")
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("X (Horizontal Position)", fontsize=13)
    plt.ylabel("Y (Altitude)", fontsize=13)
    plt.xticks(np.arange(0, width + 1, 1)); plt.yticks(np.arange(0, height + 1, 1))
    plt.grid(color='white', linestyle='-', linewidth=1, alpha=0.3)
    
    cx = [x + 0.5 for x in range(width) for _ in range(height)]
    cy = [y + 0.5 for _ in range(width) for y in range(height)]
    plt.scatter(cx, cy, color='black', s=8, alpha=0.6, label="Centri (V* vera)")
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_shaded_comparisons(results_dict, window_size=150, base_dir="img/shaded_plots"):
    os.makedirs(base_dir, exist_ok=True)
    goals = sorted(list(set([re.search(r'Goal:(.*?)\s\|', k).group(1) for k in results_dict.keys() if 'Goal:' in k])))
    gammas = sorted(list(set([re.search(r'Gamma:(.*?)\s\|', k).group(1) for k in results_dict.keys() if 'Gamma:' in k])))
    cmap = plt.get_cmap('Set1')

    for goal in goals:
        for gamma in gammas:
            plot_results = {k: v for k, v in results_dict.items() if f"Goal:{goal} |" in k and f"Gamma:{gamma} |" in k}
            if not plot_results: continue

            plt.figure(figsize=(11, 7))
            plt.title(f"Performance: {goal} | Gamma: {gamma}", fontsize=16, fontweight='bold')
            plt.axhline(y=100, color='black', linestyle=':', alpha=0.5, label='Win Threshold')
            plt.ylabel('Smoothed Episode Reward (Mean ± Std)', fontsize=12)
            plt.xlabel(f'Episode # (Moving Avg Window = {window_size})', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.4)
            
            for idx, (config_name, runs_data) in enumerate(plot_results.items()):
                if "Baseline" in config_name:
                    short_label, color, zorder = "Baseline (No Shaping)", 'black', 10
                else:
                    rew_val = re.search(r'Rew:(.*?)$', config_name).group(1) if 'Rew:' in config_name else "Unknown"
                    short_label, color, zorder = f"Shaping (Goal Reward: {rew_val})", cmap(idx % 9), 5
                
                mean, std = get_smoothed_mean_and_std(runs_data, window_size)
                x_axis = range(window_size - 1, window_size - 1 + len(mean))
                
                plt.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.15, zorder=zorder-1)
                plt.plot(x_axis, mean, color=color, linewidth=2.0, zorder=zorder, label=short_label)

            plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
            plt.tight_layout()
            filename = os.path.join(base_dir, f"shaded_{goal}_g{str(gamma).replace('.','')}.png")
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            plt.close() 


# =====================================================================
# PARTE 2: CORE TRAINING LOOP
# =====================================================================

def run_grid_search_training(env, agent, abstract_mdp, episodes, use_shaping=True):
    true_episode_rewards = []
    K = 100.0 / abstract_mdp.goal_reward if abstract_mdp.goal_reward > 0 else 1.0

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        terminated = truncated = False
        episode_true_reward = 0.0
        
        while not (terminated or truncated):
            a = agent.select_action(s_raw)
            ns_raw, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            
            abstract_ns = phi_mapping_grid(ns_raw)
            
            env_goal_reward = 0.0

            if abstract_ns in abstract_mdp.goal_states:
                env_goal_reward = 100.0
                terminated = True

            done = terminated or truncated

            episode_true_reward += env_goal_reward
                
            if use_shaping:
                px_s, py_s = get_continuous_grid_coords(s_raw, abstract_mdp.width, abstract_mdp.height)
                px_ns, py_ns = get_continuous_grid_coords(ns_raw, abstract_mdp.width, abstract_mdp.height)
                
                phi_s = get_bilinear_potential(px_s, py_s, abstract_mdp.v_star, abstract_mdp.width, abstract_mdp.height)
                phi_ns = get_bilinear_potential(px_ns, py_ns, abstract_mdp.v_star, abstract_mdp.width, abstract_mdp.height)
                
                shaping_signal = K * (agent.gamma * phi_ns - phi_s)
            else:
                shaping_signal = 0.0
            
            agent.memory.push(s_raw, a, env_goal_reward + shaping_signal, ns_raw, done)
            agent.optimize_model()
            s_raw = ns_raw

        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)
        
    return np.array(true_episode_rewards)

# =====================================================================
# PARTE 3: PIPELINE ORCHESTRATOR (MAIN)
# =====================================================================

def main():
    print("=== STARTING UNIFIED EXPERIMENT PIPELINE ===")
    os.makedirs("img/heatmaps", exist_ok=True)
    os.makedirs("img/shaded_plots", exist_ok=True)
    
    # --- IPERPARAMETRI GLOBALI ---
    NUM_SEEDS = 5
    EPISODES = 1000
    
    #goal_configs = {
    #    "2x2_Wide": [(0,0), (1,0), (0,1), (1,1)]
    #}
    #gammas = [0.99]
    #goal_rewards = [100.0]

    goal_configs = {
        "1x1_Strict": [(1,8)]
    }
    gammas = [0.99]
    goal_rewards = [100.0]
    
    results = {}
    combinations = list(itertools.product(goal_configs.items(), gammas, goal_rewards))

    # 1. RUN BASELINES (No Shaping)
    print("\n--- FASE 1: ADDESTRAMENTO BASELINES ---")
    for (goal_name, goal_states), gamma in list(itertools.product(goal_configs.items(), gammas)):
        config_name = f"Goal:{goal_name} | Gamma:{gamma} | Baseline"
        print(f"\n[Config] {config_name}")
        
        runs_data = []
        for seed in range(NUM_SEEDS):
            print(f"   -> Seed {seed+1}/{NUM_SEEDS}")
            env = gym.make("LunarLander-v3", continuous=False)
            np.random.seed(seed); torch.manual_seed(seed); env.reset(seed=seed)
            
            abstract_mdp = ConfigurableDiagonalMDP(gamma=gamma, goal_states=goal_states, goal_reward=1.0)
            abstract_mdp.value_iteration()
            
            agent = HierarchicalDQNLearner(env, abstract_mdp, phi_mapping_grid, max_episodes=EPISODES, use_ddqn=True)
            curve = run_grid_search_training(env, agent, abstract_mdp, EPISODES, use_shaping=False)
            runs_data.append(curve)
            env.close()
        results[config_name] = np.array(runs_data)

    # 2. RUN SHAPING EXPERIMENTS
    print("\n--- FASE 2: ADDESTRAMENTO CON SHAPING CONTINUO ---")
    for idx, ((goal_name, goal_states), gamma, g_rew) in enumerate(combinations):
        
        config_name = f"Goal:{goal_name} | Gamma:{gamma} | Rew:{g_rew}"
        heatmap_file = f"img/heatmaps/v_{goal_name.split('_')[0]}_g{str(gamma).replace('.','')}.png"
        print(f"\n[{idx+1}/{len(combinations)}] {config_name}")
        
        # Salva la heatmap
        abstract_mdp_temp = ConfigurableDiagonalMDP(gamma=gamma, goal_states=goal_states, goal_reward=g_rew)
        abstract_mdp_temp.value_iteration()
        save_value_function_heatmap(abstract_mdp_temp, filename=heatmap_file, title=f"V* | {goal_name} | G:{gamma}")

        runs_data = []
        for seed in range(NUM_SEEDS):
            print(f"   -> Seed {seed+1}/{NUM_SEEDS}")
            env = gym.make("LunarLander-v3", continuous=False)
            np.random.seed(seed); torch.manual_seed(seed); env.reset(seed=seed)
            
            abstract_mdp = ConfigurableDiagonalMDP(gamma=gamma, goal_states=goal_states, goal_reward=g_rew)
            abstract_mdp.value_iteration()
            
            agent = HierarchicalDQNLearner(env, abstract_mdp, phi_mapping_grid, max_episodes=EPISODES, use_ddqn=True)
            curve = run_grid_search_training(env, agent, abstract_mdp, EPISODES, use_shaping=True)
            runs_data.append(curve)
            env.close()
            
        results[config_name] = np.array(runs_data)

    # 3. PLOTTING
    plot_shaded_comparisons(results, window_size=150)
    print(">>> TUTTO COMPLETATO! Controlla le immagini nella cartella 'img/'")

if __name__ == "__main__":
    main()