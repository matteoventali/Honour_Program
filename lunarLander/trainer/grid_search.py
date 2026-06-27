import gymnasium as gym
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import os
import re

# Import the configurable diagonal MDP and agent components
from abstract_mdps import ConfigurableDiagonalMDP
from utils import phi_mapping_grid, get_continuous_grid_coords, get_bilinear_potential
from agent import HierarchicalDQNLearner

# =====================================================================
# PLOTTING UTILITIES
# =====================================================================

def save_value_function_heatmap(abstract_mdp, filename, width=12, height=12, title="Potential Map V*"):
    """
    Converts the v_star dictionary into a 2D matrix and saves it as a Heatmap PNG.
    """
    v_matrix = np.zeros((height, width))
    
    for (x, y), value in abstract_mdp.v_star.items():
        if 0 <= x < width and 0 <= y < height:
            v_matrix[y, x] = value
            
    plt.figure(figsize=(9, 8))
    im = plt.imshow(v_matrix, cmap='viridis', origin='lower')
    
    for y in range(height):
        for x in range(width):
            val = v_matrix[y, x]
            if val != 0.0: 
                text_color = 'white' if val < (np.max(v_matrix) / 2) else 'black'
                plt.text(x, y, f"{val:.2f}", ha='center', va='center', 
                         color=text_color, fontsize=8, fontweight='bold')
                
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Potential Value (V*)")
    plt.title(title, fontsize=14)
    plt.xlabel("X (Horizontal Position)", fontsize=12)
    plt.ylabel("Y (Altitude / Distance from ground)", fontsize=12)
    
    plt.xticks(np.arange(0, width, 1))
    plt.yticks(np.arange(0, height, 1))
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, height, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1, alpha=0.4)
    ax.grid(which='major', color='none') 
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def save_value_function_heatmap_with_bilinear_interpolation(abstract_mdp, filename, width=12, height=12, title="Potential Map V*"):
    """
    Converts the v_star dictionary into a high-resolution 2D matrix using 
    bilinear interpolation and saves it as a smooth Heatmap PNG.
    """
    print(f"   -> Generazione della mappa V* interpolata in corso...")
    
    # 1. Imposta la risoluzione (es. 20 punti per ogni cella logica)
    # Per una griglia 12x12, genererà un'immagine di 240x240 pixel di dati puri
    resolution = 20 
    x_continuous = np.linspace(0, width, width * resolution)
    y_continuous = np.linspace(0, height, height * resolution)
    
    Z = np.zeros((len(y_continuous), len(x_continuous)))
    
    # 2. Calcola il potenziale interpolato per ogni micro-punto
    for i, py in enumerate(y_continuous):
        for j, px in enumerate(x_continuous):
            Z[i, j] = get_bilinear_potential(px, py, abstract_mdp.v_star, width, height)
            
    # 3. Disegna e salva l'immagine
    plt.figure(figsize=(10, 9))
    im = plt.imshow(Z, cmap='viridis', origin='lower', extent=[0, width, 0, height], interpolation='none')
    
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Potential Value (V*) Interpolato")
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("X (Horizontal Position)", fontsize=13)
    plt.ylabel("Y (Altitude / Distance from ground)", fontsize=13)
    
    # --- Disegna la griglia fisica (bordi delle celle) ---
    plt.xticks(np.arange(0, width + 1, 1))
    plt.yticks(np.arange(0, height + 1, 1))
    plt.grid(color='white', linestyle='-', linewidth=1, alpha=0.3)
    
    # --- Segna i centri delle celle (i veri valori V*) ---
    cx = [x + 0.5 for x in range(width) for _ in range(height)]
    cy = [y + 0.5 for _ in range(width) for y in range(height)]
    plt.scatter(cx, cy, color='black', s=8, alpha=0.6, label="Centri delle celle (V*)")
    
    plt.legend(loc="upper right", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close() # Libera la RAM

def save_grid_search_learning_curves(results_dict, base_dir="img/grid_search_plots", window_size=50):
    """
    Generates and saves a SEPARATE plot for each (Goal, Gamma) configuration, 
    displaying the smoothed moving average of the learning curves.
    This cleanly compares the shaping agent against its specific baseline.
    """
    os.makedirs(base_dir, exist_ok=True)
    print("\n>>> Generating and saving Individual Smoothed Learning Curves Plots...")
    
    # Extract unique goals and gammas from the dictionary keys
    goals = sorted(list(set([
        re.search(r'Goal:(.*?)\s\|', k).group(1) for k in results_dict.keys() if re.search(r'Goal:(.*?)\s\|', k)
    ])))
    gammas = sorted(list(set([
        re.search(r'Gamma:(.*?)\s\|', k).group(1) for k in results_dict.keys() if re.search(r'Gamma:(.*?)\s\|', k)
    ])))
    
    if not goals or not gammas:
        print("Error: Could not parse results keys for goals or gammas. Ensure the config_name formatting is unchanged.")
        return
        
    cmap = plt.get_cmap('Set1')

    for goal in goals:
        for gamma in gammas:
            # FIX: Aggiunto " |" per forzare il match esatto e prevenire che "0.9" faccia match con "0.99"
            plot_results = {k: v for k, v in results_dict.items() if f"Goal:{goal} |" in k and f"Gamma:{gamma} |" in k}
            
            # Skip if no data exists for this specific combination
            if not plot_results:
                continue

            plt.figure(figsize=(10, 6))
            
            plt.title(f"Performance: {goal} | Gamma: {gamma}", fontsize=16, fontweight='bold', pad=10)
            plt.axhline(y=100, color='black', linestyle=':', alpha=0.5, label='Win Threshold')
            plt.ylabel('Smoothed Episode Reward', fontsize=12)
            plt.xlabel(f'Episode # (Moving Avg Window = {window_size})', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.4)
            
            for idx, (config_name, rewards) in enumerate(plot_results.items()):
                
                if "Baseline" in config_name:
                    short_label = "Baseline (No Shaping)"
                    color = 'black'
                    linestyle = '-'
                    linewidth = 1.5
                    zorder = 10
                else:
                    rew_match = re.search(r'Rew:(.*)', config_name)
                    rew_val = rew_match.group(1).strip() if rew_match else "Unknown"
                    short_label = f"Shaping (Goal Reward: {rew_val})"
                    
                    color = cmap(idx % 9)
                    linestyle = '-'
                    linewidth = 1.5
                    zorder = 5
                
                if len(rewards) >= window_size:
                    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    x_axis = range(window_size - 1, len(rewards))
                    plt.plot(x_axis, moving_avg, color=color, linestyle=linestyle, 
                             linewidth=linewidth, zorder=zorder, label=short_label)
                else:
                    plt.plot(rewards, color=color, linestyle=linestyle, 
                             linewidth=linewidth, zorder=zorder, label=short_label)

            plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
            plt.tight_layout()
            
            gamma_str = str(gamma).replace('.', '')
            filename = os.path.join(base_dir, f"learning_curve_{goal}_g{gamma_str}.png")
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            plt.close() 
            
            print(f"   [v] Saved plot for {goal} (Gamma: {gamma}) -> {filename}")

# =====================================================================
# TRAINING LOOP
# =====================================================================

def run_grid_search_training(env, agent, abstract_mdp, episodes, use_shaping=True):
    """Optimized training loop for Grid Search execution."""
    true_episode_rewards = []
    
    K = 100.0 / abstract_mdp.goal_reward if abstract_mdp.goal_reward > 0 else 1.0

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        terminated = truncated = False
        episode_true_reward = 0.0
        
        while not (terminated or truncated):
            a = agent.select_action(s_raw)
            ns_raw, _, terminated, truncated, _ = env.step(a)
            
            abstract_s = phi_mapping_grid(s_raw)
            abstract_ns = phi_mapping_grid(ns_raw)
            
            env_goal_reward = 0.0
            
            if abstract_ns in abstract_mdp.goal_states:
                env_goal_reward = 100.0
                terminated = True

            done = terminated or truncated

            episode_true_reward += env_goal_reward
                
            # Shaping signal calculation
            #if use_shaping:
            #    phi_s = abstract_mdp.v_star[abstract_s]
            #    phi_ns = abstract_mdp.v_star[abstract_ns]
            #    shaping_signal = K * (agent.gamma * phi_ns - phi_s)
            #else:
            #    shaping_signal = 0.0

            # Shaping signal calculation
            if use_shaping:
                px_s, py_s = get_continuous_grid_coords(s_raw, abstract_mdp.width, abstract_mdp.height)
                px_ns, py_ns = get_continuous_grid_coords(ns_raw, abstract_mdp.width, abstract_mdp.height)
                
                phi_s = get_bilinear_potential(px_s, py_s, abstract_mdp.v_star, abstract_mdp.width, abstract_mdp.height)
                phi_ns = get_bilinear_potential(px_ns, py_ns, abstract_mdp.v_star, abstract_mdp.width, abstract_mdp.height)
                
                shaping_signal = K * (agent.gamma * phi_ns - phi_s)
            else:
                shaping_signal = 0.0
            
            total_step_reward = env_goal_reward + shaping_signal
            agent.memory.push(s_raw, a, total_step_reward, ns_raw, done)
            agent.optimize_model()

            s_raw = ns_raw

        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            print(f"-> Progress: Episode {n_episode + 1}/{episodes} | Recent 100-eps Avg Reward: {recent_avg:6.2f} | Epsilon: {agent.eps:.3f}")
            agent._save_policy()
            
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)
        
    return np.array(true_episode_rewards)

# =====================================================================
# MAIN GRID SEARCH
# =====================================================================

def main():
    print("=== STARTING EXPERIMENTS: Diagonal Abstract MDP ===")
    
    os.makedirs("img/grid_search_plots", exist_ok=True)
    
    #goal_configurations = {
    #    "1x1_Strict": [(0,0)],
    #    "2x1_Base": [(0,0), (1,0)],
    #    "2x2_Wide": [(0,0), (1,0), (0,1), (1,1)]
    #}

    goal_configurations = {
        "1x1_Strict": [(1,8)]
    }
    gammas = [0.99, 0.90, 0.80]
    goal_rewards = [1, 100.0]

    episodes_per_run = 1000
    results = {}

    # --- 1. BASELINE EXECUTION (Iterated over Gammas) ---
    print("\n--- RUNNING BASELINES (No Shaping) ---")
    baseline_combinations = list(itertools.product(goal_configurations.items(), gammas))
    
    for (goal_name, goal_states), gamma in baseline_combinations:
        config_name = f"Goal:{goal_name} | Gamma:{gamma} | Baseline"
        goal_prefix = goal_name.split('_')[0]
        gamma_str = str(gamma).replace('.', '')
        
        print(f"\nPreparing -> {config_name}")
        env = gym.make("LunarLander-v3", continuous=False)
        
        # Initialize abstract_mdp to pass gamma and map goals
        abstract_mdp = ConfigurableDiagonalMDP(gamma=gamma, goal_states=goal_states, goal_reward=1.0)
        abstract_mdp.value_iteration()
        
        agent = HierarchicalDQNLearner(
            env, abstract_mdp, phi_mapping_grid, 
            max_episodes=episodes_per_run, use_ddqn=True, 
            policy_name=f"p_{goal_prefix}_{gamma_str}_baseline.pth"
        )
        
        print(">>> Training Baseline in progress...")
        learning_curve = run_grid_search_training(env, agent, abstract_mdp, episodes_per_run, use_shaping=False)
        results[config_name] = learning_curve
        env.close()

    # --- 2. GRID SEARCH EXECUTION (With Shaping) ---
    print("\n--- STARTING SHAPING GRID SEARCH ---")
    combinations = list(itertools.product(goal_configurations.items(), gammas, goal_rewards))
    
    for idx, ((goal_name, goal_states), gamma, g_rew) in enumerate(combinations):
        
        goal_prefix = goal_name.split('_')[0] 
        gamma_str = str(gamma).replace('.', '')
        rew_str = str(int(g_rew))
        
        config_name = f"Goal:{goal_name} | Gamma:{gamma} | Rew:{g_rew}"
        heatmap_filename = f"img/grid_search_plots/v_{goal_prefix}_{gamma_str}_{rew_str}.png"
        
        print(f"\n[{idx+1}/{len(combinations)}] Preparing -> {config_name}")
        
        env = gym.make("LunarLander-v3", continuous=False)
        
        abstract_mdp = ConfigurableDiagonalMDP(
            gamma=gamma, 
            goal_states=goal_states, 
            goal_reward=g_rew
        )
        abstract_mdp.value_iteration()
        
        print(f">>> Saving V* map to: {heatmap_filename}")
        save_value_function_heatmap_with_bilinear_interpolation(abstract_mdp, filename=heatmap_filename, title=f"V* | {config_name}")
        
        agent = HierarchicalDQNLearner(
            env, abstract_mdp, phi_mapping_grid, 
            max_episodes=episodes_per_run, use_ddqn=True, 
            policy_name=f"p_{goal_prefix}_{gamma_str}_{rew_str}.pth"
        )
        
        print(">>> Training in progress...")
        # The argument use_shaping=True is the default value
        learning_curve = run_grid_search_training(env, agent, abstract_mdp, episodes_per_run, use_shaping=True)
        results[config_name] = learning_curve
        
        env.close()
        
    print("\n=== EXPERIMENTS COMPLETED ===")
    with open('grid_search_results.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    best_config = max(results, key=lambda k: np.mean(results[k][-100:]))
    best_score = np.mean(results[best_config][-100:])
    print(f"\nBEST CONFIGURATION: {best_config} (Final Avg: {best_score:.2f})")
    
    save_grid_search_learning_curves(results, window_size=200)
    print(f">>> All plots successfully saved in the 'img/grid_search_plots' directory.")


if __name__ == "__main__":
    main()