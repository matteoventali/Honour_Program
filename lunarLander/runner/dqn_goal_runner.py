import os
import glob
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

# Import the abstract components from your project
from abstract_mdps import ConfigurableDiagonalMDP
from agent import QNetwork
from utils import phi_mapping_grid

# =====================================================================
# EVALUATION LOGIC
# =====================================================================

def evaluate_policy(env, policy_path, abstract_mdp, episodes):
    """
    Evaluates a policy purely on the Custom Goal MDP logic (+100 for winning, 0 else).
    No reward shaping is applied during evaluation.
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    q_network = QNetwork(state_dim, action_dim).to(device)
    q_network.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    q_network.eval() 

    true_rewards = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0.0
        
        while not (terminated or truncated):
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action = q_network(state_tensor).argmax(dim=1).item()
            
            obs, _, terminated, truncated, _ = env.step(action)
            
            # --- CUSTOM REWARD LOGIC (Matches your training setup) ---
            abstract_ns = phi_mapping_grid(obs)
            if abstract_ns in abstract_mdp.goal_states:
                episode_reward += 100.0
                
        true_rewards.append(episode_reward)
        
    return true_rewards

# =====================================================================
# PLOTTING & EXPORT
# =====================================================================

def extract_metadata_from_filename(filename):
    """Parses p_1x1_099_100.pth into goal='1x1', gamma_raw='099', gamma_disp='0.99'"""
    clean_name = filename.replace("p_", "").replace(".pth", "")
    parts = clean_name.split('_')
    goal = parts[0] if len(parts) > 0 else "Unknown"
    gamma_raw = parts[1] if len(parts) > 1 else "Unknown"
    
    # FIX: Converte la stringa del nome file in un valore numerico pulito
    if gamma_raw == "099": gamma_disp = "0.99"
    elif gamma_raw == "09": gamma_disp = "0.9"
    elif gamma_raw == "08": gamma_disp = "0.8"
    else: gamma_disp = gamma_raw
    
    return goal, gamma_raw, gamma_disp

def plot_evaluation_curves(results_dict, base_dir="img/eval_plots", window_size=250, max_episodes=5000):
    """
    Creates separate Moving Average Curves for each (Goal, Gamma) configuration.
    Ensures consistent coloring across all plots based on the Shaping Reward value.
    """
    os.makedirs(base_dir, exist_ok=True)
    print("\n>>> Generating evaluation charts (grouped by Goal & Gamma)...")
    
    if not results_dict:
        print("   [!] No results to plot. Skipping chart generation.")
        return

    # 1. Dynamically find all unique goals, gammas, AND rewards for consistent coloring
    goals = set()
    gammas = {}
    shaping_rewards = set()
    
    for filename in results_dict.keys():
        g, gm_raw, gm_disp = extract_metadata_from_filename(filename)
        if g != "Unknown": goals.add(g)
        if gm_raw != "Unknown": gammas[gm_raw] = gm_disp
        
        # Extract the reward value to build the global color map
        if "baseline" not in filename:
            try:
                rew = filename.replace(".pth", "").split('_')[-1]
                shaping_rewards.add(float(rew))
            except ValueError:
                pass

    goals = sorted(list(goals))
    gammas_sorted = sorted(list(gammas.keys()), reverse=True) 
    
    # 2. Create a fixed global color palette for all shaping rewards
    sorted_rewards = sorted(list(shaping_rewards))
    cmap = plt.get_cmap('tab10') 
    color_map = {rew: cmap(i % 10) for i, rew in enumerate(sorted_rewards)}

    for goal in goals:
        for gamma_raw in gammas_sorted:
            gamma_disp = gammas[gamma_raw]
            
            # Filter policies that match exactly this Goal and Gamma
            target_prefix = f"p_{goal}_{gamma_raw}_"
            plot_results = {k: v for k, v in results_dict.items() if k.startswith(target_prefix)}
            
            if not plot_results:
                continue

            sorted_policies = sorted(plot_results.keys(), key=lambda k: np.mean(plot_results[k]))

            # ---------------------------------------------------------
            # GENERATE MOVING AVERAGE CURVE
            # ---------------------------------------------------------
            plt.figure(figsize=(12, 6))
            
            for policy_name in sorted_policies:
                rewards = plot_results[policy_name]
                
                if "baseline" in policy_name:
                    label = "Baseline (No Shaping)"
                    color = 'black'
                    linewidth = 1.5
                    zorder = 10
                else:
                    rew_str = policy_name.replace(".pth", "").split('_')[-1]
                    label = f"Shaping (Reward={rew_str})"
                    
                    try:
                        # Fetch the fixed color from our global color map
                        color = color_map[float(rew_str)]
                    except ValueError:
                        color = 'gray' # Fallback
                        
                    linewidth = 1.5
                    zorder = 5
                
                # Calculates the moving average
                if len(rewards) >= window_size:
                    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    x_axis = range(window_size - 1, len(rewards))
                    plt.plot(x_axis, moving_avg, color=color, linewidth=linewidth, zorder=zorder, label=label)
                else:
                    # Fallback if total episodes are less than the window
                    plt.plot(rewards, color=color, linewidth=linewidth, zorder=zorder, label=label)

            # FIX: Utilizza gamma_disp invece di 0.{gamma}
            plt.title(f"Rolling Win-Rate | Goal: {goal} | Gamma: {gamma_disp}", fontsize=16, fontweight='bold')
            plt.xlabel(f"Evaluation Episode # (Moving Avg Window = {window_size})", fontsize=12)
            plt.ylabel("Smoothed Reward (Win Rate %)", fontsize=12)
            plt.ylim(-5, 105)
            plt.xlim(0, max_episodes)

            # Horizontal line to indicate perfect victory (100)
            plt.axhline(y=100, color='black', linestyle='--', alpha=0.4, label='Max Possible (100)')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Places the legend outside the chart
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Policies")
            plt.tight_layout()
            
            curve_filename = os.path.join(base_dir, f"eval_curve_{goal}_g{gamma_raw}.png")
            plt.savefig(curve_filename, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"   [v] Saved curve plot for {goal} (Gamma {gamma_disp})")


# =====================================================================
# MAIN FUNCTION
# =====================================================================

def main():
    policy_dir = "policy"
    eval_episodes = 5000
    
    # Redefine the exact goals used during training to reconstruct the MDP logic
    goal_configurations = {
        "1x1_Strict": [(0,0)],
        "2x1_Base": [(0,0), (1,0)],
        "2x2_Wide": [(0,0), (1,0), (0,1), (1,1)]
    }
    
    policy_files = glob.glob(os.path.join(policy_dir, "*.pth"))
    
    if not policy_files:
        print(f"Error: No .pth files found in {policy_dir}.")
        return

    results = {}
    csv_data = [["Filename", "Goal", "Gamma", "Is_Baseline", "Shaping_Reward", "Win_Rate_%", "Avg_Reward"]]
    
    print(f"--- Starting Evaluation on Custom MDP ({eval_episodes} episodes per policy) ---")

    for policy_path in policy_files:
        filename = os.path.basename(policy_path)
        print(f"Evaluating -> {filename}")
        
        # Determine which goal configuration this policy belongs to
        matched_goal = None
        for goal_name in goal_configurations.keys():
            prefix = goal_name.split('_')[0]
            if prefix in filename:
                matched_goal = goal_name
                break
                
        if not matched_goal:
            print(f"   [!] Could not determine Goal for {filename}. Skipping.")
            continue
                
        # Reconstruct the MDP logic needed for evaluation
        abstract_mdp = ConfigurableDiagonalMDP(
            gamma=0.99, 
            goal_states=goal_configurations[matched_goal],
            goal_reward=100.0
        )
        
        env = gym.make("LunarLander-v3", continuous=False)
        rewards = evaluate_policy(env, policy_path, abstract_mdp, eval_episodes)
        env.close()
        
        results[filename] = rewards
        
        # Calculate statistics
        mean_rew = np.mean(rewards)
        win_rate = (sum(1 for r in rewards if r == 100.0) / len(rewards)) * 100
        print(f"   -> Win Rate: {win_rate:.1f}% | Avg Reward: {mean_rew:.2f}")
        
        # Prepare data for CSV using the formatted gamma
        goal_val, gamma_raw, gamma_disp = extract_metadata_from_filename(filename)
        is_baseline = "Yes" if "baseline" in filename else "No"
        shaping_rew = "N/A" if is_baseline == "Yes" else filename.replace(".pth", "").split('_')[-1]
        
        csv_data.append([filename, goal_val, gamma_disp, is_baseline, shaping_rew, f"{win_rate:.2f}", f"{mean_rew:.2f}"])

    print("\n--- Evaluation Finished ---")
    
    # Save statistics to CSV
    csv_path = "evaluation_metrics.csv"
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
    print(f">>> Metrics successfully saved to {csv_path}")
    
    # Generate grouped plots
    plot_evaluation_curves(results, max_episodes=eval_episodes)
    
if __name__ == "__main__":
    main()