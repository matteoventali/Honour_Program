import os
import glob
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the abstract components from your project
from abstract_mdps import ConfigurableDiagonalMDP
from utils import phi_mapping_grid

# =====================================================================
# DQN ARCHITECTURE
# =====================================================================

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
            if terminated and abstract_ns in abstract_mdp.goal_states:
                episode_reward += 100.0
                
        true_rewards.append(episode_reward)
        
    return true_rewards

# =====================================================================
# PLOTTING
# =====================================================================

def plot_evaluation_boxplots_by_goal(results_dict, base_dir="img/eval_plots"):
    """
    Creates separate Boxplots for each Goal configuration.
    Safely handles empty results and updated Matplotlib syntax.
    """
    os.makedirs(base_dir, exist_ok=True)
    print("\n>>> Generating evaluation boxplots...")
    
    # Previene il crash se nessun file è stato valutato
    if not results_dict:
        print("   [!] No results to plot. Skipping chart generation.")
        return

    # Estrae i goal basandosi sulla presenza di "1x1", "2x1", "2x2" nel nome file
    base_goals = ["1x1", "2x1", "2x2"]
    goals = [g for g in base_goals if any(g in name for name in results_dict.keys())]

    if not goals:
        print("Warning: Could not parse goals from filenames. Falling back to a single plot.")
        goals = ["All_Policies"]
    
    for goal in goals:
        plt.figure(figsize=(10, 6))
        
        if goal == "All_Policies":
            goal_results = results_dict
        else:
            goal_results = {k: v for k, v in results_dict.items() if goal in k}
            
        # Ordina le policy dalla peggiore alla migliore per la visualizzazione
        sorted_policies = sorted(goal_results.keys(), key=lambda k: np.mean(goal_results[k]))
        data_to_plot = [goal_results[p] for p in sorted_policies]
        
        # Pulisce i nomi per il grafico (es. da "p_1x1_08_100.pth" a "1x1_08_100")
        clean_labels = [p.replace("p_", "").replace(".pth", "") for p in sorted_policies]
        
        # Gestisce la compatibilità con il nuovo parametro di Matplotlib (tick_labels)
        try:
            box = plt.boxplot(data_to_plot, patch_artist=True, tick_labels=clean_labels, vert=False)
        except TypeError:
            # Fallback per versioni di matplotlib più vecchie
            box = plt.boxplot(data_to_plot, patch_artist=True, labels=clean_labels, vert=False)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(clean_labels)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        for median in box['medians']:
            median.set(color='black', linewidth=2)

        plt.title(f"Evaluation Distribution: {goal}", fontsize=14, fontweight='bold')
        plt.xlabel("Total Episode Reward (Only +100 for Victory)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5, axis='x')
        
        plt.tight_layout()
        filename = os.path.join(base_dir, f"eval_boxplot_{goal}.png")
        plt.savefig(filename, dpi=200)
        plt.close()
        print(f"   [v] Saved plot for {goal} -> {filename}")

def plot_evaluation_curves_by_goal(results_dict, base_dir="img/eval_plots", window_size=250, max_episodes=1000):
    """
    Creates separate Moving Average curve plots for each Goal configuration
    during the evaluation phase.
    """
    os.makedirs(base_dir, exist_ok=True)
    print(f"\n>>> Generating evaluation moving average curves (Window={window_size})...")
    
    if not results_dict:
        print("   [!] No results to plot. Skipping chart generation.")
        return

    # Extract base goals dynamically based on filename presence
    base_goals = ["1x1", "2x1", "2x2"]
    goals = [g for g in base_goals if any(g in name for name in results_dict.keys())]

    if not goals:
        print("Warning: Could not parse goals from filenames. Falling back to a single plot.")
        goals = ["All_Policies"]
    
    cmap = plt.get_cmap('tab10')
    
    for goal in goals:
        plt.figure(figsize=(12, 6))
        
        if goal == "All_Policies":
            goal_results = results_dict
        else:
            goal_results = {k: v for k, v in results_dict.items() if goal in k}
            
        # Sort policies alphabetically for consistent legend
        sorted_policies = sorted(goal_results.keys())
        
        for idx, policy_name in enumerate(sorted_policies):
            rewards = goal_results[policy_name]
            clean_label = policy_name.replace("p_", "").replace(".pth", "")
            color = cmap(idx % 10)
            
            # Calcola la media mobile
            if len(rewards) >= window_size:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                x_axis = range(window_size - 1, len(rewards))
                plt.plot(x_axis, moving_avg, color=color, linewidth=2.0, alpha=0.9, label=clean_label)
            else:
                # Fallback se gli episodi totali sono meno della finestra
                plt.plot(rewards, color=color, linewidth=2.0, alpha=0.9, label=clean_label)

        plt.title(f"Evaluation Performance: {goal} (Rolling Win-Rate)", fontsize=16, fontweight='bold')
        plt.xlabel(f"Evaluation Episode # (Moving Avg Window = {window_size})", fontsize=12)
        plt.ylabel("Smoothed Reward (Win Rate %)", fontsize=12)
        

        plt.ylim(-5, 105)
        plt.xlim(0, max_episodes)

        # Linea orizzontale per indicare la vittoria perfetta (100)
        plt.axhline(y=100, color='black', linestyle='--', alpha=0.6, label='Max Possible Reward (100)')
        
        plt.grid(True, linestyle='--', alpha=0.5)
        # Posiziona la legenda fuori dal grafico per non coprire le linee
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Tested Policies")
        plt.tight_layout()
        
        filename = os.path.join(base_dir, f"eval_curve_{goal}.png")
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"   [v] Saved plot for {goal} -> {filename}")

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
    print(f"--- Starting Evaluation on Custom MDP ({eval_episodes} episodes per policy) ---")

    for policy_path in policy_files:
        filename = os.path.basename(policy_path)
        print(f"Evaluating -> {filename}")
        
        # Determine which goal configuration this policy belongs to based on its filename
        matched_goal = None
        for goal_name in goal_configurations.keys():
            # Estrae "1x1", "2x1", "2x2"
            prefix = goal_name.split('_')[0]
            
            # Verifica se "1x1" è CONTENUTO nel nome file (es. p_1x1_...)
            if prefix in filename:
                matched_goal = goal_name
                break
                
        if not matched_goal:
            print(f"   [!] Could not determine Goal for {filename}. Skipping.")
            continue
                
        # Reconstruct the MDP logic needed for evaluation
        abstract_mdp = ConfigurableDiagonalMDP(
            gamma=0.99, # Gamma doesn't matter for eval, only the goal_states matter
            goal_states=goal_configurations[matched_goal],
            goal_reward=100.0
        )
        
        env = gym.make("LunarLander-v3", continuous=False)
        rewards = evaluate_policy(env, policy_path, abstract_mdp, eval_episodes)
        env.close()
        
        results[filename] = rewards
        
        mean_rew = np.mean(rewards)
        win_rate = (sum(1 for r in rewards if r == 100.0) / len(rewards)) * 100
        print(f"   -> Win Rate: {win_rate:.1f}% | Avg Reward: {mean_rew:.2f}")

    print("\n--- Evaluation Finished ---")
    
    plot_evaluation_curves_by_goal(results, max_episodes=eval_episodes)
    
if __name__ == "__main__":
    main()