import pickle
import numpy as np
import re
import os
import gymnasium as gym
import matplotlib.pyplot as plt

# Import your project components
from abstract_mdps import ConfigurableDiagonalMDP
from utils import phi_mapping_grid
from agent import HierarchicalDQNLearner

# =====================================================================
# CONFIGURATION PARSER
# =====================================================================

def parse_configuration_string(config_str):
    """
    Parses the string (e.g., 'Goal:2x2_Wide | Gamma:0.9 | Rew:100.0') 
    back into usable Python variables.
    """
    # Define the mapping for the goal states
    goal_map = {
        "1x1_Strict": [(0, 0)],
        "2x1_Base": [(0, 0), (1, 0)],
        "2x2_Wide": [(0, 0), (1, 0), (0, 1), (1, 1)]
    }
    
    # Extract values using Regular Expressions
    goal_match = re.search(r"Goal:([a-zA-Z0-9_]+)", config_str)
    gamma_match = re.search(r"Gamma:([0-9.]+)", config_str)
    rew_match = re.search(r"Rew:([0-9.]+)", config_str)
    
    if not (goal_match and gamma_match and rew_match):
        raise ValueError(f"Could not parse configuration string: {config_str}")
        
    goal_name = goal_match.group(1)
    gamma = float(gamma_match.group(1))
    goal_reward = float(rew_match.group(1))
    
    goal_states = goal_map.get(goal_name, [(0,0)])
    
    return goal_name, goal_states, gamma, goal_reward

# =====================================================================
# TRAINING ENGINE
# =====================================================================

def run_targeted_training(env, agent, abstract_mdp, episodes):
    """Executes the training loop for the selected configuration."""
    true_episode_rewards = []
    
    # Dynamic K scaling to prevent gradient explosion
    K = 100.0 / abstract_mdp.goal_reward

    print(f"\n>>> Starting Training for {episodes} episodes...")
    
    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        terminated = truncated = False
        episode_true_reward = 0.0
        
        while not (terminated or truncated):
            a = agent.select_action(s_raw)
            ns_raw, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            
            abstract_s = phi_mapping_grid(s_raw)
            abstract_ns = phi_mapping_grid(ns_raw)
            
            env_goal_reward = 0.0
            
            if terminated and abstract_ns in abstract_mdp.goal_states:
                env_goal_reward = 100.0
                
            episode_true_reward += env_goal_reward
                
            phi_s = abstract_mdp.v_star[abstract_s]
            phi_ns = 0.0 if done else abstract_mdp.v_star[abstract_ns]
            
            shaping_signal = K * (agent.gamma * phi_ns - phi_s)
            
            total_step_reward = env_goal_reward + shaping_signal
            agent.memory.push(s_raw, a, total_step_reward, ns_raw, done)
            agent.optimize_model()
            
            s_raw = ns_raw
            
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)
        
        # Live Progress Logging
        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            print(f"-> Progress: Episode {n_episode + 1}/{episodes} | Recent 100-eps Avg: {recent_avg:6.2f} | Epsilon: {agent.eps:.3f}")
            agent._save_policy()
            
    return true_episode_rewards

def plot_targeted_learning_curve(rewards, config_name):
    """Plots and saves the learning curve for the targeted training session."""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='lightgray', alpha=0.6, label='Raw Episode Reward')
    
    window_size = 100
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(rewards)), moving_avg, color='blue', linewidth=2.5, label=f'Moving Avg ({window_size} eps)')
    
    plt.title(f"Targeted Training\n{config_name}", fontsize=14)
    plt.xlabel('Episode #')
    plt.ylabel('True Episode Reward')
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Win Threshold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    os.makedirs("targeted_runs", exist_ok=True)
    safe_name = config_name.replace(" | ", "_").replace(":", "").replace(".", "")
    plt.savefig(f"targeted_runs/curve_{safe_name}.png", dpi=150)
    plt.show()

# =====================================================================
# MAIN ANALYSIS & LAUNCHER
# =====================================================================

def analyze_and_launch(filename='policy/grid_search_results.pkl'):
    try:
        with open(filename, 'rb') as f:
            results = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print("Please run the grid search script first to generate the data.")
        return

    print("\n" + "="*60)
    print("🏆 GRID SEARCH FINAL LEADERBOARD 🏆")
    print("="*60 + "\n")

    metrics_list = []

    # Calculate metrics
    for config, rewards in results.items():
        last_50 = rewards[-50:]
        mean_score = np.mean(last_50)
        std_dev = np.std(last_50) 
        total_auc = np.sum(rewards) 
        
        metrics_list.append({
            'config': config,
            'mean': mean_score,
            'std': std_dev,
            'auc': total_auc
        })

    # Sort by Mean (highest first), tie-break with AUC
    metrics_list.sort(key=lambda x: (x['mean'], x['auc']), reverse=True)

    # Print Leaderboard
    for rank, m in enumerate(metrics_list, start=1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        print(f"{medal} RANK {rank}: {m['config']}")
        print(f"    -> Final 50-eps Mean: {m['mean']:>7.2f} / 100")
        print(f"    -> Stability (Std):   {m['std']:>7.2f}")
        print(f"    -> Global Speed (AUC):{m['auc']:>7.0f}")
        print("-" * 60)

    # --- INTERACTIVE LAUNCHER ---
    print("\n" + "="*60)
    print("🚀 TARGETED TRAINING LAUNCHER")
    print("="*60)
    
    user_input = input("\nEnter the Rank Number you want to train (or 'q' to quit): ").strip()
    
    if user_input.lower() == 'q':
        print("Exiting. Have a great day!")
        return
        
    try:
        rank_choice = int(user_input)
        if rank_choice < 1 or rank_choice > len(metrics_list):
            print("Invalid rank selection. Exiting.")
            return
    except ValueError:
        print("Invalid input. Please enter a number. Exiting.")
        return
        
    # Get the chosen configuration
    chosen_config = metrics_list[rank_choice - 1]['config']
    print(f"\n>>> Preparing to train configuration: {chosen_config}")
    
    eps_input = input("How many episodes do you want to train for? (Default 1000): ").strip()
    episodes = int(eps_input) if eps_input.isdigit() else 1000
    
    # Parse configuration
    goal_name, goal_states, gamma, goal_reward = parse_configuration_string(chosen_config)
    
    # Initialize Environment and MDP
    env = gym.make("LunarLander-v3", continuous=False)
    
    abstract_mdp = ConfigurableDiagonalMDP(
        gamma=gamma, 
        goal_states=goal_states, 
        goal_reward=goal_reward
    )
    abstract_mdp.value_iteration()
    
    # Initialize Agent
    safe_policy_name = chosen_config.replace(" | ", "_").replace(":", "").replace(".", "")
    policy_path = f"policy_{safe_policy_name}.pth"
    
    agent = HierarchicalDQNLearner(
        env, abstract_mdp, phi_mapping_grid, 
        max_episodes=episodes, use_ddqn=True, 
        policy_name=policy_path
    )
    
    # Run Training
    rewards = run_targeted_training(env, agent, abstract_mdp, episodes)
    
    env.close()
    print(f"\n>>> Training complete! Model saved to: {policy_path}")
    
    # Plot results
    plot_targeted_learning_curve(rewards, chosen_config)

if __name__ == "__main__":
    analyze_and_launch()