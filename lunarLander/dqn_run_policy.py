import os
import sys
import numpy as np
import gymnasium as gym
import time
import argparse
import multiprocessing
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# DQN ARCHITECTURE
# =====================================================================

class QNetwork(nn.Module):
    """
    Neural Network Architecture for the DQN.
    This must perfectly match the architecture used during training.
    """
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
# UTILITY FUNCTIONS
# =====================================================================

def moving_average(data, window_size):
    """
    Calculates the moving average of a 1D array to smooth the curve.
    """
    if len(data) < window_size:
        return data # Not enough data to smooth
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# =====================================================================
# WORKER FUNCTION FOR MULTIPROCESSING
# =====================================================================

def evaluate_policy_worker(policy_filename, episodes, render):
    """
    Worker function to load and evaluate a single PyTorch DQN policy.
    Returns the policy name and the list of episode rewards.
    """
    policy_path = os.path.join("./policy/", policy_filename)
    
    if not os.path.exists(policy_path):
        print(f"[{policy_filename}] Error: Could not find the file '{policy_path}'")
        return policy_filename, []
        
    print(f"[{policy_filename}] Loading and starting evaluation...")
    
    # Initialize Environment
    mode = "human" if render else None
    env = gym.make("LunarLander-v3", continuous=False, render_mode=mode)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize Device and Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = QNetwork(state_dim, action_dim).to(device)
    
    try:
        # Load the PyTorch model weights
        q_network.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
        q_network.eval() # Set the network to evaluation mode (disables dropout/batchnorm updates)
    except Exception as e:
        print(f"[{policy_filename}] Error loading the PyTorch model: {e}")
        env.close()
        return policy_filename, []

    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            # Convert observation to PyTorch tensor
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Forward pass to get Q-values and select the best action
            with torch.no_grad():
                action = q_network(state_tensor).argmax(dim=1).item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if render:
                time.sleep(0.02)
                
        rewards.append(total_reward)
        
    env.close()
    print(f"[{policy_filename}] Evaluation complete. Avg Reward: {np.mean(rewards):.2f}")
    
    return policy_filename, rewards

# =====================================================================
# PLOTTING FUNCTIONS
# =====================================================================

def plot_individual_policy(rewards, name, window_size):
    """Generates and saves a plot for a single policy."""
    plt.figure(figsize=(10, 6))
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.3, color='gray', label='Raw Reward')
    
    # Plot smoothed rewards
    smooth = moving_average(rewards, window_size)
    if len(smooth) == len(rewards) - window_size + 1:
        x_smooth = range(window_size - 1, len(rewards))
        plt.plot(x_smooth, smooth, color='blue', linewidth=2, label=f'Moving Avg (Window={window_size})')
    else:
        plt.plot(smooth, color='blue', linewidth=2, label='Smoothed')
        
    plt.title(f'Evaluation Results: {name}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    filename = f"./img/eval_{name.replace('.pth', '')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved individual plot: {filename}")

def plot_combined_comparison(results_dict, window_size, episodes):
    """Generates and saves a combined plot comparing all policies (smoothed only)."""
    plt.figure(figsize=(12, 6))
    
    # Define a set of colors for the different policies
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    for idx, (name, rewards) in enumerate(results_dict.items()):
        if not rewards: continue
        
        color = colors[idx % len(colors)]
        smooth = moving_average(rewards, window_size)
        
        if len(smooth) == len(rewards) - window_size + 1:
            x_smooth = range(window_size - 1, len(rewards))
            plt.plot(x_smooth, smooth, color=color, linewidth=2, label=f'{name} (MA Window={window_size})')
        else:
            plt.plot(smooth, color=color, linewidth=2, label=name)

    plt.title(f'Policy Comparison (Smoothed Moving Average over {episodes} Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    filename = './img/policy_comparison_combined.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved combined comparison plot: {filename}")

# =====================================================================
# MAIN FUNCTION
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare multiple trained DQN policies for LunarLander.")
    parser.add_argument("policies", nargs='+', type=str, help="List of PyTorch model files to evaluate (e.g., dqn_normally.pth dqn_shaping.pth)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run per policy (default: 100)")
    parser.add_argument("--render", action="store_true", help="Enable graphical rendering of the environment")
    parser.add_argument("--window", type=int, default=10, help="Window size for the moving average (default: 10)")
    parser.add_argument("--parallel", action="store_true", help="Run evaluations concurrently using multiprocessing")
    
    args = parser.parse_args()
    
    os.makedirs("./img", exist_ok=True)
    results_dict = {}

    if args.parallel and len(args.policies) > 1:
        print(f"--- Starting PARALLEL evaluation for {len(args.policies)} policies ---")
        
        # Use Pool to map the worker function across all policies
        with multiprocessing.Pool(processes=len(args.policies)) as pool:
            # Prepare arguments for starmap
            tasks = [(p, args.episodes, args.render) for p in args.policies]
            results_list = pool.starmap(evaluate_policy_worker, tasks)
            
        # Collect results
        for name, rewards in results_list:
            results_dict[name] = rewards
            
        print("--- Parallel evaluation finished ---\n")
        
    else:
        print(f"--- Starting SEQUENTIAL evaluation for {len(args.policies)} policies ---")
        for p in args.policies:
            name, rewards = evaluate_policy_worker(p, args.episodes, args.render)
            results_dict[name] = rewards
        print("--- Sequential evaluation finished ---\n")

    # Generating the plots
    print("--- Generating Plots ---")
    
    # 1. Generate individual plots
    for name, rewards in results_dict.items():
        if rewards: # Only plot if we successfully gathered rewards
            plot_individual_policy(rewards, name, args.window)
            
    # 2. Generate the combined comparison plot
    if len(results_dict) > 1:
        plot_combined_comparison(results_dict, args.window, args.episodes)
        
    print("\nAll done! Check the './img/' folder for your plots.")

if __name__ == "__main__":
    main()