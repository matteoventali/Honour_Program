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

# Modular imports to reuse existing components
from utils import phi_mapping_grid
from agent import QNetwork

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

def evaluate_policy_worker(policy_filename, episodes, render, goal_state, goal_reward, grid_w, grid_h):
    """
    Worker function to load and evaluate a single PyTorch DQN policy.
    Returns the policy name and the list of episode rewards.
    """
    policy_path = os.path.join("./policy/", policy_filename)
    
    if not os.path.exists(policy_path):
        print(f"[{policy_filename}] Error: Could not find the file '{policy_path}'")
        return policy_filename, [], 0
        
    print(f"[{policy_filename}] Loading and starting evaluation...")
    
    # Initialize Environment
    mode = "human" if render else None
    env = gym.make("LunarLander-v3", continuous=False, render_mode=mode)
    
    # Standard state dimension
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
        return policy_filename, [], 0

    rewards = []
    success_count = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            # Forward pass to get Q-values and select the best action
            with torch.no_grad():
                action = q_network(state_tensor).argmax(dim=1).item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # --- Goal MDP Reward Logic ---
            # If a goal_state is specified, we ONLY reward the agent for reaching that goal.
            # The intermediate rewards from the environment are ignored.
            if goal_state:
                step_reward = 0.0 # Default reward is 0
                abstract_x, abstract_y = phi_mapping_grid(obs, grid_w, grid_h)
                if (abstract_x, abstract_y) == goal_state:
                    step_reward = goal_reward # Override the reward
                    success_count += 1
                    terminated = True # End the episode successfully
            else:
                # If no goal is specified, use the standard environment reward
                step_reward = reward
                
            total_reward += step_reward
            
            if render:
                time.sleep(0.02)
                
        rewards.append(total_reward)
        
    return policy_filename, rewards, success_count

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
    # Arguments for evaluation with an abstract goal state
    parser.add_argument("--goal-x", type=int, default=None, help="X coordinate of the abstract goal state")
    parser.add_argument("--goal-y", type=int, default=None, help="Y coordinate of the abstract goal state")
    parser.add_argument("--goal-reward", type=float, default=100.0, help="Reward for reaching the abstract goal state")
    parser.add_argument("--grid-w", type=int, default=12, help="Grid width for abstract state mapping")
    parser.add_argument("--grid-h", type=int, default=12, help="Grid height for abstract state mapping")
    
    args = parser.parse_args()
    
    os.makedirs("./img", exist_ok=True)
    results_dict = {}

    if args.parallel and len(args.policies) > 1:
        print(f"--- Starting PARALLEL evaluation for {len(args.policies)} policies ---")
        
        # Use Pool to map the worker function across all policies
        with multiprocessing.Pool(processes=len(args.policies)) as pool:
            # Prepare arguments for starmap
            goal_state = (args.goal_x, args.goal_y) if args.goal_x is not None and args.goal_y is not None else None
            tasks = [(p, args.episodes, args.render, goal_state, args.goal_reward, args.grid_w, args.grid_h) for p in args.policies]
            results_list = pool.starmap(evaluate_policy_worker, tasks)
            
        # Collect results
        for name, rewards, successes in results_list:
            results_dict[name] = rewards
            print(f"[{name}] Evaluation complete. Avg Reward: {np.mean(rewards):.2f} | Successes: {successes}/{args.episodes}")
            
        print("--- Parallel evaluation finished ---\n")
        
    else:
        print(f"--- Starting SEQUENTIAL evaluation for {len(args.policies)} policies ---")
        goal_state = (args.goal_x, args.goal_y) if args.goal_x is not None and args.goal_y is not None else None

        for p in args.policies:
            name, rewards, successes = evaluate_policy_worker(
                p, 
                args.episodes, 
                args.render,
                goal_state,
                args.goal_reward,
                args.grid_w,
                args.grid_h
            )
            results_dict[name] = rewards
            print(f"[{name}] Evaluation complete. Avg Reward: {np.mean(rewards):.2f} | Successes: {successes}/{args.episodes}")

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