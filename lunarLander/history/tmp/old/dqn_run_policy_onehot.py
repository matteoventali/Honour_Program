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

def evaluate_policy_worker(policy_filename, episodes, render, goal_state, goal_reward, grid_w, grid_h, extra_dims):
    """
    Worker function to load and evaluate a single PyTorch DQN policy.
    This version handles the sequential state 'q' with one-hot encoding.
    """
    policy_path = os.path.join("./policy/", policy_filename)
    
    if not os.path.exists(policy_path):
        print(f"[{policy_filename}] Error: Could not find the file '{policy_path}'")
        return policy_filename, [], 0
        
    print(f"[{policy_filename}] Loading and starting evaluation...")
    
    # Initialize Environment
    mode = "human" if render else None
    env = gym.make("LunarLander-v3", continuous=False, render_mode=mode)
    
    # The state_dim must match the one used during training (it might have extra dimensions)
    state_dim = env.observation_space.shape[0] + extra_dims
    action_dim = env.action_space.n
    
    # Initialize Device and Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = QNetwork(state_dim, action_dim).to(device)
    
    try:
        # Load the PyTorch model weights
        q_network.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
        q_network.eval() # Set the network to evaluation mode
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
        q = 0 # Initialize sequential state 'q'
        
        while not (terminated or truncated):
            # Augment state with one-hot encoded 'q'
            q_one_hot = np.array([1.0, 0.0]) if q == 0 else np.array([0.0, 1.0])
            state_aug = np.concatenate((obs, q_one_hot)).astype(np.float32)
            state_tensor = torch.FloatTensor(state_aug).unsqueeze(0).to(device)

            # Forward pass to get Q-values and select the best action
            with torch.no_grad():
                action = q_network(state_tensor).argmax(dim=1).item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # --- State 'q' transition logic ---
            if q == 0:
                abstract_x, abstract_y = phi_mapping_grid(obs, grid_w, grid_h)
                if (abstract_x, abstract_y) == (1, 8): # Waypoint
                    q = 10
                    print(f"[{policy_filename}] Waypoint (1,8) reached! Transitioning to q=10.")

            # --- Goal MDP Reward Logic ---
            if goal_state:
                step_reward = 0.0
                abstract_x, abstract_y = phi_mapping_grid(obs, grid_w, grid_h)
                # The goal is reached only if we are in phase q=10
                if (abstract_x, abstract_y) == goal_state and q == 10:
                    step_reward = goal_reward
                    success_count += 1
                    print(f"[{policy_filename}] Final goal (8,8) reached!")
                    terminated = True
            else:
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
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, color='gray', label='Raw Reward')
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
    plt.figure(figsize=(12, 6))
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
    filename = './img/policy_comparison_combined_q.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved combined comparison plot: {filename}")

# =====================================================================
# MAIN FUNCTION
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate sequential DQN policies for LunarLander.")
    parser.add_argument("policies", nargs='+', type=str, help="List of PyTorch model files to evaluate (e.g., shaping_sequential_policy.pth)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run per policy (default: 100)")
    parser.add_argument("--render", action="store_true", help="Enable graphical rendering of the environment")
    parser.add_argument("--window", type=int, default=10, help="Window size for the moving average (default: 10)")
    parser.add_argument("--parallel", action="store_true", help="Run evaluations concurrently using multiprocessing")
    # Final goal for sequential task
    parser.add_argument("--goal-x", type=int, default=8, help="X coordinate of the final abstract goal state (default: 8)")
    parser.add_argument("--goal-y", type=int, default=8, help="Y coordinate of the final abstract goal state (default: 8)")
    parser.add_argument("--goal-reward", type=float, default=100.0, help="Reward for reaching the abstract goal state")
    parser.add_argument("--grid-w", type=int, default=12, help="Grid width for abstract state mapping")
    parser.add_argument("--grid-h", type=int, default=12, help="Grid height for abstract state mapping")
    parser.add_argument("--extra-dims", type=int, default=2, help="Number of extra dimensions for the one-hot state space (default: 2)")
    
    args = parser.parse_args()
    
    os.makedirs("./img", exist_ok=True)
    results_dict = {}

    # The goal state for the sequential task is (8, 8)
    goal_state = (args.goal_x, args.goal_y)

    if args.parallel and len(args.policies) > 1:
        print(f"--- Starting PARALLEL evaluation for {len(args.policies)} policies ---")
        with multiprocessing.Pool(processes=len(args.policies)) as pool:
            tasks = [(p, args.episodes, args.render, goal_state, args.goal_reward, args.grid_w, args.grid_h, args.extra_dims) for p in args.policies]
            results_list = pool.starmap(evaluate_policy_worker, tasks)
        for name, rewards, successes in results_list:
            results_dict[name] = rewards
            print(f"[{name}] Evaluation complete. Avg Reward: {np.mean(rewards):.2f} | Successes: {successes}/{args.episodes}")
        print("--- Parallel evaluation finished ---\n")
    else:
        print(f"--- Starting SEQUENTIAL evaluation for {len(args.policies)} policies ---")
        for p in args.policies:
            name, rewards, successes = evaluate_policy_worker(p, args.episodes, args.render, goal_state, args.goal_reward, args.grid_w, args.grid_h, args.extra_dims)
            results_dict[name] = rewards
            print(f"[{name}] Evaluation complete. Avg Reward: {np.mean(rewards):.2f} | Successes: {successes}/{args.episodes}")
        print("--- Sequential evaluation finished ---\n")

    print("--- Generating Plots ---")
    for name, rewards in results_dict.items():
        if rewards:
            plot_individual_policy(rewards, name, args.window)
    if len(results_dict) > 1:
        plot_combined_comparison(results_dict, args.window, args.episodes)
    print("\nAll done! Check the './img/' folder for your plots.")

if __name__ == "__main__":
    main()