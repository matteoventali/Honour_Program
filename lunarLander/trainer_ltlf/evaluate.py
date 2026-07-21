import os
import json
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
from agent import QNetwork
from abstract_mdps import LTLfAutomaton, LTLfWaypointMDP

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def moving_average(data, window_size):
    """Calculates the moving average of a 1D array to smooth the curve."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# =====================================================================
# WORKER FUNCTION FOR MULTIPROCESSING (N-Phases Dynamic)
# =====================================================================

def evaluate_policy_worker(policy_filename, episodes, render, formula, waypoints_dict, goal_reward, grid_w, grid_h):
    """
    Worker function to load and evaluate a single PyTorch DQN policy.
    Handles an LTLf-based task with dynamic one-hot encoding for automaton states.
    """
    policy_path = os.path.join("./policy/", policy_filename)
    automaton = LTLfAutomaton(formula)
    num_states = len(automaton.states)
    extra_dims = num_states
    
    if not os.path.exists(policy_path):
        print(f"[{policy_filename}] Error: Could not find the file '{policy_path}'")
        return policy_filename, [], []
        
    print(f"[{policy_filename}] Loading and starting evaluation ({num_states} DFA states)...")
    
    # Initialize Environment
    mode = "human" if render else None
    env = gym.make("LunarLander-v3", continuous=False, render_mode=mode)
    
    state_dim = env.observation_space.shape[0] + extra_dims
    action_dim = env.action_space.n
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = QNetwork(state_dim, action_dim).to(device)
    
    try:
        q_network.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
        q_network.eval() 
    except Exception as e:
        print(f"[{policy_filename}] Error loading the PyTorch model: {e}")
        env.close()
        return policy_filename, [], []

    rewards = []
    # Dynamic success tracker for N phases
    success_counts = [0] * num_states

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0
        
        # Get initial DFA state
        q = automaton.get_initial_q()
        q_idx = automaton.states.index(q)
        
        while not (terminated or truncated):
            # Augment state with one-hot encoded 'q'
            q_one_hot = np.zeros(num_states, dtype=np.float32)
            q_one_hot[q_idx] = 1.0
            state_aug = np.concatenate((obs, q_one_hot)).astype(np.float32)
            state_tensor = torch.FloatTensor(state_aug).unsqueeze(0).to(device)

            with torch.no_grad():
                action = q_network(state_tensor).argmax(dim=1).item()
            
            next_obs, _, terminated, truncated, _ = env.step(action)
            
            # Abstract the new physical state
            x_pos = np.clip((next_obs[0] + 1) / 2 * (grid_w - 1), 0, grid_w - 1)
            y_pos = np.clip(next_obs[1] / 1.5 * (grid_h - 1), 0, grid_h - 1)
            abstract_x_ns, abstract_y_ns = int(x_pos), int(y_pos)
            
            # --- LTLf State Transition Logic ---
            truth_assignment = {prop: (abstract_x_ns == coords[0] and abstract_y_ns == coords[1]) for prop, coords in waypoints_dict.items()}
            next_q = automaton.get_next_q(q, truth_assignment)
            
            step_reward = 0.0
            if automaton.is_goal_reached(next_q):
                step_reward = goal_reward
                print(f"[{policy_filename}] Final Goal State Reached (q={next_q})")
                terminated = True
            
            total_reward += step_reward
            
            if render:
                time.sleep(0.02)
            
            # Prepare for the next iteration
            obs = next_obs
            q = next_q
            q_idx = automaton.states.index(q)
                
        rewards.append(total_reward)
        
        # For simplicity, we just count if the final state was ever reached in the episode
        if terminated and step_reward > 0:
            # This is a proxy for success. A more detailed success count per state is possible.
            success_counts[-1] += 1
        
    env.close()
    return policy_filename, rewards, success_counts

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
    filename = './img/policy_comparison_combined_ltlf.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved combined comparison plot: {filename}")

# =====================================================================
# MAIN FUNCTION
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate LTLf-guided DQN policies for LunarLander.")
    parser.add_argument("policies", nargs='+', type=str, help="List of PyTorch model files to evaluate.")
    parser.add_argument("--config", type=str, default="trajectory.json", help="Path to the JSON config file")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run per policy (default: 100)")
    parser.add_argument("--render", action="store_true", help="Enable graphical rendering of the environment")
    parser.add_argument("--window", type=int, default=10, help="Window size for the moving average (default: 10)")
    parser.add_argument("--parallel", action="store_true", help="Run evaluations concurrently using multiprocessing")
    
    args = parser.parse_args()
    os.makedirs("./img", exist_ok=True)
    
    # 1. Load Single Source of Truth Configuration
    if not os.path.exists(args.config):
        print(f"Critical Error: Config file '{args.config}' not found.")
        return
        
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    formula = config.get('formula', 'F(goal)')
    raw_waypoints = config.get('waypoints_dict', {'goal': [5, 0]})
    waypoints_dict = {name: tuple(coords) for name, coords in raw_waypoints.items()}
    
    grid_w = config.get('grid_w', 12)
    grid_h = config.get('grid_h', 12)
    goal_reward = config.get('goal_reward', 10000.0)
    
    print(f"Loaded LTLf Formula: {formula}\nWaypoints: {waypoints_dict}")
    
    results_dict = {}

    # 2. Run Evaluations
    if args.parallel and len(args.policies) > 1:
        print(f"--- Starting PARALLEL evaluation for {len(args.policies)} policies ---")
        with multiprocessing.Pool(processes=len(args.policies)) as pool:
            tasks = [(p, args.episodes, args.render, formula, waypoints_dict, goal_reward, grid_w, grid_h) for p in args.policies]
            results_list = pool.starmap(evaluate_policy_worker, tasks)
            
        for name, rewards, success_counts in results_list:
            results_dict[name] = rewards
            hits_str = " | ".join([f"Phase {i+1}: {c}/{args.episodes}" for i, c in enumerate(success_counts)])
            print(f"[{name}] Eval complete. Avg Reward: {np.mean(rewards):.2f} | {hits_str}")
    else:
        print(f"--- Starting SEQUENTIAL evaluation for {len(args.policies)} policies ---")
        for p in args.policies:
            name, rewards, success_counts = evaluate_policy_worker(p, args.episodes, args.render, formula, waypoints_dict, goal_reward, grid_w, grid_h)
            results_dict[name] = rewards
            hits_str = " | ".join([f"Phase {i+1}: {c}/{args.episodes}" for i, c in enumerate(success_counts)])
            print(f"[{name}] Eval complete. Avg Reward: {np.mean(rewards):.2f} | {hits_str}")

    # 3. Plotting
    print("--- Generating Plots ---")
    for name, rewards in results_dict.items():
        if rewards:
            plot_individual_policy(rewards, name, args.window)
    if len(results_dict) > 1:
        plot_combined_comparison(results_dict, args.window, args.episodes)
    print("\nAll done! Check the './img/' folder for your plots.")

if __name__ == "__main__":
    main()