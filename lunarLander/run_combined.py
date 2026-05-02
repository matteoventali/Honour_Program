import os
import sys
import pickle
import numpy as np
import gymnasium as gym
import time
import argparse
import matplotlib.pyplot as plt

def discretize(obs):
    """
    Discretizes the continuous observations into a tuple of integers.
    This must exactly match the discretization used during training.
    """
    x_intervals     = [-0.5, 0.5]
    y_intervals     = [-0.1, 0.1, 1.5]
    vx_intervals    = [-7.5, -5, -0.3, -0.1, 0.1, 0.3, 5, 7.5]
    vy_intervals    = [-7.5, -5, -0.3, -0.1, 0.1, 0.3, 5, 7.5]
    theta_intervals = [-1.25663706,  -0.1, 0.1, 1.25663706]
    omega_intervals = [-7.5, -5, -0.1, 0.1, 5, 7.5]
    
    res = [
        np.digitize(obs[0], x_intervals), np.digitize(obs[1], y_intervals),
        np.digitize(obs[2], vx_intervals), np.digitize(obs[3], vy_intervals),
        np.digitize(obs[4], theta_intervals), np.digitize(obs[5], omega_intervals),
        int(obs[6]), int(obs[7])
    ]
    return tuple(res)

def moving_average(data, window_size):
    """
    Calculates the moving average of a 1D array to smooth the curve.
    """
    if len(data) < window_size:
        return data # Not enough data to smooth
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def evaluate_policy(env, q_table, episodes, render):
    """
    Runs the environment for a given number of episodes using the provided Q-table.
    Returns a list containing the total reward for each episode.
    """
    rewards = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        state = discretize(obs)
        terminated = truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            # Choose the best action from the Q-table (Greedy policy)
            action_values = q_table.get(state, np.zeros(env.action_space.n))
            action = np.argmax(action_values)
            
            # Execute the action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Update state and accumulate reward
            state = discretize(next_obs)
            total_reward += reward
            
            # Add a slight delay ONLY if rendering
            if render:
                time.sleep(0.02)
                
        rewards.append(total_reward)
        print(f"Episode {ep + 1}/{episodes} finished. Total Reward: {total_reward:.2f}")
        
    return rewards

def load_policy(policy_filename):
    """Helper function to load a policy from the ./policy/ directory."""
    policy_path = os.path.join("./policy/", policy_filename)
    if not os.path.exists(policy_path):
        print(f"Error: Could not find the file '{policy_path}'")
        sys.exit(1)
        
    print(f"Loading policy from: {policy_path}...")
    try:
        with open(policy_path, 'rb') as f:
            q_table = pickle.load(f)
        return q_table, policy_filename
    except Exception as e:
        print(f"Error loading the pickle file: {e}")
        sys.exit(1)

def main():
    # 1. Setup argument parser
    parser = argparse.ArgumentParser(description="Compare two trained Q-Learning policies for LunarLander.")
    parser.add_argument("policy1", type=str, help="Name of the first policy file (e.g., model_A.pkl)")
    parser.add_argument("policy2", type=str, help="Name of the second policy file (e.g., model_B.pkl)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run per policy (default: 100)")
    parser.add_argument("--render", action="store_true", help="Enable graphical rendering of the environment")
    parser.add_argument("--window", type=int, default=10, help="Window size for the moving average (default: 10)")
    
    args = parser.parse_args()
    
    # 2. Load both policies
    q_table1, name1 = load_policy(args.policy1)
    q_table2, name2 = load_policy(args.policy2)
    print("Both policies loaded successfully!\n")

    # 3. Initialize the environment
    mode = "human" if args.render else None
    env = gym.make("LunarLander-v3", continuous=False, render_mode=mode)
    
    # 4. Evaluate Policy 1
    print(f"--- Starting evaluation for Policy 1: {name1} ---")
    rewards1 = evaluate_policy(env, q_table1, args.episodes, args.render)
    
    # 5. Evaluate Policy 2
    print(f"\n--- Starting evaluation for Policy 2: {name2} ---")
    rewards2 = evaluate_policy(env, q_table2, args.episodes, args.render)
    
    env.close()

    # 6. Plotting the results
    print("\nGenerating comparison plot...")
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards with high transparency
    plt.plot(rewards1, alpha=0.2, color='blue')
    plt.plot(rewards2, alpha=0.2, color='orange')
    
    # Plot smoothed rewards (Moving Average)
    smooth_rewards1 = moving_average(rewards1, args.window)
    smooth_rewards2 = moving_average(rewards2, args.window)
    
    # Generate X axis values for the smoothed curve (it shifts right due to convolution)
    if len(smooth_rewards1) == len(rewards1) - args.window + 1:
        x_smooth = range(args.window - 1, args.episodes)
        plt.plot(x_smooth, smooth_rewards1, color='blue', linewidth=2, label=f'{name1} (MA Window={args.window})')
        plt.plot(x_smooth, smooth_rewards2, color='orange', linewidth=2, label=f'{name2} (MA Window={args.window})')
    else:
        # Fallback if episodes < window size
        plt.plot(smooth_rewards1, color='blue', linewidth=2, label=name1)
        plt.plot(smooth_rewards2, color='orange', linewidth=2, label=name2)

    plt.title(f'Policy Comparison over {args.episodes} Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_filename = './img/policy_comparison.png'
    plt.savefig(plot_filename)
    print(f"Plot successfully saved as '{plot_filename}'.")

if __name__ == "__main__":
    main()