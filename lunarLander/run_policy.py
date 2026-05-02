import os
import sys
import pickle
import numpy as np
import gymnasium as gym
import time
import argparse

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

def main():
    # 1. Setup argument parser
    parser = argparse.ArgumentParser(description="Run a trained Q-Learning policy for LunarLander.")
    parser.add_argument("policy_file", type=str, help="Name of the policy file (e.g., best_model.pkl)")
    parser.add_argument("--render", action="store_true", help="Enable graphical rendering of the environment")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (default: 5)")
    
    args = parser.parse_args()
    
    policy_filename = args.policy_file
    policy_path = os.path.join("./policy/", policy_filename)
    
    # 2. Verify that the file exists
    if not os.path.exists(policy_path):
        print(f"Error: Could not find the file '{policy_path}'")
        sys.exit(1)
        
    # 3. Load the saved Q-table
    print(f"Loading policy from: {policy_path}...")
    try:
        with open(policy_path, 'rb') as f:
            q_table = pickle.load(f)
    except Exception as e:
        print(f"Error loading the pickle file: {e}")
        sys.exit(1)
        
    print("Policy loaded successfully!")

    # 4. Initialize the environment with optional rendering
    # If --render is passed, render_mode="human", otherwise None (headless)
    mode = "human" if args.render else None
    print(f"Starting environment (Render mode: {mode})...")
    
    env = gym.make("LunarLander-v3", continuous=False, render_mode=mode)
    
    for ep in range(args.episodes):
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
            
            # Add a slight delay ONLY if rendering, to make animation watchable
            if args.render:
                time.sleep(0.02)
            
        print(f"Episode {ep + 1}/{args.episodes} finished. Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()