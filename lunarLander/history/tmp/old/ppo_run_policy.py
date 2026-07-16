import os
import time
import argparse
import numpy as np

# Import and set matplotlib backend BEFORE gymnasium
# This helps prevent rendering conflicts (segmentation faults)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO

# Make sure these imports match your local modules
from utils import phi_mapping_sequential

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def moving_average(data, window_size):
    """Calculates the moving average of a 1D array to smooth the curve."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def augment_state(obs, q):
    """Augments the observation with the one-hot encoded q-state."""
    q_one_hot = np.array([1.0, 0.0]) if q == 0 else np.array([0.0, 1.0])
    return np.concatenate((obs, q_one_hot)).astype(np.float32)

# =====================================================================
# EVALUATION WORKER
# =====================================================================

def evaluate_policy(policy_filename, episodes, render, goal_state, goal_reward, grid_w, grid_h):
    """
    Worker function to load and evaluate a single PPO policy.
    This version handles the sequential state 'q' manually for evaluation.
    """
    if not os.path.exists(policy_filename):
        print(f"[{policy_filename}] Error: Could not find the file '{policy_filename}'")
        return policy_filename, [], 0, 0

    print(f"[{policy_filename}] Loading and starting evaluation...")

    # Initialize Environment
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", continuous=False, render_mode=render_mode)

    # Load the PPO model
    try:
        model = PPO.load(policy_filename)
    except Exception as e:
        print(f"[{policy_filename}] Error loading the PPO model: {e}")
        env.close()
        return policy_filename, [], 0, 0

    episode_rewards = []
    success_count = 0
    waypoint_hit_count = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_episode_reward = 0
        
        # Manual state management for evaluation
        q = 0
        passed_through_waypoint = False

        while not (terminated or truncated):
            # Augment the state with 'q' before feeding it to the policy
            augmented_obs = augment_state(obs, q)
            
            # Get deterministic action from the policy
            action, _ = model.predict(augmented_obs, deterministic=True)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # --- Manual state 'q' transition logic ---
            abstract_x, abstract_y, _ = phi_mapping_sequential(obs, q, grid_w, grid_h)
            
            if q == 0 and (abstract_x, abstract_y) == (1, 8): # Waypoint coordinates
                q = 10
                passed_through_waypoint = True
                print(f"[{policy_filename}] Waypoint (1,8) reached! Transitioning to q=10.")

            # --- Goal-based reward logic for evaluation ---
            step_reward = 0.0 # Use sparse reward for evaluation
            if passed_through_waypoint and q == 10 and (abstract_x, abstract_y) == goal_state:
                step_reward = goal_reward
                success_count += 1
                print(f"[{policy_filename}] Final goal {goal_state} reached!")
                terminated = True # End episode on success
            
            total_episode_reward += step_reward
            
            if render:
                time.sleep(0.02)
                
        if passed_through_waypoint:
            waypoint_hit_count += 1
            
        episode_rewards.append(total_episode_reward)
        print(f"Episode {ep + 1}/{episodes} finished. Reward: {total_episode_reward}")

    env.close()
    return policy_filename, episode_rewards, success_count, waypoint_hit_count

# =====================================================================
# PLOTTING FUNCTION
# =====================================================================

def plot_evaluation_results(results_dict, window_size, episodes):
    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_dict)))
    
    for idx, (name, rewards) in enumerate(results_dict.items()):
        if not rewards: continue
        
        # Plot raw data points
        plt.scatter(range(len(rewards)), rewards, alpha=0.2, color=colors[idx])
        
        # Plot moving average
        smooth_rewards = moving_average(rewards, window_size)
        x_smooth = range(window_size - 1, len(rewards))
        plt.plot(x_smooth, smooth_rewards, color=colors[idx], linewidth=2.5, label=f'{os.path.basename(name)} (Avg)')

    plt.title(f'PPO Policy Evaluation (Smoothed over {window_size} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Sparse)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    filename = './img/ppo_policy_evaluation.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"\nSaved combined evaluation plot to: {filename}")

# =====================================================================
# MAIN FUNCTION
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate sequential PPO policies for LunarLander.")
    parser.add_argument("policies", nargs='+', type=str, help="List of PPO model files (.zip) to evaluate.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run per policy (default: 100).")
    parser.add_argument("--render", action="store_true", help="Enable graphical rendering of the environment.")
    parser.add_argument("--window", type=int, default=10, help="Window size for the moving average plot (default: 10).")
    
    # Abstract grid and goal parameters
    parser.add_argument("--goal-x", type=int, default=8, help="X coordinate of the final abstract goal state (default: 8).")
    parser.add_argument("--goal-y", type=int, default=8, help="Y coordinate of the final abstract goal state (default: 8).")
    parser.add_argument("--goal-reward", type=float, default=10000.0, help="Reward for reaching the final goal.")
    parser.add_argument("--grid-w", type=int, default=12, help="Grid width for abstract state mapping.")
    parser.add_argument("--grid-h", type=int, default=12, help="Grid height for abstract state mapping.")
    
    args = parser.parse_args()
    
    os.makedirs("./img", exist_ok=True)
    results_dict = {}

    goal_state = (args.goal_x, args.goal_y)

    print(f"--- Starting Evaluation for {len(args.policies)} PPO policies ---")
    for policy_file in args.policies:
        # Automatically prepend the 'models/' directory if the path doesn't exist
        if not os.path.exists(policy_file) and os.path.exists(os.path.join("models", policy_file)):
            policy_file = os.path.join("models", policy_file)

        name, rewards, successes, waypoint_hits = evaluate_policy(
            policy_file, args.episodes, args.render, goal_state,
            args.goal_reward, args.grid_w, args.grid_h
        )
        results_dict[name] = rewards
        avg_reward = np.mean(rewards) if rewards else 0
        print(f"[{os.path.basename(name)}] Evaluation complete. Avg Reward: {avg_reward:.2f}")
        print(f"  - Waypoint Reach Rate: {waypoint_hits}/{args.episodes} ({waypoint_hits/args.episodes:.2%})")
        print(f"  - Goal Reach Rate    : {successes}/{args.episodes} ({successes/args.episodes:.2%})")
    
    print("\n--- Evaluation Finished ---")

    if len(results_dict) > 0:
        print("\n--- Generating Plot ---")
        plot_evaluation_results(results_dict, args.window, args.episodes)

    print("\nAll done! Check the './img/' folder for your plot.")

if __name__ == "__main__":
    main()