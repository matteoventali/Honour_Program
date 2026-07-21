import os
import gymnasium as gym
import numpy as np
import argparse
import json
from datetime import datetime

from abstract_mdps import NPhaseWaypointMDP
from agent import HierarchicalDQNLearner
from main import run_sequential_training # Import the training function
from utils import plot_mean_std_curves # Import the new plotting function

def run_experiment(args):
    """
    Main function to run multiple training sessions and plot aggregated results.
    """
    # Create a unique directory for this experiment run based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"img/experiments/{timestamp}_{args.mode}_{args.num_runs}runs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Experiment results will be saved in: {output_dir}")

    env = gym.make("LunarLander-v3", continuous=False)

    # --- Load shared configuration ---
    with open(args.config, 'r') as f:
        config = json.load(f)
    route_waypoints = [tuple(wp) for wp in config['waypoints']]
    num_phases = len(route_waypoints)
    
    # Shared Hyperparameters
    abstract_goal_reward = 10000
    env_goal_reward = 10000
    gamma = 0.99
    plot_window_size = 100

    # --- Data Storage for all runs ---
    all_single_rewards = []
    all_multi_rewards = []

    # --- Initialize Abstract MDP once ---
    # Its properties are static across all runs for a given experiment.
    print("Initializing Abstract MDP...")
    abstract_mdp = NPhaseWaypointMDP(waypoints=route_waypoints, gamma=gamma, goal_reward=abstract_goal_reward)
    abstract_mdp.value_iteration()

    for i in range(args.num_runs):
        print("\n" + "#"*70)
        print(f"# Starting Run {i + 1} of {args.num_runs}")
        print("#"*70 + "\n")

        # --- SINGLE EPSILON RUN ---
        if args.mode in ['single', 'comparison']:
            print(f"\n--- Running: SINGLE EPSILON (Run {i+1}/{args.num_runs}) ---")
            agent_single_eps = HierarchicalDQNLearner(env=env, max_episodes=args.episodes, eps_decay=0.9996, use_ddqn=True, extra_state_dims=num_phases)
            true_rewards, _, _, _, _ = run_sequential_training(
                env, agent_single_eps, abstract_mdp, args.episodes, goal_reward=env_goal_reward, use_shaping=True, use_double_epsilon=False
            )
            all_single_rewards.append(true_rewards)

        # --- MULTI EPSILON RUN ---
        if args.mode in ['multi', 'comparison']:
            print(f"\n--- Running: MULTI EPSILON (Run {i+1}/{args.num_runs}) ---")
            agent_multi_eps = HierarchicalDQNLearner(env=env, max_episodes=args.episodes, eps_decay=0.999, use_ddqn=True, extra_state_dims=num_phases)
            true_rewards, _, _, _, _ = run_sequential_training(
                env, agent_multi_eps, abstract_mdp, args.episodes, goal_reward=env_goal_reward, use_shaping=True, use_double_epsilon=True
            )
            all_multi_rewards.append(true_rewards)

    env.close()

    # --- PLOTTING AGGREGATED RESULTS ---
    print("\n" + "="*70)
    print("All runs completed. Generating aggregated plots...")
    print("="*70 + "\n")

    plot_mean_std_curves(
        reward_histories_single=all_single_rewards if args.mode != 'multi' else None,
        reward_histories_multi=all_multi_rewards if args.mode != 'single' else None,
        window_size=plot_window_size,
        title=f"Performance Comparison over {args.num_runs} Runs",
        filename=f"{output_dir}/mean_variance_comparison.png"
    )

    print("Experiment finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple N-Phase DQN experiments and plot mean/variance.")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2,
        help="Number of times to run each experiment configuration."
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="comparison", 
        choices=['single', 'multi', 'comparison'],
        help="Execution mode: 'single' for single epsilon, 'multi' for multi-epsilon, 'comparison' for both."
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=1, # Default to a low number for quick tests
        help="Number of training episodes per run."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="trajectory.json", 
        help="Path to the trajectory configuration file."
    )
    args = parser.parse_args()
    run_experiment(args)