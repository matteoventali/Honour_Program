import os
import gymnasium as gym
import numpy as np
import argparse
import json

from abstract_mdps import NPhaseWaypointMDP
from agent import HierarchicalDQNLearner
from utils import phi_mapping_sequential, save_sequential_heatmaps, plot_buffer_fractions, plot_shaping_reward_breakdown, plot_comparison_curves

def run_sequential_training(env, agent, abstract_mdp, episodes, goal_reward=10000, save_policy=True, use_shaping=True, use_double_epsilon=True, K=1.0, log_file=None):
    num_phases = abstract_mdp.num_phases
    
    true_episode_rewards = []
    total_episode_rewards = []
    
    # Dynamic Epsilon Tracking Arrays
    epsilons = [agent.eps] * num_phases
    eps_single = agent.eps 
    
    eps_histories = [[] for _ in range(num_phases)]
    eps_single_history = []
    buffer_histories = [[] for _ in range(num_phases)]
    hits_history = [[] for _ in range(num_phases)]
    total_hits = [0] * num_phases

    log_handle = open(log_file, 'a') if log_file else None
    if log_handle: 
        log_handle.write(f"=== NEW RUN (Shaping: {use_shaping}, Multi-Eps: {use_double_epsilon}) ===\n")
        log_handle.write(f"===Trajectory: {abstract_mdp.waypoints}===\n")

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        q = 0 
        
        # Track which milestones are hit *this* episode for cascade epsilon decay
        reached_phase_this_episode = [False] * num_phases
        
        # Dynamic One-Hot encoding array
        q_one_hot = np.zeros(num_phases, dtype=np.float32)
        q_one_hot[q] = 1.0
        s_aug = np.concatenate((s_raw, q_one_hot)).astype(np.float32)
        
        terminated = truncated = False
        episode_true_reward = episode_total_reward = 0.0
        episode_hits = [0] * num_phases

        while not (terminated or truncated):
            # Select Epsilon based on current Phase (q)
            agent.eps = epsilons[q] if use_double_epsilon else eps_single

            a = agent.select_action(s_aug)
            ns_raw, _, terminated, truncated, _ = env.step(a)

            abstract_x_ns, abstract_y_ns, _ = phi_mapping_sequential(ns_raw, q)
            next_q = q
            env_goal_reward = 0.0

            # Dynamic Phase Transition Checking
            if q < num_phases - 1:
                target_x, target_y = abstract_mdp.waypoints[q]
                if abstract_x_ns == target_x and abstract_y_ns == target_y:
                    total_hits[q] += 1
                    episode_hits[q] = 1
                    reached_phase_this_episode[q] = True
                    next_q = q + 1
            else:
                # Final Goal Check
                goal_x, goal_y = abstract_mdp.waypoints[-1]
                if abstract_x_ns == goal_x and abstract_y_ns == goal_y:
                    total_hits[q] += 1
                    episode_hits[q] = 1
                    env_goal_reward = goal_reward
                    terminated = True

            abstract_ns = (abstract_x_ns, abstract_y_ns, next_q)
            done = terminated or truncated
            episode_true_reward += env_goal_reward
            
            # Next One-Hot 
            next_q_one_hot = np.zeros(num_phases, dtype=np.float32)
            next_q_one_hot[next_q] = 1.0
            ns_aug = np.concatenate((ns_raw, next_q_one_hot)).astype(np.float32)

            # Potential Based Shaping
            shaping_signal = 0.0
            if use_shaping:
                abstract_s = phi_mapping_sequential(s_raw, q)
                if abstract_s != abstract_ns:
                    phi_s = abstract_mdp.v_star.get(abstract_s, 0.0)
                    phi_ns = abstract_mdp.v_star.get(abstract_ns, 0.0)
                    shaping_signal = K * (phi_ns - phi_s)
            
            total_step_reward = env_goal_reward + shaping_signal
            episode_total_reward += total_step_reward
            
            agent.memory.push(s_aug, a, total_step_reward, ns_aug, done)
            agent.optimize_model()

            s_raw = ns_raw
            s_aug = ns_aug
            q = next_q
            
        # Cascaded Multi-Epsilon Decay
        if use_double_epsilon:
            epsilons[0] = max(0.08, epsilons[0] * agent.eps_decay)
            for i in range(1, num_phases):
                # Start decaying phase N only if phase N-1 is consistently reached and its epsilon is minimal
                if reached_phase_this_episode[i-1] and epsilons[i-1] <= 0.081:
                    epsilons[i] = max(0.08, epsilons[i] * agent.eps_decay)
        else:
            eps_single = max(agent.eps_min, eps_single * agent.eps_decay)
            
        # Logging Data Appends
        for i in range(num_phases):
            eps_histories[i].append(epsilons[i])
            buffer_histories[i].append(agent.memory.q_fraction_onehot(i, num_phases))
            hits_history[i].append(episode_hits[i])
            
        true_episode_rewards.append(episode_true_reward)
        total_episode_rewards.append(episode_total_reward)
        eps_single_history.append(eps_single)

        # Print logs
        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            recent_avg_with_shaping = np.mean(total_episode_rewards[-100:])
            mode_str = "SHAPING" if use_shaping else "BASELINE"
            eps_str = "MULTI EPS" if use_double_epsilon else "SINGLE EPS"

            # Dynamic labels: WP1, WP2, ..., Goal
            labels = [f"WP{i+1}" for i in range(num_phases - 1)] + ["Goal"]
            
            # 1. Format Epsilons String
            if use_double_epsilon:
                eps_details = ", ".join([f"q{i}({labels[i]}): {epsilons[i]:.4f}" for i in range(num_phases)])
            else:
                eps_details = f"Single: {eps_single:.4f}"

            # 2. Format Buffer Fractions String dynamically calling the dynamic q_fraction_onehot
            buffer_details = ", ".join([
                f"q{i}({labels[i]}): {agent.memory.q_fraction_onehot(i, num_phases):.2%}" 
                for i in range(num_phases)
            ])

            # 3. Format Target Hits String
            hits_details = ", ".join([f"{labels[i]}: {total_hits[i]}" for i in range(num_phases)])

            log_string = (
                f"----------------------------------------------------------------------------------------------------\n"
                f"[{mode_str} | {eps_str}] Episode {n_episode + 1}/{episodes}\n"
                f"Avg Reward                  : {recent_avg:.6f}\n" +
                (f"Avg Total Reward            : {recent_avg_with_shaping:.6f}\n" if use_shaping else "") +
                f"Epsilon Decay               : {eps_details}\n"
                f"Buffer Fractions            : {buffer_details}\n"
                f"Hits (Cumulative)           : {hits_details}\n"
                f"----------------------------------------------------------------------------------------------------\n"
            )
            
            print(log_string)
            if log_handle:
                log_handle.write(log_string + "\n")
                log_handle.flush()

        # Policy saving
        if save_policy and (n_episode + 1) % 250 == 0:
            prefix = "shaping" if use_shaping else "baseline"
            eps_suffix = "multi_eps" if use_double_epsilon else "single_eps"
            agent.policy_name = f"{prefix}_{eps_suffix}_policy_ep_{n_episode + 1}.pth"
            agent._save_policy()


    if log_handle: log_handle.close()
    
    # Return the correct epsilon history based on the mode
    final_eps_history = eps_single_history if not use_double_epsilon else eps_histories
    return true_episode_rewards, total_episode_rewards, final_eps_history, buffer_histories, hits_history

def main(args):
    print(f"=== STARTING N-PHASE ABLATION STUDY (Mode: {args.mode}) ===")
    os.makedirs("logs", exist_ok=True)
    env = gym.make("LunarLander-v3", continuous=False)

    # Hyperparameters
    abstract_goal_reward = 10000
    env_goal_reward = 10000
    gamma = 0.99
    plot_window_size = 500
    
    # Loading configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    route_waypoints = [tuple(wp) for wp in config['waypoints']]
    #route_waypoints = []
    print(f"Training for this trajectory {route_waypoints}")
    num_phases = len(route_waypoints)
    
    # Learning phase
    abstract_mdp = NPhaseWaypointMDP(waypoints=route_waypoints, gamma=gamma, goal_reward=abstract_goal_reward)
    abstract_mdp.value_iteration()
    save_sequential_heatmaps(abstract_mdp, filename_prefix=f"{args.mode}_exp")

    # --- Training Runs ---
    if args.mode == 'single' or args.mode == 'comparison':
        print("\n" + "="*50 + "\nTRAINING: SHAPING WITH SINGLE EPSILON\n" + "="*50)
        agent_single_eps = HierarchicalDQNLearner(
            env=env, max_episodes=args.episodes, eps_decay=0.9996, use_ddqn=True, 
            extra_state_dims=num_phases
        )
        true_rewards_single, total_rewards_single, eps_histories_single, buffers_single, _ = run_sequential_training(
            env, agent_single_eps, abstract_mdp, args.episodes, goal_reward=env_goal_reward, use_shaping=True, use_double_epsilon=False, 
            log_file="logs/single_epsilon_training.log"
        )        
        # Generate individual plots in both 'single' and 'comparison' modes
        print("Generating plots for single epsilon run...")
        plot_buffer_fractions(buffers_single, filename="img/buffer_fractions_single_eps.png", window_size=plot_window_size)
        plot_shaping_reward_breakdown(true_rewards_single, total_rewards_single, eps_histories_single, window_size=plot_window_size, 
            filename="img/reward_breakdown_single_eps.png"
        )

    if args.mode == 'multi' or args.mode == 'comparison':
        print("\n" + "="*50 + "\nTRAINING: SHAPING WITH MULTI EPSILON\n" + "="*50)
        agent_multi_eps = HierarchicalDQNLearner(
            env=env, max_episodes=args.episodes, eps_decay=0.999, use_ddqn=True, 
            extra_state_dims=num_phases
        )
        true_rewards_multi, total_rewards_multi, eps_histories_multi, buffers_multi, _ = run_sequential_training(
            env, agent_multi_eps, abstract_mdp, args.episodes, goal_reward=env_goal_reward, use_shaping=True, use_double_epsilon=True, 
            log_file="logs/multi_epsilon_training.log"
        )        
        # Generate individual plots in both 'multi' and 'comparison' modes
        print("Generating plots for multi epsilon run...")
        plot_buffer_fractions(buffers_multi, filename="img/buffer_fractions_multi_eps.png", window_size=plot_window_size)
        plot_shaping_reward_breakdown(true_rewards_multi, total_rewards_multi, eps_histories_multi, window_size=plot_window_size, 
            filename="img/reward_breakdown_multi_eps.png"
        )

    # --- Comparison Plot ---
    if args.mode == 'comparison':
        print("\nGenerating comparison plot...")
        plot_comparison_curves(
            baseline_rewards=true_rewards_single,
            shaping_rewards=true_rewards_multi,
            window_size=plot_window_size,
            filename="img/single_vs_multi_epsilon_comparison.png",
            title="Single Epsilon vs. Multi Epsilon Performance",
            baseline_label="Single Epsilon",
            shaping_label="Multi Epsilon"
        )

    env.close()
    print("End")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run N-Phase DQN training with different epsilon strategies.")
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
        default=1, 
        help="Number of training episodes."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="trajectory.json", 
        help="Path to the trajectory configuration file."
    )
    args = parser.parse_args()
    main(args)