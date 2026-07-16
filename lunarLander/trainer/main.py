import os
import gymnasium as gym
import numpy as np
import argparse
import json

from abstract_mdps import NPhaseWaypointMDP
from agent import HierarchicalDQNLearner
from utils import phi_mapping_sequential, save_sequential_heatmaps, plot_buffer_fractions, plot_shaping_reward_breakdown

def run_sequential_training(env, agent, abstract_mdp, episodes, use_shaping=True, use_double_epsilon=True, K=1.0, log_file=None):
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
    if log_handle: log_handle.write(f"=== NEW RUN (Shaping: {use_shaping}, Multi-Eps: {use_double_epsilon}) ===\n")

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
                    env_goal_reward = 10000.0
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

    if log_handle: log_handle.close()
    return true_episode_rewards, total_episode_rewards, eps_histories, buffer_histories, hits_history

def main():
    print("=== STARTING N-PHASE ABLATION STUDY ===")
    os.makedirs("logs", exist_ok=True)
    env = gym.make("LunarLander-v3", continuous=False)

    # Hyperparameters
    episodes = 10000
    eps_decay = 0.999
    gamma = 0.99
    
    # Loading configuration
    with open('trajectory.json', 'r') as f:
        config = json.load(f)
    route_waypoints = [tuple(wp) for wp in config['waypoints']]
    print(f"Training for this trajectory {route_waypoints}")
    num_phases = len(route_waypoints)
    
    abstract_mdp = NPhaseWaypointMDP(waypoints=route_waypoints, gamma=gamma)
    abstract_mdp.value_iteration()
    save_sequential_heatmaps(abstract_mdp)

    agent_shaping = HierarchicalDQNLearner(
        env=env, max_episodes=episodes, eps_decay=eps_decay, use_ddqn=True, 
        extra_state_dims=num_phases # Tells the agent the size of the one-hot array
    )
    
    true_rewards, total_rewards, eps_histories, buffers, hits = run_sequential_training(
        env, agent_shaping, abstract_mdp, episodes, 
        use_shaping=True, use_double_epsilon=True, 
        log_file="logs/n_phase_training.log"
    )
    
    plot_buffer_fractions(buffers, filename="img/buffer_fractions_n_phases.png")
    plot_shaping_reward_breakdown(
        true_rewards, 
        total_rewards, 
        eps_histories, 
        window_size=500, 
        filename="img/reward_breakdown_n_phases.png"
    )
    env.close()

if __name__ == "__main__":
    main()