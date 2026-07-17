import os
import gymnasium as gym
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd

from abstract_mdps import NPhaseWaypointMDP
from agent import HierarchicalDQNLearner
from utils import (phi_mapping_sequential, save_sequential_heatmaps)

# =====================================================================
# PLOTTING UTILITY FOR COMPARISON
# =====================================================================

def plot_comparison_curves(baseline_rewards, shaping_rewards, epsilon_history, window_size=100, filename="img/baseline_vs_shaping.png"):
    """
    Plots the smoothed moving average of the baseline agent against the shaping agent,
    including the epsilon decay on a secondary axis.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Calculate the moving average using pandas for correct edge handling
    baseline_ma = pd.Series(baseline_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    shaping_ma = pd.Series(shaping_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    x_axis = np.arange(len(baseline_rewards))
        
    # Plot curves on the primary y-axis (ax1)
    ax1.plot(x_axis, baseline_ma, color='black', linestyle='-', linewidth=2, label='Baseline (No Shaping)')
    ax1.plot(x_axis, shaping_ma, color='blue', linestyle='-', linewidth=2.5, label='Shaping Agent (Potential Guided)')
    
    # Formatting
    ax1.set_title("Learning Curve Comparison: Baseline vs. Shaping", fontsize=15, fontweight='bold')
    ax1.set_xlabel(f"Episode # (Moving Average Window = {window_size})", fontsize=12)
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.axhline(y=100, color='green', linestyle=':', alpha=0.6, label='Goal Threshold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Secondary axis for Epsilon
    ax2 = ax1.twinx()
    ax2.plot(x_axis, epsilon_history, color='orange', linestyle='--', linewidth=1.8, label='Epsilon Decay')
    ax2.set_ylabel("Exploration Rate (ε)", color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(0, 1.05)

    # Merge legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, 
        labels1 + labels2, 
        loc="lower right", 
        fontsize=11
    )
    
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n>>> Comparison plot successfully saved to: {filename}")
    plt.close(fig)

def plot_shaping_reward_breakdown(true_rewards, total_rewards, epsilon_history, window_size=100, filename="img/shaping_reward_breakdown.png"):
    """
    Plots the true environment reward vs the total reward (env + shaping) for the shaping agent on the same graph.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Calculate moving averages
    if len(true_rewards) >= window_size:
        true_ma = pd.Series(true_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
        total_ma = pd.Series(total_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    else:
        # Fallback if episode count is less than window size
        true_ma = true_rewards
        total_ma = total_rewards
    x_axis = np.arange(len(true_rewards))
        
    # Plot curves
    ax1.plot(x_axis, true_ma, color='green', linestyle='-', linewidth=2, label='True Environment Reward')
    ax1.plot(x_axis, total_ma, color='purple', linestyle='-', linewidth=2.5, label='Total Reward (Env + Shaping)')
    
    # Formatting
    ax1.set_title("Shaping Agent Reward Analysis", fontsize=15, fontweight='bold')
    ax1.set_xlabel(f"Episode # (Moving Average Window = {window_size})", fontsize=12)
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Secondary axis for Epsilon
    ax2 = ax1.twinx()
    ax2.plot(x_axis, epsilon_history, color='orange', linestyle='--', linewidth=1.8, label='Epsilon Decay')
    ax2.set_ylabel("Exploration Rate (ε)", color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(0, 1.05)

    # Merge legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=11)
    
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n>>> Shaping reward breakdown plot successfully saved to: {filename}")
    plt.close(fig)

def plot_buffer_fractions(buffer_histories, window_size=100, filename="img/buffer_fractions.png"):
    """
    Plots the replay buffer composition for N phases dynamically.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if not buffer_histories or not buffer_histories[0]:
        print("Warning: Buffer history is empty, skipping plot generation.")
        plt.close(fig)
        return
        
    x_axis = np.arange(len(buffer_histories[0]))
    num_phases = len(buffer_histories)
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_phases))
    for idx, history in enumerate(buffer_histories):
        ma = pd.Series(history).rolling(window=window_size, min_periods=1, center=True).mean()
        label = "Goal" if idx == num_phases - 1 else f"WP {idx + 1}"
        ax.plot(x_axis, ma, color=colors[idx], linewidth=2.5, label=f'Phase q={idx} ({label})')
    
    ax.set_title(f"Replay Buffer Composition (MA Window = {window_size})", fontsize=14, fontweight='bold')
    ax.set_ylabel("Fraction in Buffer", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=num_phases, fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n>>> Buffer fractions plot successfully saved to: {filename}")

# =====================================================================
# TRAINING LOOP
# =====================================================================

def run_sequential_training(env, agent, abstract_mdp, episodes, use_shaping=True, goal_reward=10000, use_replication=False, replication_episodes=0, K=1.0, log_file=None, debug_replication=False):
    """
    Executes the training loop for the sequential task.
    """
    num_phases = abstract_mdp.num_phases
    true_episode_rewards = []
    total_episode_rewards = []
    epsilon_history = []
    buffer_histories = [[] for _ in range(num_phases)]
    
    # Log file
    log_handle = open(log_file, 'a') if log_file else None
    if log_handle:
        log_handle.write("="*50 + f"\nSTARTING NEW TRAINING RUN\n" + "="*50 + "\n")

    # Counters
    total_hits = [0] * num_phases

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        
        # Initialize sequence variable 'q'
        q = 0 
        
        # Dynamic One-Hot encoding array
        q_one_hot = np.zeros(num_phases, dtype=np.float32)
        q_one_hot[q] = 1.0
        s_aug = np.concatenate((s_raw, q_one_hot)).astype(np.float32)
        
        terminated = truncated = False
        episode_true_reward = 0.0
        episode_total_reward = 0.0

        # Episode loop
        while not (terminated or truncated):
            q_before_transition = q
            
            # Agent selects action based on the augmented state
            a = agent.select_action(s_aug)
            ns_raw, _, terminated, truncated, _ = env.step(a)

            # Map continuous state to 3D abstract state (x, y, q)
            abstract_x_ns, abstract_y_ns, _ = phi_mapping_sequential(ns_raw, q_before_transition)
            next_q = q_before_transition
            env_goal_reward = 0.0

            # Dynamic Phase Transition Checking
            if q_before_transition < num_phases - 1:
                target_x, target_y = abstract_mdp.waypoints[q_before_transition]
                if abstract_x_ns == target_x and abstract_y_ns == target_y:
                    total_hits[q_before_transition] += 1
                    next_q = q_before_transition + 1
            else:
                # Final Goal Check
                goal_x, goal_y = abstract_mdp.waypoints[-1]
                if abstract_x_ns == goal_x and abstract_y_ns == goal_y:
                    total_hits[q_before_transition] += 1
                    env_goal_reward = goal_reward
                    terminated = True

            abstract_ns = (abstract_x_ns, abstract_y_ns, next_q)
            done = terminated or truncated
            episode_true_reward += env_goal_reward
            
            # Next One-Hot 
            next_q_one_hot = np.zeros(num_phases, dtype=np.float32)
            next_q_one_hot[next_q] = 1.0
            ns_aug = np.concatenate((ns_raw, next_q_one_hot)).astype(np.float32)

            # --- Shaping Signal Calculation (Discrete) ---
            if use_shaping:
                # Calculate current and next abstract state
                abstract_s = phi_mapping_sequential(s_raw, q_before_transition)
                
                shaping_signal = 0.0
                # Apply shaping only if the agent changes abstract cell
                if abstract_s != abstract_ns:
                    phi_s = abstract_mdp.v_star.get(abstract_s, 0.0)
                    phi_ns = abstract_mdp.v_star.get(abstract_ns, 0.0)
                    shaping_signal = K * (phi_ns - phi_s)
            else:
                shaping_signal = 0.0
            
            total_step_reward = env_goal_reward + shaping_signal
            episode_total_reward += total_step_reward
            
            # Push AUGMENTED states
            agent.memory.push(s_aug, a, total_step_reward, ns_aug, done)
            
            # --- EXPERIENCE REPLICATION ---
            # If replication is active, for each real transition (s,a,s') in phase q,
            # we replicate it for all other phases q_ != q.
            if use_replication and (n_episode < replication_episodes):
                for replicated_q_idx in range(num_phases):
                    # Skip replication for the phase where the real transition occurred
                    if replicated_q_idx != q_before_transition:
                        # 1. Create augmented states for the replicated phase
                        q_rep_one_hot = np.zeros(num_phases, dtype=np.float32)
                        q_rep_one_hot[replicated_q_idx] = 1.0
                        s_aug_rep = np.concatenate((s_raw, q_rep_one_hot)).astype(np.float32)

                        # The replicated next phase is the same, as there is no phase transition
                        ns_aug_rep = np.concatenate((ns_raw, q_rep_one_hot)).astype(np.float32)

                        # 2. Calculate the shaping signal for the replicated transition
                        abstract_s_rep = phi_mapping_sequential(s_raw, replicated_q_idx)
                        abstract_ns_rep = phi_mapping_sequential(ns_raw, replicated_q_idx)
                        phi_s_rep = abstract_mdp.v_star.get(abstract_s_rep, 0.0)
                        phi_ns_rep = abstract_mdp.v_star.get(abstract_ns_rep, 0.0)
                        shaping_signal_rep = K * (phi_ns_rep - phi_s_rep)
                        
                        # 3. Add the replicated transition to the replay buffer
                        # We use the 'done' from the original transition
                        agent.memory.push(s_aug_rep, a, shaping_signal_rep, ns_aug_rep, done)

            # Optimize model
            agent.optimize_model()

            s_raw = ns_raw
            s_aug = ns_aug
            
        # Decay epsilon at the end of the episode
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)
        total_episode_rewards.append(episode_total_reward)
        epsilon_history.append(agent.eps)
        
        # Log buffer fractions at the end of the episode
        for i in range(num_phases):
            buffer_histories[i].append(agent.memory.q_fraction_onehot(i, num_phases))

        # Print progress every 100 episodes
        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            recent_avg_with_shaping = np.mean(total_episode_rewards[-100:])
            
            # Add an indicator to know if replication is active in this block of episodes
            replication_status = "ON" if use_replication and (n_episode < replication_episodes) else "OFF"
            
            mode_str = f"SHAPING (Replica: {replication_status})" if use_shaping else "BASELINE"

            # Dynamic labels: WP1, WP2, ..., Goal
            labels = [f"WP{i+1}" for i in range(num_phases - 1)] + ["Goal"]
            buffer_details = ", ".join([f"q{i}({labels[i]}): {agent.memory.q_fraction_onehot(i, num_phases):.2%}" for i in range(num_phases)])
            hits_details = ", ".join([f"{labels[i]}: {total_hits[i]}" for i in range(num_phases)])

            log_string = (
                f"[{mode_str}] Episode {n_episode + 1}/{episodes}\n" +
                f"  Avg Reward              : {recent_avg:.6f}\n" +
                f"  Avg With Shaping Reward : {recent_avg_with_shaping:.6f}\n" +
                f"  Epsilon                 : {agent.eps:.6f}\n" +                
                f"  Buffer Fractions        : {buffer_details}\n" +
                f"  Hits (Cumulative)       : {hits_details}\n"
            )

            print(log_string)
            if log_handle:
                log_handle.write(log_string + "\n")
                log_handle.flush() # Ensure data is written immediately

        # Policy saving
        if n_episode + 1 % 250 == 0:
            agent.policy_name = f"replica_policy_ep_{n_episode + 1}.pth"
            agent._save_policy()

    
    if log_handle:
        log_handle.close()

    return np.array(true_episode_rewards), np.array(total_episode_rewards), np.array(epsilon_history), buffer_histories

# =====================================================================
# MAIN EXPERIMENT ORCHESTRATOR
# =====================================================================

def main():
    print("=== STARTING SEQUENTIAL TASK EXPERIMENT WITH REPLICATION ===")
    os.makedirs("logs", exist_ok=True)
    
    # HYPERPARAMETERS
    episodes = 100
    goal_reward = 10000
    gamma = 0.999
    eps_decay = 0.9995
    K_scaling = 1
    replication_episodes_count = episodes // 2
    
    print("\n1. Initializing Environment and Abstract MDP...")
    env = gym.make("LunarLander-v3", continuous=False)
    
    # Loading configuration from JSON
    with open('trajectory.json', 'r') as f:
        config = json.load(f)
    route_waypoints = [tuple(wp) for wp in config['waypoints']]
    print(f"Training for this trajectory: {route_waypoints}")
    num_phases = len(route_waypoints)

    # -----------------------------------------------------------------
    # PLOTTING RESULTS
    # -----------------------------------------------------------------
    abstract_mdp = NPhaseWaypointMDP(waypoints=route_waypoints, gamma=gamma, goal_reward=goal_reward)
    abstract_mdp.value_iteration()

    print("   -> Plotting Value Functions (V*) Heatmaps...")
    save_sequential_heatmaps(abstract_mdp, filename_prefix="replication_exp")
    print("\n=======================================================")
    print(f"TRAINING: AGENT WITH REPLICATION FOR THE FIRST {replication_episodes_count} EPISODES")
    print("=======================================================")
    agent_unified = HierarchicalDQNLearner(
        env=env,
        max_episodes=episodes,
        gamma=gamma,
        eps_decay=eps_decay,
        use_ddqn=True,
        policy_name="shaping_half_replication_policy.pth",
        extra_state_dims=num_phases
    )
    
    learning_curve, total_rewards, eps_history, buffer_history = run_sequential_training(
        env,
        agent_unified,
        abstract_mdp,
        episodes,
        use_shaping=True,
        use_replication=True,
        replication_episodes=replication_episodes_count,
        K=K_scaling,
        goal_reward = goal_reward,
        log_file=f"logs/shaping_and_replica_{replication_episodes_count}_episodes.log"
    )
    
    # -----------------------------------------------------------------
    # PLOTTING RESULTS
    # -----------------------------------------------------------------
    print("\n3. Generating plots...")
    plot_shaping_reward_breakdown(learning_curve, total_rewards, eps_history, window_size=500, filename="img/shaping_half_replication_breakdown.png")
    plot_buffer_fractions(buffer_history, window_size=100, filename="img/buffer_fractions_replication.png")

    env.close()

if __name__ == "__main__":
    main()