import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from abstract_mdps import SequentialWaypointMDP
from agent import HierarchicalDQNLearner
from utils import (phi_mapping_sequential, save_sequential_heatmaps,
                   plot_training_results)

# =====================================================================
# PLOTTING UTILITIES
# =====================================================================

# =====================================================================
# PLOTTING UTILITIES
# =====================================================================

def plot_comparison_curves(baseline_rewards, shaping_rewards, epsilon_history, window_size=100, filename="img/baseline_vs_shaping.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    baseline_ma = pd.Series(baseline_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    shaping_ma = pd.Series(shaping_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    x_axis = np.arange(len(baseline_rewards))
        
    ax1.plot(x_axis, baseline_ma, color='black', linestyle='-', linewidth=2, label='Baseline (No Shaping)')
    ax1.plot(x_axis, shaping_ma, color='blue', linestyle='-', linewidth=2.5, label='Shaping Agent (Potential Guided)')
    
    ax1.set_title("Learning Curve Comparison: Baseline vs Shaping", fontsize=15, fontweight='bold')
    ax1.set_xlabel(f"Episode # (Moving Average Window = {window_size})", fontsize=12)
    ax1.set_ylabel("True Episode Reward", fontsize=12)
    ax1.axhline(y=100, color='green', linestyle=':', alpha=0.6, label='Goal Threshold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    if isinstance(epsilon_history, tuple):
        eps_q0, eps_q10 = epsilon_history
        ax2.plot(x_axis, eps_q0, color='orange', linestyle='--', linewidth=1.8, label='Epsilon Decay (Shaping q=0)')
        ax2.plot(x_axis, eps_q10, color='red', linestyle=':', linewidth=1.8, label='Epsilon Decay (Shaping q=10)')
    else:
        ax2.plot(x_axis, epsilon_history, color='orange', linestyle='--', linewidth=1.8, label='Epsilon Decay (Baseline)')

    ax2.set_ylabel("Exploration Rate (ε)", color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(-0.05, 1.05)

    # LEGENDA ESTERNA SOTTO IL GRAFICO
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, 
        loc="upper center", bbox_to_anchor=(0.5, -0.15), 
        ncol=2, fontsize=11, framealpha=1.0
    )
    
    fig.tight_layout()
    # bbox_inches='tight' evita che la legenda esterna venga tagliata
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n>>> Comparison plot successfully saved to: {filename}")
    plt.close(fig)

def plot_shaping_reward_breakdown(true_rewards, total_rewards, epsilon_history, window_size=100, filename="img/shaping_reward_breakdown.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    if len(true_rewards) >= window_size:
        true_ma = pd.Series(true_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
        total_ma = pd.Series(total_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    else:
        true_ma = true_rewards
        total_ma = total_rewards
    x_axis = np.arange(len(true_rewards))
        
    ax1.plot(x_axis, true_ma, color='green', linestyle='-', linewidth=2, label='True Environment Reward')
    ax1.plot(x_axis, total_ma, color='purple', linestyle='-', linewidth=2.5, label='Total Reward (Env + Shaping)')
    
    ax1.set_title("Shaping Agent Reward Analysis", fontsize=15, fontweight='bold')
    ax1.set_xlabel(f"Episode # (Moving Average Window = {window_size})", fontsize=12)
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    eps_q0_history, eps_q10_history = epsilon_history
    
    ax2.plot(x_axis, eps_q0_history, color='orange', linestyle='--', linewidth=1.8, label='Epsilon Decay (q=0)')
    ax2.plot(x_axis, eps_q10_history, color='red', linestyle=':', linewidth=1.8, label='Epsilon Decay (q=10)')
    
    ax2.set_ylabel("Exploration Rate (ε)", color='orange', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)

    # LEGENDA ESTERNA SOTTO IL GRAFICO
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, 
        loc="upper center", bbox_to_anchor=(0.5, -0.15), 
        ncol=2, fontsize=11, framealpha=1.0
    )
    
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n>>> Shaping reward breakdown plot successfully saved to: {filename}")
    plt.close(fig)

def plot_buffer_fractions(q0_history, q10_history, window_size=100, filename="img/buffer_fractions.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    q0_ma = pd.Series(q0_history).rolling(window=window_size, min_periods=1, center=True).mean()
    q10_ma = pd.Series(q10_history).rolling(window=window_size, min_periods=1, center=True).mean()
    x_axis = np.arange(len(q0_history))
    
    ax.plot(x_axis, q0_ma, color='blue', linewidth=2.5, label='Phase q=0 (Waypoint)')
    ax.plot(x_axis, q10_ma, color='green', linewidth=2.5, label='Phase q=10 (Goal)')
    
    ax.set_title(f"Replay Buffer Composition (Moving Average Window = {window_size})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Fraction of data in Buffer", fontsize=12)
    
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Ideal Balance (50%)')
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # LEGENDA ESTERNA SOTTO IL GRAFICO
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), 
        ncol=3, fontsize=11, framealpha=1.0
    )
    
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n>>> Replay Buffer plot saved to: {filename}")
    plt.close(fig)

def plot_double_epsilon_ablation(episodes, single_eps_goals, single_eps_history, 
                                 double_eps_goals, eps_q0_history, eps_q10_history, 
                                 window_size=100, filename="img/double_epsilon_impact.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    x_axis = np.arange(episodes)
    
    single_ma = pd.Series(single_eps_goals).rolling(window=window_size, min_periods=1, center=True).mean()
    double_ma = pd.Series(double_eps_goals).rolling(window=window_size, min_periods=1, center=True).mean()
    
    ax1.plot(x_axis, single_ma, color='red', linewidth=2.5, label='Shaping Agent (Single ε) - Success Rate')
    ax1.plot(x_axis, double_ma, color='green', linewidth=2.5, label='Shaping Agent (Double ε) - Success Rate')
    
    ax1.set_xlabel("Episode", fontsize=13)
    ax1.set_ylabel(f"Goal Reached Rate (Window={window_size})", fontsize=13)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.plot(x_axis, single_eps_history, color='red', linestyle=':', alpha=0.6, linewidth=2, label='Global ε Decay')
    ax2.plot(x_axis, eps_q0_history, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Proposed ε (q=0) Decay')
    ax2.plot(x_axis, eps_q10_history, color='darkgreen', linestyle='-.', alpha=0.8, linewidth=2.5, label='Proposed ε (q=10) Decay')
    
    ax2.set_ylabel("Exploration Rate (ε)", fontsize=13)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.title("Ablation Study: Impact of the Double Epsilon Mechanism", fontsize=15, fontweight='bold')
    
    # LEGENDA ESTERNA SOTTO IL GRAFICO SU DUE COLONNE (Avendo 5 item è perfetto)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, 
        loc="upper center", bbox_to_anchor=(0.5, -0.15), 
        ncol=2, fontsize=11, framealpha=1.0
    )
    
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n>>> Ablation plot successfully saved to: {filename}")
    plt.close(fig)

# =====================================================================
# TRAINING LOOP
# =====================================================================

def run_sequential_training(env, agent, abstract_mdp, episodes, use_shaping=True, use_double_epsilon=True, K=1.0, log_file=None, save_policy=True):
    """
    Executes the training loop for the sequential task.
    Allows toggling shaping and double epsilon independently.
    """
    true_episode_rewards = []
    total_episode_rewards = []
    
    # Epsilon initialization
    eps_q0 = agent.eps 
    eps_q10 = agent.eps 
    eps_single = agent.eps 
    eps_min = agent.eps_min
    eps_decay = agent.eps_decay
    
    eps_q0_history = []
    eps_q10_history = []
    buffer_q0_history = []
    buffer_q10_history = []
    eps_single_history = []
    episode_goal_hit_history = [] 

    log_handle = open(log_file, 'a') if log_file else None
    if log_handle:
        log_handle.write("="*50 + f"\nSTARTING NEW TRAINING RUN (Shaping: {use_shaping}, Double Eps: {use_double_epsilon})\n" + "="*50 + "\n")

    # Counters
    natural_q_updates = 0
    waypoint_hits = 0
    goal_hits = 0
    truncated_episodes = 0

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        q = 0 
        passed_trough_waypoint = False
        reached_q10_this_episode = False
        
        # Augment state for the Neural Network: environment state + one-hot q state
        q_one_hot = np.array([1.0, 0.0]) if q == 0 else np.array([0.0, 1.0])
        s_aug = np.concatenate((s_raw, q_one_hot)).astype(np.float32)
        
        terminated = truncated = False
        episode_true_reward = 0.0
        episode_total_reward = 0.0
        episode_goal_hit = 0 # Local flag for this episode

        while not (terminated or truncated):
            env_goal_reward = 0.0

            # EPSILON SELECTION BASED ON FLAG
            if use_double_epsilon:
                agent.eps = eps_q0 if q == 0 else eps_q10
            else:
                agent.eps = eps_single

            # Agent selects action based on the augmented state
            a = agent.select_action(s_aug)
            ns_raw, _, terminated, truncated, _ = env.step(a)

            if truncated:
                truncated_episodes += 1

            # Map continuous next state to 2D abstract state coordinates
            abstract_x_ns, abstract_y_ns, _ = phi_mapping_sequential(ns_raw, q)
            next_q = q

            # STATE TRANSITION LOGIC: Check if the NEXT state is the waypoint
            if abstract_x_ns == 1 and abstract_y_ns == 8 and q == 0:
                passed_trough_waypoint = True
                waypoint_hits += 1
                next_q = 10
                natural_q_updates += 1
                reached_q10_this_episode = True
                
            abstract_ns = (abstract_x_ns, abstract_y_ns, next_q)

            # Check if the final goal is reached (and waypoint was passed)
            if abstract_ns == abstract_mdp.goal_state and next_q == 10 and passed_trough_waypoint:
                goal_hits += 1
                env_goal_reward = 10000.0
                terminated = True
                episode_goal_hit = 1 # Record the success
            
            done = terminated or truncated
            episode_true_reward += env_goal_reward
            
            # Build the NEXT augmented state
            next_q_one_hot = np.array([1.0, 0.0]) if next_q == 0 else np.array([0.0, 1.0])
            ns_aug = np.concatenate((ns_raw, next_q_one_hot)).astype(np.float32)

            # Shaping Signal Calculation (Without Gamma)
            if use_shaping:
                abstract_s = phi_mapping_sequential(s_raw, q)
                shaping_signal = 0.0
                # Apply shaping only if the agent changes abstract cell or q phase
                if abstract_s != abstract_ns:
                    phi_s = abstract_mdp.v_star.get(abstract_s, 0.0)
                    phi_ns = abstract_mdp.v_star.get(abstract_ns, 0.0)
                    shaping_signal = K * (phi_ns - phi_s)
            else:
                shaping_signal = 0.0
            
            total_step_reward = env_goal_reward + shaping_signal
            episode_total_reward += total_step_reward
            
            # Push AUGMENTED states to memory and optimize
            agent.memory.push(s_aug, a, total_step_reward, ns_aug, done)
            agent.optimize_model()

            # Update states for the next step
            s_raw = ns_raw
            s_aug = ns_aug
            q = next_q
            
        # EPSILON DECAY BASED ON FLAG
        if use_double_epsilon:
            # eps_q0 always decays to stabilize the first phase
            eps_q0 = max(eps_min, eps_q0 * eps_decay)
            # eps_q10 decays ONLY IF the agent entered q=10 during this episode
            if reached_q10_this_episode and eps_q0 < 0.05:
                eps_q10 = max(eps_min, eps_q10 * eps_decay)
        else:
            # Baseline uses a single epsilon that always decays
            eps_single = max(eps_min, eps_single * eps_decay)
            
        true_episode_rewards.append(episode_true_reward)
        total_episode_rewards.append(episode_total_reward)
        eps_q0_history.append(eps_q0)
        eps_q10_history.append(eps_q10)
        buffer_q0_history.append(agent.memory.q0_fraction_onehot())
        buffer_q10_history.append(agent.memory.q1_fraction_onehot())
        eps_single_history.append(eps_single)
        episode_goal_hit_history.append(episode_goal_hit)

        # Print progress every 100 episodes
        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            recent_avg_with_shaping = np.mean(total_episode_rewards[-100:])
            mode_str = "SHAPING" if use_shaping else "BASELINE"
            eps_str = "DOUBLE EPS" if use_double_epsilon else "SINGLE EPS"

            log_string = (
                f"[{mode_str} | {eps_str}] Episode {n_episode + 1}/{episodes}\n" +
                f"  Avg Reward              : {recent_avg:.6f}\n" +
                (f"  Avg With Shaping Reward : {recent_avg_with_shaping:.6f}\n" if use_shaping else "") +
                (f"  Epsilon (q0, q10)       : {eps_q0:.6f}, {eps_q10:.6f}\n" if use_double_epsilon else f"  Epsilon (single)        : {eps_single:.6f}\n") +          
                f"  Exp q0 % and q10 %      : {agent.memory.q0_fraction_onehot():.6f}, {agent.memory.q1_fraction_onehot():.6f}\n" +
                f"  Waypoint hits           : {waypoint_hits}\n" +
                f"  Goal hits               : {goal_hits}\n"
            )
            print(log_string)
            if log_handle:
                log_handle.write(log_string + "\n")
                log_handle.flush()

        # Save model policy every 500 episodes
        if save_policy and (n_episode + 1) % 500 == 0:
            prefix = "shaping" if use_shaping else "baseline"
            eps_suffix = "double_eps" if use_double_epsilon else "single_eps"
            agent.policy_name = f"{prefix}_{eps_suffix}_policy_ep_{n_episode + 1}.pth"
            agent._save_policy()

    if log_handle:
        log_handle.close()

    # Return the correct values based on use_double_epsilon flag
    if use_double_epsilon:
        return (np.array(true_episode_rewards), np.array(total_episode_rewards), (np.array(eps_q0_history), np.array(eps_q10_history)),
                (np.array(buffer_q0_history), np.array(buffer_q10_history)), np.array(episode_goal_hit_history))
    else:
        return (np.array(true_episode_rewards), np.array(total_episode_rewards), np.array(eps_single_history), 
                (np.array(buffer_q0_history), np.array(buffer_q10_history)), np.array(episode_goal_hit_history))

# =====================================================================
# MAIN EXPERIMENT ORCHESTRATOR
# =====================================================================

def main():
    print("=== STARTING SEQUENTIAL TASK EXPERIMENT: ABLATION STUDY ===")
    os.makedirs("logs", exist_ok=True)

    # HYPERPARAMETERS
    episodes = 7000
    gamma = 0.99
    eps_decay = 0.999
    K_scaling = 1
    
    print("\n1. Initializing Environment and Abstract MDP...")
    env = gym.make("LunarLander-v3", continuous=False)
    
    abstract_mdp = SequentialWaypointMDP(width=12, height=12, gamma=gamma)
    abstract_mdp.value_iteration()

    print("   -> Plotting Value Functions (V*) Heatmaps...")
    save_sequential_heatmaps(abstract_mdp, filename_prefix="seq_experiment")
    
    # -----------------------------------------------------------------
    # EXPERIMENT 1: SHAPING AGENT (SINGLE EPSILON - ABLATION BASELINE)
    # -----------------------------------------------------------------
    print("\n=======================================================")
    print("TRAINING: SHAPING AGENT (SINGLE EPSILON)")
    print("=======================================================")
    agent_shaping_single = HierarchicalDQNLearner(
        env=env,
        max_episodes=episodes,
        gamma=gamma,
        eps_decay=eps_decay, # Uses the same decay rate as the double epsilon setup
        use_ddqn=True,
        policy_name="shaping_single_eps_policy.pth",
        extra_state_dims=2
    )
    
    single_learning_curve, _, single_eps_history, _, single_goals = run_sequential_training(
        env, 
        agent_shaping_single, 
        abstract_mdp, 
        episodes, 
        use_shaping=True,            # Shaping ACTIVE
        use_double_epsilon=False,    # Double Epsilon DISABLED
        K=K_scaling,
        log_file="logs/shaping_single_eps_training.log",
        save_policy=True
    )

    # -----------------------------------------------------------------
    # EXPERIMENT 2: SHAPING AGENT (DOUBLE EPSILON - PROPOSED)
    # -----------------------------------------------------------------
    print("\n=======================================================")
    print("TRAINING: SHAPING AGENT (DOUBLE EPSILON)")
    print("=======================================================")
    agent_shaping_double = HierarchicalDQNLearner(
        env=env,
        max_episodes=episodes,
        gamma=gamma,
        eps_decay=eps_decay,
        use_ddqn=True,
        policy_name="shaping_double_eps_policy.pth",
        extra_state_dims=2
    )
    
    double_learning_curve, double_total_rewards, double_eps_history, double_buffer_history, double_goals = run_sequential_training(
        env, 
        agent_shaping_double, 
        abstract_mdp, 
        episodes, 
        use_shaping=True,           # Shaping ACTIVE
        use_double_epsilon=True,    # Double Epsilon ACTIVE
        K=K_scaling,
        log_file="logs/shaping_double_eps_training.log"
    )
    
    # -----------------------------------------------------------------
    # PLOTTING RESULTS
    # -----------------------------------------------------------------
    print("\n3. Generating plots...")
    
    # Ablation Plot (The core comparison for the thesis)
    eps_q0_history, eps_q10_history = double_eps_history
    plot_double_epsilon_ablation(
        episodes=episodes, 
        single_eps_goals=single_goals, 
        single_eps_history=single_eps_history, 
        double_eps_goals=double_goals, 
        eps_q0_history=eps_q0_history, 
        eps_q10_history=eps_q10_history,
        window_size=500, 
        filename="img/double_epsilon_ablation.png"
    )
    
    # Standard metrics for the proposed Double Epsilon agent
    buf_q0, buf_q10 = double_buffer_history
    plot_shaping_reward_breakdown(double_learning_curve, double_total_rewards, double_eps_history, window_size=500, filename="img/shaping_reward_breakdown_proposed.png")
    plot_buffer_fractions(buf_q0, buf_q10, window_size=500, filename="img/buffer_fractions_proposed.png")
    
    env.close()

if __name__ == "__main__":
    main()