import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from abstract_mdps import SequentialWaypointMDP
from agent import HierarchicalDQNLearner
from utils import (phi_mapping_sequential)

# =====================================================================
# PLOTTING UTILITIES (Aggiornate per 3 Fasi)
# =====================================================================
def save_sequential_heatmaps(abstract_mdp, filename_prefix="v_star", width=12, height=12, vmin=None, vmax=None):
    """
    Generates and saves THREE separate heatmaps for V*: 
    One for q=0 (WP1), q=1 (WP2), and q=2 (Goal).
    """
    os.makedirs("img/heatmaps", exist_ok=True)
    
    v_matrix_q0 = np.zeros((height, width))
    v_matrix_q1 = np.zeros((height, width))
    v_matrix_q2 = np.zeros((height, width))
    
    for (x, y, q), value in abstract_mdp.v_star.items():
        if 0 <= x < width and 0 <= y < height:
            if q == 0:
                v_matrix_q0[y, x] = value
            elif q == 1:
                v_matrix_q1[y, x] = value
            elif q == 2:
                v_matrix_q2[y, x] = value

    all_values = np.concatenate([v_matrix_q0.flatten(), v_matrix_q1.flatten(), v_matrix_q2.flatten()])
    computed_vmin = all_values.min() if vmin is None else vmin
    computed_vmax = all_values.max() if vmax is None else vmax

    def plot_single_heatmap(matrix, q_val, title, filename):
        plt.figure(figsize=(9, 8))
        im = plt.imshow(matrix, cmap='viridis', origin='lower', vmin=computed_vmin, vmax=computed_vmax)
        
        for y in range(height):
            for x in range(width):
                val = matrix[y, x]
                if val > 0.0: 
                    text_color = 'white' if val < (np.max(matrix) / 2) else 'black'
                    plt.text(x, y, f"{val:.2f}", ha='center', va='center', 
                             color=text_color, fontsize=7, fontweight='bold')
                    
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Potential Value (V*)")
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("X (Horizontal Position)", fontsize=12)
        plt.ylabel("Y (Altitude)", fontsize=12)
        
        plt.xticks(np.arange(0, width, 1))
        plt.yticks(np.arange(0, height, 1))
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-.5, height, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1, alpha=0.4)
        ax.grid(which='major', color='none')
        
        if q_val == 0:
            way_x, way_y = abstract_mdp.waypoint1
            plt.plot(way_x, way_y, 'ro', markersize=15, alpha=0.6, label="Waypoint 1 (q=0)")
            plt.legend(loc="upper right")
        elif q_val == 1:
            way_x, way_y = abstract_mdp.waypoint2
            plt.plot(way_x, way_y, 'mo', markersize=15, alpha=0.6, label="Waypoint 2 (q=1)")
            plt.legend(loc="upper right")
        elif q_val == 2:
            goal_x, goal_y, _ = abstract_mdp.goal_state
            plt.plot(goal_x, goal_y, 'go', markersize=15, alpha=0.6, label="Final Goal (q=2)")
            plt.legend(loc="upper right")
            
        plt.tight_layout()
        plt.savefig(f"img/heatmaps/{filename}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(" -> Generating V* Heatmap for Q=0...")
    plot_single_heatmap(v_matrix_q0, 0, "Potential Map (V*) - Phase q=0 (Seek WP1)", f"{filename_prefix}_q0")
    print(" -> Generating V* Heatmap for Q=1...")
    plot_single_heatmap(v_matrix_q1, 1, "Potential Map (V*) - Phase q=1 (Seek WP2)", f"{filename_prefix}_q1")
    print(" -> Generating V* Heatmap for Q=2...")
    plot_single_heatmap(v_matrix_q2, 2, "Potential Map (V*) - Phase q=2 (Seek Goal)", f"{filename_prefix}_q2")

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
    # Gestisce sia il caso di una singola history che della tripla per il Multi-Epsilon
    if isinstance(epsilon_history, tuple):
        eps_q0, eps_q1, eps_q2 = epsilon_history
        ax2.plot(x_axis, eps_q0, color='orange', linestyle='--', linewidth=1.8, label='ε Decay (q=0)')
        ax2.plot(x_axis, eps_q1, color='red', linestyle=':', linewidth=1.8, label='ε Decay (q=1)')
        ax2.plot(x_axis, eps_q2, color='brown', linestyle='-.', linewidth=1.8, label='ε Decay (q=2)')
    else:
        ax2.plot(x_axis, epsilon_history, color='orange', linestyle='--', linewidth=1.8, label='Epsilon Decay (Single)')
    
    ax2.set_ylabel("Exploration Rate (ε)", color='orange', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)

    ax2.tick_params(axis='y', labelcolor='orange')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, 
        loc="upper center", bbox_to_anchor=(0.5, -0.15), 
        ncol=2, fontsize=11, framealpha=1.0
    )
    
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_buffer_fractions(q0_history, q1_history, q2_history, window_size=100, filename="img/buffer_fractions.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    q0_ma = pd.Series(q0_history).rolling(window=window_size, min_periods=1, center=True).mean()
    q1_ma = pd.Series(q1_history).rolling(window=window_size, min_periods=1, center=True).mean()
    q2_ma = pd.Series(q2_history).rolling(window=window_size, min_periods=1, center=True).mean()
    x_axis = np.arange(len(q0_history))
    
    ax.plot(x_axis, q0_ma, color='blue', linewidth=2.5, label='Phase q=0 (WP1)')
    ax.plot(x_axis, q1_ma, color='green', linewidth=2.5, label='Phase q=1 (WP2)')
    ax.plot(x_axis, q2_ma, color='purple', linewidth=2.5, label='Phase q=2 (Goal)')
    
    ax.set_title(f"Replay Buffer Composition (Moving Average Window = {window_size})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Fraction of data in Buffer", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.33, color='gray', linestyle=':', alpha=0.7, label='Ideal Balance (33%)')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=11, framealpha=1.0)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_ablation_slide_phase1(episodes, single_wp, single_eps, double_wp, eps_q0, eps_q1, eps_q2, window_size=100, filename="img/slide_ablation_phase1_wp1.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x_axis = np.arange(episodes)
    
    single_wp_ma = pd.Series(single_wp).rolling(window=window_size, min_periods=1, center=True).mean() * 100
    double_wp_ma = pd.Series(double_wp).rolling(window=window_size, min_periods=1, center=True).mean() * 100
    
    ax1.plot(x_axis, single_wp_ma, color='red', linewidth=3.0, label='Single ε - WP1 Success')
    ax1.plot(x_axis, double_wp_ma, color='green', linewidth=3.0, label='Multi ε - WP1 Success')
    
    ax1.set_title("Phase 1: Waypoint 1 Reached Rate vs Exploration", fontsize=16, fontweight='bold')
    ax1.set_ylabel(f"Success Rate % (Window={window_size})", fontsize=14)
    ax1.set_xlabel("Episode", fontsize=14)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.plot(x_axis, single_eps, color='red', linestyle=':', alpha=0.5, linewidth=2, label='Global ε Decay')
    ax2.plot(x_axis, eps_q0, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='ε (q=0) Decay')
    ax2.plot(x_axis, eps_q1, color='darkgreen', linestyle='-.', alpha=0.8, linewidth=2, label='ε (q=1) Decay')
    ax2.plot(x_axis, eps_q2, color='purple', linestyle=':', alpha=0.8, linewidth=2.5, label='ε (q=2) Decay')
    ax2.set_ylabel("Exploration Rate (ε)", fontsize=14)
    ax2.set_ylim(-0.05, 1.05)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=12, framealpha=1.0)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight') 
    plt.close(fig)

def plot_ablation_slide_phase2(episodes, single_wp2, single_eps, double_wp2, eps_q0, eps_q1, eps_q2, window_size=100, filename="img/slide_ablation_phase2_wp2.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x_axis = np.arange(episodes)
    
    single_wp2_ma = pd.Series(single_wp2).rolling(window=window_size, min_periods=1, center=True).mean() * 100
    double_wp2_ma = pd.Series(double_wp2).rolling(window=window_size, min_periods=1, center=True).mean() * 100
    
    ax1.plot(x_axis, single_wp2_ma, color='red', linewidth=3.0, label='Single ε - WP2 Success')
    ax1.plot(x_axis, double_wp2_ma, color='green', linewidth=3.0, label='Multi ε - WP2 Success')
    
    ax1.set_title("Phase 2: Waypoint 2 Reached Rate vs Exploration", fontsize=16, fontweight='bold')
    ax1.set_ylabel(f"Success Rate % (Window={window_size})", fontsize=14)
    ax1.set_xlabel("Episode", fontsize=14)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.plot(x_axis, single_eps, color='red', linestyle=':', alpha=0.5, linewidth=2, label='Global ε Decay')
    ax2.plot(x_axis, eps_q0, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='ε (q=0) Decay')
    ax2.plot(x_axis, eps_q1, color='darkgreen', linestyle='-.', alpha=0.8, linewidth=2, label='ε (q=1) Decay')
    ax2.plot(x_axis, eps_q2, color='purple', linestyle=':', alpha=0.8, linewidth=2.5, label='ε (q=2) Decay')
    ax2.set_ylabel("Exploration Rate (ε)", fontsize=14)
    ax2.set_ylim(-0.05, 1.05)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=12, framealpha=1.0)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_ablation_slide_phase3(episodes, single_goal, single_eps, double_goal, eps_q0, eps_q1, eps_q2, window_size=100, filename="img/slide_ablation_phase3_goal.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x_axis = np.arange(episodes)
    
    single_gl_ma = pd.Series(single_goal).rolling(window=window_size, min_periods=1, center=True).mean() * 100
    double_gl_ma = pd.Series(double_goal).rolling(window=window_size, min_periods=1, center=True).mean() * 100
    
    ax1.plot(x_axis, single_gl_ma, color='red', linewidth=3.0, label='Single ε - Goal Success')
    ax1.plot(x_axis, double_gl_ma, color='green', linewidth=3.0, label='Multi ε - Goal Success')
    
    ax1.set_title("Phase 3: Final Goal Reached Rate vs Exploration", fontsize=16, fontweight='bold')
    ax1.set_ylabel(f"Success Rate % (Window={window_size})", fontsize=14)
    ax1.set_xlabel("Episode", fontsize=14)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.plot(x_axis, single_eps, color='red', linestyle=':', alpha=0.5, linewidth=2, label='Global ε Decay')
    ax2.plot(x_axis, eps_q0, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='ε (q=0) Decay')
    ax2.plot(x_axis, eps_q1, color='darkgreen', linestyle='-.', alpha=0.8, linewidth=2, label='ε (q=1) Decay')
    ax2.plot(x_axis, eps_q2, color='purple', linestyle=':', alpha=0.8, linewidth=2.5, label='ε (q=2) Decay')
    ax2.set_ylabel("Exploration Rate (ε)", fontsize=14)
    ax2.set_ylim(-0.05, 1.05)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=12, framealpha=1.0)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# =====================================================================
# TRAINING LOOP (Aggiornato per 3 Fasi LTLf)
# =====================================================================

def run_sequential_training(env, agent, abstract_mdp, episodes, use_shaping=True, use_double_epsilon=True, K=1.0, log_file=None, save_policy=True):
    true_episode_rewards = []
    total_episode_rewards = []
    
    # Inizializzazione Multi-Epsilon
    eps_q0 = agent.eps 
    eps_q1 = agent.eps 
    eps_q2 = agent.eps 
    eps_single = agent.eps 
    eps_min = agent.eps_min
    eps_decay = agent.eps_decay
    
    eps_q0_history, eps_q1_history, eps_q2_history, eps_single_history = [], [], [], []
    buffer_q0_history, buffer_q1_history, buffer_q2_history = [], [], []
    
    episode_wp1_hit_history = []
    episode_wp2_hit_history = []
    episode_goal_hit_history = [] 

    log_handle = open(log_file, 'a') if log_file else None
    if log_handle:
        log_handle.write("="*50 + f"\nSTARTING NEW TRAINING RUN (Shaping: {use_shaping}, Multi-Eps: {use_double_epsilon})\n" + "="*50 + "\n")

    wp1_hits = wp2_hits = goal_hits = 0

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        q = 0 
        passed_wp1 = False
        passed_wp2 = False
        
        reached_q1_this_episode = False
        reached_q2_this_episode = False
        
        # One-hot dinamico a 3 dimensioni
        q_one_hot = np.zeros(3, dtype=np.float32)
        q_one_hot[q] = 1.0
        s_aug = np.concatenate((s_raw, q_one_hot)).astype(np.float32)
        
        terminated = truncated = False
        episode_true_reward = episode_total_reward = 0.0
        episode_wp1_hit = episode_wp2_hit = episode_goal_hit = 0

        while not (terminated or truncated):
            env_goal_reward = 0.0

            # SELEZIONE EPSILON
            if use_double_epsilon:
                if q == 0: agent.eps = eps_q0
                elif q == 1: agent.eps = eps_q1
                else: agent.eps = eps_q2
            else:
                agent.eps = eps_single

            a = agent.select_action(s_aug)
            ns_raw, _, terminated, truncated, _ = env.step(a)

            abstract_x_ns, abstract_y_ns, _ = phi_mapping_sequential(ns_raw, q)
            next_q = q

            # LOGICA DI TRANSIZIONE (3 Fasi)
            if abstract_x_ns == abstract_mdp.waypoint1[0] and abstract_y_ns == abstract_mdp.waypoint1[1] and q == 0:
                passed_wp1 = True
                wp1_hits += 1
                next_q = 1
                reached_q1_this_episode = True
                episode_wp1_hit = 1 
                
            elif abstract_x_ns == abstract_mdp.waypoint2[0] and abstract_y_ns == abstract_mdp.waypoint2[1] and q == 1:
                passed_wp2 = True
                wp2_hits += 1
                next_q = 2
                reached_q2_this_episode = True
                episode_wp2_hit = 1 

            abstract_ns = (abstract_x_ns, abstract_y_ns, next_q)

            if abstract_ns == abstract_mdp.goal_state and next_q == 2 and passed_wp1 and passed_wp2:
                goal_hits += 1
                env_goal_reward = 10000.0
                terminated = True
                episode_goal_hit = 1 
            
            done = terminated or truncated
            episode_true_reward += env_goal_reward
            
            # Next One-Hot
            next_q_one_hot = np.zeros(3, dtype=np.float32)
            next_q_one_hot[next_q] = 1.0
            ns_aug = np.concatenate((ns_raw, next_q_one_hot)).astype(np.float32)

            if use_shaping:
                abstract_s = phi_mapping_sequential(s_raw, q)
                shaping_signal = 0.0
                if abstract_s != abstract_ns:
                    phi_s = abstract_mdp.v_star.get(abstract_s, 0.0)
                    phi_ns = abstract_mdp.v_star.get(abstract_ns, 0.0)
                    shaping_signal = K * (phi_ns - phi_s)
            else:
                shaping_signal = 0.0
            
            total_step_reward = env_goal_reward + shaping_signal
            episode_total_reward += total_step_reward
            
            agent.memory.push(s_aug, a, total_step_reward, ns_aug, done)
            agent.optimize_model()

            s_raw = ns_raw
            s_aug = ns_aug
            q = next_q
            
        # DECADIMENTO MULTI-EPSILON A CASCATA
        if use_double_epsilon:
            #eps_q0 = max(eps_min, eps_q0 * eps_decay)
            eps_q0 = max(0.08, eps_q0 * eps_decay)
            if reached_q1_this_episode and eps_q0 <= 0.081:
                #eps_q1 = max(eps_min, eps_q1 * eps_decay)
                eps_q1 = max(0.08, eps_q1 * eps_decay)
            if reached_q2_this_episode and eps_q1 <= 0.081:
                eps_q2 = max(eps_min, eps_q2 * eps_decay)
        else:
            eps_single = max(eps_min, eps_single * eps_decay)
            
        true_episode_rewards.append(episode_true_reward)
        total_episode_rewards.append(episode_total_reward)
        
        eps_q0_history.append(eps_q0)
        eps_q1_history.append(eps_q1)
        eps_q2_history.append(eps_q2)
        eps_single_history.append(eps_single)
        
        # Gestione errori in caso agent.py non sia stato aggiornato per la fase 3
        buffer_q0_history.append(agent.memory.q0_fraction_onehot())
        buffer_q1_history.append(agent.memory.q1_fraction_onehot())
        buffer_q2_history.append(agent.memory.q2_fraction_onehot())
        
        episode_wp1_hit_history.append(episode_wp1_hit)
        episode_wp2_hit_history.append(episode_wp2_hit)
        episode_goal_hit_history.append(episode_goal_hit)

        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            recent_avg_with_shaping = np.mean(total_episode_rewards[-100:])
            mode_str = "SHAPING" if use_shaping else "BASELINE"
            eps_str = "MULTI EPS" if use_double_epsilon else "SINGLE EPS"

            log_string = (
                f"[{mode_str} | {eps_str}] Episode {n_episode + 1}/{episodes}\n" +
                f"Avg Reward                  : {recent_avg:.6f}\n" +
                (f"Avg Total Reward           : {recent_avg_with_shaping:.6f}\n" if use_shaping else "") +
                (f"Epsilon (q0, q1, q2)       : {eps_q0:.4f}, {eps_q1:.4f}, {eps_q2:.4f}\n" if use_double_epsilon else f"  Epsilon (single)        : {eps_single:.4f}\n") +          
                f"Buffer Fractions (q0/q1/q2) : {agent.memory.q0_fraction_onehot():.2%}, {agent.memory.q1_fraction_onehot():.2%}, {agent.memory.q2_fraction_onehot():.2%}\n" +
                f"Hits (WP1/WP2/Goal)         : {wp1_hits}, {wp2_hits}, {goal_hits}\n"
            )
            print(log_string)
            if log_handle:
                log_handle.write(log_string + "\n")
                log_handle.flush()

        if save_policy and (n_episode + 1) % 500 == 0:
            prefix = "shaping" if use_shaping else "baseline"
            eps_suffix = "multi_eps" if use_double_epsilon else "single_eps"
            agent.policy_name = f"{prefix}_{eps_suffix}_policy_ep_{n_episode + 1}.pth"
            agent._save_policy()

    if log_handle: log_handle.close()

    if use_double_epsilon:
        return (np.array(true_episode_rewards), np.array(total_episode_rewards), 
                (np.array(eps_q0_history), np.array(eps_q1_history), np.array(eps_q2_history)),
                (np.array(buffer_q0_history), np.array(buffer_q1_history), np.array(buffer_q2_history)), 
                np.array(episode_wp1_hit_history), np.array(episode_wp2_hit_history), np.array(episode_goal_hit_history))
    else:
        return (np.array(true_episode_rewards), np.array(total_episode_rewards), np.array(eps_single_history), 
                (np.array(buffer_q0_history), np.array(buffer_q1_history), np.array(buffer_q2_history)), 
                np.array(episode_wp1_hit_history), np.array(episode_wp2_hit_history), np.array(episode_goal_hit_history))

# =====================================================================
# MAIN EXPERIMENT ORCHESTRATOR
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Ablation study on epsilon strategies for sequential tasks.")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="comparison", 
        choices=["single", "multi", "comparison", "baseline"],
        help="Execution mode."
    )
    args = parser.parse_args()

    print(f"=== STARTING ABLATION STUDY (3 PHASES) | MODE: {args.mode.upper()} ===")
    os.makedirs("logs", exist_ok=True)

    episodes = 10000
    gamma = 0.99
    multi_eps_decay = 0.999
    single_eps_decay = 0.9998 
    K_scaling = 1
    
    print("\n1. Initializing Environment and Abstract MDP...")
    env = gym.make("LunarLander-v3", continuous=False)
    
    abstract_mdp = SequentialWaypointMDP(width=12, height=12, gamma=gamma)
    abstract_mdp.value_iteration()

    print("\n2. Generating heatmaps...")
    save_sequential_heatmaps(abstract_mdp, filename_prefix="seq_experiment")

    if args.mode == "baseline":
        print("\n=======================================================")
        print("TRAINING: VANILLA DDQN AGENT (BASELINE - NO SHAPING)")
        print("=======================================================")
        agent_baseline = HierarchicalDQNLearner(
            env=env, max_episodes=episodes, gamma=gamma, eps_decay=single_eps_decay,
            use_ddqn=True, policy_name="baseline_policy.pth", extra_state_dims=3
        )
        
        b_curve, b_total, b_eps, b_buf, _, _, _ = run_sequential_training(
            env, agent_baseline, abstract_mdp, episodes, 
            use_shaping=False, use_double_epsilon=False, K=K_scaling,
            log_file="logs/baseline_training.log", save_policy=True
        )
        
        print("\n3. Generating plots for BASELINE mode...")
        b_buf_q0, b_buf_q1, b_buf_q2 = b_buf
        plot_shaping_reward_breakdown(b_curve, b_total, b_eps, window_size=500, filename="img/reward_breakdown_baseline.png")
        plot_buffer_fractions(b_buf_q0, b_buf_q1, b_buf_q2, window_size=500, filename="img/buffer_fractions_baseline.png")


    if args.mode in ["single", "comparison"]:
        print("\n=======================================================")
        print("TRAINING: SHAPING AGENT (SINGLE EPSILON)")
        print("=======================================================")
        agent_shaping_single = HierarchicalDQNLearner(
            env=env, max_episodes=episodes, gamma=gamma, eps_decay=single_eps_decay,
            use_ddqn=True, policy_name="shaping_single_eps_policy.pth", extra_state_dims=3
        )
        
        s_curve, s_total, s_eps, s_buf, s_wp1, s_wp2, s_goals = run_sequential_training(
            env, agent_shaping_single, abstract_mdp, episodes, 
            use_shaping=True, use_double_epsilon=False, K=K_scaling,
            log_file="logs/shaping_single_eps_training.log", save_policy=True
        )
        
        if args.mode == "single":
            print("\n3. Generating plots for SINGLE EPSILON mode...")
            s_buf_q0, s_buf_q1, s_buf_q2 = s_buf
            plot_shaping_reward_breakdown(s_curve, s_total, s_eps, window_size=500, filename="img/shaping_reward_breakdown_single_eps.png")
            plot_buffer_fractions(s_buf_q0, s_buf_q1, s_buf_q2, window_size=500, filename="img/buffer_fractions_single_eps.png")


    if args.mode in ["multi", "comparison"]:
        print("\n=======================================================")
        print("TRAINING: SHAPING AGENT (MULTI EPSILON)")
        print("=======================================================")
        agent_shaping_multi = HierarchicalDQNLearner(
            env=env, max_episodes=episodes, gamma=gamma, eps_decay=multi_eps_decay,
            use_ddqn=True, policy_name="shaping_multi_eps_policy.pth", extra_state_dims=3
        )
        
        d_curve, d_total, d_eps, d_buf, d_wp1, d_wp2, d_goals = run_sequential_training(
            env, agent_shaping_multi, abstract_mdp, episodes, 
            use_shaping=True, use_double_epsilon=True, K=K_scaling,
            log_file="logs/shaping_multi_eps_training.log", save_policy=True
        )
        
        if args.mode == "double":
            print("\n3. Generating plots for MULTI EPSILON mode...")
            d_buf_q0, d_buf_q1, d_buf_q2 = d_buf
            plot_shaping_reward_breakdown(d_curve, d_total, d_eps, window_size=500, filename="img/shaping_reward_breakdown_multi_eps.png")
            plot_buffer_fractions(d_buf_q0, d_buf_q1, d_buf_q2, window_size=500, filename="img/buffer_fractions_multi_eps.png")


    if args.mode == "comparison":
        print("\n3. Generating plots for COMPARISON mode...")
        
        eps_q0_history, eps_q1_history, eps_q2_history = d_eps
        
        # 1. Slide plot for Phase 1 (WP1)
        plot_ablation_slide_phase1(
            episodes=episodes, single_wp=s_wp1, single_eps=s_eps, 
            double_wp=d_wp1, eps_q0=eps_q0_history, eps_q1=eps_q1_history, eps_q2=eps_q2_history,
            window_size=500, filename="img/slide_ablation_phase1_wp1.png"
        )

        # 2. Slide plot for Phase 2 (WP2)
        plot_ablation_slide_phase2(
            episodes=episodes, single_wp2=s_wp2, single_eps=s_eps, 
            double_wp2=d_wp2, eps_q0=eps_q0_history, eps_q1=eps_q1_history, eps_q2=eps_q2_history,
            window_size=500, filename="img/slide_ablation_phase2_wp2.png"
        )
        
        # 3. Slide plot for Phase 3 (Goal)
        plot_ablation_slide_phase3(
            episodes=episodes, single_goal=s_goals, single_eps=s_eps, 
            double_goal=d_goals, eps_q0=eps_q0_history, eps_q1=eps_q1_history, eps_q2=eps_q2_history,
            window_size=500, filename="img/slide_ablation_phase3_goal.png"
        )
        
        # Standard metrics for the proposed Multi Epsilon agent in comparison
        d_buf_q0, d_buf_q1, d_buf_q2 = d_buf
        plot_shaping_reward_breakdown(d_curve, d_total, d_eps, window_size=500, filename="img/shaping_reward_breakdown_proposed.png")
        plot_buffer_fractions(d_buf_q0, d_buf_q1, d_buf_q2, window_size=500, filename="img/buffer_fractions_proposed.png")
    
    env.close()

if __name__ == "__main__":
    main()