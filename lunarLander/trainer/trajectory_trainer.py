import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from abstract_mdps import SequentialWaypointMDP
from agent import HierarchicalDQNLearner
from utils import (
    phi_mapping_sequential, 
    get_continuous_grid_coords, 
    get_bilinear_potential_sequential,
    save_sequential_heatmaps,
    save_sequential_interpolated_heatmaps,
    plot_training_results
)

# =====================================================================
# PLOTTING UTILITY FOR COMPARISON
# =====================================================================

def plot_comparison_curves(baseline_rewards, shaping_rewards, window_size=100, filename="img/baseline_vs_shaping.png"):
    """
    Plots the smoothed moving average of the baseline agent against the shaping agent.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # Calculate moving averages
    if len(baseline_rewards) >= window_size:
        baseline_ma = np.convolve(baseline_rewards, np.ones(window_size)/window_size, mode='valid')
        shaping_ma = np.convolve(shaping_rewards, np.ones(window_size)/window_size, mode='valid')
        x_axis = range(window_size - 1, len(baseline_rewards))
    else:
        # Fallback if episode count is less than window size
        baseline_ma = baseline_rewards
        shaping_ma = shaping_rewards
        x_axis = range(len(baseline_rewards))
        
    # Plot curves
    plt.plot(x_axis, baseline_ma, color='black', linestyle='-', linewidth=2, label='Baseline (No Shaping)')
    plt.plot(x_axis, shaping_ma, color='blue', linestyle='-', linewidth=2.5, label='Shaping Agent (Potential Guided)')
    
    # Formatting
    plt.title("Learning Curve Comparison: Baseline vs Shaping", fontsize=15, fontweight='bold')
    plt.xlabel(f"Episode # (Moving Average Window = {window_size})", fontsize=12)
    plt.ylabel("Episode Reward", fontsize=12)
    plt.axhline(y=100, color='green', linestyle=':', alpha=0.6, label='Goal Threshold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="lower right", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"\n>>> Comparison plot successfully saved to: {filename}")
    plt.close()

# =====================================================================
# TRAINING LOOP
# =====================================================================

def run_sequential_training(env, agent, abstract_mdp, episodes, use_shaping=True, K=1.0):
    """
    Executes the training loop for the sequential task.
    """
    true_episode_rewards = []

    # Hyperparameters
    mu = 0.75
    mu_min = 0.0
    mu_decay = 0.9995

    # Counters
    natural_q_updates = 0
    artificial_q_updates = 0
    artificial_q_updates_to_q1 = 0
    waypoint_hits = 0
    goal_hits = 0

    total_step_reward_array = np.array([])

    mu_values = np.array([])

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        
        # Initialize sequence variable 'q' (0 = seek waypoint, 1 = seek goal)
        q = 0 
        passed_trough_waypoint = False
        
        # Augment state for the Neural Network
        s_aug = np.append(s_raw, q) 
        
        terminated = truncated = False
        episode_true_reward = 0.0

        # Setting mu value
        #q0_frac = agent.memory.q0_fraction()
        #mu = q0_frac
        mu_values = np.append(mu_values, mu)
        
        # Episode loop
        while not (terminated or truncated):
            env_goal_reward = 0.0

            # STATE TRANSITION LOGIC
            # If the agent reaches the waypoint during phase q=0
            abstract_x, abstract_y, _ = phi_mapping_sequential(s_raw, q)
            if abstract_x == 1 and abstract_y == 8 and q == 0:
                passed_trough_waypoint = True
                waypoint_hits += 1
                q = 1
                natural_q_updates += 1
                
            # ARTIFICIAL FLIPPING of q
            #if np.random.rand() < mu:
            #    if q == 0:
            #        q = 1
            #        artificial_q_updates_to_q1 += 1
            #    else:
            #        q = 0
            #    artificial_q_updates += 1
                
            # Building the current state augmented: environment state + q state
            s_aug = np.append(s_raw, q)
            
            # Agent selects action based on the augmented state
            a = agent.select_action(s_aug)
            ns_raw, _, terminated, truncated, _ = env.step(a)

            # Map continuous state to 3D abstract state (x, y, q)
            abstract_x_ns, abstract_y_ns, _ = phi_mapping_sequential(ns_raw, q)
            ns_aug = np.append(ns_raw, q)
            abstract_ns = (abstract_x_ns, abstract_y_ns, q)

            # Check if the final goal is reached
            if abstract_ns == abstract_mdp.goal_state and passed_trough_waypoint:
                goal_hits += 1
                env_goal_reward = 100
                terminated = True
            
            done = terminated or truncated
            episode_true_reward += env_goal_reward

            # Shaping Signal Calculation
            if use_shaping:
                px_s, py_s = get_continuous_grid_coords(s_raw, abstract_mdp.width, abstract_mdp.height)
                px_ns, py_ns = get_continuous_grid_coords(ns_raw, abstract_mdp.width, abstract_mdp.height)
                
                phi_s = get_bilinear_potential_sequential(px_s, py_s, q, abstract_mdp.v_star)
                phi_ns = get_bilinear_potential_sequential(px_ns, py_ns, q, abstract_mdp.v_star)
                
                # F(s, a, s') = K * (gamma * Phi(s') - Phi(s))
                shaping_signal = K * (agent.gamma * phi_ns - phi_s)
                #print(f"Shaping phi_s: {phi_s}, phi_ns: {phi_ns}, shaping_signal: {shaping_signal}")
            else:
                shaping_signal = 0.0
            
            total_step_reward = env_goal_reward + shaping_signal
            total_step_reward_array = np.append(total_step_reward_array, total_step_reward)
            
            # Push AUGMENTED states to memory and optimize
            agent.memory.push(s_aug, a, total_step_reward, ns_aug, done)
            agent.optimize_model()

            s_raw = ns_raw
            s_aug = ns_aug
            
        # Decay epsilon at the end of the episode
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        mu = max(mu_min, mu * mu_decay)
        true_episode_rewards.append(episode_true_reward)

        # Print progress every 100 episodes
        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            mode_str = "SHAPING" if use_shaping else "BASELINE"

            #print(f"[{mode_str}] Progress: Episode {n_episode + 1}/{episodes} | 100-eps Avg Reward: {recent_avg:6.2f} | Epsilon: {agent.eps:.3f}")
            print(
                f"[{mode_str}] Episode {n_episode + 1}/{episodes}\n"
                f"  Avg Reward              : {recent_avg:.6f}\n"
                f"  Epsilon                 : {agent.eps:.6f}\n"
                f"  Mu value                : {mu:.6f}\n"
                f"  Exp q0 % and q1 %       : {agent.memory.q0_fraction():.6f}, {agent.memory.q1_fraction():.6f}\n"
                f"  Natural q→1 updates     : {natural_q_updates}\n"
                f"  Artificial flips        : {artificial_q_updates}\n"
                f"  Artificial 0→1 flips    : {artificial_q_updates_to_q1}\n"
                f"  Waypoint hits           : {waypoint_hits}\n"
                f"  Goal hits               : {goal_hits}\n"
                f"  Reward max, min, mean   : {total_step_reward_array.max():.6f}, {total_step_reward_array.min():.6f}, {total_step_reward_array.mean():.6f}"
            )
            if len(mu_values) > 100: 
                print(f"  Mu mean, Mu std         : {np.mean(mu_values[:-100]):.6f}, {np.std(mu_values[:-100]):.6f}\n")
            else:
                print("")
            
            agent._save_policy()
            
    return np.array(true_episode_rewards)

# =====================================================================
# MAIN EXPERIMENT ORCHESTRATOR
# =====================================================================

def main():
    print("=== STARTING SEQUENTIAL TASK EXPERIMENT: BASELINE VS SHAPING ===")
    
    # HYPERPARAMETERS
    episodes = 5000       
    gamma = 0.99
    eps_decay = 0.999
    K_scaling = 1
    
    print("\n1. Initializing Environment and Abstract MDP...")
    env = gym.make("LunarLander-v3", continuous=False, max_episode_steps=100000)
    
    abstract_mdp = SequentialWaypointMDP(width=12, height=12, gamma=gamma)
    abstract_mdp.value_iteration()

    print("   -> Plotting Value Functions (V*) Heatmaps...")
    save_sequential_heatmaps(abstract_mdp, filename_prefix="seq_experiment_normal")
    save_sequential_interpolated_heatmaps(abstract_mdp, filename_prefix="seq_experiment")
    
    # -----------------------------------------------------------------
    # EXPERIMENT 2: SHAPING AGENT
    # -----------------------------------------------------------------
    print("\n=======================================================")
    print("TRAINING")
    print("=======================================================")
    agent_shaping = HierarchicalDQNLearner(
        env=env,
        max_episodes=episodes,
        gamma=gamma,
        eps_decay=eps_decay,
        use_ddqn=True,
        policy_name="shaping_sequential_policy.pth",
        extra_state_dims=1
    )
    
    shaping_learning_curve = run_sequential_training(
        env, 
        agent_shaping, 
        abstract_mdp, 
        episodes, 
        use_shaping=True, 
        K=K_scaling
    )
    
    # -----------------------------------------------------------------
    # PLOTTING RESULTS
    # -----------------------------------------------------------------
    plot_training_results(shaping_learning_curve, window_size=300, filename="img/training_with_mu.png")
    env.close()

if __name__ == "__main__":
    main()