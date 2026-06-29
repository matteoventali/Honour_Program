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
    save_sequential_interpolated_heatmaps
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

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        
        # Initialize sequence variable 'q' (0 = seek waypoint, 1 = seek goal)
        q = 0 
        
        # Augment state for the Neural Network (from 8 to 9 dimensions)
        s_aug = np.append(s_raw, q) 
        
        terminated = truncated = False
        episode_true_reward = 0.0
        
        while not (terminated or truncated):
            # Agent selects action based on the augmented state
            a = agent.select_action(s_aug)
            ns_raw, _, terminated, truncated, _ = env.step(a)

            env_goal_reward = 0.0
            
            # Map continuous state to 3D abstract state (x, y, q)
            abstract_x_curr, abstract_y_curr, _ = phi_mapping_sequential(ns_raw, q)
            next_q = q
            
            # STATE TRANSITION LOGIC
            # If the agent reaches the waypoint during phase q=0
            if abstract_x_curr == 1 and abstract_y_curr == 8 and q == 0:
                env_goal_reward = 50.0  # Intermediate reward for reaching the waypoint
                next_q = 1            # Phase transition
            
            ns_aug = np.append(ns_raw, next_q)
            abstract_ns = (abstract_x_curr, abstract_y_curr, next_q)

            # Check if the final goal is reached
            if abstract_ns == abstract_mdp.goal_state:
                env_goal_reward = 100.0
                terminated = True
            # Check for crash or timeout (if not on the goal)
            elif terminated or truncated:
                # Apply crash penalty.
                env_goal_reward = -100.0
            
            done = terminated or truncated
            episode_true_reward += env_goal_reward

            # Shaping Signal Calculation
            if use_shaping:
                px_s, py_s = get_continuous_grid_coords(s_raw, abstract_mdp.width, abstract_mdp.height)
                px_ns, py_ns = get_continuous_grid_coords(ns_raw, abstract_mdp.width, abstract_mdp.height)
                
                phi_s = get_bilinear_potential_sequential(px_s, py_s, q, abstract_mdp.v_star)
                phi_ns = get_bilinear_potential_sequential(px_ns, py_ns, next_q, abstract_mdp.v_star)
                
                # F(s, a, s') = K * (gamma * Phi(s') - Phi(s))
                shaping_signal = K * (agent.gamma * phi_ns - phi_s)
            else:
                shaping_signal = 0.0
            
            total_step_reward = env_goal_reward + shaping_signal
            
            # Push AUGMENTED states to memory and optimize
            agent.memory.push(s_aug, a, total_step_reward, ns_aug, done)
            agent.optimize_model()

            s_raw = ns_raw
            s_aug = ns_aug
            q = next_q

        # Decay epsilon at the end of the episode
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)

        # Print progress every 100 episodes
        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            mode_str = "SHAPING" if use_shaping else "BASELINE"
            print(f"[{mode_str}] Progress: Episode {n_episode + 1}/{episodes} | 100-eps Avg Reward: {recent_avg:6.2f} | Epsilon: {agent.eps:.3f}")
            agent._save_policy()
            
    return np.array(true_episode_rewards)

# =====================================================================
# MAIN EXPERIMENT ORCHESTRATOR
# =====================================================================

def main():
    print("=== STARTING SEQUENTIAL TASK EXPERIMENT: BASELINE VS SHAPING ===")
    
    # HYPERPARAMETERS
    episodes = 1          
    gamma = 0.99
    eps_decay = 0.9995       
    K_scaling = 1.0          
    
    print("\n1. Initializing Environment and Abstract MDP...")
    env = gym.make("LunarLander-v3", continuous=False, max_episode_steps=100000)
    
    abstract_mdp = SequentialWaypointMDP(width=12, height=12, gamma=gamma)
    abstract_mdp.value_iteration()

    print("   -> Plotting Value Functions (V*) Heatmaps...")
    save_sequential_heatmaps(abstract_mdp, filename_prefix="seq_experiment_normal")
    save_sequential_interpolated_heatmaps(abstract_mdp, filename_prefix="seq_experiment")
    
    # -----------------------------------------------------------------
    # EXPERIMENT 1: BASELINE (NO SHAPING)
    # -----------------------------------------------------------------
    print("\n=======================================================")
    print("2. RUNNING EXPERIMENT 1: BASELINE (NO SHAPING)")
    print("=======================================================")
    agent_baseline = HierarchicalDQNLearner(
        env=env,
        max_episodes=episodes,
        gamma=gamma,
        eps_decay=eps_decay,
        use_ddqn=True,
        policy_name="baseline_sequential_policy.pth",
        extra_state_dims=1
    )
    
    baseline_learning_curve = run_sequential_training(
        env, 
        agent_baseline, 
        abstract_mdp, 
        episodes, 
        use_shaping=False, 
        K=K_scaling
    )
    
    # -----------------------------------------------------------------
    # EXPERIMENT 2: SHAPING AGENT
    # -----------------------------------------------------------------
    print("\n=======================================================")
    print("3. RUNNING EXPERIMENT 2: SHAPING AGENT (POTENTIAL GUIDED)")
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
    print("\n4. Generating Comparison Plots...")
    plot_comparison_curves(
        baseline_learning_curve, 
        shaping_learning_curve, 
        window_size=100, 
        filename="img/baseline_vs_shaping_comparison.png"
    )
    
    print("\n=== ALL EXPERIMENTS COMPLETE ===")
    env.close()

if __name__ == "__main__":
    main()