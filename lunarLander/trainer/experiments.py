import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Import components from your existing modules
from abstract_mdps import DiagonalAbstractGridMDP
from abstract_mdps import ValleyDiagonalAbstractMDP
from utils import phi_mapping_grid
from agent import HierarchicalDQNLearner

# =====================================================================
# TRAINING LOOPS
# =====================================================================

def run_sparse_goal_mdp_training(env, agent, abstract_mdp, mapping_fn, episodes, use_shaping=False):
    """
    Custom training loop for the Sparse Goal MDP experiment (Experiment 1).
    Overrides the environment's base reward, yielding 0.0 everywhere except 
    for a strict +100.0 at the final abstract goal state.
    """
    print(f"--- Starting Training | Shaping Enabled: {use_shaping} ---")
    
    if use_shaping:
        abstract_mdp.value_iteration()
        K = 100 # Scaling factor for the shaping potential

    true_episode_rewards = []

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        terminated = truncated = False
        episode_true_reward = 0.0

        while not (terminated or truncated):
            a = agent.select_action(s_raw)
            ns_raw, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            
            abstract_s = mapping_fn(s_raw)
            abstract_ns = mapping_fn(ns_raw)
            
            # 1. Strictly Sparse Reward Logic
            env_goal_reward = 0.0
            if terminated and abstract_ns == abstract_mdp.goal_state:
                env_goal_reward = 100.0
            
            episode_true_reward += env_goal_reward

            # 2. Reward Shaping Logic
            shaping_signal = 0.0
            if use_shaping:
                phi_s = abstract_mdp.v_star[abstract_s]
                
                # Grounding the potential to 0 at terminal states
                if done:
                    phi_ns = 0.0
                else:
                    phi_ns = abstract_mdp.v_star[abstract_ns]
                
                # Shaping formula: F = K * (gamma * Phi(s') - Phi(s))
                shaping_signal = K * (agent.gamma * phi_ns - phi_s)
            
            # 3. Step the agent (training relies on BOTH env reward and shaping)
            total_step_reward = env_goal_reward + shaping_signal
            agent.memory.push(s_raw, a, total_step_reward, ns_raw, done)
            agent.optimize_model()
            
            s_raw = ns_raw
            
        # Decay the exploration rate at the end of the episode
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)
        
        # Logging every 100 episodes
        if (n_episode + 1) % 100 == 0:
            avg_reward = np.mean(true_episode_rewards[-100:])
            agent._save_policy()
            print(f"Episode {n_episode+1}/{episodes} | Avg True Reward (last 100): {avg_reward:.2f}")

    return np.array(true_episode_rewards)


def run_trajectory_goal_mdp_training(env, agent, abstract_mdp, mapping_fn, episodes, trajectory_states):
    """
    Custom training loop for Experiment 2 (The Canyon).
    - Yields +100 at the goal.
    - Yields +10 for reaching a trajectory state (one-time only per episode).
    - Uses the artificially modified 'canyon' V* for Reward Shaping.
    - RETURNS ONLY THE PURE SUCCESS REWARD (0 or 100) FOR CLEAN PLOTTING.
    """
    print(f"--- Starting Trajectory Training ---")
    
    K = 100 
    true_episode_rewards = [] # Used for terminal logging (includes +10 sub-goals)
    pure_success_rewards = [] # Used for plotting (STRICTLY 0 or 100)
    all_shaping = []
    same = diff = 0

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        terminated = truncated = False
        episode_true_reward = 0.0
        episode_pure_reward = 0.0 
        
        # We use a set to prevent "Reward Hacking" (farming the same sub-goal)
        visited_abstract_states = set()

        while not (terminated or truncated):
            a = agent.select_action(s_raw)
            ns_raw, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            
            abstract_s = mapping_fn(s_raw)
            abstract_ns = mapping_fn(ns_raw)

            # Track if the agent is staying in the same abstract state or moving
            if abstract_s == abstract_ns:
                same += 1
            else:
                diff += 1
            
            env_goal_reward = 0.0
            
            # 1. Final Goal Victory
            if terminated and abstract_ns == abstract_mdp.goal_state:
                env_goal_reward = 100.0
                episode_pure_reward = 100.0 # Record pure success!
                
            # 2. Sub-goal Reached (only if not visited yet in this episode)
            elif abstract_ns in trajectory_states and abstract_ns not in visited_abstract_states:
                env_goal_reward = 10.0
                visited_abstract_states.add(abstract_ns)
            
            episode_true_reward += env_goal_reward

            # 3. Reward Shaping Logic on the Canyon V*
            phi_s = abstract_mdp.v_star[abstract_s]
            if done:
                phi_ns = 0.0
            else:
                phi_ns = abstract_mdp.v_star[abstract_ns]
                
            shaping_signal = K * (agent.gamma * phi_ns - phi_s)
            all_shaping.append(shaping_signal)
            
            # 4. Agent Training Step
            total_step_reward = env_goal_reward + shaping_signal
            agent.memory.push(s_raw, a, total_step_reward, ns_raw, done)
            agent.optimize_model()
            
            s_raw = ns_raw
            
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        
        # Save both metrics
        true_episode_rewards.append(episode_true_reward)
        pure_success_rewards.append(episode_pure_reward)
        
        # Comprehensive Logging every 100 episodes
        if (n_episode + 1) % 100 == 0:
            avg_true = np.mean(true_episode_rewards[-100:])
            avg_pure = np.mean(pure_success_rewards[-100:])
            agent._save_policy()
            
            print(f"[{agent.policy_name}] Episode {n_episode+1}/{episodes} | Avg Training Score: {avg_true:.2f} | Pure Win Rate: {avg_pure:.2f}%")
            print(f"Same States: {same}, Diff States: {diff}")
            print(f"Avg Shaping: {np.mean(all_shaping):.2f}")
            print(f"Variance Shaping: {np.var(all_shaping):.2f}")
            print(f"Max Shaping: {np.max(all_shaping):.2f}")
            print(f"Min Shaping: {np.min(all_shaping):.2f}")
            print("-" * 50)
            
            # Clear logs to prevent memory saturation
            all_shaping = []
            same = diff = 0

    return np.array(pure_success_rewards)


def run_valley_goal_mdp_training(env, agent, abstract_mdp, mapping_fn, episodes, trajectory_states):
    """
    Custom training loop for Experiment 3 (The Valley).
    - Yields +100 at the goal.
    - Yields +10 for reaching a trajectory state.
    - Uses the naturally calculated 'Valley' V* for Reward Shaping.
    """
    print(f"--- Starting Valley Training ---")
    
    K = 100 
    true_episode_rewards = [] 
    pure_success_rewards = [] 
    all_shaping = []
    same = diff = 0

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        terminated = truncated = False
        episode_true_reward = 0.0
        episode_pure_reward = 0.0 
        
        visited_abstract_states = set()

        while not (terminated or truncated):
            a = agent.select_action(s_raw)
            ns_raw, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            
            abstract_s = mapping_fn(s_raw)
            abstract_ns = mapping_fn(ns_raw)

            if abstract_s == abstract_ns:
                same += 1
            else:
                diff += 1
            
            env_goal_reward = 0.0
            
            # 1. Final Goal Victory
            if terminated and abstract_ns == abstract_mdp.goal_state:
                env_goal_reward = 100.0
                episode_pure_reward = 100.0 
                
            # 2. Sub-goal Reached
            elif abstract_ns in trajectory_states and abstract_ns not in visited_abstract_states:
                env_goal_reward = 10.0
                visited_abstract_states.add(abstract_ns)
            
            episode_true_reward += env_goal_reward

            # 3. Reward Shaping on the Valley V*
            phi_s = abstract_mdp.v_star[abstract_s]
            if done:
                phi_ns = 0.0
            else:
                phi_ns = abstract_mdp.v_star[abstract_ns]
                
            shaping_signal = K * (agent.gamma * phi_ns - phi_s)
            all_shaping.append(shaping_signal)
            
            # 4. Agent Training Step
            total_step_reward = env_goal_reward + shaping_signal
            agent.memory.push(s_raw, a, total_step_reward, ns_raw, done)
            agent.optimize_model()
            
            s_raw = ns_raw
            
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        
        true_episode_rewards.append(episode_true_reward)
        pure_success_rewards.append(episode_pure_reward)
        
        if (n_episode + 1) % 100 == 0:
            avg_true = np.mean(true_episode_rewards[-100:])
            avg_pure = np.mean(pure_success_rewards[-100:])
            agent._save_policy()
            
            print(f"[{agent.policy_name}] Episode {n_episode+1}/{episodes} | Avg Training Score: {avg_true:.2f} | Pure Win Rate: {avg_pure:.2f}%")
            print(f"Same States: {same}, Diff States: {diff}")
            if all_shaping:
                print(f"Avg Shaping: {np.mean(all_shaping):.2f}")
                print(f"Variance Shaping: {np.var(all_shaping):.2f}")
                print(f"Max Shaping: {np.max(all_shaping):.2f}")
                print(f"Min Shaping: {np.min(all_shaping):.2f}")
            print("-" * 50)
            
            all_shaping = []
            same = diff = 0

    return np.array(pure_success_rewards)

# =====================================================================
# PLOTTING UTILITIES
# =====================================================================

def plot_value_function_heatmap(abstract_mdp, width=12, height=12, title="Potential Map V*"):
    """
    Converts the v_star dictionary into a 2D matrix and plots it as a Heatmap,
    ensuring grid lines perfectly match the cell borders.
    """
    v_matrix = np.zeros((height, width))
    
    for (x, y), value in abstract_mdp.v_star.items():
        if 0 <= x < width and 0 <= y < height:
            v_matrix[y, x] = value
            
    plt.figure(figsize=(9, 8))
    im = plt.imshow(v_matrix, cmap='viridis', origin='lower')
    
    # Print textual values at the center of all non-zero cells
    for y in range(height):
        for x in range(width):
            val = v_matrix[y, x]
            if val != 0: 
                plt.text(x, y, f"{val:.2f}", ha='center', va='center', 
                         color='white' if val < 0.5 else 'black', fontsize=8, fontweight='bold')
                
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Potential Value")
    plt.title(title, fontsize=14)
    plt.xlabel("X (Horizontal Position)", fontsize=12)
    plt.ylabel("Y (Altitude / Distance from ground)", fontsize=12)
    
    # --- GRID MISALIGNMENT CORRECTION ---
    # 1. Set major ticks exactly at the center of the cells
    plt.xticks(np.arange(0, width, 1))
    plt.yticks(np.arange(0, height, 1))
    
    # 2. Create invisible "Minor Ticks" on the cell borders (-0.5, 0.5, 1.5...)
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, height, 1), minor=True)
    
    # 3. Draw the grid ONLY snapped to the Minor Ticks (on the borders)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1, alpha=0.4)
    ax.grid(which='major', color='none') # Disable grid on the numbers
    # ------------------------------------
    
    plt.tight_layout()
    plt.show()


def plot_single_result(rewards, window_size=250, title="Training Results", ylabel="True Episode Reward"):
    """Plots a single training curve without saving."""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='lightgray', alpha=0.6, label='Raw Episode Reward')
    
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(rewards)), moving_avg, color='blue', linewidth=2.5, label=f'Moving Avg ({window_size} eps)')
    
    plt.title(title)
    plt.xlabel('Episode #')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_combined_results(rewards_dict, window_size=250, title="Comparison: Baseline vs Shaping"):
    """Plots multiple training curves on the same graph for comparison without saving."""
    plt.figure(figsize=(12, 7))
    
    colors = ['red', 'green', 'blue', 'orange']
    
    for idx, (label, rewards) in enumerate(rewards_dict.items()):
        color = colors[idx % len(colors)]
        
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size - 1, len(rewards)), moving_avg, color=color, linewidth=2.5, label=f'{label} (Moving Avg)')
            
    plt.title(title)
    plt.xlabel('Episode #')
    plt.ylabel('True Episode Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# =====================================================================
# EXPERIMENTS
# =====================================================================

def experiment_1_baseline_vs_shaping(episodes=600):
    """
    Experiment 1: Compares standard Agent (Baseline) with an Agent guided 
    by Reward Shaping in a strictly sparse Goal MDP.
    """
    print("\n" + "="*50)
    print("EXPERIMENT 1: Baseline vs Reward Shaping")
    print("="*50)
    
    env = gym.make("LunarLander-v3", continuous=False)
    
    # --- 1. BASELINE (No Shaping) ---
    print("\n>>> Running Baseline Agent (No Shaping)...")
    abstract_mdp = DiagonalAbstractGridMDP() # Used only for the goal state reference
    agent_baseline = HierarchicalDQNLearner(env, abstract_mdp, phi_mapping_grid, max_episodes=episodes, use_ddqn=True, policy_name="baseline_sparse_goal.pth")
    rewards_baseline = run_sparse_goal_mdp_training(env, agent_baseline, abstract_mdp, phi_mapping_grid, episodes, use_shaping=False)
    
    # --- 2. SHAPING ---
    print("\n>>> Running Shaping Agent...")
    abstract_mdp_shaping = DiagonalAbstractGridMDP()
    agent_shaping = HierarchicalDQNLearner(env, abstract_mdp_shaping, phi_mapping_grid, max_episodes=episodes, use_ddqn=True, policy_name="shaping_sparse_goal.pth")
    rewards_shaping = run_sparse_goal_mdp_training(env, agent_shaping, abstract_mdp_shaping, phi_mapping_grid, episodes, use_shaping=True)
    
    # --- 3. PLOTS ---
    print("\n>>> Generating Plots...")
    plot_single_result(rewards_baseline, title="Sparse Goal MDP - Baseline (No Shaping)")
    plot_single_result(rewards_shaping, title="Sparse Goal MDP - With Reward Shaping")
    plot_combined_results({
        "Baseline (No Shaping)": rewards_baseline,
        "Reward Shaping": rewards_shaping
    })
    
    env.close()


def experiment_2_trajectory_vs_standard(episodes=800):
    """
    Experiment 2: Compares Baseline (No Shaping), Standard 2D Reward Shaping, 
    and a Trajectory-constrained Shaping (Canyon effect) + Sub-goal Rewards.
    """
    print("\n" + "="*50)
    print("EXPERIMENT 2: Baseline vs Standard Shaping vs Trajectory Sub-goals")
    print("="*50)
    
    env = gym.make("LunarLander-v3", continuous=False)
    
    # --- 0. BASELINE (No Shaping) ---
    print("\n>>> Running Baseline Agent (No Shaping)...")
    abstract_mdp_base = DiagonalAbstractGridMDP()
    agent_base = HierarchicalDQNLearner(
        env, abstract_mdp_base, phi_mapping_grid, 
        max_episodes=episodes, use_ddqn=True, policy_name="baseline_exp2.pth"
    )
    rewards_base = run_sparse_goal_mdp_training(
        env, agent_base, abstract_mdp_base, phi_mapping_grid, episodes, use_shaping=False
    )
    
    # --- 1. STANDARD SHAPING ---
    print("\n>>> Running Standard Shaping Agent...")
    abstract_mdp_std = DiagonalAbstractGridMDP()
    agent_std = HierarchicalDQNLearner(
        env, abstract_mdp_std, phi_mapping_grid, 
        max_episodes=episodes, use_ddqn=True, policy_name="shaping_std_exp2.pth"
    )

    abstract_mdp_std.value_iteration()
    plot_value_function_heatmap(abstract_mdp_std, title="Canyon Effect: Trajectory V*")
    
    rewards_std = run_sparse_goal_mdp_training(
        env, agent_std, abstract_mdp_std, phi_mapping_grid, episodes, use_shaping=True
    )

    # --- 2. TRAJECTORY SHAPING (The Canyon) ---
    print("\n>>> Running Trajectory Sub-goals Agent...")
    
    # Define the trajectory: Vertical descent -> Diagonal approach
    trajectory_path = [
        (5, 11), (5, 10), (5, 9), (5, 8), (5, 7), (5, 6), (5, 5), 
        (4, 4), (3, 3), (2, 2), (1, 1), (0, 0),
    ]
    
    abstract_mdp_traj = DiagonalAbstractGridMDP()
    abstract_mdp_traj.value_iteration() # Computes the full standard potential

    # V* ZEROING (Creating the Canyon effect manually)
    for s in list(abstract_mdp_traj.v_star.keys()):
        if s not in trajectory_path:
            abstract_mdp_traj.v_star[s] = 0.0

    # Display the map (ensure you close it if plt.show() blocks execution!)
    plot_value_function_heatmap(abstract_mdp_traj, title="Canyon Effect: Trajectory V*")

    agent_traj = HierarchicalDQNLearner(
        env, abstract_mdp_traj, phi_mapping_grid, 
        max_episodes=episodes, use_ddqn=True, policy_name="shaping_trajectory.pth"
    )
    
    rewards_traj = run_trajectory_goal_mdp_training(
        env, agent_traj, abstract_mdp_traj, phi_mapping_grid, episodes, trajectory_path
    )
    
    # --- 3. COMBINED PLOT ---
    print("\n>>> Generating Combined Plot...")
    plot_single_result(rewards_base, title="Exp 2 - Baseline (No Shaping)")
    plot_single_result(rewards_std, title="Exp 2 - Standard Shaping")
    plot_single_result(rewards_traj, title="Exp 2 - Trajectory Shaping (+10 Sub-goals)")
    
    plot_combined_results({
        "Baseline (No Shaping)": rewards_base,
        "Standard Shaping": rewards_std,
        "Trajectory Sub-goals": rewards_traj
    }, title="Comparison: Baseline vs Standard vs Trajectory")
    
    env.close()


def experiment_3_valley(episodes=800):
    """
    Experiment 3: Tests the Valley Abstract MDP where the potential 
    naturally degrades outside the trajectory via transition costs.
    """
    print("\n" + "="*50)
    print("EXPERIMENT 3: Valley Shaping (Natural Potential Gradient)")
    print("="*50)
    
    env = gym.make("LunarLander-v3", continuous=False)
    
    print("\n>>> Running Valley Sub-goals Agent...")
    
    trajectory_path = [
        (5, 11), (5, 10), (5, 9), (5, 8), (5, 7), (5, 6), (5, 5), 
        (4, 4), (3, 3), (2, 2), (1, 1), (0, 0),
    ]
    
    # We use the new Valley class which implements transition penalties
    abstract_mdp_valley = ValleyDiagonalAbstractMDP(trajectory_path)
    
    # Value Iteration will calculate the funnel effect automatically!
    abstract_mdp_valley.value_iteration() 
    
    # No manual zeroing! We respect the constraint of using pure V*.
    plot_value_function_heatmap(abstract_mdp_valley, title="Valley Effect: Pure Abstract MDP V*")

    agent_valley = HierarchicalDQNLearner(
        env, abstract_mdp_valley, phi_mapping_grid, 
        max_episodes=episodes, use_ddqn=True, policy_name="shaping_valley.pth"
    )
    
    rewards_valley = run_valley_goal_mdp_training(
        env, agent_valley, abstract_mdp_valley, phi_mapping_grid, episodes, trajectory_path
    )
    
    print("\n>>> Generating Plot...")
    plot_single_result(rewards_valley, title="Exp 3 - Valley Shaping (+10 Sub-goals)")
    
    env.close()


if __name__ == "__main__":
    EPISODES = 1000
    
    #experiment_1_baseline_vs_shaping(episodes=EPISODES)
    experiment_2_trajectory_vs_standard(episodes=EPISODES)
    #experiment_3_valley(episodes=EPISODES)