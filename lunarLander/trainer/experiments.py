import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Import components from your existing modules
from abstract_mdps import AbstractGridMDP
from utils import phi_mapping_grid
from agent import HierarchicalDQNLearner

def run_sparse_goal_mdp_training(env, agent, abstract_mdp, mapping_fn, episodes, use_shaping=False):
    """
    Custom training loop for the Sparse Goal MDP experiment.
    Overrides the environment's reward, yielding 0 everywhere except 
    100 at the bottom-left abstract state.
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
                
                if done:
                    phi_ns = 0.0
                else:
                    phi_ns = abstract_mdp.v_star[abstract_ns]
                
                shaping_signal = K * (agent.gamma * phi_ns - phi_s)
            
            # 3. Step the agent
            total_step_reward = env_goal_reward + shaping_signal
            agent.memory.push(s_raw, a, total_step_reward, ns_raw, done)
            agent.optimize_model()
            
            s_raw = ns_raw
            
        # Agent exploration decay
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)
        
        if (n_episode + 1) % 100 == 0:
            avg_reward = np.mean(true_episode_rewards[-100:])
            agent._save_policy()
            print(f"Episode {n_episode+1}/{episodes} | Avg True Reward (last 100): {avg_reward:.2f}")

    return np.array(true_episode_rewards)


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
    abstract_mdp = AbstractGridMDP() # Used only for the goal state reference
    agent_baseline = HierarchicalDQNLearner(env, abstract_mdp, phi_mapping_grid, max_episodes=episodes, use_ddqn=True, policy_name="baseline_sparse_goal.pth")
    rewards_baseline = run_sparse_goal_mdp_training(env, agent_baseline, abstract_mdp, phi_mapping_grid, episodes, use_shaping=False)
    
    # --- 2. SHAPING ---
    print("\n>>> Running Shaping Agent...")
    abstract_mdp_shaping = AbstractGridMDP()
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


def run_trajectory_goal_mdp_training(env, agent, abstract_mdp, mapping_fn, episodes, trajectory_states):
    """
    Custom training loop for Experiment 2.
    - Yields +100 at the goal.
    - Yields +10 for reaching a trajectory state (only once per episode).
    - Uses the modified 'canyon' V* for Reward Shaping.
    """
    print(f"--- Starting Trajectory Training ---")
    
    K = 100 # Scaling factor for the shaping potential
    true_episode_rewards = []

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        terminated = truncated = False
        episode_true_reward = 0.0
        
        # Inizializziamo il Set per evitare il Reward Hacking sui sub-goal
        visited_abstract_states = set()

        while not (terminated or truncated):
            a = agent.select_action(s_raw)
            ns_raw, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            
            abstract_s = mapping_fn(s_raw)
            abstract_ns = mapping_fn(ns_raw)
            
            # 1. Base Environment Reward
            env_goal_reward = 0.0
            
            # 2. Reward per il Goal Finale (+100)
            if terminated and abstract_ns == abstract_mdp.goal_state:
                env_goal_reward = 100.0
                
            # 3. Sub-Goal Reward per la Traiettoria (+10)
            elif abstract_ns in trajectory_states and abstract_ns not in visited_abstract_states:
                env_goal_reward = 10.0
                visited_abstract_states.add(abstract_ns)
            
            episode_true_reward += env_goal_reward

            # 4. Reward Shaping Logic (con fix per stati terminali)
            phi_s = abstract_mdp.v_star[abstract_s]
            if done:
                phi_ns = 0.0
            else:
                phi_ns = abstract_mdp.v_star[abstract_ns]
                
            shaping_signal = K * (agent.gamma * phi_ns - phi_s)
            
            # 5. Step the agent
            total_step_reward = env_goal_reward + shaping_signal
            agent.memory.push(s_raw, a, total_step_reward, ns_raw, done)
            agent.optimize_model()
            
            s_raw = ns_raw
            
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)
        
        if (n_episode + 1) % 100 == 0:
            avg_reward = np.mean(true_episode_rewards[-100:])
            agent._save_policy()
            print(f"[{agent.policy_name}] Episode {n_episode+1}/{episodes} | Avg True Reward: {avg_reward:.2f}")

    return np.array(true_episode_rewards)


def experiment_2_trajectory_vs_standard(episodes=800):
    """
    Experiment 2: Compares Standard 2D Reward Shaping with 
    a Trajectory-constrained Shaping + Sub-goal Rewards.
    """
    print("\n" + "="*50)
    print("EXPERIMENT 2: Standard Shaping vs Trajectory Sub-goals")
    print("="*50)
    
    env = gym.make("LunarLander-v3", continuous=False)
    
    # --- 1. STANDARD SHAPING (Baseline per questo esperimento) ---
    print("\n>>> Running Standard Shaping Agent...")
    abstract_mdp_std = AbstractGridMDP()
    agent_std = HierarchicalDQNLearner(
        env, abstract_mdp_std, phi_mapping_grid, 
        max_episodes=episodes, use_ddqn=True, policy_name="shaping_std_exp2.pth"
    )
    # Riutilizziamo la funzione del primo esperimento per lo standard shaping
    rewards_std = run_sparse_goal_mdp_training(
        env, agent_std, abstract_mdp_std, phi_mapping_grid, episodes, use_shaping=True
    )
    plot_single_result(rewards_std, title="Exp 2 - Standard Shaping")

    # --- 2. TRAJECTORY SHAPING ---
    print("\n>>> Running Trajectory Sub-goals Agent...")
    
    # Definiamo una traiettoria plausibile dall'alto-centro verso (0,0)
    # (Il lander spawna circa al centro-alto: x=5/6, y=7/8 nella griglia 12x12)
    trajectory_path = [(5, 7), (4, 5), (3, 4), (2, 2), (1, 1), (0, 0)]
    
    abstract_mdp_traj = AbstractGridMDP()
    abstract_mdp_traj.value_iteration() # Calcola i potenziali completi
    
    # AZZERAMENTO V* (Creazione del Canyon)
    # Manteniamo i potenziali intatti solo negli stati della traiettoria
    for s in list(abstract_mdp_traj.v_star.keys()):
        if s not in trajectory_path:
            abstract_mdp_traj.v_star[s] = 0.0

    agent_traj = HierarchicalDQNLearner(
        env, abstract_mdp_traj, phi_mapping_grid, 
        max_episodes=episodes, use_ddqn=True, policy_name="shaping_trajectory.pth"
    )
    
    rewards_traj = run_trajectory_goal_mdp_training(
        env, agent_traj, abstract_mdp_traj, phi_mapping_grid, episodes, trajectory_path
    )
    plot_single_result(rewards_traj, title="Exp 2 - Trajectory Shaping (+10 Sub-goals)")

    # --- 3. COMBINED PLOT ---
    print("\n>>> Generating Combined Plot...")
    plot_combined_results({
        "Standard Shaping": rewards_std,
        "Trajectory Sub-goals": rewards_traj
    }, title="Comparison: Standard Shaping vs Trajectory Sub-goals")
    
    env.close()


if __name__ == "__main__":
    EPISODES = 1500 
    
    experiment_1_baseline_vs_shaping(episodes=EPISODES)
    
    # experiment_2_trajectory_vs_standard()