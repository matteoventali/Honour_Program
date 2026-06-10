import gymnasium as gym
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import os

# Import the configurable diagonal MDP and agent components
from abstract_mdps import ConfigurableDiagonalMDP
from utils import phi_mapping_grid
from agent import HierarchicalDQNLearner

# =====================================================================
# PLOTTING UTILITIES
# =====================================================================

def save_value_function_heatmap(abstract_mdp, filename, width=12, height=12, title="Potential Map V*"):
    """
    Converts the v_star dictionary into a 2D matrix and saves it as a Heatmap PNG.
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
            if val != 0.0: 
                # Adapt text color for readability against the heatmap background
                text_color = 'white' if val < (np.max(v_matrix) / 2) else 'black'
                plt.text(x, y, f"{val:.2f}", ha='center', va='center', 
                         color=text_color, fontsize=8, fontweight='bold')
                
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Potential Value (V*)")
    plt.title(title, fontsize=14)
    plt.xlabel("X (Horizontal Position)", fontsize=12)
    plt.ylabel("Y (Altitude / Distance from ground)", fontsize=12)
    
    # --- GRID ALIGNMENT ---
    plt.xticks(np.arange(0, width, 1))
    plt.yticks(np.arange(0, height, 1))
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, height, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1, alpha=0.4)
    ax.grid(which='major', color='none') 
    
    plt.tight_layout()
    
    # Save the figure to disk instead of blocking execution
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close() # CRITICAL: Frees up memory so the RAM doesn't overflow during the loop


def save_grid_search_learning_curves(results_dict, filename="grid_search_learning_curves.png", window_size=50):
    """
    Takes the dictionary of results and saves smoothed learning curves 
    for every tested configuration on a single chart.
    """
    print("\n>>> Generating and saving Learning Curves Plot...")
    plt.figure(figsize=(16, 9)) 
    
    # Generate distinct colors for the different curves
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(results_dict)))
    
    for idx, (config_name, rewards) in enumerate(results_dict.items()):
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            x_axis = range(window_size - 1, len(rewards))
            plt.plot(x_axis, moving_avg, color=colors[idx], linewidth=2.0, alpha=0.85, label=config_name)
        else:
            plt.plot(rewards, color=colors[idx], linewidth=2.0, alpha=0.85, label=config_name)

    plt.title("Grid Search: Learning Curves Comparison", fontsize=16, fontweight='bold')
    plt.xlabel('Episode #', fontsize=14)
    plt.ylabel(f'True Episode Reward (Moving Avg window={window_size})', fontsize=14)
    
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Win Threshold')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    # Place legend outside the main plot area to prevent obscuring the lines
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0., fontsize=10)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()


# =====================================================================
# TRAINING LOOP
# =====================================================================

def run_grid_search_training(env, agent, abstract_mdp, episodes):
    """Optimized training loop for Grid Search execution."""
    true_episode_rewards = []
    
    # --- DYNAMIC K HANDLING ---
    # Scales K inversely to the goal_reward so the final shaping signal 
    # always matches the environment's +100 scale, preventing gradient explosion.
    K = 100.0 / abstract_mdp.goal_reward

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        terminated = truncated = False
        episode_true_reward = 0.0
        
        while not (terminated or truncated):
            a = agent.select_action(s_raw)
            ns_raw, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            
            abstract_s = phi_mapping_grid(s_raw)
            abstract_ns = phi_mapping_grid(ns_raw)
            
            env_goal_reward = 0.0
            
            # Victory Check: Is the agent in ANY of the designated goal cells?
            if terminated and abstract_ns in abstract_mdp.goal_states:
                env_goal_reward = 100.0
                
            episode_true_reward += env_goal_reward
                
            phi_s = abstract_mdp.v_star[abstract_s]
            phi_ns = 0.0 if done else abstract_mdp.v_star[abstract_ns]
            
            shaping_signal = K * (agent.gamma * phi_ns - phi_s)
            
            total_step_reward = env_goal_reward + shaping_signal
            agent.memory.push(s_raw, a, total_step_reward, ns_raw, done)
            agent.optimize_model()

            s_raw = ns_raw

        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            print(f"-> Progress: Episode {n_episode + 1}/{episodes} | Recent 100-eps Avg Reward: {recent_avg:6.2f} | Epsilon: {agent.eps:.3f}")
            
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)
        
    return np.array(true_episode_rewards)

# =====================================================================
# MAIN GRID SEARCH
# =====================================================================

def main():
    print("=== STARTING GRID SEARCH: Diagonal Abstract MDP ===")
    
    # Ensure a directory exists for our generated plots
    os.makedirs("img/grid_search_plots", exist_ok=True)
    
    # 1. Define the hyperparameters to explore
    goal_configurations = {
        "1x1_Strict": [(0,0)],
        "2x1_Base": [(0,0), (1,0)],
        "2x2_Wide": [(0,0), (1,0), (0,1), (1,1)]
    }
    gammas = [0.99, 0.90, 0.80]
    #goal_rewards = [1.0, 100.0]
    goal_rewards = [100.0]
    
    episodes_per_run = 1000
    results = {}
    
    combinations = list(itertools.product(goal_configurations.items(), gammas, goal_rewards))
    
    for idx, ((goal_name, goal_states), gamma, g_rew) in enumerate(combinations):
        
        # Format variables for clean terminal logging and file naming
        # E.g., goal_prefix: "1x1", gamma_str: "099", rew_str: "1" or "100"
        goal_prefix = goal_name.split('_')[0] 
        gamma_str = str(gamma).replace('.', '')
        rew_str = str(int(g_rew))
        
        config_name = f"Goal:{goal_name} | Gamma:{gamma} | Rew:{g_rew}"
        heatmap_filename = f"img/grid_search_plots/v_{goal_prefix}_{gamma_str}_{rew_str}.png"
        
        print(f"\n[{idx+1}/{len(combinations)}] Preparing -> {config_name}")
        
        # Fresh environment instantiation prevents memory leaks and seed contamination
        env = gym.make("LunarLander-v3", continuous=False)
        
        abstract_mdp = ConfigurableDiagonalMDP(
            gamma=gamma, 
            goal_states=goal_states, 
            goal_reward=g_rew
        )
        abstract_mdp.value_iteration()
        
        # --- SAVE V* HEATMAP AUTOMATICALLY ---
        print(f">>> Saving V* map to: {heatmap_filename}")
        save_value_function_heatmap(abstract_mdp, filename=heatmap_filename, title=f"V* | {config_name}")
        
        agent = HierarchicalDQNLearner(
            env, abstract_mdp, phi_mapping_grid, 
            max_episodes=episodes_per_run, use_ddqn=True, 
            policy_name=f"grid_model_{idx}.pth"
        )
        
        print(">>> Training in progress...")
        learning_curve = run_grid_search_training(env, agent, abstract_mdp, episodes_per_run)
        results[config_name] = learning_curve
        
        env.close()
        
        final_avg = np.mean(learning_curve[-100:])
        print(f"-> Finished. Final 100-eps Avg Reward: {final_avg:.2f}")
        
    print("\n=== GRID SEARCH COMPLETED ===")
    with open('grid_search_results.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    best_config = max(results, key=lambda k: np.mean(results[k][-50:]))
    best_score = np.mean(results[best_config][-50:])
    print(f"\n🏆 BEST CONFIGURATION: {best_config} (Final Avg: {best_score:.2f})")
    
    # Automatically save the final composite learning curves chart
    save_grid_search_learning_curves(results, filename="img/grid_search_plots/final_learning_curves.png")
    print(f">>> All plots successfully saved in the 'img/grid_search_plots' directory.")


if __name__ == "__main__":
    main()