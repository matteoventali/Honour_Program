import gymnasium as gym
import numpy as np
from abstract_mdps import SequentialWaypointMDP
from agent import HierarchicalDQNLearner
from utils import *

# =====================================================================
# TRAINING LOOP
# =====================================================================

def run_sequential_training(env, agent, abstract_mdp, episodes, use_shaping=True):
    true_episode_rewards = []
    K = 1.0  # Reward Shaping Scaling Constant

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        
        # Initialize sequence variable 'q'
        q = 0 
        
        # Augment state for the Neural Network (from 8 to 9 dimensions)
        s_aug = np.append(s_raw, q) 
        
        terminated = truncated = False
        episode_true_reward = 0.0
        
        while not (terminated or truncated):
            # Agent acts based on the augmented state
            a = agent.select_action(s_aug)
            ns_raw, _, terminated, truncated, _ = env.step(a)

            env_goal_reward = 0.0
            
            # Map continuous state to abstract state
            abstract_x_curr, abstract_y_curr, _ = phi_mapping_sequential(ns_raw, q)
            next_q = q
            
            # STATE TRANSITION LOGIC
            if abstract_x_curr == 1 and abstract_y_curr == 8 and q == 0:
                env_goal_reward = 50
                next_q = 1
            
            ns_aug = np.append(ns_raw, next_q)
            abstract_ns = (abstract_x_curr, abstract_y_curr, next_q)

            #if (abstract_ns[0], abstract_ns[1]) == abstract_mdp.waypoint and q == 0:
            #    print("Waypoint reached")
                
            if abstract_ns == abstract_mdp.goal_state:
                env_goal_reward = 100.0
                #print("Final point reached")
                terminated = True
            elif terminated or truncated:
                env_goal_reward = -100.0
            
            done = terminated or truncated
            episode_true_reward += env_goal_reward

            # Shaping Signal Calculation
            if use_shaping:
                px_s, py_s = get_continuous_grid_coords(s_raw, abstract_mdp.width, abstract_mdp.height)
                px_ns, py_ns = get_continuous_grid_coords(ns_raw, abstract_mdp.width, abstract_mdp.height)
                
                phi_s = get_bilinear_potential_sequential(px_s, py_s, q, abstract_mdp.v_star)
                phi_ns = get_bilinear_potential_sequential(px_ns, py_ns, next_q, abstract_mdp.v_star)
                
                shaping_signal = K * (agent.gamma * phi_ns - phi_s)

                #if q == 1:
                #    print(
                #        f"action={a} | "
                #        f"real=({ns_raw[0]:.3f},{ns_raw[1]:.3f}) | "
                #        f"abstract={abstract_ns} | "
                #        f"phi={phi_ns:.2f} | "
                #        f"shape={shaping_signal:.2f}"
                #    )
            else:
                shaping_signal = 0.0
            
            total_step_reward = env_goal_reward + shaping_signal
            
            # Push AUGMENTED states to memory
            agent.memory.push(s_aug, a, total_step_reward, ns_aug, done)
            agent.optimize_model()

            s_raw = ns_raw
            s_aug = ns_aug
            q = next_q

        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)

        if truncated:
            print("TRUNCATED")

        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            print(f"-> Progress: Episode {n_episode + 1}/{episodes} | 100-eps Avg Reward: {recent_avg:6.2f} | Epsilon: {agent.eps:.3f}")
            agent._save_policy()
            
    return np.array(true_episode_rewards)

def main():
    print("=== STARTING SEQUENTIAL TASK EXPERIMENT ===")
    episodes = 10000
    gamma = 0.99
    eps_decay = 0.9995
    
    print("\n1. Initializing Environment and Abstract MDP...")
    env = gym.make("LunarLander-v3", continuous=False, max_episode_steps=100000)
    
    abstract_mdp = SequentialWaypointMDP(width=12, height=12, gamma=gamma)
    abstract_mdp.value_iteration()

    print("\n -> Plotting Value Functions...")
    save_sequential_heatmaps(abstract_mdp, filename_prefix="seq_experiment_normal")
    save_sequential_interpolated_heatmaps(abstract_mdp, filename_prefix="seq_experiment")
    
    print("\n2. Initializing RL Agent...")
    agent = HierarchicalDQNLearner(
        env=env,
        max_episodes=episodes,
        gamma=gamma,
        eps_decay=eps_decay,
        use_ddqn=True,
        policy_name="sequential_policy.pth",
        extra_state_dims=1
    )
    
    print("\n3. Starting Training Loop...")
    learning_curve = run_sequential_training(env, agent, abstract_mdp, episodes, use_shaping=True)
    
    print("\n4. Saving Results...")
    plot_training_results(learning_curve, window_size=50)
    print("=== EXPERIMENT COMPLETE ===")
    
    env.close()

if __name__ == "__main__":
    main()