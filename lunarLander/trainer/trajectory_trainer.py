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
# PLOTTING UTILITY FOR COMPARISON
# =====================================================================

def plot_comparison_curves(baseline_rewards, shaping_rewards, epsilon_history, window_size=100, filename="img/baseline_vs_shaping.png"):
    """
    Plots the smoothed moving average of the baseline agent against the shaping agent,
    including the epsilon decay on a secondary axis.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Calcola la media mobile usando pandas per una gestione corretta dei bordi
    baseline_ma = pd.Series(baseline_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    shaping_ma = pd.Series(shaping_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    x_axis = np.arange(len(baseline_rewards))
        
    # Plot curves on the primary axis (ax1)
    ax1.plot(x_axis, baseline_ma, color='black', linestyle='-', linewidth=2, label='Baseline (No Shaping)')
    ax1.plot(x_axis, shaping_ma, color='blue', linestyle='-', linewidth=2.5, label='Shaping Agent (Potential Guided)')
    
    # Formatting
    ax1.set_title("Learning Curve Comparison: Baseline vs Shaping", fontsize=15, fontweight='bold')
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
    fig.savefig(filename, dpi=200)
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
    fig.savefig(filename, dpi=200)
    print(f"\n>>> Shaping reward breakdown plot successfully saved to: {filename}")
    plt.close(fig)

# =====================================================================
# TRAINING LOOP
# =====================================================================

def run_sequential_training_old(env, agent, abstract_mdp, episodes, use_shaping=True, K=1.0, log_file=None):
    """
    Executes the training loop for the sequential task.
    """
    true_episode_rewards = []
    total_episode_rewards = []
    epsilon_history = []

    log_handle = open(log_file, 'a') if log_file else None
    if log_handle:
        log_handle.write("="*50 + f"\nSTARTING NEW TRAINING RUN\n" + "="*50 + "\n")

    # Counters
    natural_q_updates = 0
    waypoint_hits = 0
    goal_hits = 0
    truncated_episodes = 0

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        
        # Initialize sequence variable 'q' (0 = seek waypoint, 10 = seek goal)
        q = 0 
        passed_trough_waypoint = False
        
        # Augment state for the Neural Network
        s_aug = np.append(s_raw, q) 
        
        terminated = truncated = False
        episode_true_reward = 0.0
        episode_total_reward = 0.0

        # Episode loop
        while not (terminated or truncated):
            env_goal_reward = 0.0

            # STATE TRANSITION LOGIC
            # If the agent reaches the waypoint during phase q=0
            abstract_x, abstract_y, _ = phi_mapping_sequential(s_raw, q)
            if abstract_x == 1 and abstract_y == 8 and q == 0:
                passed_trough_waypoint = True
                waypoint_hits += 1
                q = 10
                natural_q_updates += 1
                env_goal_reward = 10000
                
            # Building the current state augmented: environment state + q state
            s_aug = np.append(s_raw, q)
            
            # Agent selects action based on the augmented state
            a = agent.select_action(s_aug)
            ns_raw, _, terminated, truncated, _ = env.step(a)

            if truncated:
                truncated_episodes += 1

            # Map continuous state to 3D abstract state (x, y, q)
            abstract_x_ns, abstract_y_ns, _ = phi_mapping_sequential(ns_raw, q)
            ns_aug = np.append(ns_raw, q)
            abstract_ns = (abstract_x_ns, abstract_y_ns, q)

            # Check if the final goal is reached (and waypoint was passed)
            if abstract_ns == abstract_mdp.goal_state and q==10 and passed_trough_waypoint:
                goal_hits += 1
                env_goal_reward = 10000
                terminated = True
            
            done = terminated or truncated
            episode_true_reward += env_goal_reward
            
            # --- Shaping Signal Calculation (Discrete) ---
            if use_shaping:
                # Calcola lo stato astratto corrente e successivo
                abstract_s = phi_mapping_sequential(s_raw, q)
                shaping_signal = 0.0
                
                # Applica lo shaping solo se l'agente cambia cella astratta
                if abstract_s != abstract_ns:
                    phi_s = abstract_mdp.v_star.get(abstract_s, 0.0)
                    phi_ns = abstract_mdp.v_star.get(abstract_ns, 0.0)
                    shaping_signal = K * (agent.gamma * phi_ns - phi_s)
            else:
                shaping_signal = 0.0
            
            total_step_reward = env_goal_reward + shaping_signal
            episode_total_reward += total_step_reward
            
            # Push AUGMENTED states to memory and optimize
            agent.memory.push(s_aug, a, total_step_reward, ns_aug, done)
            agent.optimize_model()

            s_raw = ns_raw
            s_aug = ns_aug
            
        # Decay epsilon at the end of the episode
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)
        total_episode_rewards.append(episode_total_reward)
        epsilon_history.append(agent.eps)

        # Print progress every 100 episodes
        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            recent_avg_with_shaping = np.mean(total_episode_rewards[-100:])
            mode_str = "SHAPING" if use_shaping else "BASELINE"

            log_string = (
                f"[{mode_str}] Episode {n_episode + 1}/{episodes}\n" +
                f"  Avg Reward              : {recent_avg:.6f}\n" +
                f"  Avg With Shaping Reward : {recent_avg_with_shaping:.6f}\n" +
                f"  Epsilon                 : {agent.eps:.6f}\n" +                
                f"  Exp q0 % and q10 %      : {agent.memory.q0_fraction():.6f}, {agent.memory.q1_fraction():.6f}\n" +
                f"  Natural q=0→10 updates  : {natural_q_updates}\n" +
                f"  Waypoint hits           : {waypoint_hits}\n" +
                f"  Goal hits               : {goal_hits}\n"
                f"  Truncated episodes      : {truncated_episodes}\n"
            )

            print(log_string)
            if log_handle:
                log_handle.write(log_string + "\n")
                log_handle.flush()

            agent._save_policy()
    
    if log_handle:
        log_handle.close()

    return np.array(true_episode_rewards), np.array(total_episode_rewards), np.array(epsilon_history)

def run_sequential_training_single_eps(env, agent, abstract_mdp, episodes, use_shaping=True, K=1.0, log_file=None):
    """
    Executes the training loop for the sequential task.
    """
    true_episode_rewards = []
    total_episode_rewards = []
    epsilon_history = []

    log_handle = open(log_file, 'a') if log_file else None
    if log_handle:
        log_handle.write("="*50 + f"\nSTARTING NEW TRAINING RUN\n" + "="*50 + "\n")

    # Counters
    natural_q_updates = 0
    waypoint_hits = 0
    goal_hits = 0
    truncated_episodes = 0

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        
        # Initialize sequence variable 'q' (0 = seek waypoint, 10 = seek goal)
        q = 0 
        passed_trough_waypoint = False
        
        # Augment state for the Neural Network: environment state + q state
        s_aug = np.append(s_raw, q) 
        
        terminated = truncated = False
        episode_true_reward = 0.0
        episode_total_reward = 0.0

        # Episode loop
        while not (terminated or truncated):
            env_goal_reward = 0.0

            # 1. Agent selects action based on the augmented state
            a = agent.select_action(s_aug)
            ns_raw, _, terminated, truncated, _ = env.step(a)

            if truncated:
                truncated_episodes += 1

            # 2. Map continuous next state to 2D abstract state coordinates
            abstract_x_ns, abstract_y_ns, _ = phi_mapping_sequential(ns_raw, q)
            
            next_q = q

            # 3. STATE TRANSITION LOGIC: Check if the NEXT state is the waypoint
            if abstract_x_ns == 1 and abstract_y_ns == 8 and q == 0:
                passed_trough_waypoint = True
                waypoint_hits += 1
                next_q = 10
                natural_q_updates += 1
                env_goal_reward = 0
                
            # Build the abstract next state using next_q
            abstract_ns = (abstract_x_ns, abstract_y_ns, next_q)

            # 4. Check if the final goal is reached (and waypoint was passed)
            if abstract_ns == abstract_mdp.goal_state and next_q == 10 and passed_trough_waypoint:
                goal_hits += 1
                env_goal_reward = 10000
                terminated = True
            
            done = terminated or truncated
            episode_true_reward += env_goal_reward
            
            # 5. Build the NEXT augmented state
            ns_aug = np.append(ns_raw, next_q)

            # --- Shaping Signal Calculation (Discrete) ---
            if use_shaping:
                # Calcola lo stato astratto corrente
                abstract_s = phi_mapping_sequential(s_raw, q)
                shaping_signal = 0.0
                
                # Applica lo shaping solo se l'agente cambia cella astratta o fase q
                if abstract_s != abstract_ns:
                    phi_s = abstract_mdp.v_star.get(abstract_s, 0.0)
                    phi_ns = abstract_mdp.v_star.get(abstract_ns, 0.0)
                    shaping_signal = K * (1 * phi_ns - phi_s)
            else:
                shaping_signal = 0.0
            
            total_step_reward = env_goal_reward + shaping_signal
            episode_total_reward += total_step_reward
            
            # 6. Push AUGMENTED states to memory and optimize
            agent.memory.push(s_aug, a, total_step_reward, ns_aug, done)
            agent.optimize_model()

            # 7. Update states for the next step
            s_raw = ns_raw
            s_aug = ns_aug
            q = next_q
            
        # Decay epsilon at the end of the episode
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)
        total_episode_rewards.append(episode_total_reward)
        epsilon_history.append(agent.eps)

        # Print progress every 100 episodes
        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            recent_avg_with_shaping = np.mean(total_episode_rewards[-100:])
            mode_str = "SHAPING" if use_shaping else "BASELINE"

            log_string = (
                f"[{mode_str}] Episode {n_episode + 1}/{episodes}\n" +
                f"  Avg Reward              : {recent_avg:.6f}\n" +
                f"  Avg With Shaping Reward : {recent_avg_with_shaping:.6f}\n" +
                f"  Epsilon                 : {agent.eps:.6f}\n" +                
                f"  Exp q0 % and q10 %      : {agent.memory.q0_fraction():.6f}, {agent.memory.q1_fraction():.6f}\n" +
                f"  Natural q=0→10 updates  : {natural_q_updates}\n" +
                f"  Waypoint hits           : {waypoint_hits}\n" +
                f"  Goal hits               : {goal_hits}\n" +
                f"  Truncated episodes      : {truncated_episodes}\n"
            )

            print(log_string)
            if log_handle:
                log_handle.write(log_string + "\n")
                log_handle.flush()

            agent._save_policy()
    
    if log_handle:
        log_handle.close()

    return np.array(true_episode_rewards), np.array(total_episode_rewards), np.array(epsilon_history)

def run_sequential_training(env, agent, abstract_mdp, episodes, use_shaping=True, K=1.0, log_file=None):
    """
    Executes the training loop for the sequential task with state-dependent Epsilon.
    """
    true_episode_rewards = []
    total_episode_rewards = []
    
    # 1. Inizializzazione dei due Epsilon separati
    eps_q0 = agent.eps  # Parte dal valore iniziale (es. 1.0)
    eps_q10 = agent.eps 
    eps_min = agent.eps_min
    eps_decay = agent.eps_decay
    
    eps_q0_history = []
    eps_q10_history = []

    log_handle = open(log_file, 'a') if log_file else None
    if log_handle:
        log_handle.write("="*50 + f"\nSTARTING NEW TRAINING RUN\n" + "="*50 + "\n")

    # Counters
    natural_q_updates = 0
    waypoint_hits = 0
    goal_hits = 0
    truncated_episodes = 0

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        
        # Inizializza le variabili logiche dell'episodio
        q = 0 
        passed_trough_waypoint = False
        reached_q10_this_episode = False
        
        # Augment state for the Neural Network: environment state + q state
        s_aug = np.append(s_raw, q) 
        
        terminated = truncated = False
        episode_true_reward = 0.0
        episode_total_reward = 0.0

        # Episode loop
        while not (terminated or truncated):
            env_goal_reward = 0.0

            # 2. Imposta dinamicamente l'Epsilon corretto in base alla fase attuale
            agent.eps = eps_q0 if q == 0 else eps_q10

            # Agent selects action based on the augmented state
            a = agent.select_action(s_aug)
            ns_raw, _, terminated, truncated, _ = env.step(a)

            if truncated:
                truncated_episodes += 1

            # Map continuous next state to 2D abstract state coordinates
            abstract_x_ns, abstract_y_ns, _ = phi_mapping_sequential(ns_raw, q)
            next_q = q

            # 3. STATE TRANSITION LOGIC: Check if the NEXT state is the waypoint
            if abstract_x_ns == 1 and abstract_y_ns == 8 and q == 0:
                passed_trough_waypoint = True
                waypoint_hits += 1
                next_q = 10
                natural_q_updates += 1
                env_goal_reward = 0.0  # Nessun reward dall'ambiente, ci pensa lo shaping
                reached_q10_this_episode = True
                
            # Build the abstract next state using next_q
            abstract_ns = (abstract_x_ns, abstract_y_ns, next_q)

            # Check if the final goal is reached (and waypoint was passed)
            if abstract_ns == abstract_mdp.goal_state and next_q == 10 and passed_trough_waypoint:
                goal_hits += 1
                env_goal_reward = 10000.0
                terminated = True
            
            done = terminated or truncated
            episode_true_reward += env_goal_reward
            
            # Build the NEXT augmented state
            ns_aug = np.append(ns_raw, next_q)

            # 4. Shaping Signal Calculation (Senza Gamma)
            if use_shaping:
                abstract_s = phi_mapping_sequential(s_raw, q)
                shaping_signal = 0.0
                
                # Applica lo shaping solo se l'agente cambia cella astratta o fase q
                if abstract_s != abstract_ns:
                    phi_s = abstract_mdp.v_star.get(abstract_s, 0.0)
                    phi_ns = abstract_mdp.v_star.get(abstract_ns, 0.0)
                    # Rimossa la moltiplicazione per agent.gamma per avere un gradiente positivo
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
            
        # 5. DECAY DEGLI EPSILON A FINE EPISODIO
        
        # eps_q0 decade SEMPRE per stabilizzare la prima fase
        eps_q0 = max(eps_min, eps_q0 * eps_decay)
        
        # eps_q10 decade SOLO SE l'agente è entrato in q=10 durante questo episodio
        if reached_q10_this_episode and eps_q0 < 0.05:
            eps_q10 = max(eps_min, eps_q10 * eps_decay)

        true_episode_rewards.append(episode_true_reward)
        total_episode_rewards.append(episode_total_reward)
        eps_q0_history.append(eps_q0)
        eps_q10_history.append(eps_q10)

        # Print progress every 100 episodes
        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            recent_avg_with_shaping = np.mean(total_episode_rewards[-100:])
            mode_str = "SHAPING" if use_shaping else "BASELINE"

            log_string = (
                f"[{mode_str}] Episode {n_episode + 1}/{episodes}\n" +
                f"  Avg Reward              : {recent_avg:.6f}\n" +
                f"  Avg With Shaping Reward : {recent_avg_with_shaping:.6f}\n" +
                f"  Epsilon (q0, q10)       : {eps_q0:.6f}, {eps_q10:.6f}\n" +                
                f"  Exp q0 % and q10 %      : {agent.memory.q0_fraction():.6f}, {agent.memory.q1_fraction():.6f}\n" +
                f"  Natural q=0→10 updates  : {natural_q_updates}\n" +
                f"  Waypoint hits           : {waypoint_hits}\n" +
                f"  Goal hits               : {goal_hits}\n" +
                f"  Truncated episodes      : {truncated_episodes}\n"
            )

            print(log_string)
            if log_handle:
                log_handle.write(log_string + "\n")
                log_handle.flush()

            agent._save_policy()
    
    if log_handle:
        log_handle.close()

    # Ritorna entrambe le history degli Epsilon all'interno di una tupla
    return np.array(true_episode_rewards), np.array(total_episode_rewards), (np.array(eps_q0_history), np.array(eps_q10_history))

# =====================================================================
# MAIN EXPERIMENT ORCHESTRATOR
# =====================================================================

def main():
    print("=== STARTING SEQUENTIAL TASK EXPERIMENT: BASELINE VS SHAPING ===")
    os.makedirs("logs", exist_ok=True)
    
    # HYPERPARAMETERS
    episodes = 10000
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
    # EXPERIMENT 1: BASELINE AGENT (NO SHAPING)
    # -----------------------------------------------------------------
    #print("\n=======================================================")
    #print("TRAINING: BASELINE AGENT (NO SHAPING)")
    #print("=======================================================")
    #agent_baseline = HierarchicalDQNLearner(
    #    env=env,
    #    max_episodes=episodes,
    #    gamma=gamma,
    #    eps_decay=eps_decay,
    #    use_ddqn=True,
    #    policy_name="baseline_sequential_policy.pth",
    #    extra_state_dims=1
    #)
    #
    #baseline_learning_curve, _, baseline_eps_history = run_sequential_training(
    #    env, 
    #    agent_baseline, 
    #    abstract_mdp, 
    #    episodes, 
    #    use_shaping=False,
    #    log_file="logs/baseline_training.log"
    #)

    # -----------------------------------------------------------------
    # EXPERIMENT 2: SHAPING AGENT
    # -----------------------------------------------------------------
    print("\n=======================================================")
    print("TRAINING: SHAPING AGENT")
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
    
    shaping_learning_curve, shaping_total_rewards, shaping_eps_history = run_sequential_training(
        env, 
        agent_shaping, 
        abstract_mdp, 
        episodes, 
        use_shaping=True, 
        K=K_scaling,
        log_file="logs/shaping_training.log"
    )
    
    # -----------------------------------------------------------------
    # PLOTTING RESULTS
    # -----------------------------------------------------------------
    print("\n3. Generating plots...")
    #plot_comparison_curves(baseline_learning_curve, shaping_learning_curve, baseline_eps_history, window_size=500)
    plot_shaping_reward_breakdown(shaping_learning_curve, shaping_total_rewards, shaping_eps_history, window_size=500, filename="img/shaping_reward_breakdown.png")

    env.close()

if __name__ == "__main__":
    main()