import os
import gymnasium as gym
import numpy as np
import argparse
import json

from abstract_mdps import LTLfAutomaton, LTLfWaypointMDP
from agent import HierarchicalDQNLearner
from utils import phi_mapping_sequential, save_sequential_heatmaps, plot_buffer_fractions, plot_shaping_reward_breakdown, plot_comparison_curves

def save_training_data(filename, **kwargs):
    """
    Saves training data arrays to a compressed .npz file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Convert lists to numpy arrays for saving
    np_data = {key: np.array(value, dtype=object) for key, value in kwargs.items()}
    np.savez_compressed(filename, **np_data)
    print(f"\n>>> Training data saved to: {filename}")

def run_sequential_training(env, agent, abstract_mdp, episodes, goal_reward=10000, save_policy=True, use_shaping=True, use_double_epsilon=True, K=1.0, log_file=None):
    num_states = len(abstract_mdp.automaton.states)
    true_episode_rewards = []
    total_episode_rewards = []
    
    # Dynamic Epsilon Tracking Arrays (adattati a num_states)
    epsilons = [agent.eps] * num_states
    eps_single = agent.eps 
    
    eps_histories = [[] for _ in range(num_states)]
    eps_single_history = []
    buffer_histories = [[] for _ in range(num_states)]
    hits_history = [[] for _ in range(num_states)]
    total_hits = [0] * num_states

    log_handle = open(log_file, 'a') if log_file else None
    if log_handle: 
        log_handle.write(f"=== NEW RUN (Shaping: {use_shaping}, Multi-Eps: {use_double_epsilon}) ===\n")
        log_handle.write(f"===Trajectory: {abstract_mdp.waypoints_dict}===\n")

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        
        # Inizializzazione LTLf: recuperiamo lo stato logico iniziale ("q0", "q1", ecc.) e il suo indice
        q = abstract_mdp.automaton.get_initial_q()
        q_idx = abstract_mdp.automaton.states.index(q)
        
        # Track which milestones are hit *this* episode for cascade epsilon decay
        reached_phase_this_episode = [False] * num_states
        
        # Dynamic One-Hot encoding array basato sugli stati logici
        q_one_hot = np.zeros(num_states, dtype=np.float32)
        q_one_hot[q_idx] = 1.0
        s_aug = np.concatenate((s_raw, q_one_hot)).astype(np.float32)
        
        terminated = truncated = False
        episode_true_reward = episode_total_reward = 0.0
        episode_hits = [0] * num_states

        while not (terminated or truncated):
            # Select Epsilon based on current Phase index (q_idx)
            agent.eps = epsilons[q_idx] if use_double_epsilon else eps_single

            a = agent.select_action(s_aug)
            ns_raw, _, terminated, truncated, _ = env.step(a)

            # Estrazione coordinate fisiche astratte (passiamo 0 come fase fittizia solo per avere X e Y)
            abstract_x, abstract_y, _ = phi_mapping_sequential(s_raw, 0)
            abstract_x_ns, abstract_y_ns, _ = phi_mapping_sequential(ns_raw, 0)
            
            env_goal_reward = 0.0

            # --- Dynamic Phase Transition Checking (LTLf) ---
            # Esattamente come il vecchio codice valutava target_x e target_y su (abstract_x_ns, abstract_y_ns)
            truth_assignment = abstract_mdp._get_truth_assignment(abstract_x_ns, abstract_y_ns)
            next_q = abstract_mdp.automaton.get_next_q(q, truth_assignment)
            next_q_idx = abstract_mdp.automaton.states.index(next_q)

            # Se c'è stata una transizione logica (raggiunto un waypoint/obiettivo logico)
            if next_q != q:
                if abstract_mdp.automaton.is_goal_reached(next_q):
                    # Final Goal Check
                    total_hits[q_idx] += 1
                    episode_hits[q_idx] = 1
                    env_goal_reward = goal_reward
                    terminated = True
                else:
                    # Raggiungimento traguardo intermedio
                    total_hits[q_idx] += 1
                    episode_hits[q_idx] = 1
                    reached_phase_this_episode[q_idx] = True

            # Ricostruiamo la tripla (x, y, q) per replicare la struttura del tuo vecchio codice
            abstract_s = (abstract_x, abstract_y, q)
            abstract_ns = (abstract_x_ns, abstract_y_ns, next_q)
            
            done = terminated or truncated
            episode_true_reward += env_goal_reward
            
            # Next One-Hot 
            next_q_one_hot = np.zeros(num_states, dtype=np.float32)
            next_q_one_hot[next_q_idx] = 1.0
            ns_aug = np.concatenate((ns_raw, next_q_one_hot)).astype(np.float32)

            # --- Potential Based Shaping ---
            shaping_signal = 0.0
            if use_shaping:
                # Applica il shaping solo se lo stato astratto o logico è cambiato (replica del tuo logica originale)
                if abstract_s != abstract_ns:
                    phi_s = abstract_mdp.v_star.get(abstract_s, 0.0)
                    phi_ns = abstract_mdp.v_star.get(abstract_ns, 0.0)
                    shaping_signal = K * (abstract_mdp.gamma * phi_ns - phi_s)
            
            total_step_reward = env_goal_reward + shaping_signal
            episode_total_reward += total_step_reward
            
            agent.memory.push(s_aug, a, total_step_reward, ns_aug, done)
            agent.optimize_model()

            s_raw = ns_raw
            s_aug = ns_aug
            q = next_q
            q_idx = next_q_idx
            
        # Cascaded Multi-Epsilon Decay (Usa i nuovi indici LTLf)
        if use_double_epsilon:
            epsilons[0] = max(0.08, epsilons[0] * agent.eps_decay)
            for i in range(1, num_states):
                # Start decaying phase N only if phase N-1 is consistently reached and its epsilon is minimal
                if reached_phase_this_episode[i-1] and epsilons[i-1] <= 0.081:
                    epsilons[i] = max(0.08, epsilons[i] * agent.eps_decay)
        else:
            eps_single = max(agent.eps_min, eps_single * agent.eps_decay)
            
        # Logging Data Appends
        for i in range(num_states):
            eps_histories[i].append(epsilons[i])
            buffer_histories[i].append(agent.memory.q_fraction_onehot(i, num_states))
            hits_history[i].append(episode_hits[i])
            
        true_episode_rewards.append(episode_true_reward)
        total_episode_rewards.append(episode_total_reward)
        eps_single_history.append(eps_single)

        # Print logs
        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            recent_avg_with_shaping = np.mean(total_episode_rewards[-100:])
            mode_str = "SHAPING" if use_shaping else "BASELINE"
            eps_str = "MULTI EPS" if use_double_epsilon else "SINGLE EPS"

            # Dynamic labels generate direttamente dagli stati logici (e.g., q0, q1, ecc.)
            labels = [f"q{s}" for s in abstract_mdp.automaton.states]
            
            # 1. Format Epsilons String
            if use_double_epsilon:
                eps_details = ", ".join([f"{labels[i]}: {epsilons[i]:.4f}" for i in range(num_states)])
            else:
                eps_details = f"Single: {eps_single:.4f}"

            # 2. Format Buffer Fractions String dynamically calling the dynamic q_fraction_onehot
            buffer_details = ", ".join([
                f"{labels[i]}: {agent.memory.q_fraction_onehot(i, num_states):.2%}" 
                for i in range(num_states)
            ])

            # 3. Format Target Hits String
            hits_details = ", ".join([f"{labels[i]}: {total_hits[i]}" for i in range(num_states)])

            log_string = (
                f"----------------------------------------------------------------------------------------------------\n"
                f"[{mode_str} | {eps_str}] Episode {n_episode + 1}/{episodes}\n"
                f"Avg Reward                  : {recent_avg:.6f}\n" +
                (f"Avg Total Reward            : {recent_avg_with_shaping:.6f}\n" if use_shaping else "") +
                f"Epsilon Decay               : {eps_details}\n"
                f"Buffer Fractions            : {buffer_details}\n"
                f"Hits (Cumulative)           : {hits_details}\n"
                f"----------------------------------------------------------------------------------------------------\n"
            )
            
            print(log_string)
            if log_handle:
                log_handle.write(log_string + "\n")
                log_handle.flush()

        # Policy saving
        if save_policy and (n_episode + 1) % 250 == 0:
            prefix = "shaping" if use_shaping else "baseline"
            eps_suffix = "multi_eps" if use_double_epsilon else "single_eps"
            agent.policy_name = f"{prefix}_{eps_suffix}_policy_ep_{n_episode + 1}.pth"
            agent._save_policy()

    if log_handle: log_handle.close()
    
    # Return the correct epsilon history based on the mode
    final_eps_history = eps_single_history if not use_double_epsilon else eps_histories
    return true_episode_rewards, total_episode_rewards, final_eps_history, buffer_histories, hits_history

def main(args):
    # Create a unique directory for this experiment run
    data_dir = "results"
    img_dir = "img"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True) # For log files
    
    # Determine plot directory based on execution mode
    plot_dir = data_dir if args.post_process else img_dir
    
    print(f"=== STARTING LTLf STUDY (Mode: {args.mode}) ===")
    print(f"Data will be saved in: '{data_dir}/'. Plots will be saved in: '{plot_dir}/'")

    abstract_goal_reward = 10000
    env_goal_reward = 10000
    gamma = 0.99
    plot_window_size = 500
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    # Extract LTLf formula and waypoint dictionary
    formula_str = config.get('formula', 'F(goal)')
    raw_waypoints = config.get('waypoints_dict', {'goal': [5, 0]})
    
    # Convert lists in JSON to tuples for the MDP
    waypoints_dict = {name: tuple(coords) for name, coords in raw_waypoints.items()}
    
    print(f"\nLoaded LTLf Formula: {formula_str}")
    print(f"Waypoint Dictionary: {waypoints_dict}")
    
    # Inizializziamo l'automa sempre, così num_states è corretto anche per il post-process
    print("\nGenerating LTLf Automaton ...")
    automaton = LTLfAutomaton(formula_str)
    num_states = len(automaton.states)
    
    if not args.post_process:
        automaton.render_graph()
        env = gym.make("LunarLander-v3", continuous=False)
        
        # --- Abstraction ---
        print("\nAbstract Value Iteration ...")
        abstract_mdp = LTLfWaypointMDP(
            waypoints_dict=waypoints_dict, 
            ltlf_automaton=automaton, 
            gamma=gamma, 
            goal_reward=abstract_goal_reward
        )
        abstract_mdp.value_iteration()
        save_sequential_heatmaps(abstract_mdp, filename_prefix=f"{plot_dir}/{args.mode}_exp")

        # --- Training Runs ---
        if args.mode in ['single', 'comparison']:
            print("\n" + "="*50 + "\nTRAINING: SHAPING WITH SINGLE EPSILON\n" + "="*50)
            agent_single_eps = HierarchicalDQNLearner(
                env=env, max_episodes=args.episodes, eps_decay=0.9996, use_ddqn=True, 
                extra_state_dims=num_states
            )
            s_true, s_total, s_eps, s_bufs, _ = run_sequential_training(
                env, agent_single_eps, abstract_mdp, args.episodes, goal_reward=env_goal_reward, use_shaping=True, use_double_epsilon=False, 
                log_file="logs/single_epsilon_training.log"
            )
            save_training_data(f"{data_dir}/single_eps_data.npz", true_rewards=s_true, total_rewards=s_total, eps_histories=s_eps, buffer_histories=s_bufs)

        if args.mode in ['multi', 'comparison']:
            print("\n" + "="*50 + "\nTRAINING: SHAPING WITH MULTI EPSILON\n" + "="*50)
            agent_multi_eps = HierarchicalDQNLearner(
                env=env, max_episodes=args.episodes, eps_decay=0.999, use_ddqn=True, 
                extra_state_dims=num_states
            )
            m_true, m_total, m_eps, m_bufs, _ = run_sequential_training(
                env, agent_multi_eps, abstract_mdp, args.episodes, goal_reward=env_goal_reward, use_shaping=True, use_double_epsilon=True, 
                log_file="logs/multi_epsilon_training.log"
            )
            save_training_data(f"{data_dir}/multi_eps_data.npz", true_rewards=m_true, total_rewards=m_total, eps_histories=m_eps, buffer_histories=m_bufs)

        env.close()

    # --- POST-PROCESSING / PLOTTING ---
    print("\n" + "="*50 + "\nGENERATING PLOTS\n" + "="*50)
    
    if args.mode in ['single', 'comparison']:
        print("Loading data for 'single epsilon' and generating plots...")
        data = np.load(f"{data_dir}/single_eps_data.npz", allow_pickle=True)
        true_rewards_single = data['true_rewards']
        total_rewards_single = data['total_rewards']
        eps_histories_single = data['eps_histories']
        buffers_single = data['buffer_histories']
        
        plot_buffer_fractions(buffers_single, filename=f"{plot_dir}/buffer_fractions_single_eps.png", window_size=plot_window_size)
        plot_shaping_reward_breakdown(true_rewards_single, total_rewards_single, eps_histories_single, window_size=plot_window_size, 
            filename=f"{plot_dir}/reward_breakdown_single_eps.png"
        )

    if args.mode in ['multi', 'comparison']:
        print("Loading data for 'multi epsilon' and generating plots...")
        data = np.load(f"{data_dir}/multi_eps_data.npz", allow_pickle=True)
        true_rewards_multi = data['true_rewards']
        total_rewards_multi = data['total_rewards']
        eps_histories_multi = data['eps_histories']
        buffers_multi = data['buffer_histories']

        plot_buffer_fractions(buffers_multi, filename=f"{plot_dir}/buffer_fractions_multi_eps.png", window_size=plot_window_size)
        plot_shaping_reward_breakdown(true_rewards_multi, total_rewards_multi, eps_histories_multi, window_size=plot_window_size, 
            filename=f"{plot_dir}/reward_breakdown_multi_eps.png"
        )

    if args.mode == 'comparison':
        print("\nGenerating comparison plot...")
        # Reload single epsilon data to be safe, even if it's already in memory
        data_single = np.load(f"{data_dir}/single_eps_data.npz", allow_pickle=True)
        true_rewards_single = data_single['true_rewards']
        data_multi = np.load(f"{data_dir}/multi_eps_data.npz", allow_pickle=True)
        true_rewards_multi = data_multi['true_rewards']

        plot_comparison_curves(
            baseline_rewards=true_rewards_single,
            shaping_rewards=true_rewards_multi,
            window_size=plot_window_size,
            filename=f"{plot_dir}/single_vs_multi_epsilon_comparison.png",
            title="Single Epsilon vs. Multi Epsilon Performance",
            baseline_label="Single Epsilon",
            shaping_label="Multi Epsilon"
        )

    print("\nFinished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LTLf DQN training with different epsilon strategies.")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="comparison", 
        choices=['single', 'multi', 'comparison', 'post-process'],
        help="Execution mode: 'single' for single epsilon, 'multi' for multi-epsilon, 'comparison' for both."
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=1, 
        help="Number of training episodes."
    )
    parser.add_argument(
        "--config", 
        type=str,
        default="trajectory.json", 
        help="Path to the trajectory configuration file."
    )
    parser.add_argument(
        "--post-process",
        action="store_true",
        help="Run only the plotting part, loading data from the 'results' directory."
    )
    args = parser.parse_args()
    main(args)