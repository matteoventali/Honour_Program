# ==============================
# Standard library imports
# ==============================

import argparse
import json
import os
from collections import Counter

# ==============================
# External and project imports
# ==============================

import gymnasium as gym
import numpy as np

from abstract_mdps import ManualWaypointMDP
from agent import HierarchicalDQNLearner
from manual_automaton import AlternatingGoalsAutomaton
from utils import phi_mapping_sequential, plot_buffer_fractions, plot_shaping_reward_breakdown, save_sequential_heatmaps


# ==============================
# Data and state helpers
# ==============================

def save_training_data(filename, **kwargs):
    """Convert training metrics to arrays and save them in a compressed NPZ file."""
    # Preserve numeric dtypes and rectangular shapes for direct plotting.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np_data = {key: np.asarray(value) for key, value in kwargs.items()}
    if any(array.dtype == object for array in np_data.values()):
        raise ValueError("Training metrics must be rectangular numeric arrays")
    np.savez_compressed(filename, **np_data)
    print(f"\nTraining data saved to: {filename}")


def _abstract_position(observation, abstract_mdp):
    """Map a raw environment observation to its abstract spatial coordinates."""
    x, y, _ = phi_mapping_sequential(
        observation, 0, abstract_mdp.width, abstract_mdp.height
    )
    return x, y


def _augment_state(observation, q, state_to_index):
    """Append a one-hot encoding of the current automaton state."""
    one_hot = np.zeros(len(state_to_index), dtype=np.float32)
    one_hot[state_to_index[q]] = 1.0
    return np.concatenate((observation, one_hot)).astype(np.float32)


def _evaluate_initial_automaton_state(observation, abstract_mdp):
    """Consume the initial observation and return the first active automaton state."""
    initial_x, initial_y = _abstract_position(observation, abstract_mdp)
    initial_truth_assignment = abstract_mdp._get_truth_assignment(initial_x, initial_y)
    pre_trace_q = abstract_mdp.automaton.get_initial_q()
    return abstract_mdp.automaton.advance(
        pre_trace_q, initial_truth_assignment
    ).next_state


def _format_counter(counter):
    """Convert an automaton transition counter into a readable string."""
    if not counter:
        return "none"
    return ", ".join(f"{source}->{destination}: {count}" for (source, destination), count in sorted(counter.items()))


# ==============================
# Logging and checkpoint helpers
# ==============================

def _write_log(message, log_handle=None):
    """Print a message and optionally append it to the active log file."""
    print(message)
    if log_handle:
        log_handle.write(message)
        log_handle.flush()


def _write_run_header(log_handle, episodes, use_shaping, K, goal_reward, abstract_mdp, automaton_states):
    """Write the configuration and automaton metadata for a training run."""
    if not log_handle:
        return
    automaton = abstract_mdp.automaton
    header = (
        "\n=== NEW RUN ===\n"
        f"episodes={episodes}, shaping={use_shaping}, K={K}, goal_reward={goal_reward}, gamma={abstract_mdp.gamma}\n"
        f"waypoints={abstract_mdp.waypoints_dict}\n"
        f"automaton_states={automaton_states}, initial={automaton.get_initial_q()}, accepting={sorted(automaton.accepting_states)}\n"
    )
    log_handle.write(header)
    log_handle.flush()


def _should_log(episode, episodes, log_interval):
    """Return whether the current episode requires a periodic training report."""
    return episode == 0 or episode + 1 == episodes or (episode + 1) % log_interval == 0


def _build_training_log(episode, episodes, log_interval, automaton_states, agent, histories, cumulative_counters):
    """Build a report containing recent metrics and automaton counters."""
    window = min(log_interval, episode + 1)
    recent_slice = slice(-window, None)
    recent_transitions = Counter()
    for transitions in histories["transition_counters"][-window:]:
        recent_transitions.update(transitions)

    recent_state_visits = np.asarray(histories["state_visits"], dtype=np.int64)[:, -window:].sum(axis=1)
    recent_state_entries = np.asarray(histories["state_entries"], dtype=np.int64)[:, -window:].sum(axis=1)
    buffer_details = ", ".join(f"{q}: {agent.memory.q_fraction_onehot(index, len(automaton_states)):.1%}" for index, q in enumerate(automaton_states))
    recent_visits_details = ", ".join(f"{q}: {recent_state_visits[index]}" for index, q in enumerate(automaton_states))
    recent_entries_details = ", ".join(f"{q}: {recent_state_entries[index]}" for index, q in enumerate(automaton_states))
    cumulative_visits_details = ", ".join(f"{q}: {cumulative_counters['state_visits'][q]}" for q in automaton_states)
    cumulative_entries_details = ", ".join(f"{q}: {cumulative_counters['state_entries'][q]}" for q in automaton_states)

    return (
        "\n"
        f"[Episode {episode + 1}/{episodes} | last {window}]\n"
        f"success rate                : {np.mean(histories['successes'][recent_slice]):.1%} (cumulative {np.mean(histories['successes']):.1%})\n"
        f"synthetic task reward       : {np.mean(histories['task_rewards'][recent_slice]):.3f}\n"
        f"shaping reward              : {np.mean(histories['shaping_rewards'][recent_slice]):.3f}\n"
        f"learning reward             : {np.mean(histories['learning_rewards'][recent_slice]):.3f}\n"
        f"episode length              : {np.mean(histories['episode_lengths'][recent_slice]):.1f}\n"
        f"abstract changes / episode  : {np.mean(histories['abstract_changes'][recent_slice]):.1f}\n"
        f"completed cycles / episode   : {np.mean(histories['completed_cycles'][recent_slice]):.2f}\n"
        f"automaton changes / episode  : {np.mean(histories['automaton_transitions'][recent_slice]):.2f}\n"
        f"automaton changes in window  : {_format_counter(recent_transitions)}\n"
        f"epsilon (next episode)       : {histories['epsilons'][-1]:.5f}\n"
        f"replay buffer                : {len(agent.memory)} samples [{buffer_details}]\n"
        f"state visits in window        : {recent_visits_details}\n"
        f"state visits cumulative       : {cumulative_visits_details}\n"
        f"state entries in window       : {recent_entries_details}\n"
        f"state entries cumulative      : {cumulative_entries_details}\n"
        f"transitions cumulative       : {_format_counter(cumulative_counters['transitions'])}\n"
        f"Gym endings cumulative       : terminated={cumulative_counters['env_terminated']}, truncated={cumulative_counters['env_truncated']}\n"
    )


def _save_named_policy(agent, policy_name):
    """Save the current policy using a stable descriptive filename."""
    agent.policy_name = policy_name
    agent._save_policy()


def _monitoring_average(values, episode, log_interval):
    """Return the mean over the active monitoring window."""
    window = min(log_interval, episode + 1)
    return float(np.mean(values[-window:]))


def _validate_training_setup(automaton, state_to_index, episodes, log_interval):
    """Validate automaton consistency and numeric training parameters."""
    if automaton.get_initial_q() not in state_to_index:
        raise ValueError("The initial state is missing from automaton.states")
    if set(state_to_index) != set(automaton.active_states):
        raise ValueError("Network phases must match the stable automaton states")
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")
    if log_interval <= 0:
        raise ValueError("log_interval must be greater than zero")


def _build_training_results(histories, buffer_histories, automaton_states, best_mean_reward, best_policy_episode):
    """Select and name the numeric histories returned by the training loop."""
    return {
        "task_rewards": histories["task_rewards"],
        "learning_rewards": histories["learning_rewards"],
        "shaping_rewards": histories["shaping_rewards"],
        "epsilon_history": histories["epsilons"],
        "buffer_histories": buffer_histories,
        "state_visit_histories": histories["state_visits"],
        "state_entry_histories": histories["state_entries"],
        "successes": histories["successes"],
        "completed_cycles": histories["completed_cycles"],
        "episode_lengths": histories["episode_lengths"],
        "abstract_changes": histories["abstract_changes"],
        "automaton_transitions": histories["automaton_transitions"],
        "automaton_states": automaton_states,
        "best_mean_learning_reward": best_mean_reward,
        "best_policy_episode": best_policy_episode,
    }


# ==============================
# Training loop
# ==============================

def run_sequential_training(env, agent, abstract_mdp, episodes, goal_reward=10000, save_policy=True, use_shaping=True, K=1.0, log_file=None, log_interval=100):
    """
    Train the DDQN agent with the manual automaton and one global epsilon.

    The Gym reward is deliberately discarded. The learning reward is the
    synthetic goal reward plus potential-based shaping. Shaping is evaluated
    only when the complete abstract state (x, y, q) changes.
    """
    # Build a stable mapping between automaton states and network features.
    automaton = abstract_mdp.automaton
    automaton_states = list(automaton.active_states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}
    num_states = len(automaton_states)

    # Fail early if the automaton or training parameters are inconsistent.
    _validate_training_setup(automaton, state_to_index, episodes, log_interval)

    # Store episode-level metrics for plots and post-processing.
    task_reward_history = []
    learning_reward_history = []
    shaping_reward_history = []
    epsilon_history = []
    episode_length_history = []
    success_history = []
    completed_cycle_history = []
    abstract_change_history = []
    automaton_transition_history = []
    transition_counter_history = []
    buffer_histories = [[] for _ in automaton_states]
    state_visit_histories = [[] for _ in automaton_states]
    state_entry_histories = [[] for _ in automaton_states]
    histories = {
        "task_rewards": task_reward_history,
        "learning_rewards": learning_reward_history,
        "shaping_rewards": shaping_reward_history,
        "epsilons": epsilon_history,
        "episode_lengths": episode_length_history,
        "successes": success_history,
        "completed_cycles": completed_cycle_history,
        "abstract_changes": abstract_change_history,
        "automaton_transitions": automaton_transition_history,
        "transition_counters": transition_counter_history,
        "state_visits": state_visit_histories,
        "state_entries": state_entry_histories,
    }

    # Keep cumulative counters for diagnostics shown during training.
    cumulative_state_visits = Counter()
    cumulative_state_entries = Counter()
    cumulative_transitions = Counter()
    cumulative_env_terminated = 0
    cumulative_env_truncated = 0
    best_mean_reward = -np.inf
    best_policy_episode = 0

    # Open one append-only log file for the complete run.
    log_handle = open(log_file, "a", encoding="utf-8") if log_file else None
    _write_run_header(log_handle, episodes, use_shaping, K, goal_reward, abstract_mdp, automaton_states)

    try:
        for episode in range(episodes):
            # Reset the environment and consume s0 before selecting an action.
            raw_state, _ = env.reset()
            q = _evaluate_initial_automaton_state(raw_state, abstract_mdp)
            if q not in state_to_index:
                raise RuntimeError(f"Automaton returned unknown initial state {q!r}")
            augmented_state = _augment_state(raw_state, q, state_to_index)

            # Reset counters local to the current episode.
            succeeded = False
            episode_done = False
            episode_steps = 0
            episode_task_reward = 0.0
            episode_shaping_reward = 0.0
            episode_abstract_changes = 0
            episode_automaton_transitions = 0
            episode_completed_cycles = 0
            episode_state_visits = [0] * num_states
            episode_state_visits[state_to_index[q]] = 1
            # Count s0 as an entry from the virtual pre-episode state.
            episode_state_entries = [0] * num_states
            episode_state_entries[state_to_index[q]] = 1
            episode_transitions = Counter()
            cumulative_state_visits[q] += 1
            cumulative_state_entries[q] += 1

            while not episode_done:
                # Select an action using the single global epsilon.
                agent.eps = epsilon_history[-1] if epsilon_history else agent.eps
                action = agent.select_action(augmented_state)

                # The environment reward is intentionally not part of training.
                next_raw_state, _ignored_env_reward, env_terminated, env_truncated, _ = env.step(action)

                # Map the transition to abstract spatial states.
                x, y = _abstract_position(raw_state, abstract_mdp)
                next_x, next_y = _abstract_position(next_raw_state, abstract_mdp)
                abstract_state = (x, y, q)

                # Advance the automaton using propositions true on arrival.
                truth_assignment = abstract_mdp._get_truth_assignment(next_x, next_y)
                automaton_step = automaton.advance(q, truth_assignment)
                next_q = automaton_step.next_state
                if next_q not in state_to_index:
                    raise RuntimeError(f"Automaton returned unknown state {next_q!r} from state {q!r}")

                # Count every arrival in an automaton state, including self-loops.
                episode_state_visits[state_to_index[next_q]] += 1
                cumulative_state_visits[next_q] += 1

                # Track physical abstraction and automaton changes separately.
                abstract_next_state = (next_x, next_y, next_q)
                abstract_changed = abstract_state != abstract_next_state
                automaton_changed = next_q != q

                if abstract_changed:
                    episode_abstract_changes += 1
                if automaton_changed:
                    transition = (q, next_q)
                    episode_automaton_transitions += 1
                    episode_state_entries[state_to_index[next_q]] += 1
                    episode_transitions[transition] += 1
                    cumulative_state_entries[next_q] += 1
                    cumulative_transitions[transition] += 1

                # Reward every q2 -> q3 transition, once per completed cycle.
                completed_cycle = automaton_step.completed_cycle
                synthetic_goal_reward = (
                    float(goal_reward) if completed_cycle else 0.0
                )
                if completed_cycle:
                    succeeded = True
                    episode_completed_cycles += 1

                # Acceptance does not end an episode. Only Gymnasium can do so.
                # A truncation (for example Gym's time limit) ends data
                # collection, but it is not an MDP terminal state: DDQN must
                # still bootstrap from its final observation.
                episode_done = env_terminated or env_truncated
                bootstrap_terminal = env_terminated
                next_augmented_state = _augment_state(next_raw_state, next_q, state_to_index)

                # Evaluate shaping only when the complete abstract state changes.
                shaping_signal = 0.0
                if use_shaping and abstract_changed:
                    phi_state = abstract_mdp.v_star.get(abstract_state, 0.0)
                    phi_next_state = abstract_mdp.v_star.get(abstract_next_state, 0.0)
                    shaping_signal = K * (abstract_mdp.gamma * phi_next_state - phi_state)

                # Store the transition and perform one DDQN optimization step.
                learning_reward = synthetic_goal_reward + shaping_signal
                agent.memory.push(
                    augmented_state,
                    action,
                    learning_reward,
                    next_augmented_state,
                    bootstrap_terminal,
                )
                agent.optimize_model()

                # Update the episode totals and move to the next state.
                episode_steps += 1
                episode_task_reward += synthetic_goal_reward
                episode_shaping_reward += shaping_signal
                raw_state = next_raw_state
                augmented_state = next_augmented_state
                q = next_q

                # Count Gym endings for diagnostics without using its reward.
                if env_terminated:
                    cumulative_env_terminated += 1
                if env_truncated:
                    cumulative_env_truncated += 1

            # Decay the single epsilon once at the end of the episode.
            next_epsilon = max(agent.eps_min, agent.eps * agent.eps_decay)
            agent.eps = next_epsilon

            # Save the metrics collected for this episode.
            episode_learning_reward = episode_task_reward + episode_shaping_reward
            task_reward_history.append(episode_task_reward)
            shaping_reward_history.append(episode_shaping_reward)
            learning_reward_history.append(episode_learning_reward)
            epsilon_history.append(next_epsilon)
            episode_length_history.append(episode_steps)
            success_history.append(int(succeeded))
            completed_cycle_history.append(episode_completed_cycles)
            abstract_change_history.append(episode_abstract_changes)
            automaton_transition_history.append(episode_automaton_transitions)
            transition_counter_history.append(episode_transitions)

            # Record replay-buffer composition, state visits, and entries from other states.
            for index in range(num_states):
                buffer_histories[index].append(agent.memory.q_fraction_onehot(index, num_states))
                state_visit_histories[index].append(episode_state_visits[index])
                state_entry_histories[index].append(episode_state_entries[index])

            # Print recent and cumulative diagnostics at the requested interval.
            if _should_log(episode, episodes, log_interval):
                cumulative_counters = {"state_visits": cumulative_state_visits, "state_entries": cumulative_state_entries, "transitions": cumulative_transitions, "env_terminated": cumulative_env_terminated, "env_truncated": cumulative_env_truncated}
                _write_log(_build_training_log(episode, episodes, log_interval, automaton_states, agent, histories, cumulative_counters), log_handle)

                # Replace the best policy when the monitored mean reward improves.
                monitored_mean_reward = _monitoring_average(learning_reward_history, episode, log_interval)
                if monitored_mean_reward > best_mean_reward:
                    best_mean_reward = monitored_mean_reward
                    best_policy_episode = episode + 1
                    if save_policy:
                        _save_named_policy(agent, "best_policy.pth")
                        _write_log(f"Best policy updated at episode {best_policy_episode}: mean learning reward={best_mean_reward:.3f}\n", log_handle)

        # Save the final policy independently from its monitored performance.
        if save_policy:
            _save_named_policy(agent, "last_policy.pth")
            _write_log(f"Last policy saved after episode {episodes}. Best policy: episode {best_policy_episode}, mean learning reward={best_mean_reward:.3f}\n", log_handle)
    finally:
        # Always close the log, including when training raises an exception.
        if log_handle:
            log_handle.close()

    # Return named histories to avoid ambiguous tuple positions.
    return _build_training_results(histories, buffer_histories, automaton_states, best_mean_reward, best_policy_episode)


# ==============================
# Experiment setup and outputs
# ==============================

def main(args):
    """Configure the experiment, run or load training, and generate diagnostic plots."""
    # Prepare output directories shared by training and post-processing.
    data_dir = "results"
    image_dir = "img"
    log_dir = "logs"
    for directory in (data_dir, image_dir, log_dir):
        os.makedirs(directory, exist_ok=True)
    plot_dir = data_dir if args.post_process else image_dir

    # Load the manual task and optional training parameters.
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    waypoints = {
        name: tuple(coordinates)
        for name, coordinates in config.get(
            "waypoints_dict", {"g1": [1, 8], "g2": [8, 8]}
        ).items()
    }
    gamma = float(config.get("gamma", 0.99))
    goal_reward = float(config.get("goal_reward", 10000))
    grid_w = int(config.get("grid_w", 12))
    grid_h = int(config.get("grid_h", 12))

    # Build and validate the fixed three-state automaton.
    automaton = AlternatingGoalsAutomaton()
    automaton.validate_waypoints(waypoints, width=grid_w, height=grid_h)
    print(
        "=== MANUAL AUTOMATON TRAINING (single epsilon) ===\n"
        f"Waypoints: {waypoints}\n"
        f"Automaton: states={automaton.states}, stable={automaton.active_states}, "
        f"initial={automaton.initial_state}, "
        f"accepting={sorted(automaton.accepting_states)}\n"
        "Cycle: q1 --g1--> q2 --g2/reward--> q3 --epsilon--> q1\n"
        "Gym reward is ignored; acceptance does not end an episode."
    )

    if not args.post_process:
        # Create the environment and abstract MDP used to compute the potential.
        automaton.render_graph()
        env = gym.make("LunarLander-v3", continuous=False)
        try:
            abstract_mdp = ManualWaypointMDP(
                waypoints_dict=waypoints,
                automaton=automaton,
                width=grid_w,
                height=grid_h,
                gamma=gamma,
                goal_reward=goal_reward,
            )
            abstract_mdp.value_iteration()
            save_sequential_heatmaps(abstract_mdp, filename_prefix="single_epsilon_exp")

            # Initialize one DDQN agent with a single exploration schedule.
            agent = HierarchicalDQNLearner(
                env=env,
                max_episodes=args.episodes,
                eps_decay=args.eps_decay,
                gamma=gamma,
                extra_state_dims=len(automaton.active_states),
                use_polyak=args.polyak,
                tau=args.polyak_tau,
                target_update_freq=args.target_update_freq,
                network_type=args.network_type,
            )

            # Run training and persist all collected metrics.
            metrics = run_sequential_training(env=env, agent=agent, abstract_mdp=abstract_mdp, episodes=args.episodes, goal_reward=goal_reward, use_shaping=not args.no_shaping, K=args.shaping_scale, log_file=f"{log_dir}/single_epsilon_training.log", log_interval=args.log_interval)
            save_training_data(f"{data_dir}/single_epsilon_data.npz", **metrics)
        finally:
            # Release environment resources even if training fails.
            env.close()

    # Load saved metrics and generate the final diagnostic plots.
    data = np.load(f"{data_dir}/single_epsilon_data.npz", allow_pickle=False)
    plot_buffer_fractions(data["buffer_histories"], filename=f"{plot_dir}/buffer_fractions_single_epsilon.png", window_size=args.plot_window, state_labels=data["automaton_states"])
    plot_shaping_reward_breakdown(data["task_rewards"], data["learning_rewards"], data["epsilon_history"], window_size=args.plot_window, filename=f"{plot_dir}/reward_breakdown_single_epsilon.png")
    print("\nFinished.")


# ==============================
# Command-line entry point
# ==============================

if __name__ == "__main__":
    # Expose the main training and post-processing options.
    parser = argparse.ArgumentParser(description="Manual-automaton DDQN training with one global epsilon.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--config", default="trajectory.json")
    parser.add_argument("--eps-decay", type=float, default=0.9996)
    parser.add_argument("--shaping-scale", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--plot-window", type=int, default=500)
    parser.add_argument(
        "--polyak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Polyak target updates (disable with --no-polyak).",
    )
    parser.add_argument("--polyak-tau", type=float, default=0.005)
    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=1000,
        help="Hard target-network update interval used with --no-polyak.",
    )
    parser.add_argument(
        "--network-type",
        choices=["standard", "dueling"],
        default="standard",
        help="Q-network architecture: standard MLP or dueling value/advantage streams.",
    )
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--post-process", action="store_true")
    main(parser.parse_args())
