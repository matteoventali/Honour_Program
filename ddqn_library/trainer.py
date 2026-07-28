# ==============================
# Standard library imports
# ==============================

import argparse
import json
import os
import warnings
from collections import Counter

# Keep Tianshou and transitive dependency diagnostics out of framework output.
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore", module=r"tianshou(\.|$)")

# ==============================
# External and project imports
# ==============================

import gymnasium as gym
import numpy as np
from tianshou.algorithm.algorithm_base import TrainingStats
from tianshou.data import Collector
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams

from abstract_mdps import LTLfAutomaton, LTLfWaypointMDP
from agent import HierarchicalDQNLearner
from automaton_validator import validate_automaton
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
    """Append a one-hot encoding of the current DFA state to an observation."""
    one_hot = np.zeros(len(state_to_index), dtype=np.float32)
    one_hot[state_to_index[q]] = 1.0
    return np.concatenate((observation, one_hot)).astype(np.float32)


def _evaluate_initial_automaton_state(observation, abstract_mdp):
    """Consume the initial observation from the DFA pre-trace state and return the first active state."""
    initial_x, initial_y = _abstract_position(observation, abstract_mdp)
    initial_truth_assignment = abstract_mdp._get_truth_assignment(initial_x, initial_y)
    pre_trace_q = abstract_mdp.automaton.get_initial_q()
    return abstract_mdp.automaton.get_next_q(pre_trace_q, initial_truth_assignment)


def _format_counter(counter):
    """Convert a DFA transition counter into a compact human-readable string."""
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
    """Write the configuration and DFA metadata at the beginning of a training run."""
    if not log_handle:
        return
    automaton = abstract_mdp.automaton
    header = (
        "\n=== NEW RUN ===\n"
        f"episodes={episodes}, shaping={use_shaping}, K={K}, goal_reward={goal_reward}, gamma={abstract_mdp.gamma}\n"
        f"formula={automaton.formula_str}\n"
        f"waypoints={abstract_mdp.waypoints_dict}\n"
        f"dfa_states={automaton_states}, pre_trace={automaton.get_initial_q()}, accepting={sorted(automaton.accepting_states)}\n"
    )
    log_handle.write(header)
    log_handle.flush()


def _should_log(episode, episodes, log_interval):
    """Return whether the current episode requires a periodic training report."""
    return episode == 0 or episode + 1 == episodes or (episode + 1) % log_interval == 0


def _build_training_log(episode, episodes, log_interval, automaton_states, agent, histories, cumulative_counters):
    """Build a report containing recent metrics and cumulative DFA counters."""
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
        f"DFA transitions / episode   : {np.mean(histories['dfa_transitions'][recent_slice]):.2f}\n"
        f"DFA transitions in window   : {_format_counter(recent_transitions)}\n"
        f"epsilon (next episode)       : {histories['epsilons'][-1]:.5f}\n"
        f"replay buffer                : {len(agent.memory)} samples [{buffer_details}]\n"
        f"DFA state visits in window   : {recent_visits_details}\n"
        f"DFA state visits cumulative  : {cumulative_visits_details}\n"
        f"DFA state entries in window  : {recent_entries_details}\n"
        f"DFA state entries cumulative : {cumulative_entries_details}\n"
        f"transitions cumulative       : {_format_counter(cumulative_counters['transitions'])}\n"
        f"accepted directly from s0    : {cumulative_counters['initial_acceptances']}\n"
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
    """Validate DFA consistency and the numeric parameters required by training."""
    if automaton.get_initial_q() not in state_to_index:
        raise ValueError("The DFA initial state is missing from automaton.states")
    if not automaton.accepting_states.issubset(state_to_index):
        raise ValueError("At least one accepting DFA state is missing from automaton.states")
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero")
    if log_interval <= 0:
        raise ValueError("log_interval must be greater than zero")


def _build_training_results(histories, initial_acceptance_history, buffer_histories, automaton_states, best_mean_reward, best_policy_episode):
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
        "initial_acceptances": initial_acceptance_history,
        "episode_lengths": histories["episode_lengths"],
        "abstract_changes": histories["abstract_changes"],
        "dfa_transitions": histories["dfa_transitions"],
        "automaton_states": automaton_states,
        "best_mean_learning_reward": best_mean_reward,
        "best_policy_episode": best_policy_episode,
    }


# ==============================
# Training loop
# ==============================

class LTLfTrainingWrapper(gym.Wrapper):
    """Expose the original augmented state, reward, and termination logic to Tianshou."""

    def __init__(self, env, abstract_mdp, goal_reward, use_shaping, shaping_scale):
        super().__init__(env)
        self.abstract_mdp = abstract_mdp
        self.automaton = abstract_mdp.automaton
        self.automaton_states = list(self.automaton.states)
        self.state_to_index = {q: index for index, q in enumerate(self.automaton_states)}
        self.goal_reward = float(goal_reward)
        self.use_shaping = use_shaping
        self.shaping_scale = shaping_scale
        state_dim = env.observation_space.shape[0] + len(self.automaton_states)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (state_dim,), dtype=np.float32)
        self.completed_summaries = []
        self.cumulative_state_visits = Counter()
        self.cumulative_state_entries = Counter()
        self.cumulative_transitions = Counter()
        self.cumulative_env_terminated = 0
        self.cumulative_env_truncated = 0
        self.cumulative_initial_acceptances = 0

    def reset(self, **kwargs):
        self.raw_state, info = self.env.reset(**kwargs)
        self.q = _evaluate_initial_automaton_state(self.raw_state, self.abstract_mdp)
        if self.automaton.is_goal_reached(self.q):
            raise RuntimeError("Tianshou cannot collect the original zero-step accepting episode")
        count = len(self.automaton_states)
        self.steps = 0
        self.task_reward = 0.0
        self.shaping_reward = 0.0
        self.abstract_changes = 0
        self.dfa_transitions = 0
        self.state_visits = [0] * count
        self.state_visits[self.state_to_index[self.q]] = 1
        self.state_entries = [0] * count
        self.state_entries[self.state_to_index[self.q]] = 1
        self.transitions = Counter()
        # Defer cumulative accounting until the first action. Tianshou resets a
        # finished environment before the episode hook runs; deferral prevents
        # the next episode's initial state from leaking into the previous log.
        self.initial_state_counted = False
        return _augment_state(self.raw_state, self.q, self.state_to_index), info

    def step(self, action):
        if not self.initial_state_counted:
            self.cumulative_state_visits[self.q] += 1
            self.cumulative_state_entries[self.q] += 1
            self.initial_state_counted = True
        next_raw, _ignored_reward, env_terminated, env_truncated, info = self.env.step(action)
        x, y = _abstract_position(self.raw_state, self.abstract_mdp)
        next_x, next_y = _abstract_position(next_raw, self.abstract_mdp)
        abstract_state = (x, y, self.q)
        truth = self.abstract_mdp._get_truth_assignment(next_x, next_y)
        next_q = self.automaton.get_next_q(self.q, truth)
        self.state_visits[self.state_to_index[next_q]] += 1
        self.cumulative_state_visits[next_q] += 1
        abstract_next_state = (next_x, next_y, next_q)
        abstract_changed = abstract_state != abstract_next_state
        dfa_changed = next_q != self.q
        if abstract_changed:
            self.abstract_changes += 1
        if dfa_changed:
            transition = (self.q, next_q)
            self.dfa_transitions += 1
            self.state_entries[self.state_to_index[next_q]] += 1
            self.transitions[transition] += 1
            self.cumulative_state_entries[next_q] += 1
            self.cumulative_transitions[transition] += 1
        synthetic_reward = self.goal_reward if self.automaton.is_goal_reached(next_q) else 0.0
        succeeded = synthetic_reward != 0.0
        shaping_signal = 0.0
        if self.use_shaping and abstract_changed:
            phi_state = self.abstract_mdp.v_star.get(abstract_state, 0.0)
            phi_next = self.abstract_mdp.v_star.get(abstract_next_state, 0.0)
            shaping_signal = self.shaping_scale * (self.abstract_mdp.gamma * phi_next - phi_state)
        learning_reward = synthetic_reward + shaping_signal
        self.steps += 1
        self.task_reward += synthetic_reward
        self.shaping_reward += shaping_signal
        self.raw_state, self.q = next_raw, next_q
        if env_terminated:
            self.cumulative_env_terminated += 1
        if env_truncated:
            self.cumulative_env_truncated += 1
        terminated = bool(env_terminated or succeeded)
        truncated = bool(env_truncated)
        if terminated or truncated:
            self.completed_summaries.append({
                "task_reward": self.task_reward,
                "shaping_reward": self.shaping_reward,
                "learning_reward": self.task_reward + self.shaping_reward,
                "steps": self.steps,
                "success": int(succeeded),
                "initial_acceptance": 0,
                "abstract_changes": self.abstract_changes,
                "dfa_transitions": self.dfa_transitions,
                "transitions": self.transitions.copy(),
                "state_visits": self.state_visits.copy(),
                "state_entries": self.state_entries.copy(),
            })
        augmented = _augment_state(next_raw, next_q, self.state_to_index)
        return augmented, learning_reward, terminated, truncated, info


class EpisodeLimitedOffPolicyTrainer(OffPolicyTrainer):
    """Use Tianshou's online trainer while stopping at the requested episode count."""

    def __init__(self, algorithm, params, wrapped_env, episodes, episode_done_callback):
        super().__init__(algorithm, params)
        self.wrapped_env = wrapped_env
        self.episodes = episodes
        self.episode_done_callback = episode_done_callback
        self.processed_episodes = 0

    def _update_step(self, collect_stats):
        if len(self.params.training_collector.buffer) < self.params.batch_size:
            return TrainingStats()
        return super()._update_step(collect_stats)

    def _training_step(self):
        result = super()._training_step()
        while self.processed_episodes < len(self.wrapped_env.completed_summaries):
            self.episode_done_callback()
            self.processed_episodes += 1
        if len(self.wrapped_env.completed_summaries) >= self.episodes:
            result._is_training_done = True
        return result


def run_sequential_training(env, agent, abstract_mdp, episodes, goal_reward=10000, save_policy=True, use_shaping=True, K=1.0, log_file=None, log_interval=100):
    """Train through Tianshou while preserving the original experiment semantics."""
    automaton_states = list(abstract_mdp.automaton.states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}
    _validate_training_setup(abstract_mdp.automaton, state_to_index, episodes, log_interval)
    histories = {
        "task_rewards": [], "learning_rewards": [], "shaping_rewards": [],
        "epsilons": [], "episode_lengths": [], "successes": [],
        "abstract_changes": [], "dfa_transitions": [], "transition_counters": [],
        "state_visits": [[] for _ in automaton_states],
        "state_entries": [[] for _ in automaton_states],
    }
    initial_acceptance_history = []
    buffer_histories = [[] for _ in automaton_states]
    best_mean_reward = -np.inf
    best_policy_episode = 0
    wrapped_env = LTLfTrainingWrapper(env, abstract_mdp, goal_reward, use_shaping, K)
    log_handle = open(log_file, "a", encoding="utf-8") if log_file else None
    _write_run_header(log_handle, episodes, use_shaping, K, goal_reward, abstract_mdp, automaton_states)

    def process_completed_episode():
        nonlocal best_mean_reward, best_policy_episode
        summary = wrapped_env.completed_summaries[-1]
        histories["task_rewards"].append(summary["task_reward"])
        histories["shaping_rewards"].append(summary["shaping_reward"])
        histories["learning_rewards"].append(summary["learning_reward"])
        histories["episode_lengths"].append(summary["steps"])
        histories["successes"].append(summary["success"])
        histories["abstract_changes"].append(summary["abstract_changes"])
        histories["dfa_transitions"].append(summary["dfa_transitions"])
        histories["transition_counters"].append(summary["transitions"])
        initial_acceptance_history.append(summary["initial_acceptance"])
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        agent.algorithm.policy.set_eps_training(agent.eps)
        histories["epsilons"].append(agent.eps)
        for index in range(len(automaton_states)):
            buffer_histories[index].append(agent.memory.q_fraction_onehot(index, len(automaton_states)))
            histories["state_visits"][index].append(summary["state_visits"][index])
            histories["state_entries"][index].append(summary["state_entries"][index])
        episode = len(histories["successes"]) - 1
        if _should_log(episode, episodes, log_interval):
            cumulative = {
                "state_visits": wrapped_env.cumulative_state_visits,
                "state_entries": wrapped_env.cumulative_state_entries,
                "transitions": wrapped_env.cumulative_transitions,
                "initial_acceptances": wrapped_env.cumulative_initial_acceptances,
                "env_terminated": wrapped_env.cumulative_env_terminated,
                "env_truncated": wrapped_env.cumulative_env_truncated,
            }
            _write_log(_build_training_log(episode, episodes, log_interval, automaton_states, agent, histories, cumulative), log_handle)
            monitored = _monitoring_average(histories["learning_rewards"], episode, log_interval)
            if monitored > best_mean_reward:
                best_mean_reward, best_policy_episode = monitored, episode + 1
                if save_policy:
                    _save_named_policy(agent, "best_policy.pth")
                    _write_log(f"Best policy updated at episode {best_policy_episode}: mean learning reward={best_mean_reward:.3f}\n", log_handle)
        return None

    collector = Collector(
        agent.algorithm,
        wrapped_env,
        buffer=agent.memory,
        exploration_noise=True,
    )
    params = OffPolicyTrainerParams(
        max_epochs=1,
        epoch_num_steps=episodes * 1001,
        training_collector=collector,
        collection_step_num_env_steps=1,
        collection_step_num_episodes=None,
        batch_size=agent.batch_size,
        update_step_num_gradient_steps_per_sample=1.0,
        logger=None,
        verbose=False,
        show_progress=False,
        test_in_training=False,
    )
    try:
        EpisodeLimitedOffPolicyTrainer(
            agent.algorithm,
            params,
            wrapped_env,
            episodes,
            process_completed_episode,
        ).run()
        if save_policy:
            _save_named_policy(agent, "last_policy.pth")
            _write_log(f"Last policy saved after episode {episodes}. Best policy: episode {best_policy_episode}, mean learning reward={best_mean_reward:.3f}\n", log_handle)
    finally:
        if log_handle:
            log_handle.close()
    return _build_training_results(histories, initial_acceptance_history, buffer_histories, automaton_states, best_mean_reward, best_policy_episode)


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

    # Load the temporal task and optional training parameters.
    with open(args.config, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    formula = config.get("formula", "F(goal)")
    waypoints = {name: tuple(coordinates) for name, coordinates in config.get("waypoints_dict", {"goal": [5, 0]}).items()}
    gamma = float(config.get("gamma", 0.99))
    goal_reward = float(config.get("goal_reward", 10000))

    # Build the DFA once for both training and post-processing.
    automaton = LTLfAutomaton(formula)
    validation_report = validate_automaton(automaton, waypoints, width=int(config.get("grid_w", 12)), height=int(config.get("grid_h", 12)))
    print(
        "=== LTLf TRAINING (single epsilon) ===\n"
        f"Formula: {formula}\n"
        f"Waypoints: {waypoints}\n"
        f"DFA: states={automaton.states}, pre-trace={automaton.initial_state}, "
        f"accepting={sorted(automaton.accepting_states)}\n"
        "Gym reward is ignored by design.\n"
        f"{validation_report.format()}"
    )

    if not args.post_process:
        # Create the environment and abstract MDP used to compute the potential.
        automaton.render_graph()
        env = gym.make("LunarLander-v3", continuous=False)
        try:
            abstract_mdp = LTLfWaypointMDP(waypoints_dict=waypoints, ltlf_automaton=automaton, width=int(config.get("grid_w", 12)), height=int(config.get("grid_h", 12)), gamma=gamma, goal_reward=goal_reward)
            abstract_mdp.value_iteration()
            save_sequential_heatmaps(abstract_mdp, filename_prefix="single_epsilon_exp")

            # Initialize Tianshou's Double DQN with the original network shape.
            agent = HierarchicalDQNLearner(
                env=env,
                max_episodes=args.episodes,
                eps_decay=args.eps_decay,
                extra_state_dims=len(automaton.states),
                target_update_freq=args.target_update_freq,
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
    parser = argparse.ArgumentParser(description="LTLf training with Tianshou Double DQN.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--config", default="trajectory.json")
    parser.add_argument("--eps-decay", type=float, default=0.9996)
    parser.add_argument("--target-update-freq", type=int, default=100)
    parser.add_argument("--shaping-scale", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--plot-window", type=int, default=500)
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--post-process", action="store_true")
    main(parser.parse_args())
