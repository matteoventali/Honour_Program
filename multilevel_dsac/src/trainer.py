# ==============================
# Standard library imports
# ==============================

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path

# ==============================
# External and project imports
# ==============================

import gymnasium as gym
import numpy as np
import torch

from abstraction import AbstractionConfig
from abstract_mdps import LTLfAutomaton, MultiLevelWaypointMDP
from agent import DiscreteSACAgent
from automaton_validator import validate_automaton
from utils import (
    phi_mapping_sequential,
    plot_buffer_fractions,
    plot_buffer_variance,
    plot_shaping_reward_breakdown,
    plot_training_variance,
    save_multilevel_heatmaps,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR) == "src"
    else SCRIPT_DIR
)


# ==============================
# Data and state helpers
# ==============================

def _positive_int(value):
    """Parse a strictly positive command-line integer."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _optimization_reward(learning_reward, goal_reward, reward_scaling=True):
    """Return the reward stored in replay while preserving raw reporting metrics."""
    if not reward_scaling:
        return float(learning_reward)
    if not np.isfinite(goal_reward) or goal_reward <= 0.0:
        raise ValueError(
            "goal_reward must be finite and greater than zero when reward scaling is enabled"
        )
    return float(learning_reward) / float(goal_reward)


def _experiment_name(value):
    """Validate a safe single-directory experiment name."""
    name = str(value).strip()
    if len(name) > 100 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise argparse.ArgumentTypeError(
            "must start with a letter or digit and contain only letters, digits, '.', '_' or '-'"
        )
    return name


def _resolve_config_path(requested_path, default_filename, experiment_dir, post_process):
    """Resolve a config, preferring the experiment snapshot during post-processing."""
    requested = Path(requested_path).expanduser()
    framework_default = Path(SCRIPT_DIR) / default_filename
    uses_default = (
        str(requested_path) == default_filename
        or requested.resolve() == framework_default.resolve()
    )
    candidates = []
    if post_process and uses_default:
        candidates.extend(
            [
                Path(experiment_dir) / default_filename,
                Path(experiment_dir) / "results" / default_filename,
            ]
        )
    candidates.append(requested)
    if not requested.is_absolute():
        candidates.append(Path(SCRIPT_DIR) / requested)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Configuration file not found. Checked:\n  - {checked}")


def _archive_config(config_path, experiment_dir, filename):
    """Store the exact training configuration beside the experiment outputs."""
    destination = Path(experiment_dir) / filename
    if Path(config_path).resolve() != destination.resolve():
        shutil.copy2(config_path, destination)


def _resolve_metrics_path(experiment_dir, filename):
    """Find metrics in the results subfolder or the legacy experiment root."""
    candidates = [
        Path(experiment_dir) / "results" / filename,
        Path(experiment_dir) / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Training data not found. Checked:\n  - {checked}")


def _organize_policy_files(policy_dir):
    """Move checkpoints from legacy layouts into policy/best and policy/last."""
    policy_root = Path(policy_dir)
    destinations = {
        "best": policy_root / "best",
        "last": policy_root / "last",
    }
    for destination in destinations.values():
        destination.mkdir(parents=True, exist_ok=True)

    for category, destination in destinations.items():
        for source in policy_root.glob(f"{category}_policy*"):
            if source.is_file() and not (destination / source.name).exists():
                shutil.move(str(source), destination / source.name)


def _organize_legacy_seed_plots(image_dir):
    """Move legacy per-seed plots from img/ into img/seed_<seed>/ folders."""
    pattern = re.compile(r"^((?:reward_breakdown|buffer_fractions)_.+)_seed_(-?\d+)\.png$")
    for source in Path(image_dir).glob("*.png"):
        match = pattern.fullmatch(source.name)
        if not match:
            continue
        destination_dir = Path(image_dir) / f"seed_{match.group(2)}"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{match.group(1)}.png"
        if not destination.exists():
            shutil.move(str(source), destination)


def save_training_data(filename, **kwargs):
    """Convert training metrics to arrays and save them in a compressed NPZ file."""
    # Preserve numeric dtypes and rectangular shapes for direct plotting.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np_data = {key: np.asarray(value) for key, value in kwargs.items()}
    if any(array.dtype == object for array in np_data.values()):
        raise ValueError("Training metrics must be rectangular numeric arrays")
    np.savez_compressed(filename, **np_data)
    print(f"\nTraining data saved to: {filename}")


def _set_training_seed(seed, env=None):
    """Seed every random generator used by SAC-Discrete and LunarLander."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.action_space.seed(seed)


def _aggregate_seed_metrics(seed_metrics, seeds):
    """Keep first-run compatibility and add every metric stacked by seed."""
    if not seed_metrics:
        raise ValueError("At least one seed run is required")
    aggregated = dict(seed_metrics[0])
    aggregated["seeds"] = np.asarray(seeds, dtype=np.int64)
    for key in seed_metrics[0]:
        if key == "automaton_states":
            continue
        try:
            aggregated[f"{key}_runs"] = np.stack(
                [np.asarray(metrics[key]) for metrics in seed_metrics]
            )
        except ValueError as error:
            raise ValueError(f"Metric {key!r} has inconsistent shapes across seeds") from error
    for key in ("task_rewards", "learning_rewards", "shaping_rewards"):
        runs = aggregated[f"{key}_runs"]
        aggregated[f"{key}_mean"] = np.mean(runs, axis=0)
        aggregated[f"{key}_variance"] = np.var(runs, axis=0)
    return aggregated


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


def _write_run_header(log_handle, episodes, use_shaping, K, goal_reward, abstract_mdp, automaton_states, training_shaping_gamma, reward_scaling, agent):
    """Write the configuration and DFA metadata at the beginning of a training run."""
    if not log_handle:
        return
    automaton = abstract_mdp.automaton
    shaping_formula = (
        "K*(gamma*Phi(next)-Phi(state))"
        if training_shaping_gamma
        else "K*(Phi(next)-Phi(state))"
    )
    header = (
        "\n=== NEW CLEANRL SAC-DISCRETE RUN ===\n"
        f"episodes={episodes}, shaping={use_shaping}, K={K}, goal_reward={goal_reward}, gamma={abstract_mdp.gamma}\n"
        f"reward_scaling={reward_scaling}, optimization_reward_divisor={goal_reward if reward_scaling else 1.0}\n"
        f"autotune={agent.autotune}, alpha={agent.alpha}, target_entropy={agent.target_entropy}\n"
        f"learning_starts={agent.learning_starts}, batch_size={agent.batch_size}, "
        f"update_frequency={agent.update_frequency}, target_network_frequency={agent.target_network_frequency}, tau={agent.tau}\n"
        f"training_shaping_gamma={training_shaping_gamma}, shaping_formula={shaping_formula}\n"
        f"inter_level_shaping={abstract_mdp.upper_level_mdp is not None}, "
        f"inter_level_K={abstract_mdp.inter_level_shaping_scale}\n"
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
        f"DSAC optimization reward   : {np.mean(histories['optimization_rewards'][recent_slice]):.6f}\n"
        f"episode length              : {np.mean(histories['episode_lengths'][recent_slice]):.1f}\n"
        f"abstract changes / episode  : {np.mean(histories['abstract_changes'][recent_slice]):.1f}\n"
        f"DFA transitions / episode   : {np.mean(histories['dfa_transitions'][recent_slice]):.2f}\n"
        f"DFA transitions in window   : {_format_counter(recent_transitions)}\n"
        f"entropy coefficient alpha    : {histories['alphas'][-1]:.6f}\n"
        f"critic loss (Q1 / Q2)        : {histories['qf1_losses'][-1]:.5g} / {histories['qf2_losses'][-1]:.5g}\n"
        f"actor / alpha loss           : {histories['actor_losses'][-1]:.5g} / {histories['alpha_losses'][-1]:.5g}\n"
        f"environment / update steps   : {agent.environment_steps} / {agent.optimization_steps}\n"
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
    category = "best" if policy_name.startswith("best_policy") else "last"
    os.makedirs(os.path.join(agent.policy_dir, category), exist_ok=True)
    agent.policy_name = os.path.join(category, policy_name)
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


def _build_training_results(histories, initial_acceptance_history, buffer_histories, automaton_states, best_mean_reward, best_policy_episode, reward_scaling, goal_reward):
    """Select and name the numeric histories returned by the training loop."""
    return {
        "task_rewards": histories["task_rewards"],
        "learning_rewards": histories["learning_rewards"],
        "shaping_rewards": histories["shaping_rewards"],
        "optimization_rewards": histories["optimization_rewards"],
        "entropy_coefficient_history": histories["alphas"],
        "qf1_loss_history": histories["qf1_losses"],
        "qf2_loss_history": histories["qf2_losses"],
        "actor_loss_history": histories["actor_losses"],
        "alpha_loss_history": histories["alpha_losses"],
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
        "reward_scaling_enabled": int(reward_scaling),
        "optimization_reward_divisor": float(goal_reward) if reward_scaling else 1.0,
    }


# ==============================
# Training loop
# ==============================

def run_sequential_training(env, agent, abstract_mdp, episodes, goal_reward=10000, save_policy=True, use_shaping=True, K=1.0, log_file=None, log_interval=100, training_shaping_gamma=True, reward_scaling=True, seed=None, policy_suffix=""):
    """
    Train CleanRL-style SAC-Discrete with the LTLf multilevel potential.

    The Gym reward is deliberately discarded. The learning reward is the
    synthetic goal reward plus potential-based shaping. By default the reward
    stored in replay is divided by ``goal_reward``; reported metrics retain the
    original scale. Shaping is evaluated only when the complete abstract state
    changes.
    """
    # Build a stable mapping between DFA states and neural-network features.
    automaton = abstract_mdp.automaton
    automaton_states = list(automaton.states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}
    num_states = len(automaton_states)
    # Validate the divisor before any environment interaction.
    _optimization_reward(0.0, goal_reward, reward_scaling)

    # Fail early if the DFA or training parameters are inconsistent.
    _validate_training_setup(automaton, state_to_index, episodes, log_interval)

    # Store episode-level metrics for plots and post-processing.
    task_reward_history = []
    learning_reward_history = []
    shaping_reward_history = []
    optimization_reward_history = []
    alpha_history = []
    qf1_loss_history = []
    qf2_loss_history = []
    actor_loss_history = []
    alpha_loss_history = []
    episode_length_history = []
    success_history = []
    initial_acceptance_history = []
    abstract_change_history = []
    dfa_transition_history = []
    transition_counter_history = []
    buffer_histories = [[] for _ in automaton_states]
    state_visit_histories = [[] for _ in automaton_states]
    state_entry_histories = [[] for _ in automaton_states]
    histories = {
        "task_rewards": task_reward_history,
        "learning_rewards": learning_reward_history,
        "shaping_rewards": shaping_reward_history,
        "optimization_rewards": optimization_reward_history,
        "alphas": alpha_history,
        "qf1_losses": qf1_loss_history,
        "qf2_losses": qf2_loss_history,
        "actor_losses": actor_loss_history,
        "alpha_losses": alpha_loss_history,
        "episode_lengths": episode_length_history,
        "successes": success_history,
        "abstract_changes": abstract_change_history,
        "dfa_transitions": dfa_transition_history,
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
    cumulative_initial_acceptances = 0
    best_mean_reward = -np.inf
    best_policy_episode = 0

    # Open one append-only log file for the complete run.
    log_handle = open(log_file, "a", encoding="utf-8") if log_file else None
    _write_run_header(log_handle, episodes, use_shaping, K, goal_reward, abstract_mdp, automaton_states, training_shaping_gamma, reward_scaling, agent)

    try:
        for episode in range(episodes):
            # Reset the environment and consume s0 before selecting the first action.
            raw_state, _ = env.reset(seed=seed if episode == 0 else None)
            q = _evaluate_initial_automaton_state(raw_state, abstract_mdp)
            if q not in state_to_index:
                raise RuntimeError(f"DFA returned unknown initial state {q!r} after evaluating s0")
            augmented_state = _augment_state(raw_state, q, state_to_index)

            # Reset counters local to the current episode.
            succeeded = automaton.is_goal_reached(q)
            episode_done = succeeded
            episode_steps = 0
            episode_task_reward = float(goal_reward) if succeeded else 0.0
            episode_shaping_reward = 0.0
            episode_optimization_reward = 0.0
            episode_abstract_changes = 0
            episode_dfa_transitions = 0
            episode_state_visits = [0] * num_states
            episode_state_visits[state_to_index[q]] = 1
            # Count s0 as an entry from the virtual pre-trace state.
            episode_state_entries = [0] * num_states
            episode_state_entries[state_to_index[q]] = 1
            episode_transitions = Counter()
            episode_losses = []
            if succeeded:
                cumulative_initial_acceptances += 1
            cumulative_state_visits[q] += 1
            cumulative_state_entries[q] += 1

            while not episode_done:
                # CleanRL uses random warm-up, then samples its categorical actor.
                action = agent.select_action(augmented_state)

                # The environment reward is intentionally not part of training.
                next_raw_state, _ignored_env_reward, env_terminated, env_truncated, _ = env.step(action)

                # Map the transition to abstract spatial states.
                x, y = _abstract_position(raw_state, abstract_mdp)
                next_x, next_y = _abstract_position(next_raw_state, abstract_mdp)
                abstract_state = (x, y, q)

                # Advance the DFA using propositions true in the arrival state.
                truth_assignment = abstract_mdp._get_truth_assignment(next_x, next_y)
                next_q = automaton.get_next_q(q, truth_assignment)
                if next_q not in state_to_index:
                    raise RuntimeError(f"DFA returned unknown state {next_q!r} from state {q!r}")

                # Count every arrival in a DFA state, including self-transitions.
                episode_state_visits[state_to_index[next_q]] += 1
                cumulative_state_visits[next_q] += 1

                # Track physical abstraction changes separately from DFA changes.
                abstract_next_state = (next_x, next_y, next_q)
                abstract_changed = abstract_state != abstract_next_state
                dfa_changed = next_q != q

                if abstract_changed:
                    episode_abstract_changes += 1
                if dfa_changed:
                    transition = (q, next_q)
                    episode_dfa_transitions += 1
                    episode_state_entries[state_to_index[next_q]] += 1
                    episode_transitions[transition] += 1
                    cumulative_state_entries[next_q] += 1
                    cumulative_transitions[transition] += 1

                # Assign the synthetic task reward only on DFA acceptance.
                synthetic_goal_reward = 0.0
                if automaton.is_goal_reached(next_q):
                    synthetic_goal_reward = float(goal_reward)
                    succeeded = True

                # Stop data collection on any Gym ending or DFA success.
                # A truncation (for example Gym's time limit) ends data
                # collection, but it is not an MDP terminal state: SAC must
                # still bootstrap from its final observation.
                episode_done = env_terminated or env_truncated or succeeded
                bootstrap_terminal = env_terminated or succeeded
                next_augmented_state = _augment_state(next_raw_state, next_q, state_to_index)

                # Evaluate shaping only when the complete abstract state changes.
                shaping_signal = 0.0
                if use_shaping and abstract_changed:
                    phi_state = abstract_mdp.v_star.get(abstract_state, 0.0)
                    phi_next_state = abstract_mdp.v_star.get(abstract_next_state, 0.0)
                    training_discount = abstract_mdp.gamma if training_shaping_gamma else 1.0
                    shaping_signal = K * (training_discount * phi_next_state - phi_state)

                # Store the transition and run an update when CleanRL's schedule is due.
                learning_reward = synthetic_goal_reward + shaping_signal
                optimization_reward = _optimization_reward(
                    learning_reward, goal_reward, reward_scaling
                )
                agent.store_transition(
                    augmented_state,
                    action,
                    optimization_reward,
                    next_augmented_state,
                    bootstrap_terminal,
                )
                update_losses = agent.optimize_model()
                if update_losses is not None:
                    episode_losses.append(update_losses)

                # Update the episode totals and move to the next state.
                episode_steps += 1
                episode_task_reward += synthetic_goal_reward
                episode_shaping_reward += shaping_signal
                episode_optimization_reward += optimization_reward
                raw_state = next_raw_state
                augmented_state = next_augmented_state
                q = next_q

                # Count Gym endings for diagnostics without using its reward.
                if env_terminated:
                    cumulative_env_terminated += 1
                if env_truncated:
                    cumulative_env_truncated += 1

            # Save the metrics collected for this episode.
            episode_learning_reward = episode_task_reward + episode_shaping_reward
            task_reward_history.append(episode_task_reward)
            shaping_reward_history.append(episode_shaping_reward)
            learning_reward_history.append(episode_learning_reward)
            optimization_reward_history.append(episode_optimization_reward)
            alpha_history.append(agent.alpha)
            for key, history in (
                ("qf1_loss", qf1_loss_history),
                ("qf2_loss", qf2_loss_history),
                ("actor_loss", actor_loss_history),
                ("alpha_loss", alpha_loss_history),
            ):
                finite_values = [loss[key] for loss in episode_losses if np.isfinite(loss[key])]
                history.append(float(np.mean(finite_values)) if finite_values else np.nan)
            episode_length_history.append(episode_steps)
            success_history.append(int(succeeded))
            initial_acceptance_history.append(int(episode_steps == 0 and succeeded))
            abstract_change_history.append(episode_abstract_changes)
            dfa_transition_history.append(episode_dfa_transitions)
            transition_counter_history.append(episode_transitions)

            # Record replay-buffer composition, state visits, and entries from other states.
            for index in range(num_states):
                buffer_histories[index].append(agent.memory.q_fraction_onehot(index, num_states))
                state_visit_histories[index].append(episode_state_visits[index])
                state_entry_histories[index].append(episode_state_entries[index])

            # Print recent and cumulative diagnostics at the requested interval.
            if _should_log(episode, episodes, log_interval):
                cumulative_counters = {"state_visits": cumulative_state_visits, "state_entries": cumulative_state_entries, "transitions": cumulative_transitions, "initial_acceptances": cumulative_initial_acceptances, "env_terminated": cumulative_env_terminated, "env_truncated": cumulative_env_truncated}
                _write_log(_build_training_log(episode, episodes, log_interval, automaton_states, agent, histories, cumulative_counters), log_handle)

                # Replace the best policy when the monitored mean reward improves.
                monitored_mean_reward = _monitoring_average(learning_reward_history, episode, log_interval)
                if monitored_mean_reward > best_mean_reward:
                    best_mean_reward = monitored_mean_reward
                    best_policy_episode = episode + 1
                    if save_policy:
                        _save_named_policy(agent, f"best_policy{policy_suffix}.pth")
                        _write_log(f"Best policy updated at episode {best_policy_episode}: mean learning reward={best_mean_reward:.3f}\n", log_handle)

        # Save the final policy independently from its monitored performance.
        if save_policy:
            _save_named_policy(agent, f"last_policy{policy_suffix}.pth")
            _write_log(f"Last policy saved after episode {episodes}. Best policy: episode {best_policy_episode}, mean learning reward={best_mean_reward:.3f}\n", log_handle)
    finally:
        # Always close the log, including when training raises an exception.
        if log_handle:
            log_handle.close()

    # Return named histories to avoid ambiguous tuple positions.
    return _build_training_results(histories, initial_acceptance_history, buffer_histories, automaton_states, best_mean_reward, best_policy_episode, reward_scaling, goal_reward)


# ==============================
# Experiment setup and outputs
# ==============================

def main(args):
    """Configure the experiment, run or load training, and generate diagnostic plots."""
    if args.num_seeds <= 0:
        raise ValueError("num_seeds must be greater than zero")
    # Keep every artifact isolated under results/<experiment-name>/.
    experiment_dir = os.path.join(FRAMEWORK_DIR, "results", args.experiment_name)
    if args.post_process and not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    data_dir = os.path.join(experiment_dir, "results")
    image_dir = os.path.join(experiment_dir, "img")
    log_dir = os.path.join(experiment_dir, "logs")
    policy_dir = os.path.join(experiment_dir, "policy")
    best_policy_dir = os.path.join(policy_dir, "best")
    last_policy_dir = os.path.join(policy_dir, "last")
    for directory in (
        data_dir,
        image_dir,
        log_dir,
        best_policy_dir,
        last_policy_dir,
    ):
        os.makedirs(directory, exist_ok=True)
    _organize_policy_files(policy_dir)
    _organize_legacy_seed_plots(image_dir)
    plot_dir = image_dir
    print(f"Experiment outputs: {experiment_dir}")

    # Load the temporal task and optional training parameters.
    config_path = _resolve_config_path(
        args.config, "trajectory.json", experiment_dir, args.post_process
    )
    abstraction_config_path = _resolve_config_path(
        args.abstraction_config,
        "abstraction.json",
        experiment_dir,
        args.post_process,
    )
    print(f"Task configuration: {config_path}")
    print(f"Abstraction configuration: {abstraction_config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    abstraction_config = AbstractionConfig.load(abstraction_config_path)
    if not args.post_process:
        _archive_config(config_path, experiment_dir, "trajectory.json")
        _archive_config(abstraction_config_path, experiment_dir, "abstraction.json")

    formula = config.get("formula", "F(goal)")
    waypoints = {name: tuple(coordinates) for name, coordinates in config.get("waypoints_dict", {"goal": [5, 0]}).items()}
    gamma = float(config.get("gamma", 0.99))
    goal_reward = float(config.get("goal_reward", 10000))
    primary_level = abstraction_config.primary

    # Build the DFA once for both training and post-processing.
    automaton = LTLfAutomaton(formula)
    validation_report = validate_automaton(
        automaton,
        waypoints,
        width=primary_level.width,
        height=primary_level.height,
    )
    level_summary = ", ".join(
        f"{index}:{level.name}={level.width}x{level.height}"
        for index, level in enumerate(abstraction_config.levels, start=1)
    )
    print(
        "=== MULTILEVEL LTLf TRAINING (CleanRL SAC-Discrete) ===\n"
        f"Formula: {formula}\n"
        f"Waypoints: {waypoints}\n"
        f"Abstractions: {level_summary}\n"
        f"Inter-level shaping scale: "
        f"{abstraction_config.inter_level_shaping_scale}\n"
        "Automaton coordinates and training potential: level1\n"
        "Policy: categorical over LunarLander's four native discrete actions.\n"
        f"DFA: states={automaton.states}, pre-trace={automaton.initial_state}, "
        f"accepting={sorted(automaton.accepting_states)}\n"
        "Gym reward is ignored by design.\n"
        f"{validation_report.format()}"
    )

    if not args.post_process:
        automaton.render_graph(directory=image_dir)

    # Heatmaps depend only on the saved task configuration, not on agent training.
    multilevel_mdp = MultiLevelWaypointMDP(
        waypoints_dict=waypoints,
        ltlf_automaton=automaton,
        abstraction_config=abstraction_config,
        gamma=gamma,
        goal_reward=goal_reward,
    )
    multilevel_mdp.compute_value_functions()
    save_multilevel_heatmaps(
        multilevel_mdp,
        filename_prefix="dsac_exp",
        output_root=os.path.join(image_dir, "heatmaps"),
    )
    abstract_mdp = multilevel_mdp.primary_mdp

    if not args.post_process:
        # Create LunarLander only when agent training is requested.
        seeds = [args.seed + index for index in range(args.num_seeds)]
        seed_metrics = []
        for run_index, run_seed in enumerate(seeds, start=1):
            print(f"\n=== SEED RUN {run_index}/{args.num_seeds}: seed={run_seed} ===")
            _set_training_seed(run_seed)
            env = gym.make("LunarLander-v3", continuous=False)
            try:
                _set_training_seed(run_seed, env)
                agent = DiscreteSACAgent(
                    env=env,
                    gamma=gamma,
                    extra_state_dims=len(automaton.states),
                    hidden_dim=args.hidden_dim,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    learning_starts=args.learning_starts,
                    policy_lr=args.policy_lr,
                    q_lr=args.q_lr,
                    update_frequency=args.update_frequency,
                    target_network_frequency=args.target_network_frequency,
                    tau=args.tau,
                    alpha=args.alpha,
                    autotune=args.autotune,
                    target_entropy_scale=args.target_entropy_scale,
                    policy_dir=policy_dir,
                    device=args.device,
                )
                policy_suffix = "" if args.num_seeds == 1 else f"_seed_{run_seed}"
                metrics = run_sequential_training(env=env, agent=agent, abstract_mdp=abstract_mdp, episodes=args.episodes, goal_reward=goal_reward, use_shaping=not args.no_shaping, K=args.shaping_scale, log_file=f"{log_dir}/dsac_training_seed_{run_seed}.log", log_interval=args.log_interval, training_shaping_gamma=args.training_shaping_gamma, reward_scaling=args.reward_scaling, seed=run_seed, policy_suffix=policy_suffix)
                seed_metrics.append(metrics)
                save_training_data(f"{data_dir}/dsac_data_seed_{run_seed}.npz", **metrics)
            finally:
                env.close()
        save_training_data(f"{data_dir}/dsac_data.npz", **_aggregate_seed_metrics(seed_metrics, seeds))

    # Load saved metrics and generate the final diagnostic plots.
    data_path = (
        _resolve_metrics_path(experiment_dir, "dsac_data.npz")
        if args.post_process
        else Path(data_dir) / "dsac_data.npz"
    )
    print(f"Training data: {data_path}")
    data = np.load(data_path, allow_pickle=False)
    task_reward_runs = data["task_rewards_runs"] if "task_rewards_runs" in data else data["task_rewards"][np.newaxis, :]
    learning_reward_runs = data["learning_rewards_runs"] if "learning_rewards_runs" in data else data["learning_rewards"][np.newaxis, :]
    alpha_runs = data["entropy_coefficient_history_runs"] if "entropy_coefficient_history_runs" in data else data["entropy_coefficient_history"][np.newaxis, ...]
    buffer_runs = data["buffer_histories_runs"] if "buffer_histories_runs" in data else data["buffer_histories"][np.newaxis, ...]
    seed_values = data["seeds"] if "seeds" in data else np.asarray([args.seed])
    for obsolete_name in ("buffer_fractions_dsac.png", "reward_breakdown_dsac.png"):
        (Path(plot_dir) / obsolete_name).unlink(missing_ok=True)
    for run_index, (run_seed, task_rewards, learning_rewards, alpha_history) in enumerate(zip(seed_values, task_reward_runs, learning_reward_runs, alpha_runs)):
        seed_plot_dir = os.path.join(plot_dir, f"seed_{int(run_seed)}")
        os.makedirs(seed_plot_dir, exist_ok=True)
        plot_shaping_reward_breakdown(task_rewards, learning_rewards, alpha_history, window_size=args.plot_window, filename=f"{seed_plot_dir}/reward_breakdown_dsac.png", title=f"DSAC Reward Breakdown — Seed {int(run_seed)}", exploration_label="Entropy coefficient (alpha)")
        if run_index < len(buffer_runs):
            plot_buffer_fractions(buffer_runs[run_index], filename=f"{seed_plot_dir}/buffer_fractions_dsac.png", window_size=args.plot_window, state_labels=data["automaton_states"], title=f"Replay Buffer Composition — Seed {int(run_seed)}")
    plot_training_variance(
        learning_reward_runs,
        window_size=args.plot_window,
        filename=f"{plot_dir}/training_variance_dsac.png",
        epsilon_histories=alpha_runs,
        exploration_label="Entropy coefficient (alpha)",
    )
    plot_buffer_variance(buffer_runs, window_size=args.plot_window, filename=f"{plot_dir}/buffer_variance_dsac.png", state_labels=data["automaton_states"])
    print("\nFinished.")


# ==============================
# Command-line entry point
# ==============================

if __name__ == "__main__":
    # Expose the main training and post-processing options.
    parser = argparse.ArgumentParser(description="Multilevel LTLf training with CleanRL SAC-Discrete.")
    parser.add_argument("--experiment-name", type=_experiment_name, required=True, help="Output directory name under results/.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-seeds", type=_positive_int, default=1, help="Number of training runs with consecutive seeds.")
    parser.add_argument("--seed", type=int, default=42, help="First training seed.")
    parser.add_argument("--config", default="trajectory.json")
    parser.add_argument(
        "--abstraction-config",
        default="abstraction.json",
        help="Ordered grid hierarchy (level1 defines automaton coordinates).",
    )
    parser.add_argument("--shaping-scale", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--plot-window", type=int, default=500)
    parser.add_argument("--hidden-dim", type=_positive_int, default=128)
    parser.add_argument("--buffer-size", type=_positive_int, default=300000)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--learning-starts", type=int, default=20000)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--q-lr", type=float, default=3e-4)
    parser.add_argument("--update-frequency", type=_positive_int, default=4)
    parser.add_argument("--target-network-frequency", type=_positive_int, default=8000)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-entropy-scale", type=float, default=0.89)
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--reward-scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Divide goal and shaping rewards by goal_reward before storing them in replay (default: enabled).",
    )
    parser.add_argument(
        "--training-shaping-gamma",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use gamma*Phi(next)-Phi(state) during training; disable to use Phi(next)-Phi(state).",
    )
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--post-process", action="store_true")
    main(parser.parse_args())
