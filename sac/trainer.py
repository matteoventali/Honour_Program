"""Train Stable-Baselines3 SAC on the LTLf LunarLander task."""

# ==============================
# Standard library imports
# ==============================

import argparse
import json
import os
from collections import Counter
from pathlib import Path

# ==============================
# External and project imports
# ==============================

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from abstract_mdps import LTLfAutomaton, LTLfWaypointMDP
from agent import DiscreteToContinuousActionWrapper, LTLfTaskWrapper
from automaton_validator import validate_automaton
from utils import (
    plot_buffer_fractions,
    plot_shaping_reward_breakdown,
    plot_training_variance,
    save_sequential_heatmaps,
)


SCRIPT_DIR = Path(__file__).resolve().parent


# ==============================
# Data helpers
# ==============================

def _positive_int(value):
    """Parse a strictly positive command-line integer."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _parse_entropy_coefficient(value):
    """Accept SB3's automatic modes or a positive fixed coefficient."""
    text = str(value).strip()
    if text == "auto" or text.startswith("auto_"):
        if text != "auto":
            try:
                initial_value = float(text.removeprefix("auto_"))
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    "must be 'auto', 'auto_<initial_value>' or a positive number"
                ) from error
            if initial_value <= 0:
                raise argparse.ArgumentTypeError(
                    "the automatic entropy coefficient must start above zero"
                )
        return text

    try:
        coefficient = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be 'auto', 'auto_<initial_value>' or a positive number"
        ) from error
    if coefficient <= 0:
        raise argparse.ArgumentTypeError(
            "the entropy coefficient must be greater than zero"
        )
    return coefficient

def save_training_data(filename, **kwargs):
    """Convert training metrics to arrays and save them in a compressed NPZ."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    np_data = {key: np.asarray(value) for key, value in kwargs.items()}
    if any(array.dtype == object for array in np_data.values()):
        raise ValueError("Training metrics must be rectangular numeric arrays")
    np.savez_compressed(filename, **np_data)
    print(f"\nTraining data saved to: {filename}")


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


def _format_counter(counter):
    """Convert a DFA transition counter into a compact readable string."""
    if not counter:
        return "none"
    return ", ".join(
        f"{source}->{destination}: {count}"
        for (source, destination), count in sorted(counter.items())
    )


def _replay_buffer_fractions(model, num_states):
    """Return the fraction of stored observations belonging to each DFA state."""
    replay_buffer = model.replay_buffer
    fractions = np.zeros(num_states, dtype=np.float64)
    if replay_buffer is None or replay_buffer.size() == 0:
        return fractions

    # SB3 stores observations as [buffer index, environment index, feature].
    valid_size = replay_buffer.buffer_size if replay_buffer.full else replay_buffer.pos
    observations = replay_buffer.observations[:valid_size]
    observations = observations.reshape(-1, observations.shape[-1])
    phase_features = observations[:, -num_states:]
    phase_indices = np.argmax(phase_features, axis=1)
    counts = np.bincount(phase_indices, minlength=num_states)
    return counts.astype(np.float64) / len(phase_indices)


def _entropy_coefficient(model):
    """Read SAC's current entropy coefficient for diagnostics."""
    if model.log_ent_coef is not None:
        return float(model.log_ent_coef.detach().exp().cpu().item())
    return float(model.ent_coef)


def _build_training_results(task_wrapper, callback):
    """Collect numeric episode histories under stable descriptive names."""
    metrics = task_wrapper.episode_metrics
    return {
        "task_rewards": [episode["task_reward"] for episode in metrics],
        "learning_rewards": [episode["learning_reward"] for episode in metrics],
        "shaping_rewards": [episode["shaping_reward"] for episode in metrics],
        "entropy_coefficient_history": callback.entropy_history,
        "buffer_histories": np.asarray(callback.buffer_histories).T,
        "state_visit_histories": np.asarray(
            [episode["state_visits"] for episode in metrics]
        ).T,
        "state_entry_histories": np.asarray(
            [episode["state_entries"] for episode in metrics]
        ).T,
        "successes": [int(episode["success"]) for episode in metrics],
        "initial_acceptances": [
            int(episode["initial_acceptance"]) for episode in metrics
        ],
        "episode_lengths": [episode["episode_length"] for episode in metrics],
        "abstract_changes": [episode["abstract_changes"] for episode in metrics],
        "dfa_transitions": [episode["dfa_transitions"] for episode in metrics],
        "automaton_states": task_wrapper.automaton_states,
        "best_mean_learning_reward": callback.best_mean_reward,
        "best_policy_episode": callback.best_policy_episode,
    }


# ==============================
# SAC callback and training
# ==============================

class SACTrainingCallback(BaseCallback):
    """Collect episode metrics, print diagnostics and stop at an exact count."""

    def __init__(self, task_wrapper, episodes, log_interval, policy_dir, save_policy=True, log_file=None):
        super().__init__(verbose=0)
        if episodes <= 0:
            raise ValueError("episodes must be greater than zero")
        if log_interval <= 0:
            raise ValueError("log_interval must be greater than zero")

        self.task_wrapper = task_wrapper
        self.episodes = int(episodes)
        self.log_interval = int(log_interval)
        self.policy_dir = Path(policy_dir)
        self.save_policy = bool(save_policy)
        self.log_file = Path(log_file) if log_file else None
        self.processed_episodes = 0
        self.buffer_histories = []
        self.entropy_history = []
        self.best_mean_reward = -np.inf
        self.best_policy_episode = 0
        self.cumulative_state_visits = Counter()
        self.cumulative_state_entries = Counter()
        self.cumulative_transitions = Counter()
        self.cumulative_initial_acceptances = 0
        self.cumulative_env_terminated = 0
        self.cumulative_env_truncated = 0
        self._log_handle = None

    def _write(self, message):
        """Print a message and append it to the run log when configured."""
        print(message)
        if self._log_handle:
            self._log_handle.write(message)
            self._log_handle.flush()

    def _on_training_start(self):
        """Open the log and record the immutable experiment configuration."""
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_file.open("a", encoding="utf-8")

        abstract_mdp = self.task_wrapper.abstract_mdp
        automaton = abstract_mdp.automaton
        self._write(
            "\n=== NEW SAC RUN ===\n"
            f"episodes={self.episodes}, shaping={self.task_wrapper.use_shaping}, "
            f"K={self.task_wrapper.shaping_scale}, "
            f"goal_reward={self.task_wrapper.goal_reward}, gamma={abstract_mdp.gamma}\n"
            f"formula={automaton.formula_str}\n"
            f"waypoints={abstract_mdp.waypoints_dict}\n"
            f"dfa_states={self.task_wrapper.automaton_states}, "
            f"pre_trace={automaton.get_initial_q()}, "
            f"accepting={sorted(automaton.accepting_states)}\n"
        )

    def _update_cumulative_counters(self, episode):
        """Merge one completed episode into cumulative diagnostic counters."""
        for index, q in enumerate(self.task_wrapper.automaton_states):
            self.cumulative_state_visits[q] += episode["state_visits"][index]
            self.cumulative_state_entries[q] += episode["state_entries"][index]
        self.cumulative_transitions.update(episode["transition_counter"])
        self.cumulative_initial_acceptances += int(episode["initial_acceptance"])
        self.cumulative_env_terminated += int(episode["env_terminated"])
        self.cumulative_env_truncated += int(episode["env_truncated"])

    def _should_log(self):
        """Return whether the latest episode closes a monitoring window."""
        return (
            self.processed_episodes == 1
            or self.processed_episodes == self.episodes
            or self.processed_episodes % self.log_interval == 0
        )

    def _training_log(self):
        """Build a report from the latest monitoring window."""
        metrics = self.task_wrapper.episode_metrics[: self.processed_episodes]
        window = min(self.log_interval, self.processed_episodes)
        recent = metrics[-window:]
        recent_transitions = Counter()
        for episode in recent:
            recent_transitions.update(episode["transition_counter"])

        states = self.task_wrapper.automaton_states
        recent_visits = np.asarray(
            [episode["state_visits"] for episode in recent], dtype=np.int64
        ).sum(axis=0)
        recent_entries = np.asarray(
            [episode["state_entries"] for episode in recent], dtype=np.int64
        ).sum(axis=0)
        latest_fractions = self.buffer_histories[-1]

        def average(key):
            return float(np.mean([episode[key] for episode in recent]))

        return (
            "\n"
            f"[Episode {self.processed_episodes}/{self.episodes} | last {window}]\n"
            f"success rate                : {average('success'):.1%} "
            f"(cumulative {np.mean([e['success'] for e in metrics]):.1%})\n"
            f"synthetic task reward       : {average('task_reward'):.3f}\n"
            f"shaping reward              : {average('shaping_reward'):.3f}\n"
            f"learning reward             : {average('learning_reward'):.3f}\n"
            f"episode length              : {average('episode_length'):.1f}\n"
            f"abstract changes / episode  : {average('abstract_changes'):.1f}\n"
            f"DFA transitions / episode   : {average('dfa_transitions'):.2f}\n"
            f"DFA transitions in window   : {_format_counter(recent_transitions)}\n"
            f"entropy coefficient          : {self.entropy_history[-1]:.6f}\n"
            f"replay buffer                : {self.model.replay_buffer.size()} samples "
            f"[{', '.join(f'{q}: {latest_fractions[i]:.1%}' for i, q in enumerate(states))}]\n"
            f"DFA state visits in window   : "
            f"{', '.join(f'{q}: {recent_visits[i]}' for i, q in enumerate(states))}\n"
            f"DFA state visits cumulative  : "
            f"{', '.join(f'{q}: {self.cumulative_state_visits[q]}' for q in states)}\n"
            f"DFA state entries in window  : "
            f"{', '.join(f'{q}: {recent_entries[i]}' for i, q in enumerate(states))}\n"
            f"DFA state entries cumulative : "
            f"{', '.join(f'{q}: {self.cumulative_state_entries[q]}' for q in states)}\n"
            f"transitions cumulative       : {_format_counter(self.cumulative_transitions)}\n"
            f"accepted directly from s0    : {self.cumulative_initial_acceptances}\n"
            f"Gym endings cumulative       : terminated={self.cumulative_env_terminated}, "
            f"truncated={self.cumulative_env_truncated}\n"
        )

    def _save_best_policy(self):
        """Replace the best checkpoint when monitored learning reward improves."""
        metrics = self.task_wrapper.episode_metrics[: self.processed_episodes]
        window = min(self.log_interval, self.processed_episodes)
        monitored_mean = float(
            np.mean([episode["learning_reward"] for episode in metrics[-window:]])
        )
        if monitored_mean <= self.best_mean_reward:
            return

        self.best_mean_reward = monitored_mean
        self.best_policy_episode = self.processed_episodes
        if self.save_policy:
            self.policy_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.policy_dir / "best_policy")
        self._write(
            f"Best policy updated at episode {self.best_policy_episode}: "
            f"mean learning reward={self.best_mean_reward:.3f}\n"
        )

    def _process_completed_episode(self, episode):
        """Record replay, entropy and task metrics for one completed episode."""
        self.processed_episodes += 1
        self._update_cumulative_counters(episode)
        self.buffer_histories.append(
            _replay_buffer_fractions(self.model, len(self.task_wrapper.automaton_states))
        )
        self.entropy_history.append(_entropy_coefficient(self.model))
        if self._should_log():
            self._write(self._training_log())
            self._save_best_policy()

    def _on_step(self):
        """Process newly completed episodes and stop at the requested total."""
        while self.processed_episodes < len(self.task_wrapper.episode_metrics):
            episode = self.task_wrapper.episode_metrics[self.processed_episodes]
            self._process_completed_episode(episode)
        return self.processed_episodes < self.episodes

    def _on_training_end(self):
        """Close the append-only log even when training stops via the callback."""
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None


def run_sequential_training(env, task_wrapper, abstract_mdp, episodes, policy_dir, log_file, log_interval=100, learning_rate=3e-4, buffer_size=300000, learning_starts=100, batch_size=256, tau=0.005, ent_coef="auto", seed=None, device="auto", save_policy=True):
    """Construct and train SB3 SAC; all optimization logic lives here."""
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=abstract_mdp.gamma,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef=ent_coef,
        policy_kwargs={"net_arch": [128, 128]},
        seed=seed,
        device=device,
        verbose=0,
    )
    print(f"Using device: {model.device}")

    callback = SACTrainingCallback(
        task_wrapper=task_wrapper,
        episodes=episodes,
        log_interval=log_interval,
        policy_dir=policy_dir,
        save_policy=save_policy,
        log_file=log_file,
    )

    max_episode_steps = getattr(env.spec, "max_episode_steps", None) or 1000
    model.learn(
        total_timesteps=int(episodes) * int(max_episode_steps),
        callback=callback,
        log_interval=log_interval,
    )

    if callback.processed_episodes != episodes:
        raise RuntimeError(
            f"Training stopped after {callback.processed_episodes} episodes; "
            f"expected {episodes}"
        )

    if save_policy:
        policy_dir = Path(policy_dir)
        policy_dir.mkdir(parents=True, exist_ok=True)
        model.save(policy_dir / "last_policy")
        print(
            f"Last policy saved after episode {episodes}. Best policy: episode "
            f"{callback.best_policy_episode}, mean learning reward="
            f"{callback.best_mean_reward:.3f}"
        )

    return _build_training_results(task_wrapper, callback)


# ==============================
# Experiment setup and outputs
# ==============================

def main(args):
    """Configure SAC training or regenerate plots from saved numeric metrics."""
    if args.num_seeds <= 0:
        raise ValueError("num_seeds must be greater than zero")
    data_dir = SCRIPT_DIR / "results"
    image_dir = SCRIPT_DIR / "img"
    log_dir = SCRIPT_DIR / "logs"
    policy_dir = SCRIPT_DIR / "policy"
    for directory in (data_dir, image_dir, log_dir, policy_dir):
        directory.mkdir(parents=True, exist_ok=True)
    plot_dir = data_dir if args.post_process else image_dir

    with Path(args.config).expanduser().open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    formula = config.get("formula", "F(goal)")
    waypoints = {
        name: tuple(coordinates)
        for name, coordinates in config.get(
            "waypoints_dict", {"goal": [5, 0]}
        ).items()
    }
    grid_w = int(config.get("grid_w", 12))
    grid_h = int(config.get("grid_h", 12))
    gamma = float(config.get("gamma", 0.99))
    goal_reward = float(config.get("goal_reward", 10000))

    automaton = LTLfAutomaton(formula)
    validation_report = validate_automaton(
        automaton, waypoints, width=grid_w, height=grid_h
    )
    print(
        "=== LTLf TRAINING (SB3 SAC) ===\n"
        f"Formula: {formula}\n"
        f"Waypoints: {waypoints}\n"
        f"DFA: states={automaton.states}, pre-trace={automaton.initial_state}, "
        f"accepting={sorted(automaton.accepting_states)}\n"
        "SAC action: Box(-1, 1, (1,)) -> floor-scaled LunarLander action.\n"
        "Gym reward is ignored by design.\n"
        f"{validation_report.format()}"
    )

    data_path = data_dir / "sac_data.npz"
    if not args.post_process:
        automaton.render_graph()
        abstract_mdp = LTLfWaypointMDP(
            waypoints_dict=waypoints,
            ltlf_automaton=automaton,
            width=grid_w,
            height=grid_h,
            gamma=gamma,
            goal_reward=goal_reward,
        )
        abstract_mdp.value_iteration()

        previous_directory = Path.cwd()
        try:
            # Existing plotting utilities use relative output paths.
            os.chdir(SCRIPT_DIR)
            save_sequential_heatmaps(
                abstract_mdp, filename_prefix="sac_experiment"
            )
        finally:
            os.chdir(previous_directory)

        seeds = [args.seed + index for index in range(args.num_seeds)]
        seed_metrics = []
        for run_index, run_seed in enumerate(seeds, start=1):
            print(f"\n=== SEED RUN {run_index}/{args.num_seeds}: seed={run_seed} ===")
            base_env = gym.make("LunarLander-v3", continuous=False)
            continuous_env = DiscreteToContinuousActionWrapper(base_env)
            task_env = LTLfTaskWrapper(
                continuous_env,
                abstract_mdp,
                use_shaping=not args.no_shaping,
                shaping_scale=args.shaping_scale,
                goal_reward=goal_reward,
            )
            run_policy_dir = policy_dir if args.num_seeds == 1 else policy_dir / f"seed_{run_seed}"
            try:
                metrics = run_sequential_training(
                    env=task_env,
                    task_wrapper=task_env,
                    abstract_mdp=abstract_mdp,
                    episodes=args.episodes,
                    policy_dir=run_policy_dir,
                    log_file=log_dir / f"sac_training_seed_{run_seed}.log",
                    log_interval=args.log_interval,
                    learning_rate=args.learning_rate,
                    buffer_size=args.buffer_size,
                    learning_starts=args.learning_starts,
                    batch_size=args.batch_size,
                    tau=args.tau,
                    ent_coef=args.ent_coef,
                    seed=run_seed,
                    device=args.device,
                )
                seed_metrics.append(metrics)
                save_training_data(data_dir / f"sac_data_seed_{run_seed}.npz", **metrics)
            finally:
                task_env.close()
        save_training_data(data_path, **_aggregate_seed_metrics(seed_metrics, seeds))

    data = np.load(data_path, allow_pickle=False)
    plot_buffer_fractions(
        data["buffer_histories"],
        filename=plot_dir / "buffer_fractions_sac.png",
        window_size=args.plot_window,
        state_labels=data["automaton_states"],
    )
    plot_shaping_reward_breakdown(
        data["task_rewards"],
        data["learning_rewards"],
        data["entropy_coefficient_history"],
        window_size=args.plot_window,
        filename=plot_dir / "reward_breakdown_sac.png",
        exploration_label="Entropy coefficient (alpha)",
    )
    task_reward_runs = data["task_rewards_runs"] if "task_rewards_runs" in data else data["task_rewards"][np.newaxis, :]
    learning_reward_runs = data["learning_rewards_runs"] if "learning_rewards_runs" in data else data["learning_rewards"][np.newaxis, :]
    entropy_runs = data["entropy_coefficient_history_runs"] if "entropy_coefficient_history_runs" in data else data["entropy_coefficient_history"][np.newaxis, :]
    seed_values = data["seeds"] if "seeds" in data else np.asarray([args.seed])
    for run_seed, task_rewards, learning_rewards, entropy_history in zip(seed_values, task_reward_runs, learning_reward_runs, entropy_runs):
        plot_shaping_reward_breakdown(
            task_rewards,
            learning_rewards,
            entropy_history,
            window_size=args.plot_window,
            filename=plot_dir / f"reward_breakdown_sac_seed_{int(run_seed)}.png",
            exploration_label="Entropy coefficient (alpha)",
        )
    plot_training_variance(
        learning_reward_runs,
        window_size=args.plot_window,
        filename=plot_dir / "training_variance_sac.png",
    )
    print("\nFinished.")


# ==============================
# Command-line entry point
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LTLf LunarLander training with Stable-Baselines3 SAC."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-seeds", type=_positive_int, default=1, help="Number of training runs with consecutive seeds.")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "trajectory.json")
    parser.add_argument("--shaping-scale", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--plot-window", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=300000)
    parser.add_argument("--learning-starts", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument(
        "--ent-coef",
        type=_parse_entropy_coefficient,
        default="auto",
        help="SAC entropy coefficient, for example 'auto', 'auto_0.1' or 0.05.",
    )
    parser.add_argument("--seed", type=int, default=42, help="First training seed.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--post-process", action="store_true")
    main(parser.parse_args())
