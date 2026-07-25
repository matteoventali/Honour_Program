"""Train discrete SAC on LunarLander tasks specified by an LTLf formula."""

import argparse
import json
import os
from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from d3rlpy.algos import DiscreteSACConfig
from d3rlpy.dataset import create_fifo_replay_buffer
from d3rlpy.models.encoders import VectorEncoderFactory

from abstract_mdps import LTLfAutomaton, LTLfWaypointMDP
from automaton_validator import validate_automaton
from utils import phi_mapping_sequential, save_sequential_heatmaps


# =============================================================================
# Discrete SAC algorithm configuration
# =============================================================================

def build_discrete_sac(
    *,
    hidden_sizes,
    learning_rate,
    batch_size,
    gamma,
    initial_temperature,
    target_update_interval,
    device,
):
    """Create a configured d3rlpy DiscreteSAC learner."""
    # d3rlpy builds the actor, twin critics, target networks and optimizers
    # internally. Separate encoder factories keep actor and critic parameters
    # independent while giving both networks the same architecture.
    actor_encoder = VectorEncoderFactory(hidden_units=hidden_sizes)
    critic_encoder = VectorEncoderFactory(hidden_units=hidden_sizes)
    config = DiscreteSACConfig(
        actor_learning_rate=learning_rate,
        critic_learning_rate=learning_rate,
        temp_learning_rate=learning_rate,
        actor_encoder_factory=actor_encoder,
        critic_encoder_factory=critic_encoder,
        batch_size=batch_size,
        gamma=gamma,
        initial_temperature=initial_temperature,
        target_update_interval=target_update_interval,
    )
    return config.create(device=device)


# =============================================================================
# Episode metrics
# =============================================================================

class TrainingMetrics:
    """Own all mutable training metrics without relying on module globals."""

    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.enabled = False
        self.clear()

    def clear(self):
        """Discard all recorded episodes and DFA transitions."""
        self.task_rewards = []
        self.total_rewards = []
        self.episode_lengths = []
        self.episode_end_reasons = []
        self.dfa_transition_counts = defaultdict(int)

    def record_transition(self, source, destination):
        """Count a DFA transition when metric collection is enabled."""
        if self.enabled:
            self.dfa_transition_counts[(source, destination)] += 1

    def record_episode(self, task_reward, total_reward, length, end_reason, use_shaping):
        """Append one completed episode and print periodic diagnostics."""
        if not self.enabled:
            return
        self.task_rewards.append(task_reward)
        self.total_rewards.append(total_reward)
        self.episode_lengths.append(length)
        self.episode_end_reasons.append(end_reason)
        if len(self.task_rewards) % self.log_interval == 0:
            self._print_progress(use_shaping)

    def _print_progress(self, use_shaping):
        """Print recent and cumulative learning diagnostics."""
        window = min(self.log_interval, len(self.task_rewards))
        recent_rewards = np.asarray(self.task_rewards[-window:])
        recent_reasons = self.episode_end_reasons[-window:]
        successes = int(np.count_nonzero(np.asarray(self.task_rewards) > 0))
        transitions = ", ".join(
            f"{source}->{destination}: {count}"
            for (source, destination), count in sorted(self.dfa_transition_counts.items())
        ) or "none"
        mode = "SHAPING" if use_shaping else "BASELINE"
        training_reward_line = (
            f"Average training reward      : {np.mean(self.total_rewards[-window:]):.6f}\n"
            if use_shaping
            else ""
        )
        print(
            f"\n[{mode} | DISCRETE SAC] Completed episode {len(self.task_rewards)}\n"
            f"Average task reward          : {np.mean(recent_rewards):.6f}\n"
            f"Cumulative successes         : {successes}/{len(self.task_rewards)} "
            f"({successes / len(self.task_rewards):.2%})\n"
            f"Success rate (last {window}) : {np.mean(recent_rewards > 0):.2%}\n"
            f"Endings (last {window})      : "
            f"environment={recent_reasons.count('environment_terminated')}, "
            f"truncated={recent_reasons.count('truncated')}, "
            f"success={recent_reasons.count('success')}\n"
            f"Average episode length       : {np.mean(self.episode_lengths[-window:]):.1f}\n"
            f"{training_reward_line}"
            f"DFA transitions (cumulative) : {transitions}"
        )

    def validate(self):
        """Ensure that every episode-level metric has the same length."""
        lengths = {
            "task_rewards": len(self.task_rewards),
            "total_rewards": len(self.total_rewards),
            "episode_lengths": len(self.episode_lengths),
            "episode_end_reasons": len(self.episode_end_reasons),
        }
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"Inconsistent episode metrics: {lengths}.")

    def as_dict(self):
        """Return defensive copies suitable for serialization."""
        return {
            "task_rewards": self.task_rewards.copy(),
            "total_rewards": self.total_rewards.copy(),
            "episode_lengths": self.episode_lengths.copy(),
            "episode_end_reasons": self.episode_end_reasons.copy(),
            "dfa_transition_counts": dict(self.dfa_transition_counts),
        }


# =============================================================================
# LTLf observation and reward wrapper
# =============================================================================

class LTLfShapingWrapper(gym.Wrapper):
    """Apply the synthetic task reward and potential-based reward shaping."""

    def __init__(
        self,
        env,
        abstract_mdp,
        metrics,
        use_shaping=True,
        shaping_scale=1.0,
        goal_reward=10000.0,
    ):
        super().__init__(env)
        self.abstract_mdp = abstract_mdp
        self.metrics = metrics
        self.use_shaping = use_shaping
        self.shaping_scale = shaping_scale
        self.goal_reward = goal_reward
        self.automaton_states = list(abstract_mdp.automaton.states)
        self.state_to_index = {
            state: index for index, state in enumerate(self.automaton_states)
        }
        self.current_dfa_state = None
        self.previous_observation = None
        self.episode_task_reward = 0.0
        self.episode_total_reward = 0.0
        self.episode_length = 0

        # The policy receives the physical LunarLander state followed by a
        # one-hot encoding of the active DFA state.
        observation_size = env.observation_space.shape[0] + len(self.automaton_states)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

    @staticmethod
    def _abstract_position(observation):
        x, y, _ = phi_mapping_sequential(observation, 0)
        return x, y

    def _augment_observation(self, observation, dfa_state):
        one_hot = np.zeros(len(self.automaton_states), dtype=np.float32)
        one_hot[self.state_to_index[dfa_state]] = 1.0
        return np.concatenate((observation, one_hot)).astype(np.float32)

    def reset(self, **kwargs):
        """Reset the environment and evaluate the initial observation in the DFA."""
        observation, info = self.env.reset(**kwargs)
        x, y = self._abstract_position(observation)
        valuation = self.abstract_mdp._get_truth_assignment(x, y)

        # The DFA initial node represents the empty trace. Consuming s0 here
        # ensures that the first policy observation carries the correct state.
        pre_trace_state = self.abstract_mdp.automaton.get_initial_q()
        self.current_dfa_state = self.abstract_mdp.automaton.get_next_q(
            pre_trace_state,
            valuation,
        )
        self.previous_observation = observation
        self.episode_task_reward = 0.0
        self.episode_total_reward = 0.0
        self.episode_length = 0
        return self._augment_observation(observation, self.current_dfa_state), info

    def step(self, action):
        """Advance the environment, DFA and potential-based reward process."""
        # The native LunarLander reward is intentionally discarded: this
        # experiment learns exclusively from the temporal task and shaping.
        observation, _, terminated, truncated, info = self.env.step(action)
        environment_terminated = terminated
        self.episode_length += 1

        x, y = self._abstract_position(self.previous_observation)
        next_x, next_y = self._abstract_position(observation)
        valuation = self.abstract_mdp._get_truth_assignment(next_x, next_y)
        next_dfa_state = self.abstract_mdp.automaton.get_next_q(
            self.current_dfa_state,
            valuation,
        )

        task_reward = 0.0
        task_success = False

        # Entering an accepting DFA state completes the temporal task and ends
        # the Gymnasium episode with the configured synthetic goal reward.
        if next_dfa_state != self.current_dfa_state:
            self.metrics.record_transition(self.current_dfa_state, next_dfa_state)
            if self.abstract_mdp.automaton.is_goal_reached(next_dfa_state):
                task_reward = self.goal_reward
                task_success = True
                terminated = True

        abstract_state = (x, y, self.current_dfa_state)
        next_abstract_state = (next_x, next_y, next_dfa_state)
        shaping_reward = 0.0

        # Potential-based shaping is applied only when the complete abstract
        # state changes, using F(s,s') = K * (gamma * V*(s') - V*(s)).
        if self.use_shaping and abstract_state != next_abstract_state:
            current_potential = self.abstract_mdp.v_star.get(abstract_state, 0.0)
            next_potential = self.abstract_mdp.v_star.get(next_abstract_state, 0.0)
            shaping_reward = self.shaping_scale * (
                self.abstract_mdp.gamma * next_potential - current_potential
            )

        total_reward = task_reward + shaping_reward
        self.episode_task_reward += task_reward
        self.episode_total_reward += total_reward

        # Store one coherent record only when an episode has actually ended.
        if terminated or truncated:
            if task_success:
                end_reason = "success"
            elif environment_terminated:
                end_reason = "environment_terminated"
            else:
                end_reason = "truncated"
            self.metrics.record_episode(
                self.episode_task_reward,
                self.episode_total_reward,
                self.episode_length,
                end_reason,
                self.use_shaping,
            )

        self.current_dfa_state = next_dfa_state
        self.previous_observation = observation
        augmented_observation = self._augment_observation(
            observation,
            next_dfa_state,
        )
        return augmented_observation, total_reward, terminated, truncated, info


# =============================================================================
# Plotting helpers
# =============================================================================

def _moving_average(values, window_size):
    """Smooth an episode sequence while retaining boundary observations."""
    return pd.Series(values).rolling(
        window=window_size,
        min_periods=1,
        center=True,
    ).mean()


def plot_reward_breakdown(
    task_rewards,
    total_rewards,
    window_size=100,
    filename="img/shaping_reward_breakdown.png",
):
    """Plot synthetic task rewards against the total learning rewards."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    task_average = _moving_average(task_rewards, window_size)
    total_average = _moving_average(total_rewards, window_size)
    episodes = np.arange(len(task_rewards))

    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(
        episodes,
        task_average,
        color="green",
        linewidth=2,
        label="Goal-MDP task reward",
    )
    axis.plot(
        episodes,
        total_average,
        color="purple",
        linewidth=2.5,
        label="Training reward (task + shaping)",
    )
    axis.set_title(
        f"Discrete SAC reward analysis (moving-average window: {window_size})",
        fontsize=15,
        fontweight="bold",
    )
    axis.set_xlabel("Episode")
    axis.set_ylabel("Episode reward")
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    figure.tight_layout()
    figure.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Reward plot saved to {filename}")


def plot_comparison(
    baseline_rewards,
    shaping_rewards,
    window_size=100,
    filename="img/baseline_vs_shaping.png",
):
    """Compare baseline and reward-shaped discrete SAC learning curves."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    episodes = np.arange(min(len(baseline_rewards), len(shaping_rewards)))
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(
        episodes,
        _moving_average(baseline_rewards[: len(episodes)], window_size),
        color="black",
        linewidth=2,
        label="Discrete SAC without shaping",
    )
    axis.plot(
        episodes,
        _moving_average(shaping_rewards[: len(episodes)], window_size),
        color="blue",
        linewidth=2.5,
        label="Discrete SAC with shaping",
    )
    axis.set_title("Discrete SAC performance comparison")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Episode reward")
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Comparison plot saved to {filename}")


# =============================================================================
# Experiment configuration
# =============================================================================

def load_experiment_config(filename):
    """Load and normalize the temporal-task configuration."""
    with open(filename, "r", encoding="utf-8") as config_file:
        raw_config = json.load(config_file)
    return {
        "formula": raw_config.get("formula", "F(goal)"),
        "waypoints": {
            name: tuple(coordinates)
            for name, coordinates in raw_config.get(
                "waypoints_dict",
                {"goal": [5, 0]},
            ).items()
        },
        "grid_width": int(raw_config.get("grid_w", 12)),
        "grid_height": int(raw_config.get("grid_h", 12)),
        "gamma": float(raw_config.get("gamma", 0.99)),
        "goal_reward": float(raw_config.get("goal_reward", 10000)),
    }


# =============================================================================
# Abstract model construction
# =============================================================================

def build_abstract_mdp(config, image_directory):
    """Build, validate and solve the abstract LTLf-guided MDP."""
    # Compile the finite-trace temporal formula into a deterministic automaton.
    automaton = LTLfAutomaton(config["formula"])

    # Reject malformed automata and waypoint definitions before allocating the
    # neural networks or starting an expensive training run.
    validation_report = validate_automaton(
        automaton,
        config["waypoints"],
        width=config["grid_width"],
        height=config["grid_height"],
    )
    print(
        "=== LTLf discrete SAC experiment ===\n"
        f"Formula: {config['formula']}\n"
        f"Waypoints: {config['waypoints']}\n"
        f"DFA states: {automaton.states}\n"
        f"Pre-trace state: {automaton.get_initial_q()}\n"
        f"Accepting states: {sorted(automaton.accepting_states)}\n"
        f"{validation_report.format()}"
    )
    automaton.render_graph(directory=image_directory)

    # Value iteration produces the potential function V*, later used by the
    # environment wrapper to compute potential-based shaping rewards.
    abstract_mdp = LTLfWaypointMDP(
        waypoints_dict=config["waypoints"],
        ltlf_automaton=automaton,
        width=config["grid_width"],
        height=config["grid_height"],
        gamma=config["gamma"],
        goal_reward=config["goal_reward"],
    )
    abstract_mdp.value_iteration()
    save_sequential_heatmaps(
        abstract_mdp,
        filename_prefix=f"{image_directory}/heatmap_V_star",
    )
    return abstract_mdp


# =============================================================================
# Environment construction
# =============================================================================

def make_environment(abstract_mdp, metrics, args, goal_reward):
    """Create the Gymnasium environment consumed directly by d3rlpy."""
    environment = gym.make("LunarLander-v3", continuous=False)
    return LTLfShapingWrapper(
        environment,
        abstract_mdp,
        metrics,
        use_shaping=args.use_shaping,
        shaping_scale=args.shaping_scale,
        goal_reward=goal_reward,
    )


# =============================================================================
# Discrete SAC training
# =============================================================================

def train_discrete_sac(environment, metrics, args, gamma):
    """Train d3rlpy's DiscreteSAC for the requested environment steps."""
    device = args.device
    print(
        f"\nInitializing discrete SAC on {device} "
        f"(reward shaping: {args.use_shaping})."
    )
    algorithm = build_discrete_sac(
        hidden_sizes=args.hidden_sizes,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=gamma,
        initial_temperature=args.initial_temperature,
        target_update_interval=args.target_update_interval,
        device=device,
    )
    replay_buffer = create_fifo_replay_buffer(
        limit=args.buffer_size,
        env=environment,
    )

    try:
        # d3rlpy owns action selection, replay collection and gradient updates.
        # Random actions are used until the replay buffer is sufficiently warm.
        metrics.enabled = True
        print(
            f"Training for {args.steps} environment steps "
            f"({args.random_steps} initial random steps)."
        )
        algorithm.fit_online(
            environment,
            buffer=replay_buffer,
            n_steps=args.steps,
            n_steps_per_epoch=min(args.steps_per_epoch, args.steps),
            update_interval=args.update_interval,
            n_updates=args.updates_per_interval,
            update_start_step=args.random_steps,
            random_steps=args.random_steps,
            experiment_name="ltlf_discrete_sac",
            with_timestamp=False,
            show_progress=args.show_progress,
        )
        mode = "shaping" if args.use_shaping else "baseline"
        algorithm.save(f"{args.output_directory}/{mode}_discrete_sac.d3")
    finally:
        metrics.enabled = False
        environment.close()

    metrics.validate()
    print("Training completed.")
    return metrics.as_dict()


# =============================================================================
# Result persistence
# =============================================================================

def save_metrics(metrics, output_directory, use_shaping):
    """Persist numeric metrics in a format suitable for later analysis."""
    task_rewards = np.asarray(metrics["task_rewards"], dtype=np.float64)
    total_rewards = np.asarray(metrics["total_rewards"], dtype=np.float64)
    success_flags = task_rewards > 0
    transitions = sorted(metrics["dfa_transition_counts"].items())
    prefix = "shaping" if use_shaping else "baseline"
    filename = f"{output_directory}/{prefix}_discrete_sac_data.npz"
    np.savez_compressed(
        filename,
        task_rewards=task_rewards,
        total_rewards=total_rewards,
        success_flags=success_flags,
        success_rate=float(np.mean(success_flags)) if len(success_flags) else 0.0,
        episode_lengths=np.asarray(metrics["episode_lengths"], dtype=np.int64),
        episode_end_reasons=np.asarray(metrics["episode_end_reasons"]),
        dfa_transition_labels=np.asarray(
            [f"{source}->{destination}" for (source, destination), _ in transitions],
        ),
        dfa_transition_counts=np.asarray(
            [count for _, count in transitions],
            dtype=np.int64,
        ),
        true_rewards=task_rewards,
    )
    print(f"Training metrics saved to {filename}")
    return task_rewards, total_rewards


# =============================================================================
# Experiment orchestration
# =============================================================================

def main(args):
    """Execute the complete configuration, training and reporting pipeline."""
    os.makedirs(args.output_directory, exist_ok=True)
    os.makedirs(args.image_directory, exist_ok=True)
    config = load_experiment_config(args.config)
    abstract_mdp = build_abstract_mdp(config, args.image_directory)
    metrics = TrainingMetrics(log_interval=args.log_interval)
    environment = make_environment(
        abstract_mdp,
        metrics,
        args,
        config["goal_reward"],
    )
    results = train_discrete_sac(
        environment,
        metrics,
        args,
        config["gamma"],
    )
    task_rewards, total_rewards = save_metrics(
        results,
        args.output_directory,
        args.use_shaping,
    )
    success_rate = float(np.mean(task_rewards > 0)) if len(task_rewards) else 0.0
    print(f"Overall success rate: {success_rate:.2%}")
    mode = "shaping" if args.use_shaping else "baseline"
    plot_reward_breakdown(
        task_rewards,
        total_rewards,
        window_size=min(args.plot_window, max(1, len(task_rewards))),
        filename=(
            f"{args.image_directory}/discrete_sac_{mode}_reward_breakdown.png"
        ),
    )


# =============================================================================
# Command-line interface
# =============================================================================

def build_argument_parser():
    """Define all reproducible experiment parameters exposed by the CLI."""
    parser = argparse.ArgumentParser(description="LTLf discrete SAC training.")
    parser.add_argument("--steps", type=int, default=250000)
    parser.add_argument("--config", default="trajectory.json")
    parser.add_argument(
        "--use-shaping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable potential-based reward shaping.",
    )
    parser.add_argument("--shaping-scale", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--initial-temperature", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--random-steps", type=int, default=5000)
    parser.add_argument("--target-update-interval", type=int, default=8000)
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--updates-per-interval", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=10000)
    parser.add_argument("--device", default="cpu:0")
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--plot-window", type=int, default=50)
    parser.add_argument("--output-directory", default="results")
    parser.add_argument("--image-directory", default="img")
    return parser


if __name__ == "__main__":
    main(build_argument_parser().parse_args())
