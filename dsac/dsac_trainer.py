"""Train discrete SAC on LunarLander tasks specified by an LTLf formula."""

import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv

from abstract_mdps import LTLfAutomaton, LTLfWaypointMDP
from automaton_validator import validate_automaton
from dsac_environment import LTLfShapingWrapper
from dsac_metrics import TrainingMetrics
from dsac_plotting import plot_reward_breakdown
from dsac_policy import build_discrete_sac_policy
from utils import save_sequential_heatmaps


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

def make_environment_factory(abstract_mdp, metrics, args, goal_reward):
    """Return the environment factory required by Tianshou."""

    def make_environment():
        # A fresh base environment is required whenever the vectorized
        # environment invokes this factory.
        environment = gym.make("LunarLander-v3", continuous=False)
        return LTLfShapingWrapper(
            environment,
            abstract_mdp,
            metrics,
            use_shaping=args.use_shaping,
            shaping_scale=args.shaping_scale,
            goal_reward=goal_reward,
        )

    return make_environment


# =============================================================================
# Tianshou compatibility helpers
# =============================================================================

def _collected_steps(collection_result):
    """Read the step count from supported Tianshou collector result formats."""
    # Tianshou 0.x returns a dictionary, while later collector releases expose
    # the same information through a typed statistics object.
    if isinstance(collection_result, dict):
        return int(collection_result["n/st"])
    if hasattr(collection_result, "n_collected_steps"):
        return int(collection_result.n_collected_steps)
    raise TypeError(f"Unsupported collector result: {type(collection_result).__name__}")


# =============================================================================
# Discrete SAC training
# =============================================================================

def train_dsac(environment_factory, metrics, args, gamma):
    """Run the exact requested number of DSAC training episodes."""
    # CPU execution is intentionally used for reproducibility across Kaggle
    # runtimes, independently of the accelerator assigned to the notebook.
    device = "cpu"
    print(
        f"\nInitializing discrete SAC on {device} "
        f"(reward shaping: {args.use_shaping})."
    )
    training_environments = DummyVectorEnv([environment_factory])

    # Infer dimensions from the wrapped environment. The observation already
    # contains the one-hot DFA state appended to the LunarLander state.
    state_shape = training_environments.observation_space[0].shape
    action_shape = training_environments.action_space[0].n
    policy = build_discrete_sac_policy(
        state_shape=state_shape,
        action_shape=action_shape,
        device=device,
        learning_rate=args.learning_rate,
        hidden_sizes=args.hidden_sizes,
        gamma=gamma,
        alpha=args.alpha,
        tau=args.tau,
    )
    replay_buffer = VectorReplayBuffer(args.buffer_size, len(training_environments))
    collector = Collector(
        policy,
        training_environments,
        replay_buffer,
        exploration_noise=True,
    )

    try:
        # Warm-up transitions populate the replay buffer only. Disabling the
        # recorder prevents partial random episodes from polluting metrics.
        print(f"Collecting {args.warmup_steps} random warm-up steps.")
        metrics.enabled = False
        collector.collect(n_step=args.warmup_steps, random=True)
        metrics.clear()
        collector.reset_env()
        metrics.enabled = True

        print(f"Training for exactly {args.episodes} episodes.")
        for _ in range(args.episodes):
            # Collect one complete episode so that experiment-level metrics
            # always contain exactly the requested number of episodes.
            collection_result = collector.collect(n_episode=1)
            gradient_steps = _collected_steps(collection_result)

            # Preserve the original update-to-data ratio: one gradient update
            # for every newly collected environment transition.
            for _ in range(gradient_steps):
                policy.update(sample_size=args.batch_size, buffer=replay_buffer)
    finally:
        metrics.enabled = False
        training_environments.close()

    metrics.validate_episode_count(args.episodes)
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
    filename = f"{output_directory}/{prefix}_dsac_data.npz"
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
    metrics = TrainingMetrics(
        expected_episodes=args.episodes,
        log_interval=args.log_interval,
    )
    environment_factory = make_environment_factory(
        abstract_mdp,
        metrics,
        args,
        config["goal_reward"],
    )
    results = train_dsac(
        environment_factory,
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
        filename=f"{args.image_directory}/dsac_{mode}_reward_breakdown.png",
    )


# =============================================================================
# Command-line interface
# =============================================================================

def build_argument_parser():
    """Define all reproducible experiment parameters exposed by the CLI."""
    parser = argparse.ArgumentParser(description="LTLf discrete SAC training.")
    parser.add_argument("--episodes", type=int, default=500)
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
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--plot-window", type=int, default=50)
    parser.add_argument("--output-directory", default="results")
    parser.add_argument("--image-directory", default="img")
    return parser


if __name__ == "__main__":
    main(build_argument_parser().parse_args())
