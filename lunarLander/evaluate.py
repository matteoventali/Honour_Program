"""Evaluate one or more LTLf-guided LunarLander DQN policies."""

# ==============================
# Standard library imports
# ==============================

import argparse
import json
import time
from pathlib import Path

# ==============================
# External and project imports
# ==============================

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from abstract_mdps import LTLfAutomaton, LTLfWaypointMDP
from agent import QNetwork
from utils import phi_mapping_sequential


# ==============================
# Paths and generic helpers
# ==============================

SCRIPT_DIR = Path(__file__).resolve().parent


def moving_average(data, window_size):
    """Return a moving average, or the original values when the window is larger."""
    values = np.asarray(data, dtype=np.float64)
    if len(values) < window_size:
        return values
    return np.convolve(values, np.ones(window_size) / window_size, mode="valid")


def _resolve_policy_path(policy, policy_dir):
    """Accept explicit paths as well as filenames relative to the policy directory."""
    supplied_path = Path(policy).expanduser()
    if supplied_path.is_file():
        return supplied_path.resolve()

    policy_path = Path(policy_dir).expanduser() / supplied_path
    if policy_path.is_file():
        return policy_path.resolve()

    raise FileNotFoundError(f"Policy '{policy}' not found either as an explicit path or under '{policy_dir}'.")


def _load_state_dict(policy_path, device):
    """Load both plain state dictionaries and common wrapped checkpoints."""
    checkpoint = torch.load(policy_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("policy_state_dict", "state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _abstract_position(observation, q, grid_w, grid_h):
    """Map an environment observation to its abstract grid coordinates."""
    x, y, _ = phi_mapping_sequential(observation, q, grid_w, grid_h)
    return x, y


# ==============================
# Policy evaluation
# ==============================

def evaluate_policy(policy, policy_dir, episodes, render, formula, waypoints_dict, goal_reward, grid_w, grid_h, seed):
    """Load and evaluate one policy using the same DFA semantics as training."""
    # Rebuild the same automaton and abstract MDP used during training.
    policy_path = _resolve_policy_path(policy, policy_dir)
    policy_name = policy_path.name
    automaton = LTLfAutomaton(formula)
    abstract_mdp = LTLfWaypointMDP(waypoints_dict=waypoints_dict, ltlf_automaton=automaton, width=grid_w, height=grid_h, goal_reward=goal_reward)
    automaton_states = list(automaton.states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}

    # Create the environment and a network with one extra feature per DFA state.
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", continuous=False, render_mode=render_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = QNetwork(env.observation_space.shape[0] + len(automaton_states), env.action_space.n).to(device)

    # Load the trained parameters before starting any episode.
    try:
        network.load_state_dict(_load_state_dict(policy_path, device))
        network.eval()
    except Exception:
        env.close()
        raise

    task_returns = []
    environment_returns = []
    episode_lengths = []
    successes = 0
    state_reach_counts = {q: 0 for q in automaton_states}

    # Run every requested episode sequentially.
    try:
        for episode in range(episodes):
            episode_seed = None if seed is None else seed + episode
            observation, _ = env.reset(seed=episode_seed)

            # Training consumes the valuation at s0 before choosing the first action.
            initial_q = automaton.get_initial_q()
            initial_x, initial_y = _abstract_position(observation, initial_q, grid_w, grid_h)
            initial_truth_assignment = abstract_mdp._get_truth_assignment(initial_x, initial_y)
            q = automaton.get_next_q(initial_q, initial_truth_assignment)
            if q not in state_to_index:
                raise RuntimeError(f"DFA returned unknown state {q!r}")

            reached_states = {q}
            success = automaton.is_goal_reached(q)
            terminated = truncated = False
            environment_return = 0.0
            steps = 0

            while not (success or terminated or truncated):
                # Append the current DFA state as a one-hot vector.
                one_hot = np.zeros(len(automaton_states), dtype=np.float32)
                one_hot[state_to_index[q]] = 1.0
                augmented_state = np.concatenate((observation, one_hot)).astype(np.float32)

                # Evaluation is greedy: always select the action with maximum Q-value.
                with torch.inference_mode():
                    state_tensor = torch.as_tensor(augmented_state, device=device).unsqueeze(0)
                    action = network(state_tensor).argmax(dim=1).item()

                next_observation, env_reward, terminated, truncated, _ = env.step(action)
                environment_return += float(env_reward)
                steps += 1

                # Advance the DFA using the propositions true in the arrival state.
                x, y = _abstract_position(next_observation, q, grid_w, grid_h)
                truth_assignment = abstract_mdp._get_truth_assignment(x, y)
                next_q = automaton.get_next_q(q, truth_assignment)
                if next_q not in state_to_index:
                    raise RuntimeError(f"DFA returned unknown state {next_q!r}")

                # Report every effective DFA transition during evaluation.
                if next_q != q:
                    if automaton.is_goal_reached(next_q):
                        print(f"[{policy_name} | Episode {episode + 1}] DFA transition {q} -> {next_q}: final goal reached.")
                    else:
                        print(f"[{policy_name} | Episode {episode + 1}] DFA transition {q} -> {next_q}: intermediate waypoint reached.")

                reached_states.add(next_q)

                observation = next_observation
                q = next_q
                success = automaton.is_goal_reached(q)

                if render:
                    time.sleep(0.02)

            # Store episode-level metrics and count every DFA state reached at least once.
            successes += int(success)
            for reached_q in reached_states:
                state_reach_counts[reached_q] += 1
            task_returns.append(float(goal_reward) if success else 0.0)
            environment_returns.append(environment_return)
            episode_lengths.append(steps)
    finally:
        env.close()

    return {
        "policy": policy_name,
        "path": str(policy_path),
        "task_returns": task_returns,
        "environment_returns": environment_returns,
        "episode_lengths": episode_lengths,
        "successes": successes,
        "state_reach_counts": state_reach_counts,
    }


# ==============================
# Plotting helpers
# ==============================

def _safe_stem(name):
    """Create a filesystem-safe plot stem from a checkpoint filename."""
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in Path(name).stem)


def plot_policy(result, window_size, output_dir):
    """Plot Gym returns for one policy."""
    returns = result["environment_returns"]
    smooth = moving_average(returns, window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(returns, alpha=0.3, color="gray", label="Raw Gym return")
    start = window_size - 1 if len(returns) >= window_size else 0
    plt.plot(range(start, start + len(smooth)), smooth, color="blue", linewidth=2, label=f"Moving average (window={window_size})")
    plt.title(f"Evaluation: {result['policy']}")
    plt.xlabel("Episode")
    plt.ylabel("Gym return")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    output_path = output_dir / f"eval_{_safe_stem(result['policy'])}.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def plot_comparison(results, window_size, output_dir):
    """Plot smoothed Gym returns for multiple policies."""
    plt.figure(figsize=(12, 6))
    for result in results:
        returns = result["environment_returns"]
        smooth = moving_average(returns, window_size)
        start = window_size - 1 if len(returns) >= window_size else 0
        plt.plot(range(start, start + len(smooth)), smooth, linewidth=2, label=result["policy"])
    plt.title("Policy comparison")
    plt.xlabel("Episode")
    plt.ylabel("Gym return")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    output_path = output_dir / "policy_comparison.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


# ==============================
# Command-line interface
# ==============================

def _positive_int(value):
    """Parse and validate a strictly positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args():
    """Build and parse the evaluator command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LTLf-guided DQN policies for LunarLander.")
    parser.add_argument("policies", nargs="+", help="Checkpoint filenames or explicit checkpoint paths.")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "trajectory.json", help="Experiment JSON configuration.")
    parser.add_argument("--policy-dir", type=Path, default=SCRIPT_DIR / "policy", help="Directory used to resolve checkpoint filenames.")
    parser.add_argument("--episodes", type=_positive_int, default=100)
    parser.add_argument("--window", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "img" / "evaluation")
    return parser.parse_args()


# ==============================
# Main program
# ==============================

def main():
    """Load the configuration, evaluate the policies, and generate the plots."""
    args = parse_args()

    # Load the LTLf task shared with the trainer.
    with args.config.expanduser().open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    formula = config["formula"]
    raw_waypoints = config["waypoints_dict"]
    waypoints_dict = {name: tuple(coordinates) for name, coordinates in raw_waypoints.items()}
    grid_w = int(config.get("grid_w", 12))
    grid_h = int(config.get("grid_h", 12))
    goal_reward = float(config.get("goal_reward", 10000.0))

    # Evaluate policies one at a time to keep rendering and output deterministic.
    results = []
    for policy in args.policies:
        result = evaluate_policy(policy, args.policy_dir, args.episodes, args.render, formula, waypoints_dict, goal_reward, grid_w, grid_h, args.seed)
        results.append(result)

    # Print the summary and create one plot for each evaluated policy.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        success_rate = result["successes"] / args.episodes
        mean_gym_return = np.mean(result["environment_returns"])
        mean_length = np.mean(result["episode_lengths"])
        reached = ", ".join(f"q={q}: {count}/{args.episodes}" for q, count in result["state_reach_counts"].items())
        print(f"[{result['policy']}] success={success_rate:.1%}, mean Gym return={mean_gym_return:.2f}, mean length={mean_length:.1f} | reached: {reached}")
        print(f"Plot saved to: {plot_policy(result, args.window, args.output_dir)}")

    # Add a combined comparison when more than one policy was requested.
    if len(results) > 1:
        print(f"Comparison saved to: {plot_comparison(results, args.window, args.output_dir)}")


if __name__ == "__main__":
    main()
