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
from grid_overlay import (
    abstract_cell_to_pixel,
    draw_abstract_grid,
    geometry_from_env,
)
from utils import phi_mapping_sequential


# ==============================
# Paths and generic helpers
# ==============================

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent / "experiments"


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

def evaluate_policy(policy, policy_dir, episodes, render, formula, waypoints_dict,
                    goal_reward, grid_w, grid_h, seed, trace_episodes=0):
    """Load and evaluate one policy using the same DFA semantics as training."""
    # Rebuild the same automaton and abstract MDP used during training.
    policy_path = _resolve_policy_path(policy, policy_dir)
    policy_name = policy_path.name
    automaton = LTLfAutomaton(formula)
    abstract_mdp = LTLfWaypointMDP(waypoints_dict=waypoints_dict, ltlf_automaton=automaton, width=grid_w, height=grid_h, goal_reward=goal_reward)
    automaton_states = list(automaton.states)
    state_to_index = {q: index for index, q in enumerate(automaton_states)}

    # Create the environment and a network with one extra feature per DFA state.
    render_mode = "human" if render else ("rgb_array" if trace_episodes else None)
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
    grid_traces = []
    trace_frames = []
    trace_geometries = []

    # Run every requested episode sequentially.
    try:
        for episode in range(episodes):
            episode_seed = None if seed is None else seed + episode
            observation, _ = env.reset(seed=episode_seed)
            tracing = episode < trace_episodes
            if tracing:
                trace_frames.append(env.render())
                trace_geometries.append(geometry_from_env(env))
                initial_cell = _abstract_position(observation, automaton.get_initial_q(), grid_w, grid_h)
                cell_trace = [initial_cell]

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
                    action_values, _ = network(state_tensor)
                    action = action_values.argmax(dim=1).item()

                next_observation, env_reward, terminated, truncated, _ = env.step(action)
                environment_return += float(env_reward)
                steps += 1

                # Advance the DFA using the propositions true in the arrival state.
                x, y = _abstract_position(next_observation, q, grid_w, grid_h)
                if tracing and (x, y) != cell_trace[-1]:
                    cell_trace.append((x, y))
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
            if tracing:
                grid_traces.append(cell_trace)
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
        "grid_traces": grid_traces,
        "trace_frames": trace_frames,
        "trace_geometries": trace_geometries,
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


def plot_grid_traces(result, waypoints_dict, grid_w, grid_h, output_dir):
    """Save one abstract-grid path image for every recorded episode."""
    output_paths = []
    trace_data = zip(
        result["grid_traces"],
        result["trace_frames"],
        result["trace_geometries"],
    )
    for episode_index, (cells, frame, geometry) in enumerate(trace_data, start=1):
        figure = draw_abstract_grid(
            frame=frame,
            geometry=geometry,
            grid_w=grid_w,
            grid_h=grid_h,
            waypoints=waypoints_dict,
            title=f"Agent Abstract-Cell Trace — Episode {episode_index}",
        )
        axis = figure.axes[0]
        points = [
            abstract_cell_to_pixel(x, y, grid_w, grid_h, geometry)
            for x, y in cells
        ]
        if points:
            pixel_x, pixel_y = zip(*points)
            axis.plot(
                pixel_x,
                pixel_y,
                color="#00e5ff",
                linewidth=2.8,
                marker="o",
                markersize=5,
                label="Visited-cell path",
                zorder=4,
            )
            for change_index, ((cell_x, cell_y), (point_x, point_y)) in enumerate(
                zip(cells, points)
            ):
                axis.annotate(
                    str(change_index),
                    (point_x, point_y),
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="black",
                    zorder=7,
                )
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
        figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        output_path = output_dir / (
            f"grid_trace_{_safe_stem(result['policy'])}_episode_{episode_index}.png"
        )
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def format_waypoint_trace(cells, waypoints_dict):
    """Report the first cell-change index at which each waypoint was visited."""
    first_visit = {}
    for index, cell in enumerate(cells):
        first_visit.setdefault(tuple(cell), index)
    return ", ".join(
        f"{name}=reached@{first_visit[tuple(position)]}"
        if tuple(position) in first_visit
        else f"{name}=missed"
        for name, position in waypoints_dict.items()
    )


# ==============================
# Command-line interface
# ==============================

def _positive_int(value):
    """Parse and validate a strictly positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _select_files_graphically(policy_dir, config_path):
    """Select policy checkpoints and the experiment configuration with native dialogs."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError(
            "The graphical selector requires tkinter. Install python3-tk or pass "
            "the policy paths and --config from the command line."
        ) from error

    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(
            "The graphical selector could not be opened. Make sure a desktop "
            "session is available, or use the command-line arguments."
        ) from error
    root.withdraw()
    root.update()

    try:
        initial_directory = EXPERIMENTS_DIR if EXPERIMENTS_DIR.is_dir() else SCRIPT_DIR
        policies = filedialog.askopenfilenames(
            parent=root,
            title="Select one or more policy files",
            initialdir=str(initial_directory),
            filetypes=[
                ("PyTorch checkpoints", "*.pt *.pth *.ckpt"),
                ("All files", "*"),
            ],
        )
        if not policies:
            raise RuntimeError("No policy file was selected.")

        config = filedialog.askopenfilename(
            parent=root,
            title="Select trajectory.json",
            initialdir=str(initial_directory),
            initialfile=Path(config_path).name,
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*"),
            ],
        )
        if not config:
            raise RuntimeError("No trajectory configuration was selected.")
    finally:
        root.destroy()

    return list(policies), Path(config)


def parse_args():
    """Build and parse the evaluator command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LTLf-guided DQN policies for LunarLander.")
    parser.add_argument(
        "policies",
        nargs="*",
        help="Checkpoint filenames or explicit checkpoint paths. If omitted, graphical file selectors are opened.",
    )
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "trajectory.json", help="Experiment JSON configuration.")
    parser.add_argument("--policy-dir", type=Path, default=SCRIPT_DIR / "policy", help="Directory used to resolve checkpoint filenames.")
    parser.add_argument("--gui", action="store_true", help="Select policies and trajectory.json using graphical dialogs.")
    parser.add_argument("--episodes", type=_positive_int, default=100)
    parser.add_argument("--window", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--trace-grid",
        action="store_true",
        help="Save the sequence of abstract cells visited during evaluation.",
    )
    parser.add_argument(
        "--trace-episodes",
        type=_positive_int,
        default=1,
        help="Number of episodes to trace when --trace-grid is enabled (default: 1).",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "img" / "evaluation")
    return parser.parse_args()


# ==============================
# Main program
# ==============================

def main():
    """Load the configuration, evaluate the policies, and generate the plots."""
    args = parse_args()
    if args.render and args.trace_grid:
        raise SystemExit(
            "--render and --trace-grid cannot be used together because Gymnasium "
            "requires a single render mode. Run them as separate evaluations."
        )

    # Open native file dialogs when requested or when no policy was supplied.
    if args.gui or not args.policies:
        try:
            args.policies, args.config = _select_files_graphically(args.policy_dir, args.config)
        except RuntimeError as error:
            raise SystemExit(f"Selection cancelled: {error}") from error

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
        traced_episodes = min(args.trace_episodes, args.episodes) if args.trace_grid else 0
        result = evaluate_policy(
            policy, args.policy_dir, args.episodes, args.render, formula,
            waypoints_dict, goal_reward, grid_w, grid_h, args.seed,
            trace_episodes=traced_episodes,
        )
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
        if args.trace_grid:
            trace_paths = plot_grid_traces(
                result, waypoints_dict, grid_w, grid_h, args.output_dir
            )
            for episode_index, (cells, trace_path) in enumerate(
                zip(result["grid_traces"], trace_paths), start=1
            ):
                waypoint_status = format_waypoint_trace(cells, waypoints_dict)
                print(
                    f"Grid trace episode {episode_index}: {waypoint_status} | "
                    f"saved to: {trace_path}"
                )

    # Add a combined comparison when more than one policy was requested.
    if len(results) > 1:
        print(f"Comparison saved to: {plot_comparison(results, args.window, args.output_dir)}")


if __name__ == "__main__":
    main()
