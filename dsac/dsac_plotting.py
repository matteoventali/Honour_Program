"""Diagnostic plots for discrete SAC experiments."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        f"DSAC reward analysis (moving-average window: {window_size})",
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
    """Compare baseline and reward-shaped DSAC learning curves."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    episodes = np.arange(min(len(baseline_rewards), len(shaping_rewards)))
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(
        episodes,
        _moving_average(baseline_rewards[: len(episodes)], window_size),
        color="black",
        linewidth=2,
        label="DSAC without shaping",
    )
    axis.plot(
        episodes,
        _moving_average(shaping_rewards[: len(episodes)], window_size),
        color="blue",
        linewidth=2.5,
        label="DSAC with shaping",
    )
    axis.set_title("DSAC performance comparison")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Episode reward")
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Comparison plot saved to {filename}")
