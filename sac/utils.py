import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


# Legacy discretization. To reactivate it, uncomment this function and comment
# out the active phi_mapping_grid implementation immediately below.
#
# def phi_mapping_grid(obs, grid_w=12, grid_h=12):
#     """Map coordinates using the original grid_size - 1 discretization."""
#     x, y = float(obs[0]), float(obs[1])
#     abstract_x = int(np.clip((x + 1.0) / 2.0 * (grid_w - 1), 0, grid_w - 1))
#     abstract_y = int(np.clip(y / 1.5 * (grid_h - 1), 0, grid_h - 1))
#     return abstract_x, abstract_y


def phi_mapping_grid(obs, grid_w=12, grid_h=12):
    """Map LunarLander coordinates to uniform bins over x=[-1,1], y=[0,1.5]."""
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError("grid_w and grid_h must be positive")

    x, y = float(obs[0]), float(obs[1])
    abstract_x = int(np.floor((x + 1.0) / 2.0 * grid_w))
    abstract_y = int(np.floor(y / 1.5 * grid_h))
    abstract_x = int(np.clip(abstract_x, 0, grid_w - 1))
    abstract_y = int(np.clip(abstract_y, 0, grid_h - 1))
    return abstract_x, abstract_y


def _axis_boundaries(map_axis, size, lower, upper):
    """Infer bin boundaries from the active mapper."""
    if size <= 0:
        raise ValueError("grid dimensions must be positive")

    boundaries = [float(lower)]
    iterations = 60
    for target_index in range(1, size):
        left, right = float(lower), float(upper)
        for _ in range(iterations):
            midpoint = (left + right) / 2.0
            if map_axis(midpoint) < target_index:
                left = midpoint
            else:
                right = midpoint
        boundaries.append(right)
    boundaries.append(float(upper))
    return np.asarray(boundaries, dtype=float)


def spatial_grid_boundaries(grid_w=12, grid_h=12):
    """Return x/y bin boundaries implied by the active phi_mapping_grid."""
    x_boundaries = _axis_boundaries(
        lambda x: phi_mapping_grid((x, 0.0), grid_w, grid_h)[0],
        grid_w,
        -1.0,
        1.0,
    )
    y_boundaries = _axis_boundaries(
        lambda y: phi_mapping_grid((0.0, y), grid_w, grid_h)[1],
        grid_h,
        0.0,
        1.5,
    )
    return x_boundaries, y_boundaries


def phi_mapping_sequential(obs, q, grid_w=12, grid_h=12):
    abstract_x, abstract_y = phi_mapping_grid(obs, grid_w, grid_h)
    return abstract_x, abstract_y, q


def lunar_lander_visible_observation_bounds():
    """Return the normalised x/y bounds covered by LunarLander's RGB viewport."""
    from gymnasium.envs.box2d import lunar_lander

    viewport_world_width = lunar_lander.VIEWPORT_W / lunar_lander.SCALE
    viewport_world_height = lunar_lander.VIEWPORT_H / lunar_lander.SCALE
    helipad_y = viewport_world_height / 4.0
    lander_y_offset = helipad_y + lunar_lander.LEG_DOWN / lunar_lander.SCALE
    half_world_height = viewport_world_height / 2.0
    visible_y_min = (0.0 - lander_y_offset) / half_world_height
    visible_y_max = (viewport_world_height - lander_y_offset) / half_world_height
    return -1.0, 1.0, visible_y_min, visible_y_max


def _draw_visible_area_overlay(axis, width, height):
    """Mark which portion of the active abstract grid lies in the RGB viewport."""
    visible_x_min, visible_x_max, visible_y_min, visible_y_max = (
        lunar_lander_visible_observation_bounds()
    )
    x_boundaries, y_boundaries = spatial_grid_boundaries(width, height)

    def to_plot(value, boundaries):
        value = float(np.clip(value, boundaries[0], boundaries[-1]))
        index = int(np.searchsorted(boundaries, value, side="right") - 1)
        index = int(np.clip(index, 0, len(boundaries) - 2))
        lower, upper = boundaries[index], boundaries[index + 1]
        fraction = 0.0 if upper <= lower else (value - lower) / (upper - lower)
        return index - 0.5 + fraction

    left = float(to_plot(visible_x_min, x_boundaries))
    right = float(to_plot(visible_x_max, x_boundaries))
    bottom = float(to_plot(visible_y_min, y_boundaries))
    top = float(to_plot(visible_y_max, y_boundaries))

    axis.add_patch(
        Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            fill=False,
            edgecolor="#ff1744",
            linewidth=1.4,
            linestyle="--",
            label="Visible RGB viewport",
            zorder=5,
        )
    )
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

def save_sequential_heatmaps(
    abstract_mdp,
    filename_prefix="v_star",
    output_dir=None,
):
    """
    Generates and saves a separate heatmap for V* for each phase defined in the MDP,
    without any waypoint or goal markers (clean heatmap).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    output_dir = output_dir or os.path.join("img", "heatmaps")
    os.makedirs(output_dir, exist_ok=True)
    filename_prefix = os.path.basename(filename_prefix)
    
    width, height = abstract_mdp.width, abstract_mdp.height
    
    # Extract global min/max for consistent colormap scaling
    all_values = np.array(list(abstract_mdp.v_star.values()))
    computed_vmin = all_values.min() if len(all_values) > 0 else 0
    computed_vmax = all_values.max() if len(all_values) > 0 else 1

    for current_q in abstract_mdp.automaton.states:
        matrix = np.zeros((height, width))
        for (x, y, q), value in abstract_mdp.v_star.items():
            if q == current_q and 0 <= x < width and 0 <= y < height:
                matrix[y, x] = value
                
        plt.figure(figsize=(9, 8))
        im = plt.imshow(matrix, cmap='viridis', origin='lower', vmin=computed_vmin, vmax=computed_vmax)
        
        for y in range(height):
            for x in range(width):
                val = matrix[y, x]
                if val > 0.0: 
                    text_color = 'white' if val < (computed_vmax / 2) else 'black'
                    plt.text(x, y, f"{val:.1f}", ha='center', va='center', color=text_color, fontsize=7)
                    
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Potential Value (V*)")
        
        is_goal_state = abstract_mdp.automaton.is_goal_reached(current_q)
        phase_label = "Goal Reached" if is_goal_state else "Seeking Targets"
        plt.title(f"Potential Map (V*) - DFA State q={current_q} ({phase_label})", fontsize=14, fontweight='bold')
        
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-.5, height, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1, alpha=0.4)
        _draw_visible_area_overlay(ax, width, height)
        
        # Keep the heatmap free of waypoint and goal markers.
            
        plt.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        plt.savefig(os.path.join(output_dir, f"{filename_prefix}_q{current_q}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f" -> Generated V* Heatmap for DFA State q={current_q}")

def plot_comparison_curves(baseline_rewards, shaping_rewards, epsilon_history=None, window_size=100, filename="img/baseline_vs_shaping.png", title="Learning Curve Comparison", baseline_label="Baseline", shaping_label="Shaping"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    baseline_ma = pd.Series(baseline_rewards).rolling(window=window_size, min_periods=1, center=False).mean()
    shaping_ma = pd.Series(shaping_rewards).rolling(window=window_size, min_periods=1, center=False).mean()
    x_axis = np.arange(len(baseline_rewards))
    ax1.plot(x_axis, baseline_ma, color='black', linestyle='-', linewidth=2, label=baseline_label)
    ax1.plot(x_axis, shaping_ma, color='blue', linestyle='-', linewidth=2.5, label=shaping_label)
    ax1.set_title(title, fontsize=15, fontweight='bold')
    ax1.set_xlabel(f"Episode (mean over the last {window_size} episodes)", fontsize=12)
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    if epsilon_history:
        ax2 = ax1.twinx()
        ax2.plot(x_axis, epsilon_history, color='orange', linestyle='--', linewidth=1.8, label='Epsilon Decay')
        ax2.set_ylabel("Exploration Rate (ε)", color='orange', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=11)
    else:
        ax1.legend(loc="lower right", fontsize=11)

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n>>> Comparison plot successfully saved to: {filename}")
    plt.close(fig)

def plot_mean_std_curves(reward_histories_single=None, reward_histories_multi=None, window_size=100, title="Mean Performance with Variance", filename="img/mean_std_plot.png"):
    """
    Plots the mean and standard deviation of reward histories for one or two sets of runs.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    def plot_single_curve(reward_histories, label, color):
        if not reward_histories:
            return
        
        # Ensure all histories have the same length by padding with NaNs if necessary
        max_len = max(len(h) for h in reward_histories)
        padded_histories = [np.pad(h, (0, max_len - len(h)), 'constant', constant_values=np.nan) for h in reward_histories]
        
        rewards_df = pd.DataFrame(padded_histories).T
        smoothed_rewards = rewards_df.rolling(
            window=window_size, min_periods=1, center=False
        ).mean()
        mean_ma = smoothed_rewards.mean(axis=1)
        std_ma = smoothed_rewards.std(axis=1, ddof=0)

        x_axis = np.arange(len(mean_ma))
        ax.plot(x_axis, mean_ma, label=f"Mean {label}", color=color, linewidth=2.5)
        ax.fill_between(x_axis, mean_ma - std_ma, mean_ma + std_ma, color=color, alpha=0.2, label=f"Std Dev {label}")

    if reward_histories_single:
        plot_single_curve(reward_histories_single, "Single Epsilon", "black")

    if reward_histories_multi:
        plot_single_curve(reward_histories_multi, "Multi Epsilon", "blue")

    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel(f"Episode (mean over the last {window_size} episodes)", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n>>> Mean/Std plot successfully saved to: {filename}")
    plt.close(fig)


def plot_training_variance(reward_histories, window_size=100, title="Training Performance Across Seeds", filename="img/training_variance.png", label="Learning reward"):
    """Plot the smoothed mean reward and its variance across training seeds."""
    runs = np.asarray(reward_histories, dtype=np.float64)
    if runs.ndim == 1:
        runs = runs[np.newaxis, :]
    if runs.ndim != 2 or runs.shape[0] == 0 or runs.shape[1] == 0:
        raise ValueError("reward_histories must have shape (num_seeds, episodes)")
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")

    smoothed_runs = (
        pd.DataFrame(runs.T)
        .rolling(window=window_size, min_periods=1, center=False)
        .mean()
        .to_numpy()
        .T
    )
    mean_reward = np.mean(smoothed_runs, axis=0)
    variance = np.var(smoothed_runs, axis=0)
    std_reward = np.sqrt(variance)
    episodes = np.arange(1, runs.shape[1] + 1)

    output_directory = os.path.dirname(os.fspath(filename))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(episodes, mean_reward, color="tab:blue", linewidth=2.5, label=f"Mean {label}")
    ax.fill_between(
        episodes,
        mean_reward - std_reward,
        mean_reward + std_reward,
        color="tab:blue",
        alpha=0.2,
        label="±1 std across seeds (variance = σ²)",
    )
    ax.set_title(f"{title} ({runs.shape[0]} seeds)", fontsize=15, fontweight="bold")
    ax.set_xlabel(f"Episode (mean over the last {window_size} episodes)", fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"\n>>> Training variance plot saved to: {filename}")
    plt.close(fig)

def plot_buffer_fractions(buffer_histories, window_size=100, filename="img/buffer_fractions.png", state_labels=None):
    """
    Plots the replay buffer composition for N phases dynamically.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    x_axis = np.arange(len(buffer_histories[0]))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(buffer_histories)))
    for idx, history in enumerate(buffer_histories):
        ma = pd.Series(history).rolling(window=window_size, min_periods=1, center=False).mean()
        state_label = state_labels[idx] if state_labels is not None else idx
        ax.plot(x_axis, ma, color=colors[idx], linewidth=2.5, label=f'DFA state q={state_label}')
    
    ax.set_title(f"Replay Buffer Composition (MA Window = {window_size})", fontsize=14, fontweight='bold')
    ax.set_ylabel("Fraction in Buffer", fontsize=12)
    ax.set_ylim(0, 1.05)
    
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(buffer_histories), fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_shaping_reward_breakdown(true_rewards, total_rewards, exploration_history, window_size=100, filename="img/shaping_reward_breakdown.png", exploration_label="Entropy coefficient (alpha)"):
    """
    Plot reward moving averages and SAC's entropy coefficient.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Moving Average Calculation
    true_ma = pd.Series(true_rewards).rolling(window=window_size, min_periods=1, center=False).mean()
    total_ma = pd.Series(total_rewards).rolling(window=window_size, min_periods=1, center=False).mean()
        
    x_axis = np.arange(len(true_rewards))
        
    # Plot Rewards (Left Y-Axis)
    ax1.plot(x_axis, true_ma, color='green', linestyle='-', linewidth=2, label='Synthetic Goal Reward')
    ax1.plot(x_axis, total_ma, color='purple', linestyle='-', linewidth=2.5, label='Learning Reward (Goal + Shaping)')
    
    ax1.set_title(f"SAC Reward Analysis (MA Window = {window_size})", fontsize=15, fontweight='bold')
    ax1.set_xlabel("Episode #", fontsize=12)
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # SAC does not use epsilon-greedy exploration.  Its learned entropy
    # coefficient is the corresponding exploration diagnostic.
    ax2 = ax1.twinx()
    ax2.plot(
        x_axis,
        exploration_history,
        color='orange',
        linestyle='--',
        linewidth=1.8,
        label=exploration_label,
    )
    ax2.set_ylabel(exploration_label, color='black', fontsize=12)

    # Combine Legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Dynamically calculate legend columns based on number of items to keep it compact
    legend_cols = max(2, (len(labels1) + len(labels2)) // 2)
    
    ax1.legend(
        lines1 + lines2, labels1 + labels2, 
        loc="upper center", bbox_to_anchor=(0.5, -0.15), 
        ncol=legend_cols, fontsize=11, framealpha=1.0
    )
    
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
