import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def phi_mapping_grid(obs, grid_w=12, grid_h=12):
    x, y = obs[0], obs[1]
    abstract_x = int(np.clip((x + 1) / 2 * (grid_w - 1), 0, grid_w - 1))
    abstract_y = int(np.clip(y / 1.5 * (grid_h - 1), 0, grid_h - 1))
    return abstract_x, abstract_y

def phi_mapping_sequential(obs, q, grid_w=12, grid_h=12):
    abstract_x, abstract_y = phi_mapping_grid(obs, grid_w, grid_h)
    return abstract_x, abstract_y, q

def save_sequential_heatmaps(abstract_mdp, filename_prefix="v_star"):
    """
    Generates and saves a separate heatmap for V* for each phase defined in the MDP.
    """
    # Construct the full path for the heatmap and ensure the directory exists.
    base_output_dir = os.path.dirname(filename_prefix)
    os.makedirs(f"img/heatmaps/{base_output_dir}", exist_ok=True)
    width, height, num_phases = abstract_mdp.width, abstract_mdp.height, abstract_mdp.num_phases
    
    # Extract global min/max for consistent colormap scaling
    all_values = np.array(list(abstract_mdp.v_star.values()))
    computed_vmin = all_values.min() if len(all_values) > 0 else 0
    computed_vmax = all_values.max() if len(all_values) > 0 else 1

    for current_q in range(num_phases):
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
        
        is_final_phase = (current_q == num_phases - 1)
        target_name = "Goal" if is_final_phase else f"Waypoint {current_q + 1}"
        plt.title(f"Potential Map (V*) - Phase q={current_q} (Seek {target_name})", fontsize=14, fontweight='bold')
        
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-.5, height, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1, alpha=0.4)
        
        # Plot target point
        target_x, target_y = abstract_mdp.waypoints[current_q]
        color = 'go' if is_final_phase else 'ro'
        plt.plot(target_x, target_y, color, markersize=15, alpha=0.6, label=f"Target: {target_name}")
        plt.legend(loc="upper right")
            
        plt.tight_layout()
        plt.savefig(f"img/heatmaps/{filename_prefix}_q{current_q}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f" -> Generated V* Heatmap for Phase q={current_q}")

def plot_comparison_curves(baseline_rewards, shaping_rewards, epsilon_history=None, window_size=100, filename="img/baseline_vs_shaping.png", title="Learning Curve Comparison", baseline_label="Baseline", shaping_label="Shaping"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    baseline_ma = pd.Series(baseline_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    shaping_ma = pd.Series(shaping_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    x_axis = np.arange(len(baseline_rewards))
    ax1.plot(x_axis, baseline_ma, color='black', linestyle='-', linewidth=2, label=baseline_label)
    ax1.plot(x_axis, shaping_ma, color='blue', linestyle='-', linewidth=2.5, label=shaping_label)
    ax1.set_title(title, fontsize=15, fontweight='bold')
    ax1.set_xlabel(f"Episode # (Moving Average Window = {window_size})", fontsize=12)
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
        mean_rewards = rewards_df.mean(axis=1)
        std_rewards = rewards_df.std(axis=1)

        # Apply moving average
        mean_ma = mean_rewards.rolling(window=window_size, min_periods=1, center=True).mean()
        std_ma = std_rewards.rolling(window=window_size, min_periods=1, center=True).mean()

        x_axis = np.arange(len(mean_ma))
        ax.plot(x_axis, mean_ma, label=f"Mean {label}", color=color, linewidth=2.5)
        ax.fill_between(x_axis, mean_ma - std_ma, mean_ma + std_ma, color=color, alpha=0.2, label=f"Std Dev {label}")

    if reward_histories_single:
        plot_single_curve(reward_histories_single, "Single Epsilon", "black")

    if reward_histories_multi:
        plot_single_curve(reward_histories_multi, "Multi Epsilon", "blue")

    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel(f"Episode # (Moving Average Window = {window_size})", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n>>> Mean/Std plot successfully saved to: {filename}")
    plt.close(fig)

def plot_buffer_fractions(buffer_histories, window_size=100, filename="img/buffer_fractions.png"):
    """
    Plots the replay buffer composition for N phases dynamically.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    x_axis = np.arange(len(buffer_histories[0]))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(buffer_histories)))
    for idx, history in enumerate(buffer_histories):
        ma = pd.Series(history).rolling(window=window_size, min_periods=1, center=True).mean()
        label = "Goal" if idx == len(buffer_histories)-1 else f"WP {idx+1}"
        ax.plot(x_axis, ma, color=colors[idx], linewidth=2.5, label=f'Phase q={idx} ({label})')
    
    ax.set_title(f"Replay Buffer Composition (MA Window = {window_size})", fontsize=14, fontweight='bold')
    ax.set_ylabel("Fraction in Buffer", fontsize=12)
    ax.set_ylim(0, 1.05)
    
    ideal_balance = 1.0 / len(buffer_histories)
    ax.axhline(y=ideal_balance, color='gray', linestyle=':', alpha=0.7, label=f'Ideal Balance ({ideal_balance:.0%})')
    
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(buffer_histories)+1, fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_shaping_reward_breakdown(true_rewards, total_rewards, eps_histories, window_size=100, filename="img/shaping_reward_breakdown.png"):
    """
    Plots the moving average of rewards (True vs Total) and overlays the N-phase Epsilon decay dynamically.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Moving Average Calculation
    if len(true_rewards) >= window_size:
        true_ma = pd.Series(true_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
        total_ma = pd.Series(total_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    else:
        true_ma = true_rewards
        total_ma = total_rewards
        
    x_axis = np.arange(len(true_rewards))
        
    # Plot Rewards (Left Y-Axis)
    ax1.plot(x_axis, true_ma, color='green', linestyle='-', linewidth=2, label='True Environment Reward')
    ax1.plot(x_axis, total_ma, color='purple', linestyle='-', linewidth=2.5, label='Total Reward (Env + Shaping)')
    
    ax1.set_title(f"Shaping Agent Reward Analysis (MA Window = {window_size})", fontsize=15, fontweight='bold')
    ax1.set_xlabel("Episode #", fontsize=12)
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot Epsilon Decays (Right Y-Axis)
    ax2 = ax1.twinx()
    
    # Check if it's multi-epsilon or single epsilon history
    is_multi_eps = any(isinstance(i, list) for i in eps_histories)

    if is_multi_eps:
        num_phases = len(eps_histories)
        colors = plt.cm.plasma(np.linspace(0, 0.8, num_phases))
        for idx in range(num_phases):
            label = "Goal" if idx == num_phases - 1 else f"WP {idx + 1}"
            ax2.plot(x_axis, eps_histories[idx], color=colors[idx], linestyle='--', linewidth=2, alpha=0.8, label=f'ε Decay (q={idx}: {label})')
    else: # Single epsilon history
        # The history might be wrapped in another list, get the first element if so.
        single_eps_history = eps_histories[0] if isinstance(eps_histories[0], list) else eps_histories
        ax2.plot(x_axis, single_eps_history, color='orange', linestyle='--', linewidth=1.8, label='Epsilon Decay')

    ax2.set_ylabel("Exploration Rate (ε)", color='black', fontsize=12)
    ax2.set_ylim(-0.05, 1.05)

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