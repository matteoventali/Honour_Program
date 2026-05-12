import numpy as np
import matplotlib.pyplot as plt

def plot_training_results(rewards, window_size=100, title="Training Results", ylabel="Reward", filename=None):
    """
    Plots the raw rewards and a moving average to show the learning trend.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(rewards, color='lightgray', alpha=0.6, label='Raw Episode Reward')
    
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(rewards)), moving_avg, color='red', linewidth=2.5, label=f'Moving Average ({window_size} eps)')
    
    plt.title(title)
    plt.xlabel('Episode #')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        print(f"Plot saved to: {filename}")
    else:
        plt.show()
    plt.close()

def phi_mapping_grid(obs, grid_w=12, grid_h=12):
    """
    Maps continuous LunarLander state to 2D Abstract Grid State.
    """
    x, y = obs[0], obs[1]
    abstract_x = int(np.clip((x + 1) / 2 * (grid_w - 1), 0, grid_w - 1))
    abstract_y = int(np.clip(y / 1.5 * (grid_h - 1), 0, grid_h - 1))
    return (abstract_x, abstract_y)

def phi_mapping_kinematic(obs, grid_w=12, grid_h=12):
    """
    Maps continuous LunarLander state to a 4D Abstract State: (x, y, vy, angle)
    obs: [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]
    """
    x, y, vx, vy, angle, v_angle, left_leg, right_leg = obs
    
    abstract_x = int(np.clip((x + 1) / 2 * (grid_w - 1), 0, grid_w - 1))
    abstract_y = int(np.clip(y / 1.5 * (grid_h - 1), 0, grid_h - 1))
    
    # Vertical Velocity (vy)
    if vy < -0.4: abstract_vy = 0   # Falling too fast
    elif vy > 0.1: abstract_vy = 2  # Going up
    else: abstract_vy = 1           # Safe landing speed
        
    # Angle (theta)
    if angle < -0.2: abstract_angle = 0   # Tilted Left
    elif angle > 0.2: abstract_angle = 2  # Tilted Right
    else: abstract_angle = 1              # Straight
        
    return (abstract_x, abstract_y, abstract_vy, abstract_angle)