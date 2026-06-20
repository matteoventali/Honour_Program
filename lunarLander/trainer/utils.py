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


def get_continuous_grid_coords(obs, grid_w=12, grid_h=12):
    x, y = obs[0], obs[1]
    px = (x + 1) / 2 * (grid_w - 1)
    py = y / 1.5 * (grid_h - 1)
    return px, py

def get_bilinear_potential(px, py, v_star_dict, grid_w=12, grid_h=12):
    """
    Calcola il potenziale interpolato bilinearmente.
    Usa direttamente il dizionario v_star_dict dell'abstract MDP.
    """
    # 1. Trasliamo per allinearci ai centri delle celle (situati a 0.5, 0.5)
    x_base = px - 0.5
    y_base = py - 0.5
    
    # 2. Indici interi del centro in basso a sinistra
    x1 = int(np.floor(x_base))
    y1 = int(np.floor(y_base))
    
    # 3. Indici del centro in alto a destra
    x2 = x1 + 1
    y2 = y1 + 1
    
    # 4. Pesi (distanze dal centro in basso a sinistra)
    u = x_base - x1
    v = y_base - y1
    
    # 5. Clamping per i bordi: impedisce di cercare fuori dalla griglia
    x1_c = int(np.clip(x1, 0, grid_w - 1))
    x2_c = int(np.clip(x2, 0, grid_w - 1))
    y1_c = int(np.clip(y1, 0, grid_h - 1))
    y2_c = int(np.clip(y2, 0, grid_h - 1))
    
    # 6. Estraiamo i 4 valori dai vertici (v_star_dict restituisce 0.0 di default se manca)
    Q11 = v_star_dict[(x1_c, y1_c)]
    Q21 = v_star_dict[(x2_c, y1_c)]
    Q12 = v_star_dict[(x1_c, y2_c)]
    Q22 = v_star_dict[(x2_c, y2_c)]
    
    # 7. Formula matematica dell'interpolazione
    interpolated_value = (
        Q11 * (1 - u) * (1 - v) +
        Q21 * u * (1 - v) +
        Q12 * (1 - u) * v +
        Q22 * u * v
    )
    
    return interpolated_value