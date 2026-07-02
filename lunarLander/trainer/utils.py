import os
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# SECTION 1: STATE MAPPING FUNCTIONS
# =====================================================================

def phi_mapping_grid(obs, grid_w=12, grid_h=12):
    """
    Maps the continuous LunarLander state to a 2D Abstract Grid State.
    """
    x, y = obs[0], obs[1]
    abstract_x = int(np.clip((x + 1) / 2 * (grid_w - 1), 0, grid_w - 1))
    abstract_y = int(np.clip(y / 1.5 * (grid_h - 1), 0, grid_h - 1))
    return abstract_x, abstract_y

def phi_mapping_sequential(obs, q, grid_w=12, grid_h=12):
    """
    Maps the continuous state to a 3D Abstract State (x, y, q).
    Reuses the 2D grid mapping logic to maintain consistency.
    """
    abstract_x, abstract_y = phi_mapping_grid(obs, grid_w, grid_h)
    return abstract_x, abstract_y, q

def get_continuous_grid_coords(obs, grid_w=12, grid_h=12):
    """
    Converts raw observation coordinates into continuous grid coordinates.
    """
    x, y = obs[0], obs[1]
    px = (x + 1) / 2 * (grid_w - 1)
    py = y / 1.5 * (grid_h - 1)
    return px, py

# Backup old
def phi_mapping_kinematic(obs, grid_w=12, grid_h=12):
    """
    Maps the continuous LunarLander state to a 4D Abstract State: (x, y, vy, angle)
    Expected obs format: [x, y, vx, vy, angle, angular_velocity, left_leg, right_leg]
    """
    x, y, vx, vy, angle, v_angle, left_leg, right_leg = obs
    
    abstract_x, abstract_y = phi_mapping_grid(obs, grid_w, grid_h)
    
    # Vertical Velocity (vy) discretization
    if vy < -0.4:
        abstract_vy = 0     # Falling too fast
    elif vy > 0.1:
        abstract_vy = 2     # Going up
    else:
        abstract_vy = 1     # Safe landing speed
        
    # Angle (theta) discretization
    if angle < -0.2:
        abstract_angle = 0  # Tilted Left
    elif angle > 0.2:
        abstract_angle = 2  # Tilted Right
    else:
        abstract_angle = 1  # Straight
        
    return abstract_x, abstract_y, abstract_vy, abstract_angle

def get_continuous_grid_coords_alternative(obs, grid_w=12, grid_h=12):
    """
    Mappa l'intero spazio giocabile nativo di LunarLander su coordinate continue.
    X copre il range orizzontale [-2.5, 2.5]. 
    Y copre il range verticale dal suolo [0.0] al tetto [2.5].
    """
    x, y = obs[0], obs[1]
    
    # Normalizzazione Asse X:
    # (x + 2.5) trasla il range da [-2.5, 2.5] a [0.0, 5.0].
    # Dividendo per 5.0 otteniamo una percentuale [0, 1], poi moltiplicata per grid_w.
    px = ((x + 2.5) / 5.0) * grid_w
    
    # Normalizzazione Asse Y:
    # y traslato da [0.0, 2.5] a una percentuale [0, 1], moltiplicata per grid_h.
    py = (y / 2.5) * grid_h
    
    # Clipping di Sicurezza:
    # Previene errori matematici (es. out-of-bounds nell'interpolazione) se il 
    # motore fisico di Gym genera collisioni che spingono il Lander fuori mappa.
    px = np.clip(px, 0.0, grid_w - 0.001)
    py = np.clip(py, 0.0, grid_h - 0.001)
    
    return px, py

def phi_mapping_grid_alternative(obs, grid_w=12, grid_h=12):
    """
    Converte le coordinate continue della griglia nel rispettivo stato astratto discreto.
    """
    px, py = get_continuous_grid_coords(obs, grid_w, grid_h)
    
    # Estraiamo l'indice della cella arrotondando per difetto.
    # Grazie al clipping precedente, questi valori non supereranno mai (grid_w - 1).
    abstract_x = int(np.floor(px))
    abstract_y = int(np.floor(py))
    
    return abstract_x, abstract_y

# =====================================================================
# SECTION 2: POTENTIAL & INTERPOLATION FUNCTIONS
# =====================================================================

def get_bilinear_potential(px, py, v_star_dict, grid_w=12, grid_h=12):
    """
    Calculates the bilinearly interpolated potential for a 2D state.
    Uses the v_star_dict from the abstract MDP (defaults to 0.0 if missing).
    """
    # 1. Shift to align with cell centers (located at 0.5, 0.5)
    x_base, y_base = px - 0.5, py - 0.5
    
    # 2. Bottom-left integer indices
    x1, y1 = int(np.floor(x_base)), int(np.floor(y_base))
    
    # 3. Top-right integer indices
    x2, y2 = x1 + 1, y1 + 1
    
    # 4. Interpolation weights (distances from the bottom-left center)
    u, v = x_base - x1, y_base - y1
    
    # 5. Clamping edges to prevent out-of-bounds lookups
    x1_c, x2_c = int(np.clip(x1, 0, grid_w - 1)), int(np.clip(x2, 0, grid_w - 1))
    y1_c, y2_c = int(np.clip(y1, 0, grid_h - 1)), int(np.clip(y2, 0, grid_h - 1))
    
    # 6. Extract corner values
    Q11 = v_star_dict.get((x1_c, y1_c), 0.0)
    Q21 = v_star_dict.get((x2_c, y1_c), 0.0)
    Q12 = v_star_dict.get((x1_c, y2_c), 0.0)
    Q22 = v_star_dict.get((x2_c, y2_c), 0.0)
    
    # 7. Bilinear interpolation formula
    interpolated_value = (
        Q11 * (1 - u) * (1 - v) +
        Q21 * u * (1 - v) +
        Q12 * (1 - u) * v +
        Q22 * u * v
    )
    
    return interpolated_value

def get_bilinear_potential_sequential(px, py, q, v_star_dict, grid_w=12, grid_h=12):
    """
    Calculates the bilinearly interpolated potential for a 3D sequential state.
    Keeps the sequence identifier 'q' locked during interpolation.
    """
    x_base, y_base = px - 0.5, py - 0.5
    
    x1, y1 = int(np.floor(x_base)), int(np.floor(y_base))
    x2, y2 = x1 + 1, y1 + 1
    
    u, v = x_base - x1, y_base - y1
    
    x1_c, x2_c = int(np.clip(x1, 0, grid_w - 1)), int(np.clip(x2, 0, grid_w - 1))
    y1_c, y2_c = int(np.clip(y1, 0, grid_h - 1)), int(np.clip(y2, 0, grid_h - 1))
    
    Q11 = v_star_dict.get((x1_c, y1_c, q), 0.0)
    Q21 = v_star_dict.get((x2_c, y1_c, q), 0.0)
    Q12 = v_star_dict.get((x1_c, y2_c, q), 0.0)
    Q22 = v_star_dict.get((x2_c, y2_c, q), 0.0)
    
    interpolated_value = (
        Q11 * (1 - u) * (1 - v) +
        Q21 * u * (1 - v) +
        Q12 * (1 - u) * v +
        Q22 * u * v
    )
    return interpolated_value

def get_idw_potential(px, py, v_star_dict, grid_w=12, grid_h=12, p=2.0):
    """
    Calcola il potenziale continuo usando l'Inverse Distance Weighting (IDW) 
    basandosi sui centri delle 4 celle adiacenti. 
    Garantisce la restituzione del valore V* esatto se calcolato al centro di una cella.
    """
    # 1. Spostiamo il sistema di riferimento per ancorarci ai centri geometrici
    x_base, y_base = px - 0.5, py - 0.5
    
    # 2. Troviamo gli indici interi delle 4 celle adiacenti
    x1, y1 = int(np.floor(x_base)), int(np.floor(y_base))
    x2, y2 = x1 + 1, y1 + 1
    
    # 3. Applichiamo il clipping per gestire i margini della mappa (evita errori out-of-bounds)
    x1_c, x2_c = int(np.clip(x1, 0, grid_w - 1)), int(np.clip(x2, 0, grid_w - 1))
    y1_c, y2_c = int(np.clip(y1, 0, grid_h - 1)), int(np.clip(y2, 0, grid_h - 1))
    
    # 4. Definiamo le coordinate fisiche (continue) dei 4 centri e i rispettivi valori V*
    centers = [
        (x1_c + 0.5, y1_c + 0.5, v_star_dict.get((x1_c, y1_c), 0.0)), # In basso a sinistra
        (x2_c + 0.5, y1_c + 0.5, v_star_dict.get((x2_c, y1_c), 0.0)), # In basso a destra
        (x1_c + 0.5, y2_c + 0.5, v_star_dict.get((x1_c, y2_c), 0.0)), # In alto a sinistra
        (x2_c + 0.5, y2_c + 0.5, v_star_dict.get((x2_c, y2_c), 0.0))  # In alto a destra
    ]
    
    numerator = 0.0
    denominator = 0.0
    epsilon = 1e-8 # Tolleranza vitale per evitare la divisione per zero
    
    # 5. Applichiamo la formula IDW
    for cx, cy, v in centers:
        # Calcolo distanza euclidea tra il Lander e il centro corrente
        dist = np.sqrt((px - cx)**2 + (py - cy)**2)
        
        # Se il Lander è esattamente (o quasi) sul centro, bypassa la formula 
        # e restituisci il valore discreto puro.
        if dist < epsilon:
            return v
            
        # Calcolo del peso (inversamente proporzionale alla distanza elevata a p)
        weight = 1.0 / (dist ** p)
        
        numerator += weight * v
        denominator += weight
        
    return numerator / denominator

def get_idw_potential_sequential(px, py, q, v_star_dict, grid_w=12, grid_h=12, p=2.0):
    """
    Calcola il potenziale continuo IDW per uno stato sequenziale 3D (x, y, q).
    Mantiene l'identificatore di sequenza 'q' bloccato durante l'interpolazione spaziale.
    """
    x_base, y_base = px - 0.5, py - 0.5
    
    x1, y1 = int(np.floor(x_base)), int(np.floor(y_base))
    x2, y2 = x1 + 1, y1 + 1
    
    x1_c, x2_c = int(np.clip(x1, 0, grid_w - 1)), int(np.clip(x2, 0, grid_w - 1))
    y1_c, y2_c = int(np.clip(y1, 0, grid_h - 1)), int(np.clip(y2, 0, grid_h - 1))
    
    # Nota l'uso di (..., q) per l'estrazione dal dizionario V*
    centers = [
        (x1_c + 0.5, y1_c + 0.5, v_star_dict.get((x1_c, y1_c, q), 0.0)),
        (x2_c + 0.5, y1_c + 0.5, v_star_dict.get((x2_c, y1_c, q), 0.0)),
        (x1_c + 0.5, y2_c + 0.5, v_star_dict.get((x1_c, y2_c, q), 0.0)),
        (x2_c + 0.5, y2_c + 0.5, v_star_dict.get((x2_c, y2_c, q), 0.0))
    ]
    
    numerator = 0.0
    denominator = 0.0
    epsilon = 1e-8
    
    for cx, cy, v in centers:
        dist = np.sqrt((px - cx)**2 + (py - cy)**2)
        if dist < epsilon:
            return v
            
        weight = 1.0 / (dist ** p)
        numerator += weight * v
        denominator += weight
        
    return numerator / denominator

# =====================================================================
# SECTION 3: VISUALIZATION & PLOTTING
# =====================================================================

def plot_training_results(rewards, window_size=100, title="Training Results", ylabel="Reward", filename=None):
    """
    Plots the raw rewards and a moving average to show the learning trend.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(rewards, color='lightgray', alpha=0.6, label='Raw Episode Reward')
    
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(rewards)), moving_avg, color='red', linewidth=2.5, label=f'Moving Average ({window_size} eps)')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Episode #')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"Plot saved to: {filename}")
    else:
        plt.show()
    plt.close()

def save_sequential_heatmaps(abstract_mdp, filename_prefix="v_star", width=12, height=12, vmin = 0, vmax = 100):
    """
    Generates and saves TWO separate heatmaps for V*: 
    One for q=0 (searching waypoint) and one for q=1 (searching final goal).
    """
    os.makedirs("img/heatmaps", exist_ok=True)
    
    v_matrix_q0 = np.zeros((height, width))
    v_matrix_q1 = np.zeros((height, width))
    
    # Populate matrices from the 3D dictionary
    for (x, y, q), value in abstract_mdp.v_star.items():
        if 0 <= x < width and 0 <= y < height:
            if q == 0:
                v_matrix_q0[y, x] = value
            elif q == 1:
                v_matrix_q1[y, x] = value

    all_values = np.concatenate([
        v_matrix_q0.flatten(), 
        v_matrix_q1.flatten()
    ])

    # Calcola i limiti globali
    computed_vmin = all_values.min()
    computed_vmax = all_values.max()

    def plot_single_heatmap(matrix, q_val, title, filename):
        plt.figure(figsize=(9, 8))
        im = plt.imshow(matrix, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        
        # Overlay numerical values on cells
        for y in range(height):
            for x in range(width):
                val = matrix[y, x]
                if val > 0.0: 
                    text_color = 'white' if val < (np.max(matrix) / 2) else 'black'
                    plt.text(x, y, f"{val:.2f}", ha='center', va='center', 
                             color=text_color, fontsize=7, fontweight='bold')
                    
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Potential Value (V*)")
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("X (Horizontal Position)", fontsize=12)
        plt.ylabel("Y (Altitude)", fontsize=12)
        
        # Grid setup
        plt.xticks(np.arange(0, width, 1))
        plt.yticks(np.arange(0, height, 1))
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-.5, height, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1, alpha=0.4)
        ax.grid(which='major', color='none')
        
        # Visually highlight Waypoint and Goal
        if q_val == 0:
            way_x, way_y = abstract_mdp.waypoint
            plt.plot(way_x, way_y, 'ro', markersize=15, alpha=0.6, label="Waypoint (Target Q0)")
            plt.legend(loc="upper right")
        elif q_val == 1:
            goal_x, goal_y, _ = abstract_mdp.goal_state
            plt.plot(goal_x, goal_y, 'go', markersize=15, alpha=0.6, label="Final Goal (Target Q1)")
            plt.legend(loc="upper right")
            
        plt.tight_layout()
        plt.savefig(f"img/heatmaps/{filename}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(" -> Generating V* Heatmap for Q=0...")
    plot_single_heatmap(v_matrix_q0, 0, "Potential Map (V*) - Phase q=0 (Seek Waypoint)", f"{filename_prefix}_q0")
    
    print(" -> Generating V* Heatmap for Q=1...")
    plot_single_heatmap(v_matrix_q1, 1, "Potential Map (V*) - Phase q=1 (Seek Goal)", f"{filename_prefix}_q1")

def save_sequential_idw_heatmaps(abstract_mdp, filename_prefix="v_star", width=12, height=12, vmin=0, vmax=100, p=2.0):
    """
    Genera DUE heatmap ad alta risoluzione usando l'Inverse Distance Weighting (IDW):
    Una per q=0 (Seek Waypoint) e una per q=1 (Seek Goal).
    """
    os.makedirs("img/heatmaps", exist_ok=True)
    
    # Alta risoluzione per una visualizzazione fluida (20 punti per cella)
    resolution = 20 
    x_continuous = np.linspace(0, width, width * resolution)
    y_continuous = np.linspace(0, height, height * resolution)
    
    Z_q0 = np.zeros((len(y_continuous), len(x_continuous)))
    Z_q1 = np.zeros((len(y_continuous), len(x_continuous)))
    
    print(f" -> Generazione matrici IDW interpolate (p={p})...")
    for i, py in enumerate(y_continuous):
        for j, px in enumerate(x_continuous):
            Z_q0[i, j] = get_idw_potential_sequential(px, py, 0, abstract_mdp.v_star, width, height, p=p)
            Z_q1[i, j] = get_idw_potential_sequential(px, py, 1, abstract_mdp.v_star, width, height, p=p)

    def plot_smooth_heatmap(Z_matrix, q_val, title, filename):
        plt.figure(figsize=(10, 9))
        im = plt.imshow(Z_matrix, cmap='viridis', origin='lower', extent=[0, width, 0, height], interpolation='none', vmin=vmin, vmax=vmax)
        
        plt.colorbar(im, fraction=0.046, pad=0.04, label="IDW Interpolated Potential Value (V*)")
        plt.title(title, fontsize=15, fontweight='bold')
        plt.xlabel("X (Horizontal Position)", fontsize=13)
        plt.ylabel("Y (Altitude)", fontsize=13)
        
        # Griglia fisica
        plt.xticks(np.arange(0, width + 1, 1))
        plt.yticks(np.arange(0, height + 1, 1))
        plt.grid(color='white', linestyle='-', linewidth=1, alpha=0.3)
        
        # Centri logici delle celle (che ora fungono da ancoraggi esatti per l'IDW)
        cx = [x + 0.5 for x in range(width) for _ in range(height)]
        cy = [y + 0.5 for _ in range(width) for y in range(height)]
        plt.scatter(cx, cy, color='black', s=8, alpha=0.4, label="Cell Centers (Anchors)")
        
        # Highlight Waypoint e Goal
        if q_val == 0:
            way_x, way_y = abstract_mdp.waypoint
            plt.plot(way_x + 0.5, way_y + 0.5, 'ro', markersize=18, alpha=0.8, markeredgecolor='white', label="Waypoint (q=0)")
        elif q_val == 1:
            goal_x, goal_y, _ = abstract_mdp.goal_state
            plt.plot(goal_x + 0.5, goal_y + 0.5, 'go', markersize=18, alpha=0.8, markeredgecolor='white', label="Final Goal (q=1)")
            
        plt.legend(loc="upper right", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"img/heatmaps/{filename}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(" -> Salvataggio heatmap IDW per q=0...")
    plot_smooth_heatmap(Z_q0, 0, f"IDW V* - Phase q=0 (Seek Waypoint) | p={p}", f"{filename_prefix}_q0_idw")
    
    print(" -> Salvataggio heatmap IDW per q=1...")
    plot_smooth_heatmap(Z_q1, 1, f"IDW V* - Phase q=1 (Seek Goal) | p={p}", f"{filename_prefix}_q1_idw")

def save_sequential_interpolated_heatmaps(abstract_mdp, filename_prefix="v_star", width=12, height=12, vmin = 0, vmax = 100):
    """
    Generates TWO high-resolution heatmaps using bilinear interpolation:
    One for q=0 and one for q=1.
    """
    os.makedirs("img/heatmaps", exist_ok=True)
    
    # Higher resolution for smooth visualization (e.g., 20 points per cell)
    resolution = 20 
    x_continuous = np.linspace(0, width, width * resolution)
    y_continuous = np.linspace(0, height, height * resolution)
    
    Z_q0 = np.zeros((len(y_continuous), len(x_continuous)))
    Z_q1 = np.zeros((len(y_continuous), len(x_continuous)))
    
    print(" -> Generating interpolated matrices...")
    for i, py in enumerate(y_continuous):
        for j, px in enumerate(x_continuous):
            Z_q0[i, j] = get_bilinear_potential_sequential(px, py, 0, abstract_mdp.v_star, width, height)
            Z_q1[i, j] = get_bilinear_potential_sequential(px, py, 1, abstract_mdp.v_star, width, height)

    all_values = np.concatenate([
        Z_q0.flatten(), 
        Z_q1.flatten()
    ])

    # Calcola i limiti globali
    computed_vmin = all_values.min()
    computed_vmax = all_values.max()

    def plot_smooth_heatmap(Z_matrix, q_val, title, filename):
        plt.figure(figsize=(10, 9))
        im = plt.imshow(Z_matrix, cmap='viridis', origin='lower', extent=[0, width, 0, height], interpolation='none', vmin=vmin, vmax=vmax)
        
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Interpolated Potential Value (V*)")
        plt.title(title, fontsize=15, fontweight='bold')
        plt.xlabel("X (Horizontal Position)", fontsize=13)
        plt.ylabel("Y (Altitude)", fontsize=13)
        
        # Physical grid
        plt.xticks(np.arange(0, width + 1, 1))
        plt.yticks(np.arange(0, height + 1, 1))
        plt.grid(color='white', linestyle='-', linewidth=1, alpha=0.3)
        
        # Logical cell centers
        cx = [x + 0.5 for x in range(width) for _ in range(height)]
        cy = [y + 0.5 for _ in range(width) for y in range(height)]
        plt.scatter(cx, cy, color='black', s=8, alpha=0.4, label="Cell Centers")
        
        # Highlight Waypoint and Goal (offset by +0.5 to center in continuous space)
        if q_val == 0:
            way_x, way_y = abstract_mdp.waypoint
            plt.plot(way_x + 0.5, way_y + 0.5, 'ro', markersize=18, alpha=0.8, markeredgecolor='white', label="Waypoint (q=0)")
        elif q_val == 1:
            goal_x, goal_y, _ = abstract_mdp.goal_state
            plt.plot(goal_x + 0.5, goal_y + 0.5, 'go', markersize=18, alpha=0.8, markeredgecolor='white', label="Final Goal (q=1)")
            
        plt.legend(loc="upper right", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"img/heatmaps/{filename}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(" -> Saving heatmap for q=0...")
    plot_smooth_heatmap(Z_q0, 0, "Interpolated V* - Phase q=0 (Seek Waypoint)", f"{filename_prefix}_q0_smooth")
    
    print(" -> Saving heatmap for q=1...")
    plot_smooth_heatmap(Z_q1, 1, "Interpolated V* - Phase q=1 (Seek Goal)", f"{filename_prefix}_q1_smooth")