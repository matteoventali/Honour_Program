import gymnasium as gym
import numpy as np
import random as ran
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict, deque
import matplotlib.pyplot as plt

# =====================================================================
# SECTION 1: SEQUENTIAL ABSTRACT MDP
# =====================================================================

class SequentialWaypointMDP:
    """
    MDP for sequential tasks.
    The abstract state is 3D: (x, y, q)
    q = 0: searching for the waypoint (1, 8)
    q = 1: searching for the final goal (8, 8)
    """
    def __init__(self, width=12, height=12, gamma=0.99):
        self.width = width
        self.height = height
        self.gamma = gamma
        # 8-way diagonal movement
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7]
        
        # 3D State: adding the sequence variable 'q' (0 or 1)
        self.states = [(x, y, q) for x in range(width) for y in range(height) for q in (0, 1)]
        
        self.waypoint = (1, 8)
        self.goal_state = (3, 8, 1) # Final goal is only valid when q=1
        
        self.v_star = defaultdict(float)

    def get_transitions(self, state, action):
        x, y, q = state
        
        # Base movement calculation
        next_y = y
        if action in [0, 4, 5]:    next_y = min(y + 1, self.height - 1)
        elif action in [1, 6, 7]:  next_y = max(y - 1, 0)
            
        next_x = x
        if action in [2, 4, 6]:    next_x = max(x - 1, 0)
        elif action in [3, 5, 7]:  next_x = min(x + 1, self.width - 1)
        
        next_q = q
        
        # THE SEQUENTIAL LOGIC: If we touch the waypoint during q=0, we transition to q=1
        if next_x == self.waypoint[0] and next_y == self.waypoint[1] and q == 0:
            next_q = 1
            
        next_state = (next_x, next_y, next_q)
        reward = 100.0 if next_state == self.goal_state else 0.0
        
        return next_state, reward

    def value_iteration(self, theta=0.001):
        print("Solving Sequential MDP with Value Iteration...")
        
        while True:
            delta = 0
            new_v = self.v_star.copy()
            for s in self.states:
                if s == self.goal_state: 
                    continue
                v_actions = [self.get_transitions(s, a)[1] + self.gamma * self.v_star[self.get_transitions(s, a)[0]] for a in self.actions]
                best_v = max(v_actions)
                delta = max(delta, abs(best_v - self.v_star[s]))
                new_v[s] = best_v
            self.v_star = new_v
            if delta < theta: break
        
        #self.v_star[self.goal_state] = 100.0

        print("Value Iteration Complete.")


# =====================================================================
# SECTION 2: UTILS (MAPPING & INTERPOLATION)
# =====================================================================

def phi_mapping_sequential(obs, q, grid_w=12, grid_h=12):
    """Maps continuous state to 3D Abstract State (x, y, q)."""
    x, y = obs[0], obs[1]
    abstract_x = int(np.clip((x + 1) / 2 * (grid_w - 1), 0, grid_w - 1))
    abstract_y = int(np.clip(y / 1.5 * (grid_h - 1), 0, grid_h - 1))
    return (abstract_x, abstract_y, q)

def get_continuous_grid_coords(obs, grid_w=12, grid_h=12):
    x, y = obs[0], obs[1]
    px = (x + 1) / 2 * (grid_w - 1)
    py = y / 1.5 * (grid_h - 1)
    return px, py

def get_bilinear_potential_sequential(px, py, q, v_star_dict, grid_w=12, grid_h=12):
    """Calculates bilinear interpolation while keeping 'q' locked."""
    x_base = px - 0.5
    y_base = py - 0.5
    
    x1, y1 = int(np.floor(x_base)), int(np.floor(y_base))
    x2, y2 = x1 + 1, y1 + 1
    
    u, v = x_base - x1, y_base - y1
    
    x1_c, x2_c = int(np.clip(x1, 0, grid_w - 1)), int(np.clip(x2, 0, grid_w - 1))
    y1_c, y2_c = int(np.clip(y1, 0, grid_h - 1)), int(np.clip(y2, 0, grid_h - 1))
    
    Q11 = v_star_dict[(x1_c, y1_c, q)]
    Q21 = v_star_dict[(x2_c, y1_c, q)]
    Q12 = v_star_dict[(x1_c, y2_c, q)]
    Q22 = v_star_dict[(x2_c, y2_c, q)]
    
    interpolated_value = (
        Q11 * (1 - u) * (1 - v) + Q21 * u * (1 - v) +
        Q12 * (1 - u) * v + Q22 * u * v
    )
    return interpolated_value

def plot_training_results(rewards, window_size=50, title="Sequential Training Results", filename="sequential_learning_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='lightgray', alpha=0.6, label='Raw Episode Reward')
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(rewards)), moving_avg, color='red', linewidth=2.5, label=f'Moving Avg ({window_size} eps)')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Episode #')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to: {filename}")
    plt.close()

def save_sequential_heatmaps(abstract_mdp, filename_prefix="v_star", width=12, height=12):
    """
    Genera e salva DUE heatmap separate per V*: una per q=0 (ricerca waypoint) 
    e una per q=1 (ricerca goal).
    """
    os.makedirs("img/heatmaps", exist_ok=True)
    
    # Prepariamo due matrici 2D vuote
    v_matrix_q0 = np.zeros((height, width))
    v_matrix_q1 = np.zeros((height, width))
    
    # Riempiamo le matrici estraendo i valori dal dizionario 3D
    for (x, y, q), value in abstract_mdp.v_star.items():
        if 0 <= x < width and 0 <= y < height:
            if q == 0:
                v_matrix_q0[y, x] = value
            elif q == 1:
                v_matrix_q1[y, x] = value

    # Funzione interna per evitare di duplicare il codice di plottaggio
    def plot_single_heatmap(matrix, q_val, title, filename):
        plt.figure(figsize=(9, 8))
        im = plt.imshow(matrix, cmap='viridis', origin='lower')
        
        # Aggiungiamo i valori numerici su ogni cella (opzionale ma utile)
        for y in range(height):
            for x in range(width):
                val = matrix[y, x]
                if val > 0.0: 
                    text_color = 'white' if val < (np.max(matrix) / 2) else 'black'
                    plt.text(x, y, f"{val:.2f}", ha='center', va='center', 
                             color=text_color, fontsize=7, fontweight='bold')
                    
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Potential Value (V*)")
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("X (Posizione Orizzontale)", fontsize=12)
        plt.ylabel("Y (Altitudine)", fontsize=12)
        
        # Disegno della griglia
        plt.xticks(np.arange(0, width, 1))
        plt.yticks(np.arange(0, height, 1))
        ax = plt.gca()
        ax.set_xticks(np.arange(-.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-.5, height, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1, alpha=0.4)
        ax.grid(which='major', color='none')
        
        # Evidenziamo visivamente Waypoint e Goal
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

    # Disegniamo e salviamo le due mappe
    print(f" -> Generazione Mappa V* per Q=0 in corso...")
    plot_single_heatmap(v_matrix_q0, 0, "Mappa Potenziale (V*) - Fase q=0 (Cerca Waypoint)", f"{filename_prefix}_q0")
    
    print(f" -> Generazione Mappa V* per Q=1 in corso...")
    plot_single_heatmap(v_matrix_q1, 1, "Mappa Potenziale (V*) - Fase q=1 (Cerca Goal)", f"{filename_prefix}_q1")

def save_sequential_interpolated_heatmaps(abstract_mdp, filename_prefix="v_star", width=12, height=12):
    """
    Genera DUE heatmap ad alta risoluzione tramite interpolazione bilineare: 
    una per q=0 e una per q=1.
    """
    os.makedirs("img/heatmaps", exist_ok=True)
    
    # 1. Imposta la risoluzione per avere un'immagine fluida (es. 20 punti per cella)
    resolution = 20 
    x_continuous = np.linspace(0, width, width * resolution)
    y_continuous = np.linspace(0, height, height * resolution)
    
    # Matrici ad alta risoluzione
    Z_q0 = np.zeros((len(y_continuous), len(x_continuous)))
    Z_q1 = np.zeros((len(y_continuous), len(x_continuous)))
    
    # 2. Calcola il potenziale interpolato per ogni micro-punto e per entrambe le fasi
    print(" -> Generazione matrici interpolate in corso...")
    for i, py in enumerate(y_continuous):
        for j, px in enumerate(x_continuous):
            Z_q0[i, j] = get_bilinear_potential_sequential(px, py, 0, abstract_mdp.v_star, width, height)
            Z_q1[i, j] = get_bilinear_potential_sequential(px, py, 1, abstract_mdp.v_star, width, height)

    # 3. Funzione interna per il plottaggio standardizzato
    def plot_smooth_heatmap(Z_matrix, q_val, title, filename):
        plt.figure(figsize=(10, 9))
        im = plt.imshow(Z_matrix, cmap='viridis', origin='lower', extent=[0, width, 0, height], interpolation='none')
        
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Potential Value (V*) Interpolato")
        plt.title(title, fontsize=15, fontweight='bold')
        plt.xlabel("X (Posizione Orizzontale)", fontsize=13)
        plt.ylabel("Y (Altitudine)", fontsize=13)
        
        # Disegna la griglia fisica
        plt.xticks(np.arange(0, width + 1, 1))
        plt.yticks(np.arange(0, height + 1, 1))
        plt.grid(color='white', linestyle='-', linewidth=1, alpha=0.3)
        
        # Segna i centri logici delle celle
        cx = [x + 0.5 for x in range(width) for _ in range(height)]
        cy = [y + 0.5 for _ in range(width) for y in range(height)]
        plt.scatter(cx, cy, color='black', s=8, alpha=0.4, label="Centri delle celle")
        
        # Evidenzia Waypoint e Goal
        if q_val == 0:
            way_x, way_y = abstract_mdp.waypoint
            # Aggiungiamo +0.5 per centrare il pallino nel mezzo della cella sul grafico continuo
            plt.plot(way_x + 0.5, way_y + 0.5, 'ro', markersize=18, alpha=0.8, markeredgecolor='white', label="Waypoint (q=0)")
        elif q_val == 1:
            goal_x, goal_y, _ = abstract_mdp.goal_state
            plt.plot(goal_x + 0.5, goal_y + 0.5, 'go', markersize=18, alpha=0.8, markeredgecolor='white', label="Final Goal (q=1)")
            
        plt.legend(loc="upper right", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"img/heatmaps/{filename}.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Genera e salva le immagini
    print(" -> Salvataggio mappa q=0...")
    plot_smooth_heatmap(Z_q0, 0, "V* Interpolata - Fase q=0 (Cerca Waypoint)", f"{filename_prefix}_q0_smooth")
    
    print(" -> Salvataggio mappa q=1...")
    plot_smooth_heatmap(Z_q1, 1, "V* Interpolata - Fase q=1 (Cerca Goal)", f"{filename_prefix}_q1_smooth")

# =====================================================================
# SECTION 3: REINFORCEMENT LEARNING AGENT
# =====================================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = ran.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class HierarchicalDQNLearner:
    def __init__(self, env, max_episodes=1000, gamma=0.99, policy_name="policy", use_ddqn=True):
        self.env = env
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.policy_name = policy_name
        self.use_ddqn = use_ddqn
        
        self.batch_size = 64
        self.lr = 1e-3
        self.tau = 0.005 
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995
        
        # INCREASING STATE DIMENSION BY 1 TO ACCOMMODATE 'q'
        state_dim = self.env.observation_space.shape[0] + 1
        action_dim = self.env.action_space.n
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        
        if self.use_ddqn:
            self.target_net = QNetwork(state_dim, action_dim).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(capacity=100000)

    def select_action(self, state):
        if ran.random() < self.eps:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size: return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            if self.use_ddqn:
                best_actions = self.policy_net(next_states).argmax(dim=1).unsqueeze(1)
                next_q_values = self.target_net(next_states).gather(1, best_actions)
            else:
                next_q_values = self.policy_net(next_states).max(1)[0].unsqueeze(1)
                
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.use_ddqn:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def save_policy(self):
        os.makedirs("./policy", exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"./policy/{self.policy_name}")


# =====================================================================
# SECTION 4: MAIN TRAINING LOOP
# =====================================================================

def run_sequential_training(env, agent, abstract_mdp, episodes, use_shaping=True):
    true_episode_rewards = []
    K = 1.0  # Reward Shaping Scaling Constant

    for n_episode in range(episodes):
        s_raw, _ = env.reset()
        
        # Initialize sequence variable 'q'
        q = 0 
        
        # Augment state for the Neural Network (from 8 to 9 dimensions)
        s_aug = np.append(s_raw, q) 
        
        terminated = truncated = False
        episode_true_reward = 0.0
        
        while not (terminated or truncated):
            # Agent acts based on the augmented state
            a = agent.select_action(s_aug)
            ns_raw, _, terminated, truncated, _ = env.step(a)
            
            # Map continuous state to abstract state
            abstract_x_curr, abstract_y_curr, _ = phi_mapping_sequential(ns_raw, q)
            next_q = q

            # STATE TRANSITION LOGIC
            if abstract_x_curr == 1 and abstract_y_curr == 8 and q == 0:
                next_q = 1
            
            ns_aug = np.append(ns_raw, next_q)
            abstract_ns = (abstract_x_curr, abstract_y_curr, next_q)

            # Final Goal Check
            env_goal_reward = 0.0

            if (abstract_ns[0], abstract_ns[1]) == abstract_mdp.waypoint and q == 0:
                print("Waypoint reached")
            
            if abstract_ns == abstract_mdp.goal_state:
                env_goal_reward = 100.0
                print("Final point reached")
                terminated = True
            
            done = terminated or truncated
            episode_true_reward += env_goal_reward

            # Shaping Signal Calculation
            if use_shaping:
                px_s, py_s = get_continuous_grid_coords(s_raw, abstract_mdp.width, abstract_mdp.height)
                px_ns, py_ns = get_continuous_grid_coords(ns_raw, abstract_mdp.width, abstract_mdp.height)
                
                phi_s = get_bilinear_potential_sequential(px_s, py_s, q, abstract_mdp.v_star)
                phi_ns = get_bilinear_potential_sequential(px_ns, py_ns, next_q, abstract_mdp.v_star)
                
                shaping_signal = K * (agent.gamma * phi_ns - phi_s)

                if q == 1:
                    print(
                        f"real=({ns_raw[0]:.3f},{ns_raw[1]:.3f}) | abstract={abstract_ns}"
                    )
            else:
                shaping_signal = 0.0
            
            total_step_reward = env_goal_reward + shaping_signal
            
            # Push AUGMENTED states to memory
            agent.memory.push(s_aug, a, total_step_reward, ns_aug, done)
            agent.optimize_model()

            s_raw = ns_raw
            s_aug = ns_aug
            q = next_q

        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        true_episode_rewards.append(episode_true_reward)

        if truncated:
            print("TRUNCATED")

        if (n_episode + 1) % 100 == 0:
            recent_avg = np.mean(true_episode_rewards[-100:])
            print(f"-> Progress: Episode {n_episode + 1}/{episodes} | 100-eps Avg Reward: {recent_avg:6.2f} | Epsilon: {agent.eps:.3f}")
            agent.save_policy()
            
    return np.array(true_episode_rewards)

def main():
    print("=== STARTING SEQUENTIAL TASK EXPERIMENT ===")
    episodes = 10000
    gamma = 0.8
    
    print("\n1. Initializing Environment and Abstract MDP...")
    env = gym.make("LunarLander-v3", continuous=False)
    
    abstract_mdp = SequentialWaypointMDP(width=12, height=12, gamma=gamma)
    abstract_mdp.value_iteration()

    print("\n -> Plotting Value Functions...")
    save_sequential_heatmaps(abstract_mdp, filename_prefix="seq_experiment_normal")
    save_sequential_interpolated_heatmaps(abstract_mdp, filename_prefix="seq_experiment")
    
    print("\n2. Initializing RL Agent...")
    agent = HierarchicalDQNLearner(
        env=env,
        max_episodes=episodes,
        gamma=gamma,
        use_ddqn=True,
        policy_name="sequential_policy.pth"
    )
    
    print("\n3. Starting Training Loop...")
    learning_curve = run_sequential_training(env, agent, abstract_mdp, episodes, use_shaping=True)
    
    print("\n4. Saving Results...")
    plot_training_results(learning_curve, window_size=50)
    print("=== EXPERIMENT COMPLETE ===")
    
    env.close()

if __name__ == "__main__":
    main()