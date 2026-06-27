import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

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

def run_policy(policy_path, env_name="LunarLander-v3", grid_w=12, grid_h=12):
    # 1. Setup Ambiente
    env = gym.make(env_name, render_mode="human")
    
    # 2. Caricamento della Rete (stessa architettura di HierarchicalDQNLearner)
    # Nota: la dimensione è 9 perché avevamo aggiunto la variabile 'q'
    state_dim = env.observation_space.shape[0] + 1
    action_dim = env.action_space.n
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = QNetwork(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load(policy_path, map_location=device))
    policy_net.eval() # Modalità valutazione (fondamentale!)
    
    print(f"Policy caricata da: {policy_path}")
    
    # 3. Ciclo di test
    s_raw, _ = env.reset()
    q = 0 # Iniziamo dalla fase di ricerca waypoint
    s_aug = np.append(s_raw, q)
    
    terminated = truncated = False
    total_reward = 0
    
    print("Inizio esecuzione...")
    while not (terminated or truncated):
        # Inferenza pura (niente eps-greedy)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(s_aug).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        ns_raw, reward, terminated, truncated, _ = env.step(action)
        
        # LOGICA DI STATO (Dobbiamo replicare la transizione di fase anche qui)
        from utils import phi_mapping_grid
        abs_x, abs_y = phi_mapping_grid(ns_raw, grid_w, grid_h)
        
        if q == 0 and abs_x == 1 and abs_y == 8:
            q = 1
            print(">>> Waypoint (1,8) raggiunto! Passaggio a fase q=1.")
            
        s_aug = np.append(ns_raw, q)
        total_reward += reward
        
        # Piccolo delay per vedere l'animazione umana
        time.sleep(0.02)
        
    print(f"Episodio terminato. Ricompensa totale: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    # Sostituisci con il nome del tuo file policy
    
    for i in range(0, 100):
        run_policy("./policy/sequential_policy.pth")