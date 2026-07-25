import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn

import tianshou as ts
from tianshou.policy import DiscreteSACPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv

# Importiamo le classi base che non cambiano
from abstract_mdps import LTLfAutomaton, LTLfWaypointMDP
from utils import phi_mapping_sequential, save_sequential_heatmaps

# Variabili globali per estrarre facilmente i dati di logging per i nostri plot custom
global_true_rewards = []
global_total_rewards = []

# ==========================================
# 1. FUNZIONI DI PLOTTING AGGIORNATE
# ==========================================
def plot_shaping_reward_breakdown(true_rewards, total_rewards, window_size=100, filename="img/shaping_reward_breakdown.png"):
    """
    Versione aggiornata per SAC: rimosso il doppio asse per l'Epsilon.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    if len(true_rewards) >= window_size:
        true_ma = pd.Series(true_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
        total_ma = pd.Series(total_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    else:
        true_ma = true_rewards
        total_ma = total_rewards
        
    x_axis = np.arange(len(true_rewards))
        
    ax1.plot(x_axis, true_ma, color='green', linestyle='-', linewidth=2, label='True Environment Reward')
    ax1.plot(x_axis, total_ma, color='purple', linestyle='-', linewidth=2.5, label='Total Reward (Env + Shaping)')
    
    ax1.set_title(f"Discrete SAC Reward Analysis (MA Window = {window_size})", fontsize=15, fontweight='bold')
    ax1.set_xlabel("Episode #", fontsize=12)
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=11, framealpha=1.0)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f">>> Plot salvato in: {filename}")

def plot_comparison_curves(baseline_rewards, shaping_rewards, window_size=100, filename="img/baseline_vs_shaping.png"):
    """
    Confronto diretto senza parametri epsilon.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    baseline_ma = pd.Series(baseline_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    shaping_ma = pd.Series(shaping_rewards).rolling(window=window_size, min_periods=1, center=True).mean()
    x_axis = np.arange(len(baseline_rewards))
    
    ax1.plot(x_axis, baseline_ma, color='black', linestyle='-', linewidth=2, label="Discrete SAC (No Shaping)")
    ax1.plot(x_axis, shaping_ma, color='blue', linestyle='-', linewidth=2.5, label="Discrete SAC (With Shaping)")
    
    ax1.set_title("Discrete SAC Performance Comparison", fontsize=15, fontweight='bold')
    ax1.set_xlabel(f"Episode # (Moving Average Window = {window_size})", fontsize=12)
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc="lower right", fontsize=11)
    
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f">>> Plot salvato in: {filename}")


# ==========================================
# 2. LTLF SHAPING WRAPPER
# ==========================================
class LTLfShapingWrapper(gym.Wrapper):
    def __init__(self, env, abstract_mdp, use_shaping=True, K=1.0, goal_reward=10000):
        super().__init__(env)
        self.abstract_mdp = abstract_mdp
        self.use_shaping = use_shaping
        self.K = K
        self.goal_reward = goal_reward
        self.num_states = len(abstract_mdp.automaton.states)
        self.current_q = None
        self.last_s_raw = None
        
        self.episode_count = 0
        self.total_hits = [0] * self.num_states
        self.labels = [f"q{s}" for s in self.abstract_mdp.automaton.states]
        
        self.ep_true_reward = 0.0
        self.ep_total_reward = 0.0

        obs_shape = env.observation_space.shape[0] + self.num_states
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_s_raw = obs
        self.current_q = self.abstract_mdp.automaton.get_initial_q()
        
        self.ep_true_reward = 0.0
        self.ep_total_reward = 0.0
        
        q_idx = self.abstract_mdp.automaton.states.index(self.current_q)
        q_one_hot = np.zeros(self.num_states, dtype=np.float32)
        q_one_hot[q_idx] = 1.0
        
        return np.concatenate((obs, q_one_hot)).astype(np.float32), info

    def step(self, action):
        ns_raw, env_reward, terminated, truncated, info = self.env.step(action)
        
        abstract_x, abstract_y, _ = phi_mapping_sequential(self.last_s_raw, 0)
        abstract_x_ns, abstract_y_ns, _ = phi_mapping_sequential(ns_raw, 0)
        
        truth_assignment = self.abstract_mdp._get_truth_assignment(abstract_x_ns, abstract_y_ns)
        next_q = self.abstract_mdp.automaton.get_next_q(self.current_q, truth_assignment)
        
        custom_env_reward = 0.0
        
        if next_q != self.current_q:
            q_idx = self.abstract_mdp.automaton.states.index(self.current_q)
            self.total_hits[q_idx] += 1
            
            if self.abstract_mdp.automaton.is_goal_reached(next_q):
                custom_env_reward = self.goal_reward
                terminated = True
                
        abstract_s = (abstract_x, abstract_y, self.current_q)
        abstract_ns = (abstract_x_ns, abstract_y_ns, next_q)
        
        shaping_signal = 0.0
        if self.use_shaping and abstract_s != abstract_ns:
            phi_s = self.abstract_mdp.v_star.get(abstract_s, 0.0)
            phi_ns = self.abstract_mdp.v_star.get(abstract_ns, 0.0)
            shaping_signal = self.K * (self.abstract_mdp.gamma * phi_ns - phi_s)
            
        total_reward = custom_env_reward + shaping_signal
        
        self.ep_true_reward += custom_env_reward
        self.ep_total_reward += total_reward
        
        if terminated or truncated:
            global_true_rewards.append(self.ep_true_reward)
            global_total_rewards.append(self.ep_total_reward)
            self.episode_count += 1
            
            if self.episode_count % 100 == 0:
                recent_avg = np.mean(global_true_rewards[-100:])
                recent_avg_with_shaping = np.mean(global_total_rewards[-100:])
                mode_str = "SHAPING" if self.use_shaping else "BASELINE"
                hits_details = ", ".join([f"{self.labels[i]}: {self.total_hits[i]}" for i in range(self.num_states)])
                
                log_string = (
                    f"----------------------------------------------------------------------------------------------------\n"
                    f"[{mode_str} | DISCRETE SAC] Episode {self.episode_count}/{args.episodes}\n"
                    f"Avg Reward                  : {recent_avg:.6f}\n" +
                    (f"Avg Total Reward            : {recent_avg_with_shaping:.6f}\n" if self.use_shaping else "") +
                    f"Hits (Cumulative)           : {hits_details}\n"
                    f"----------------------------------------------------------------------------------------------------"
                )
                print(log_string)
            
        next_q_idx = self.abstract_mdp.automaton.states.index(next_q)
        next_q_one_hot = np.zeros(self.num_states, dtype=np.float32)
        next_q_one_hot[next_q_idx] = 1.0
        ns_aug = np.concatenate((ns_raw, next_q_one_hot)).astype(np.float32)
        
        self.current_q = next_q
        self.last_s_raw = ns_raw
        
        return ns_aug, total_reward, terminated, truncated, info


# ==========================================
# 3. RETI NEURALI CUSTOM (Puro PyTorch)
# ==========================================
class CustomActor(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.device = device
        input_dim = int(np.prod(state_shape))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, action_shape)
        ).to(device)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        logits = self.net(obs)
        return logits, state

class CustomCritic(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.device = device
        input_dim = int(np.prod(state_shape))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, action_shape)
        ).to(device)

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        q_values = self.net(obs)
        return q_values


# ==========================================
# 4. TRAINING LOOP E MAIN
# ==========================================
def train_discrete_sac(env_fn, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n---> Inizializzazione Discrete SAC su {device} (Shaping: {args.use_shaping})")
    
    train_envs = DummyVectorEnv([env_fn])
    
    state_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].n
    
    actor = CustomActor(state_shape, action_shape, device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    
    critic1 = CustomCritic(state_shape, action_shape, device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-3)
    
    critic2 = CustomCritic(state_shape, action_shape, device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-3)
    
    policy = DiscreteSACPolicy(
        actor=actor, actor_optim=actor_optim,
        critic1=critic1, critic1_optim=critic1_optim,
        critic2=critic2, critic2_optim=critic2_optim,
        tau=0.005, gamma=0.99, alpha=0.05, estimation_step=1
    ).to(device)
    
    buffer = VectorReplayBuffer(100000, len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    
    print(">>> Pre-campionamento dati casuali nel buffer...")
    train_collector.collect(n_step=2000, random=True)
    
    print(f"\n=== INIZIO ADDESTRAMENTO: ESATTAMENTE {args.episodes} EPISODI ===")
    
    # ---------------------------------------------------------
    # CICLO MANUALE E PRECISO (Sostituisce offpolicy_trainer)
    # ---------------------------------------------------------
    for n_episode in range(1, args.episodes + 1):
        # 1. Raccoglie ESATTAMENTE 1 episodio dall'ambiente
        collect_result = train_collector.collect(n_episode=1)
        steps_this_episode = collect_result["n/st"]
        
        # 2. Ottimizza la rete (esattamente 1 volta per ogni step compiuto, come nel tuo DQN originale)
        for _ in range(steps_this_episode):
            policy.update(sample_size=64, buffer=buffer)
            
    print("\n=== ADDESTRAMENTO COMPLETATO ===")
    
    return global_true_rewards.copy(), global_total_rewards.copy()


def main(args):
    data_dir = "results"
    img_dir = "img"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    print("=== STARTING LTLf STUDY CON DISCRETE SAC ===")
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    formula_str = config.get('formula', 'F(goal)')
    raw_waypoints = config.get('waypoints_dict', {'goal': [5, 0]})
    waypoints_dict = {name: tuple(coords) for name, coords in raw_waypoints.items()}
    
    print(f"LTLf Formula: {formula_str}")
    
    automaton = LTLfAutomaton(formula_str)
    
    abstract_mdp = LTLfWaypointMDP(
        waypoints_dict=waypoints_dict, 
        ltlf_automaton=automaton, 
        gamma=0.99, 
        goal_reward=10000
    )
    abstract_mdp.value_iteration()
    
    save_sequential_heatmaps(abstract_mdp, filename_prefix=f"{img_dir}/heatmap_V_star")
    
    global global_true_rewards, global_total_rewards
    
    def make_env():
        base_env = gym.make("LunarLander-v3", continuous=False)
        return LTLfShapingWrapper(base_env, abstract_mdp, use_shaping=args.use_shaping)

    global_true_rewards.clear()
    global_total_rewards.clear()
    
    true_rewards, total_rewards = train_discrete_sac(make_env, args)
    
    prefix = "shaping" if args.use_shaping else "baseline"
    
    np.savez_compressed(f"{data_dir}/{prefix}_discrete_sac_data.npz", true_rewards=true_rewards, total_rewards=total_rewards)
    
    plot_shaping_reward_breakdown(
        true_rewards, total_rewards, 
        window_size=min(50, max(1, len(true_rewards) // 10)), 
        filename=f"{img_dir}/discrete_sac_{prefix}_reward_breakdown.png"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTLf Discrete SAC Training")
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=500, 
        help="Numero esatto di episodi per l'addestramento."
    )
    parser.add_argument(
        "--config", 
        type=str,
        default="trajectory.json", 
        help="Path alla configurazione della traiettoria."
    )
    parser.add_argument(
        "--use_shaping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abilita o disabilita il Potential-Based Reward Shaping (--use_shaping / --no-use_shaping)."
    )
    args = parser.parse_args()
    main(args)
