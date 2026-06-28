import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Move the import to the top for cleaner code
from utils import phi_mapping_grid 
from agent import QNetwork


def load_policy(policy_path, state_dim, action_dim, device):
    """Module 1: Loads and returns the model (executed only once)."""
    policy_net = QNetwork(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load(policy_path, map_location=device))
    policy_net.eval() # Evaluation mode
    return policy_net

def run_single_episode(env, policy_net, device, grid_w=12, grid_h=12, render_delay=0.02):
    """Module 2: Runs a single episode using an existing environment and model."""
    s_raw, _ = env.reset()
    q = 0
    s_aug = np.append(s_raw, q)
    
    terminated = truncated = False
    total_reward = 0
    waypoint_reached = False
    
    while not (terminated or truncated):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(s_aug).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        ns_raw, reward, terminated, truncated, _ = env.step(action)
        
        # State Logic (Phase Transition)
        abs_x, abs_y = phi_mapping_grid(ns_raw, grid_w, grid_h)
        
        if q == 0 and abs_x == 1 and abs_y == 8:
            q = 1
            waypoint_reached = True
            print("   >>> Waypoint (1,8) reached! Transitioning to phase q=1.")
            
        s_aug = np.append(ns_raw, q)
        total_reward += reward
        
        if render_delay > 0:
            time.sleep(render_delay)
            
    return total_reward, waypoint_reached

def evaluate_policy(policy_path, num_episodes=100, env_name="LunarLander-v3", grid_w=12, grid_h=12):
    """Module 3: Orchestrator. Initializes everything and manages the testing loop."""
    
    # 1. Environment Setup (Executed ONCE)
    env = gym.make(env_name, render_mode="human")
    state_dim = env.observation_space.shape[0] + 1
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Network Loading (Executed ONCE)
    print(f"Loading policy from: {policy_path} on {device}...")
    policy_net = load_policy(policy_path, state_dim, action_dim, device)
    
    # 3. Evaluation Loop
    print(f"\nStarting evaluation for {num_episodes} episodes...")
    success_count = 0
    rewards_history = []
    
    for i in range(1, num_episodes + 1):
        print(f"\n--- Episode {i}/{num_episodes} ---")
        
        reward, wp_reached = run_single_episode(
            env=env, 
            policy_net=policy_net, 
            device=device, 
            grid_w=grid_w, 
            grid_h=grid_h,
            render_delay=0.02 # Set to 0 if you want a lightning-fast test without watching it
        )
        
        rewards_history.append(reward)
        if wp_reached:
            success_count += 1
            
        print(f"Episode {i} finished | Reward: {reward:.2f}")

    # 4. Teardown and Results
    env.close()
    
    print("\n======================================")
    print("        EVALUATION COMPLETED          ")
    print("======================================")
    print(f"Episodes tested:   {num_episodes}")
    print(f"Waypoints reached: {success_count}/{num_episodes} ({(success_count/num_episodes)*100:.1f}%)")
    print(f"Average Reward:    {np.mean(rewards_history):.2f}")
    print("======================================")

if __name__ == "__main__":
    evaluate_policy("./policy/sequential_policy.pth", num_episodes=100)