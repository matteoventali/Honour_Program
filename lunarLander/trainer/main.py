import os
import argparse
import multiprocessing
import gymnasium as gym

from utils import plot_training_results, phi_mapping_grid, phi_mapping_kinematic
from abstract_mdps import AbstractGridMDP, DiagonalAbstractGridMDP, KinematicAbstractMDP
from agent import HierarchicalDQNLearner

def run_training(mode, episodes, use_ddqn):
    env = gym.make("LunarLander-v3", continuous=False)
    
    algo_str = "ddqn" if use_ddqn else "single_dqn"
    algo_display = "DDQN" if use_ddqn else "True Single DQN"
    
    # ---------------- NORMAL ----------------
    if mode == "normal":
        abstract = AbstractGridMDP()
        policy_name = f"{algo_str}_normally.pth"
        agent = HierarchicalDQNLearner(env, abstract, phi_mapping_grid, max_episodes=episodes, policy_name=policy_name, use_ddqn=use_ddqn)
        rewards = agent.train_normal()
        plot_training_results(rewards, title=f"Normal {algo_display}", filename=f"./img/{algo_str}_normal.png")
        
    # ---------------- SHAPING (2D GRID) ----------------
    elif mode == "shaping":
        abstract = AbstractGridMDP()
        policy_name = f"{algo_str}_shaping.pth"
        agent = HierarchicalDQNLearner(env, abstract, phi_mapping_grid, max_episodes=episodes, policy_name=policy_name, use_ddqn=use_ddqn)
        rewards = agent.train_shaping()
        plot_training_results(rewards, title=f"Shaping {algo_display}", filename=f"./img/{algo_str}_shaping.png")
        
    elif mode == "extended":
        abstract = DiagonalAbstractGridMDP()
        policy_name = f"{algo_str}_shaping_ext.pth"
        agent = HierarchicalDQNLearner(env, abstract, phi_mapping_grid, max_episodes=episodes, policy_name=policy_name, use_ddqn=use_ddqn)
        rewards = agent.train_shaping()
        plot_training_results(rewards, title=f"Extended Shaping {algo_display}", filename=f"./img/{algo_str}_shaping_ext.png")

    # ---------------- SHAPING (4D KINEMATIC) ----------------
    elif mode == "kinematic":
        abstract = KinematicAbstractMDP()
        policy_name = f"{algo_str}_kinematic.pth"
        agent = HierarchicalDQNLearner(env, abstract, phi_mapping_kinematic, max_episodes=episodes, policy_name=policy_name, use_ddqn=use_ddqn)
        rewards = agent.train_shaping()
        plot_training_results(rewards, title=f"Kinematic Shaping {algo_display}", filename=f"./img/{algo_str}_kinematic.png")

    # ---------------- GOAL MDP (2D GRID) ----------------
    elif mode == "goal_mdp":
        abstract = AbstractGridMDP()
        policy_name = f"{algo_str}_goal_mdp.pth"
        agent = HierarchicalDQNLearner(env, abstract, phi_mapping_grid, max_episodes=episodes, policy_name=policy_name, use_ddqn=use_ddqn)
        true_r, goal_r = agent.train_goal_mdp()
        plot_training_results(true_r, title=f"Goal MDP {algo_display} - True Env", filename=f"./img/{algo_str}_goal_true.png")
        plot_training_results(goal_r, title=f"Goal MDP {algo_display} - Sparse", filename=f"./img/{algo_str}_goal_sparse.png")

    elif mode == "goal_mdp_extended":
        abstract = DiagonalAbstractGridMDP()
        policy_name = f"{algo_str}_goal_mdp_ext.pth"
        agent = HierarchicalDQNLearner(env, abstract, phi_mapping_grid, max_episodes=episodes, policy_name=policy_name, use_ddqn=use_ddqn)
        true_r, goal_r = agent.train_goal_mdp()
        plot_training_results(true_r, title=f"Goal MDP Ext {algo_display} - True Env", filename=f"./img/{algo_str}_goal_ext_true.png")
        plot_training_results(goal_r, title=f"Goal MDP Ext {algo_display} - Sparse", filename=f"./img/{algo_str}_goal_ext_sparse.png")

    # ---------------- GOAL MDP (4D KINEMATIC) ----------------
    elif mode == "goal_mdp_kinematic":
        abstract = KinematicAbstractMDP()
        policy_name = f"{algo_str}_goal_mdp_kinematic.pth"
        agent = HierarchicalDQNLearner(env, abstract, phi_mapping_kinematic, max_episodes=episodes, policy_name=policy_name, use_ddqn=use_ddqn)
        true_r, goal_r = agent.train_goal_mdp()
        plot_training_results(true_r, title=f"Goal MDP Kinematic {algo_display} - True Env", filename=f"./img/{algo_str}_goal_kinematic_true.png")
        plot_training_results(goal_r, title=f"Goal MDP Kinematic {algo_display} - Sparse", filename=f"./img/{algo_str}_goal_kinematic_sparse.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular HRL training for LunarLander.")
    parser.add_argument("--mode", type=str, 
                        choices=["normal", "shaping", "extended", "kinematic", "goal_mdp", "goal_mdp_extended", "goal_mdp_kinematic", "all"], 
                        default="normal", help="Choose training mode")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of episodes")
    parser.add_argument("--parallel", action="store_true", help="Run chosen models in parallel")
    parser.add_argument("--ddqn", action="store_true", help="Enable Double DQN (DDQN)")
    
    args = parser.parse_args()
    
    os.makedirs("./img", exist_ok=True)
    os.makedirs("./policy", exist_ok=True)

    modes_to_run = ["normal", "shaping", "extended", "kinematic", "goal_mdp", "goal_mdp_extended", "goal_mdp_kinematic"] if args.mode == "all" else [args.mode]

    if args.parallel and len(modes_to_run) > 1:
        print(f"--- Starting PARALLEL training ---")
        processes = []
        for m in modes_to_run:
            p = multiprocessing.Process(target=run_training, args=(m, args.episodes, args.ddqn))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        print(f"--- Starting SEQUENTIAL training ---")
        for m in modes_to_run:
            print(f"\n-> Launching mode: {m.upper()}")
            run_training(m, args.episodes, args.ddqn)