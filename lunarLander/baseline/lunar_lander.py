# Authors: Matteo Ventali and Valerio Spagnoli
# ML Project : tabular Q-learning for Lunar Lander envinronment

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from collections import defaultdict
import random as ran
import pickle

window_size = 500

def accuracy_plot(reward, types):
    x = range(len(reward))
    mean = np.mean(reward)
    plt.scatter(x,reward)
    plt.title(f'Cumulative episode reward {types} policy')
    plt.xlabel('Episode #')
    plt.ylabel('Episode reward')
    plt.axhline(y=mean, color='red', linestyle='--', label=f'Mean = {mean:.2f}')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

def function_plot_combined(reward_eps, reward_random, epsilon_value):
    mean_mobile = np.convolve(reward_eps, np.ones(window_size)/window_size, mode="valid")
    mean_mobile_random = np.convolve(reward_random, np.ones(window_size)/window_size, mode="valid")

    # Organizing the graph with two y axis
    fig, ax1 = plt.subplots()

    # First axis
    ax1.plot(mean_mobile, color='red', label='reward_greedy_policy')
    ax1.plot(mean_mobile_random, color='green', label='reward_greedy_policy')
    ax1.set_ylabel('policies')
    ax1.set_xlabel('episodes')

    # Second axis
    ax2 = ax1.twinx()
    ax2.plot(epsilon_value, color='orange', label='reward_greedy_policy')
    ax2.set_ylabel('epsilon value')
    
    plt.title("Learning trend")
    plt.show()

def function_plot_tuning(reward_eps1, reward_eps2, reward_eps3):
    mean_mobile1 = np.convolve(reward_eps1, np.ones(window_size)/window_size, mode="valid")
    mean_mobile2 = np.convolve(reward_eps2, np.ones(window_size)/window_size, mode="valid")
    mean_mobile3 = np.convolve(reward_eps3, np.ones(window_size)/window_size, mode="valid")

    plt.plot(mean_mobile1, color='red', label='15k_y99_9995')
    plt.plot(mean_mobile2, color='blue', label='15k_y9_999')
    plt.plot(mean_mobile3, color='green', label='15k_y9_99')
    plt.ylabel('policies')
    plt.xlabel('episodes')

    plt.title("Different learning trends")
    plt.legend(loc='upper left')
    plt.show()

def discretize(obs):
        result = []
        
        # Interval array
        x_intervals     = [-0.5, 0.5]
        y_intervals     = [-0.1, 0.1, 1.5]
        vx_intervals    = [-7.5, -5, -0.3, -0.1, 0.1, 0.3, 5, 7.5]
        vy_intervals    = [-7.5, -5, -0.3, -0.1, 0.1, 0.3, 5, 7.5]
        theta_intervals = [-1.25663706,  -0.1, 0.1, 1.25663706]
        omega_intervals = [-7.5, -5, -0.1, 0.1, 5, 7.5]
        
        result.append(np.digitize(obs[0], x_intervals))
        result.append(np.digitize(obs[1], y_intervals))
        result.append(np.digitize(obs[2], vx_intervals))
        result.append(np.digitize(obs[3], vy_intervals))
        result.append(np.digitize(obs[4], theta_intervals))
        result.append(np.digitize(obs[5], omega_intervals))

        # No need to discretize boolean variables
        result.append(int(obs[6]))
        result.append(int(obs[7]))
        return tuple(result)
    

class QLearner():
    def __init__(self, env:gym.Env, max_episodes=15000, gamma=0.99, alpha=0.1, end_eps=0.01, start_eps=1.0,  eps_decay=0.9995, policy_name="policy_lunar_lander"):
        self.env = env
        self.max_episodes = max_episodes        
        self.gamma = gamma
        self.alpha = alpha
        self.end_eps = end_eps
        self.eps = start_eps
        self.eps_decay = eps_decay
        self.policy_name = "./policy/" + policy_name
        
    def _espilon_update(self):
        self.eps = max(self.eps_decay * self.eps, self.end_eps)

    def _next_action(self, current_state, modality):
        n = ran.random()
        if modality == 0  or n < self.eps: # Exploration
            return self.env.action_space.sample()
        else: # Exploitation
            return np.argmax(self.q_table[current_state])

    def tabular_QLearning(self, modality=1): # 0 for random, 1 for eps-greedy policy
        # Q-table creation dynamically. It is implemented as a dictionary in which:
        # - the keys are the states (observation collected)
        # - the values are array of 4 components (number of actions available), representing Q(s,a)
        # - when a state is not already in the dictionary, then it is added with a value of (0,0,0,0). This happens the first time
        #   the agent ends up in that state
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.eps = 1.0

        # Starting of the environment
        s, _ = self.env.reset()
        s = discretize(s)

        # Array for collecting total rewards
        total_rewards = []
        eps_per_episode = []
        
        # Training of episodes
        for n_episode in range(self.max_episodes):
            
            terminated = truncated = False
            episode_reward = 0
            
            while not (terminated or truncated): # Single episode
                # Select the action to be executed
                a = self._next_action(s, modality)

                # Execution of a
                ns, reward, terminated, truncated, _ = self.env.step(a)
                episode_reward += reward
                ns = discretize(ns)

                # Update Q_table
                # Q(s,a) = (1 - alpha)*Q(s,a) + alpha*[r + y max_a'{Q(s',a')}]
                # Q(s,a) = Q(s,a) + alpha[r + y*max_a'{Q(s',a')} - Q(s,a)]
                self.q_table[s][a] += self.alpha * (reward + self.gamma * max(self.q_table[ns]) - self.q_table[s][a])

                # Updating new state
                if not (terminated or truncated):
                    s = ns
            
            # Stats of the episode
            eps_per_episode.append(self.eps)
            self._espilon_update()
            s, _ = self.env.reset()
            s = discretize(s)
            total_rewards.append(episode_reward)
            if n_episode > 50:
                print(f"(m={modality} episode {n_episode} {np.mean(total_rewards[-50:])} {self.eps})")
            
        # Dump q_table
        if modality == 1:
            with open(self.policy_name, "wb") as f:
                pickle.dump(dict(ql.q_table), f)

        return np.array(total_rewards), np.array(eps_per_episode)
    
    def load_policy(self):
        with open(self.policy_name, "rb") as f:
            self.policy = defaultdict(lambda: np.zeros(self.env.action_space.n),pickle.load(f))
        
    def run_policy(self):
        self.load_policy()
        
        episodes_reward = []
        episodes_length = []
        success_100 = 0 # Times reward > 100
        success_200 = 0 # Times reward > 200
        n_episodes = 1000

        for i in range(0,n_episodes):
            s, _ = self.env.reset()
            s = discretize(s)

            terminated = truncated = False
            rw = 0
            length_ep = 0

            while not (terminated or truncated):
                a = np.argmax(self.policy[s])
                ns, r, terminated, truncated, _ = self.env.step(a)
                length_ep += 1
                ns = discretize(ns)
                rw += r
                s = ns

            episodes_reward.append(rw)
            episodes_length.append(length_ep)
            if rw >= 100:
                success_100 += 1
            if rw >= 200:
                success_200 += 1
            
        print(f"Mean Episode Reward: {np.mean(episodes_reward)}")
        print(f"Success >= 100: {success_100}")
        print(f"Success >= 200: {success_200}")
        print(f"Mean Episode Length: {np.mean(episodes_length)}")
        return episodes_reward

    def run_random(self, n_ep=None):
        total_reward = 0
        episodes_reward = []
        n_episodes = 1000

        if n_ep != None:
            n_episodes = n_ep
        else:
            n_episodes = self.max_episodes

        for i in range(0,n_episodes):
            s, _ = self.env.reset()
            s = discretize(s)
            
            terminated = truncated = False
            rw = 0

            while not (terminated or truncated):
                a = self._next_action(s, 0)
                ns, r, terminated, truncated, _ = self.env.step(a)
                ns = discretize(ns)
                rw += r
                s = ns

            total_reward += rw
            episodes_reward.append(rw)
        
        print(f"Mean Episode Reward: {np.mean(episodes_reward)}")        
        return episodes_reward


if __name__ == "__main__":
    # Lunar Lander Environment
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    #env = gym.make("LunarLander-v3", continuous=False)
    
    # Menu
    mode = input("Select modality (0 = training, 1 = running, 2 = tuning): ").strip()
    policy_file = input("File policy (empty for default):").strip()
    
    # Learner object
    if policy_file == "": # Apply default name
        ql = QLearner(env)
    else:
        ql = QLearner(env, policy_name=policy_file)    

    if mode == "0": # Training
        rw_random = ql.run_random()
        rw_eps, eps_values = ql.tabular_QLearning()
        function_plot_combined(rw_eps, rw_random, eps_values)
    elif mode == "1": # Running
        rw_policy = ql.run_policy()
        rw_random = ql.run_random(n_ep=1000)
        accuracy_plot(rw_policy, 'epsilon-greedy')
        accuracy_plot(rw_random, 'random')
    elif mode == "2": # Tuning
        ql = QLearner(env, policy_name="tuning_15k_y99_9995_2", max_episodes=15000, gamma=0.99, eps_decay=0.9995)
        rw_eps1, eps_values1 = ql.tabular_QLearning()

        ql2 = QLearner(env, policy_name="tuning_15k_y9_999_2", max_episodes=15000, gamma=0.9, eps_decay=0.999)
        rw_eps2, eps_values2 = ql2.tabular_QLearning()

        ql3 = QLearner(env, policy_name="tuning_15k_y9_99_2", max_episodes=15000, gamma=0.9, eps_decay=0.99)
        rw_eps3, eps_values3 = ql3.tabular_QLearning()

        function_plot_tuning(rw_eps1, rw_eps2, rw_eps3)
    else:
        print("Input not valid")
