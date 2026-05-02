# Lunar Lander Q-Learning and Hierarchical Reinforcement Learning

This repository contains implementations of Tabular Q-Learning and Hierarchical Reinforcement Learning (HRL) with reward shaping for the Lunar Lander environment from Gymnasium. The project explores different training methodologies to learn optimal landing policies.


## Features

- **Tabular Q-Learning Baseline**: A standard Q-learning agent used as a reference.
- **Hierarchical Reinforcement Learning (HRL)**: Uses an abstract grid-world MDP to guide the low-level Q-learning agent.
- **Reward Shaping**: Incorporates a potential-based shaping signal derived from the value function of the abstract MDP to accelerate learning.
- **Flexible Abstract MDPs**: Supports different grid abstractions, including variants with diagonal movements.
- **Policy Management**: Save and load trained Q-tables (policies).
- **Visualization**: Generates plots for training progress (raw rewards and moving averages) and policy comparisons.
- **Parallel Execution**: Supports multiprocessing for training and evaluating multiple policies concurrently.


## Project Structure

````

lunarLander/
├── baseline/
│   ├── lunar_lander.py         # Original baseline Q-learner (possibly deprecated)
│   └── baseVersion.py          # Improved baseline Q-learner
├── policy/                     # Stores trained Q-tables (.pkl files)
├── img/                        # Stores generated plots (.png files)
├── trainer.py                  # HRL training script (sequential)
├── trainer_improved.py         # HRL training script (with multiprocessing)
├── run_policy.py               # Run a single trained policy
├── run_combined.py             # Compare two trained policies
├── run_combined_improved.py    # Compare multiple policies (with multiprocessing)

````