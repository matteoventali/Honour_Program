# Lunar Lander with LTLf Reward Shaping

This repository contains implementations of Tabular Q-Learning and Hierarchical Reinforcement Learning (HRL) with reward shaping for the Lunar Lander environment from Gymnasium. The project explores different training methodologies to learn optimal landing policies.

The `discrete_sac` module provides a neural baseline based on d3rlpy's Discrete Soft
Actor-Critic implementation. It keeps LunarLander's four-action discrete
interface and augments each observation with the current LTLf automaton state.
Its `trainer.py` entry point contains separate documented sections for the
algorithm configuration, environment wrapper, metrics, plotting and training
orchestration.

Install the Python dependencies and start an experiment from the `discrete_sac`
directory:

```bash
python3 -m pip install -r ../requirements.txt
python3 trainer.py --steps 250000 --config trajectory.json
```

Use `--no-use-shaping` to train the corresponding unshaped baseline. Models,
numeric episode metrics and plots are written to `results` and `img`.

A streamlined Kaggle workflow is available in
`notebook/discrete_sac_training_kaggle.ipynb`. It contains separate source,
trajectory, training, validation, execution and result-export cells.


## Features

- **Tabular Q-Learning Baseline**: A standard Q-learning agent used as a reference.
- **Hierarchical Reinforcement Learning (HRL)**: Uses an abstract grid-world MDP to guide the low-level Q-learning agent.
- **Reward Shaping**: Incorporates a potential-based shaping signal derived from the value function of the abstract MDP to accelerate learning.
- **Flexible Abstract MDPs**: Supports different grid abstractions, including variants with diagonal movements.
- **Policy Management**: Save and load trained Q-tables (policies).
- **Visualization**: Generates plots for training progress (raw rewards and moving averages) and policy comparisons.
- **Parallel Execution**: Supports multiprocessing for training and evaluating multiple policies concurrently.

## Abstract grid overlay

To render the same abstract grid used by the trainer directly over a
LunarLander RGB frame, run:

```bash
cd lunarLander
python3 grid_overlay.py --config trajectory.json \
  --output img/abstract_grid_overlay.png --seed 0
```

The generated image includes the configured waypoints and highlights the
abstract cell occupied by the lander after reset. The reusable
`draw_abstract_grid` function can also annotate frames captured later in an
episode.

During policy evaluation, the cells visited by the agent can be recorded and
drawn over the same grid:

```bash
cd lunarLander
python3 evaluate.py policy/example.pt --config trajectory.json \
  --episodes 10 --trace-grid --trace-episodes 3
```

This saves one numbered cell path per traced episode under `img/evaluation`
and prints whether each configured waypoint was reached or missed. Grid
tracing and interactive `--render` use different Gymnasium render modes and
must be run separately.


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
