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

## Esecuzione degli esperimenti con Docker

Ogni framework attivo contiene un `Dockerfile` indipendente e usa la propria
directory come contesto di build:

| Framework | Immagine suggerita |
| --- | --- |
| `lunarLander` | `tesi-lunar-lander` |
| `manual_experiment` | `tesi-manual` |
| `multilevel_framework` | `tesi-multilevel` |
| `multilevel_framework_convention` | `tesi-multilevel-convention` |
| `multilevel_multieps` | `tesi-multilevel-multieps` |
| `sac` | `tesi-sac` |

Per esempio, dalla radice della repository:

```bash
docker build -t tesi-lunar-lander lunarLander
docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$(realpath lunarLander)",target=/workspace \
  tesi-lunar-lander \
  --episodes 1000 --num-seeds 5 --seed 42
```

Il bind mount collega `/workspace` alla directory fisica `lunarLander`
dell'host. Di conseguenza, i file prodotti dal container sono disponibili
direttamente nelle sottodirectory host `results`, `img`, `logs` e `policy`.
L'opzione `--user` fa sì che i nuovi file appartengano all'utente corrente.

Lo stesso schema vale per gli altri framework. Per esempio:

```bash
docker build -t tesi-manual manual_experiment
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$(realpath manual_experiment)",target=/workspace \
  tesi-manual --episodes 1000

docker build -t tesi-multilevel multilevel_framework
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$(realpath multilevel_framework)",target=/workspace \
  tesi-multilevel --episodes 1000

docker build -t tesi-multilevel-convention multilevel_framework_convention
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$(realpath multilevel_framework_convention)",target=/workspace \
  tesi-multilevel-convention --episodes 1000

docker build -t tesi-multilevel-multieps multilevel_multieps
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$(realpath multilevel_multieps)",target=/workspace \
  tesi-multilevel-multieps --episodes 1000

docker build -t tesi-sac sac
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$(realpath sac)",target=/workspace \
  tesi-sac --episodes 1000 --device auto
```

Gli argomenti posti dopo il nome dell'immagine vengono inoltrati direttamente
a `trainer.py`, quindi sono disponibili anche opzioni come `--config`,
`--no-shaping` e `--post-process`. Le immagini installano PyTorch con supporto
CUDA 12.8 e configurano Matplotlib e SDL per l'esecuzione headless. Il server
deve avere NVIDIA Container Toolkit configurato per usare `--gpus all`.

### Script di avvio

Ogni framework contiene anche un `run_experiment.sh`. Lo script costruisce
l'immagine, avvia il container in background, monta la directory del framework
e inoltra gli argomenti al trainer. Prima dell'avvio verifica che PyTorch rilevi
CUDA e stampa il modello della GPU; se CUDA non è disponibile, termina senza
avviare un training destinato a ricadere sulla CPU. Può essere eseguito dalla
radice della repository, per esempio:

```bash
./lunarLander/run_experiment.sh \
  --episodes 1000 --num-seeds 5 --seed 42

./multilevel_multieps/run_experiment.sh \
  --episodes 1000 --epsilon-strategy visited

./sac/run_experiment.sh \
  --episodes 1000 --device auto
```

Al termine dell'avvio lo script stampa il nome univoco del container e i
comandi per seguirne i log, controllarne lo stato e fermarlo. Per assegnare un
nome esplicito al container:

```bash
CONTAINER_NAME=esperimento-seed-42 \
  ./lunarLander/run_experiment.sh --episodes 1000 --seed 42

docker logs -f esperimento-seed-42
```

## Multi-seed training and variance plots

Every active trainer accepts `--num-seeds` and `--seed`. The first option is
the number of independent training runs; the second is the first seed (42 by
default), and the following runs use consecutive values. For example:

```bash
cd lunarLander
python3 trainer.py --episodes 1000 --num-seeds 5 --seed 42
```

Each run saves its own metrics and policy. The combined `.npz` file also stores
the seed list, every stacked metric under a `_runs` key, and reward summaries
under `_mean` and `_variance` keys. At the end of training, the trainer writes
a `training_variance_*.png` plot with the smoothed mean learning reward and a
shaded `±1σ` band across seeds (`σ²` is the cross-seed variance). The same
options are available in `manual_experiment`,
`multilevel_framework`, `multilevel_framework_convention`,
`multilevel_multieps`, and `sac`.

The original reward-breakdown plot is preserved, and an additional
`reward_breakdown_*_seed_<seed>.png` is generated for every individual run.
Replay-buffer plots show only the observed DFA-state fractions, without an
ideal-load-balance reference line.

The Kaggle notebooks expose the same settings as `NUM_SEEDS` and `SEED` in
their configuration cell. Their embedded trainers and plotting utilities can
be refreshed after source changes with `python3 notebook/sync_training_notebooks.py`.

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

The project uses a uniform `12 x 12` discretization by default. The `x`
domain `[-1, 1]` and the `y` domain `[0, 1.5]` are each divided into twelve
equal-width bins; values outside these domains are clipped into the nearest
edge cell. Training, evaluation and the overlay all use this same mapping.
The previous `grid_size - 1` implementation remains as a commented
`phi_mapping_grid` block in `utils.py`. Swapping the comments between the two
implementations automatically updates training, evaluation, heatmaps and the
grid overlay.

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


## Manual cycles with N waypoints

The `manual_experiment` variant accepts an ordered cycle of any positive length.
Declare waypoint coordinates in `waypoints_dict` and their required visit order
in `waypoint_cycle`:

```json
{
  "grid_w": 12,
  "grid_h": 12,
  "goal_reward": 10000,
  "waypoint_cycle": ["g1", "g2", "g3", "g4"],
  "waypoints_dict": {
    "g1": [1, 8],
    "g2": [4, 10],
    "g3": [8, 10],
    "g4": [10, 8]
  }
}
```

The reward is emitted only after the last waypoint, then the automaton resets
immediately and starts the next cycle. If `waypoint_cycle` is omitted, the
insertion order of `waypoints_dict` is used. Existing two-waypoint experiments
remain valid. Because N determines the neural network input size, a checkpoint
must be evaluated with the same `waypoint_cycle` used during its training.


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
