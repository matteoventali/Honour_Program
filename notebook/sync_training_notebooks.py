"""Synchronize Kaggle notebook training cells with the active frameworks."""

from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"

NOTEBOOK_SOURCES = {
    "notebook_manual_experiment.ipynb": "manual_experiment/src",
    "notebook_multilevel_framework.ipynb": "multilevel_framework/src",
}

MULTILEVEL_NOTEBOOKS = {
    "notebook_multilevel_framework.ipynb",
}


def _replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {description}, found {count}")
    return text.replace(old, new, 1)


def _multilevel_trainer_variant(source: str) -> str:
    """Preserve the notebooks' configurable training-time shaping discount."""
    source = _replace_once(
        source,
        "def _write_run_header(log_handle, episodes, use_shaping, K, goal_reward, abstract_mdp, automaton_states):",
        "def _write_run_header(log_handle, episodes, use_shaping, K, goal_reward, abstract_mdp, automaton_states, training_shaping_gamma):",
        "run-header signature",
    )
    source = _replace_once(
        source,
        '        f"episodes={episodes}, shaping={use_shaping}, K={K}, goal_reward={goal_reward}, gamma={abstract_mdp.gamma}\\n"\n',
        '        f"episodes={episodes}, shaping={use_shaping}, K={K}, goal_reward={goal_reward}, gamma={abstract_mdp.gamma}\\n"\n'
        '        f"training_shaping_gamma={training_shaping_gamma}\\n"\n',
        "training-shaping log entry",
    )
    source = _replace_once(
        source,
        'def run_sequential_training(env, agent, abstract_mdp, episodes, goal_reward=10000, save_policy=True, use_shaping=True, K=1.0, log_file=None, log_interval=100, seed=None, policy_suffix=""):',
        'def run_sequential_training(env, agent, abstract_mdp, episodes, goal_reward=10000, save_policy=True, use_shaping=True, K=1.0, log_file=None, log_interval=100, training_shaping_gamma=True, seed=None, policy_suffix=""):',
        "training-loop signature",
    )
    source = _replace_once(
        source,
        "    _write_run_header(log_handle, episodes, use_shaping, K, goal_reward, abstract_mdp, automaton_states)\n",
        "    _write_run_header(log_handle, episodes, use_shaping, K, goal_reward, abstract_mdp, automaton_states, training_shaping_gamma)\n",
        "run-header invocation",
    )
    source = _replace_once(
        source,
        "                    shaping_signal = K * (abstract_mdp.gamma * phi_next_state - phi_state)\n",
        "                    training_discount = abstract_mdp.gamma if training_shaping_gamma else 1.0\n"
        "                    shaping_signal = K * (training_discount * phi_next_state - phi_state)\n",
        "training shaping equation",
    )
    source = _replace_once(
        source,
        "log_interval=args.log_interval, seed=run_seed, policy_suffix=policy_suffix)",
        "log_interval=args.log_interval, training_shaping_gamma=args.training_shaping_gamma, seed=run_seed, policy_suffix=policy_suffix)",
        "multi-seed training invocation",
    )
    source = _replace_once(
        source,
        '    parser.add_argument("--no-shaping", action="store_true")\n',
        '    parser.add_argument(\n'
        '        "--training-shaping-gamma",\n'
        '        action=argparse.BooleanOptionalAction,\n'
        '        default=True,\n'
        '        help="Use gamma*Phi(next)-Phi(state) during DDQN training.",\n'
        '    )\n'
        '    parser.add_argument("--no-shaping", action="store_true")\n',
        "training-shaping CLI option",
    )
    return source


def _writefile_source(filename: str, body: str) -> list[str]:
    return [f"%%writefile {filename}\n", *body.splitlines(keepends=True)]


def _update_configuration(source: str) -> str:
    if not re.search(r"(?m)^NUM_SEEDS\s*=", source):
        source = re.sub(
            r"(?m)^(EPISODES\s*=.*\n)",
            r"\1NUM_SEEDS = 1\n",
            source,
            count=1,
        )
    if re.search(r"(?m)^SEED\s*=", source):
        source = re.sub(r"(?m)^SEED\s*=.*$", "SEED = 42", source, count=1)
        source = source.replace(
            "# None keeps stochastic resets; use an integer for a reproducible run.\n",
            "# First fixed seed; later runs use consecutive values.\n",
        )
    else:
        source = re.sub(
            r"(?m)^(NUM_SEEDS\s*=.*\n)",
            r"\1SEED = 42\n",
            source,
            count=1,
        )

    if "Number of seeds:" not in source:
        source = re.sub(
            r'(?m)^(print\(f"Episodes: \{EPISODES\}"\)\n)',
            r'\1print(f"Number of seeds: {NUM_SEEDS}")\n',
            source,
            count=1,
        )
    source = re.sub(
        r'(?:print\(f"First seed: \{SEED\}"\)\n)+',
        'print(f"First seed: {SEED}")\n',
        source,
    )
    if (
        'print(f"Seed: {SEED}")' not in source
        and 'print(f"First seed: {SEED}")' not in source
    ):
        source = re.sub(
            r'(?m)^(print\(f"Number of seeds: \{NUM_SEEDS\}"\)\n)',
            r'\1print(f"First seed: {SEED}")\n',
            source,
            count=1,
        )

    if not re.search(r"(?m)^TRAINING_USE_GAMMA\s*=", source):
        source = re.sub(
            r"(?m)^(SHAPING_SCALE\s*=.*\n)",
            r"\1TRAINING_USE_GAMMA = True\n",
            source,
            count=1,
        )
    if 'print(f"Training uses gamma: {TRAINING_USE_GAMMA}")' not in source:
        source = re.sub(
            r'(?m)^(print\(f"Shaping scale: \{SHAPING_SCALE\}"\)\n)',
            r'\1print(f"Training uses gamma: {TRAINING_USE_GAMMA}")\n',
            source,
            count=1,
        )
    return source


def _update_training_command(source: str) -> str:
    old_conditional = (
        'if SEED is not None:\n'
        '    command.extend(["--seed", str(SEED)])\n'
    )
    source = source.replace(old_conditional, "")
    extension = 'command.extend(["--num-seeds", str(NUM_SEEDS), "--seed", str(SEED)])\n'
    if extension not in source:
        marker = "if DISABLE_SHAPING:\n"
        if marker not in source:
            raise RuntimeError("Training command has no DISABLE_SHAPING marker")
        source = source.replace(marker, extension + marker, 1)

    gamma_switch = (
        "if not TRAINING_USE_GAMMA:\n"
        '    command.append("--no-training-shaping-gamma")\n'
    )
    if gamma_switch not in source:
        marker = "if DISABLE_SHAPING:\n"
        if marker not in source:
            raise RuntimeError("Training command has no DISABLE_SHAPING marker")
        source = source.replace(marker, gamma_switch + marker, 1)

    if 'environment["PYTHONHASHSEED"]' not in source:
        source = source.replace(
            'environment["PYTHONUNBUFFERED"] = "1"\n',
            'environment["PYTHONUNBUFFERED"] = "1"\n'
            'environment["PYTHONHASHSEED"] = str(SEED)\n',
            1,
        )
    return source


def _update_result_preview(source: str) -> str:
    if "plot_paths = [" not in source:
        return source
    # Frameworks with named experiments write below results/<experiment-name>;
    # older notebook variants write directly below WORK_DIR.
    plot_root = "EXPERIMENT_DIR" if "EXPERIMENT_DIR" in source else "WORK_DIR"
    for root in ("WORK_DIR", "EXPERIMENT_DIR"):
        for pattern in (
            "training_variance_*.png",
            "*_variance_*.png",
            "reward_breakdown_*_seed_*.png",
            "seed_*/*.png",
        ):
            source = source.replace(
                f'    *sorted(({root} / "img").glob("{pattern}")),\n',
                "",
            )
    variance_glob = f'    *sorted(({plot_root} / "img").glob("*_variance_*.png")),\n'
    seed_plot_glob = f'    *sorted(({plot_root} / "img").glob("seed_*/*.png")),\n'
    if variance_glob not in source:
        source = source.replace("plot_paths = [\n", "plot_paths = [\n" + variance_glob, 1)
    if seed_plot_glob not in source:
        source = source.replace("plot_paths = [\n", "plot_paths = [\n" + seed_plot_glob, 1)
    return source


def synchronize_notebook(notebook_name: str, source_directory: str) -> None:
    notebook_path = NOTEBOOK_DIR / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    writefile_targets = set()
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if source.startswith("%%writefile "):
            writefile_targets.add(source.splitlines()[0].split(maxsplit=1)[1])

    source_root = PROJECT_ROOT / source_directory
    source_by_filename = {
        filename: (source_root / filename).read_text(encoding="utf-8")
        for filename in writefile_targets
        if (source_root / filename).is_file()
    }
    missing_sources = writefile_targets.difference(source_by_filename)
    if missing_sources:
        raise RuntimeError(
            f"Missing source files for {notebook_name}: {sorted(missing_sources)}"
        )

    trainer_source = source_by_filename["trainer.py"]
    if (
        notebook_name in MULTILEVEL_NOTEBOOKS
        and "--training-shaping-gamma" not in trainer_source
    ):
        trainer_source = _multilevel_trainer_variant(trainer_source)
    source_by_filename["trainer.py"] = trainer_source

    replaced = {filename: 0 for filename in source_by_filename}
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if source.startswith("%%writefile "):
            filename = source.splitlines()[0].split(maxsplit=1)[1]
            cell["source"] = _writefile_source(
                filename, source_by_filename[filename]
            )
            replaced[filename] += 1
        elif re.search(r"(?m)^EPISODES\s*=", source):
            cell["source"] = _update_configuration(source).splitlines(keepends=True)
        elif "trainer.py" in source and "subprocess" in source and "command" in source:
            cell["source"] = _update_training_command(source).splitlines(keepends=True)
        elif "plot_paths = [" in source:
            cell["source"] = _update_result_preview(source).splitlines(keepends=True)

    expected = {filename: 1 for filename in replaced}
    if replaced != expected:
        raise RuntimeError(f"Unexpected writefile cells in {notebook_name}: {replaced}")

    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for notebook_name, source_directory in NOTEBOOK_SOURCES.items():
        synchronize_notebook(notebook_name, source_directory)
        print(f"Updated {notebook_name} from {source_directory}")


if __name__ == "__main__":
    main()
