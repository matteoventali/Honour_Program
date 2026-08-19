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

REQUIRED_WRITEFILES = {
    "notebook_manual_experiment.ipynb": ("spatial_regions.py",),
    "notebook_multilevel_framework.ipynb": ("spatial_regions.py",),
}


def _writefile_source(filename: str, body: str) -> list[str]:
    return [f"%%writefile {filename}\n", *body.splitlines(keepends=True)]


def _update_configuration(source: str) -> str:
    source = re.sub(r"(?m)^SHAPING_SCALE\s*=.*\n", "", source)
    source = re.sub(
        r'(?m)^print\(f"Shaping scale: \{SHAPING_SCALE\}"\)\n',
        "",
        source,
    )
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
            r"(?m)^(EPSILON_DECAY\s*=.*\n)",
            r"\1TRAINING_USE_GAMMA = True\n",
            source,
            count=1,
        )
    if 'print(f"Training uses gamma: {TRAINING_USE_GAMMA}")' not in source:
        source = re.sub(
            r'(?m)^(print\(f"Epsilon decay: \{EPSILON_DECAY\}"\)\n)',
            r'\1print(f"Training uses gamma: {TRAINING_USE_GAMMA}")\n',
            source,
            count=1,
        )
    return source


def _update_training_command(source: str) -> str:
    source = source.replace(
        '    "--shaping-scale",\n    str(SHAPING_SCALE),\n',
        "",
    )
    source = source.replace(
        '    "--shaping-scale", str(SHAPING_SCALE),\n',
        "",
    )
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
    source_root = PROJECT_ROOT / source_directory

    existing_targets = {
        "".join(cell.get("source", [])).splitlines()[0].split(maxsplit=1)[1]
        for cell in notebook["cells"]
        if "".join(cell.get("source", [])).startswith("%%writefile ")
    }
    for filename in REQUIRED_WRITEFILES.get(notebook_name, ()):
        if filename in existing_targets:
            continue
        body = (source_root / filename).read_text(encoding="utf-8")
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _writefile_source(filename, body),
        }
        insertion_index = next(
            index
            for index, cell in enumerate(notebook["cells"])
            if "".join(cell.get("source", [])).startswith("%%writefile abstract_mdps.py")
        )
        notebook["cells"].insert(insertion_index, new_cell)
        existing_targets.add(filename)

    writefile_targets = set()
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if source.startswith("%%writefile "):
            writefile_targets.add(source.splitlines()[0].split(maxsplit=1)[1])

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
