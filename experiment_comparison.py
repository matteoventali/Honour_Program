"""Interactive comparison of training trends stored below a framework's results/.

The module is shared by the small ``compare_experiments.py`` entry points in
each active framework.  It can also run without a display, which is useful for
servers and automated checks.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np


METRIC_LABELS = {
    "learning_rewards": "Learning reward",
    "task_rewards": "Task reward",
    "shaping_rewards": "Shaping reward",
    "successes": "Success rate",
    "episode_lengths": "Episode length",
    "abstract_changes": "Abstract-state changes",
    "dfa_transitions": "DFA transitions",
}
DEFAULT_METRIC = "learning_rewards"
SEED_FILE_RE = re.compile(r"_seed_-?\d+\.npz$")


@dataclass(frozen=True)
class Experiment:
    """A named result directory and the metrics available in its NPZ files."""

    name: str
    directory: Path
    files: tuple[Path, ...]
    metrics: frozenset[str]


@dataclass(frozen=True)
class TrainingSummary:
    """Smoothed aggregate used to draw one experiment."""

    mean: np.ndarray
    standard_deviation: np.ndarray
    run_count: int
    source: str


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _available_metrics(path: Path) -> set[str]:
    """Read only the array names needed to identify plottable metrics."""
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
    except (OSError, ValueError):
        return set()

    available = set()
    for metric in METRIC_LABELS:
        if metric in keys or f"{metric}_runs" in keys or f"{metric}_mean" in keys:
            available.add(metric)
    return available


def discover_experiments(results_dir: Path) -> list[Experiment]:
    """Discover valid experiment directories immediately below ``results_dir``."""
    results_dir = results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        return []

    experiments: list[Experiment] = []
    for directory in sorted(
        (item for item in results_dir.iterdir() if item.is_dir()),
        key=lambda item: item.name.casefold(),
    ):
        files = tuple(sorted(directory.rglob("*.npz")))
        metrics: set[str] = set()
        for path in files:
            metrics.update(_available_metrics(path))
        if metrics:
            experiments.append(
                Experiment(
                    name=directory.name,
                    directory=directory,
                    files=files,
                    metrics=frozenset(metrics),
                )
            )

    # Keep compatibility with old layouts that wrote NPZ files directly in results/.
    for path in sorted(results_dir.glob("*.npz")):
        metrics = _available_metrics(path)
        if metrics:
            experiments.append(
                Experiment(
                    name=path.stem,
                    directory=results_dir,
                    files=(path,),
                    metrics=frozenset(metrics),
                )
            )
    return experiments


def _as_runs(array: np.ndarray, source: Path, key: str) -> np.ndarray:
    runs = np.asarray(array, dtype=np.float64)
    if runs.ndim == 1:
        runs = runs[np.newaxis, :]
    if runs.ndim != 2 or runs.shape[0] == 0 or runs.shape[1] == 0:
        raise ValueError(
            f"{source}: metric {key!r} must have shape (runs, episodes)"
        )
    return runs


def _centered_moving_average(runs: np.ndarray, window: int) -> np.ndarray:
    """Apply a centered, NaN-aware moving average while preserving length."""
    window = min(window, runs.shape[1])
    kernel = np.ones(window, dtype=np.float64)
    smoothed = np.empty_like(runs, dtype=np.float64)
    for index, run in enumerate(runs):
        valid = np.isfinite(run)
        sums = np.convolve(np.where(valid, run, 0.0), kernel, mode="same")
        counts = np.convolve(valid.astype(np.float64), kernel, mode="same")
        smoothed[index] = np.divide(
            sums,
            counts,
            out=np.full_like(sums, np.nan, dtype=np.float64),
            where=counts > 0,
        )
    return smoothed


def _summary_from_runs(
    runs: np.ndarray, window: int, source: str
) -> TrainingSummary:
    smoothed = _centered_moving_average(runs, window)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(smoothed, axis=0)
        variance = np.nanvar(smoothed, axis=0)
    return TrainingSummary(
        mean=mean,
        standard_deviation=np.sqrt(variance),
        run_count=runs.shape[0],
        source=source,
    )


def load_training_summary(
    experiment: Experiment, metric: str, window: int
) -> TrainingSummary:
    """Load one metric, preferring aggregate multi-seed data when available."""
    if metric not in METRIC_LABELS:
        raise ValueError(f"Unsupported metric: {metric}")
    if metric not in experiment.metrics:
        raise ValueError(
            f"Experiment {experiment.name!r} does not contain {metric!r}"
        )
    if window <= 0:
        raise ValueError("The smoothing window must be greater than zero")

    runs_key = f"{metric}_runs"
    aggregate_candidates: list[tuple[int, Path, np.ndarray]] = []
    for path in experiment.files:
        try:
            with np.load(path, allow_pickle=False) as data:
                if runs_key in data:
                    runs = _as_runs(data[runs_key], path, runs_key)
                    aggregate_candidates.append((runs.shape[0], path, runs))
        except (OSError, ValueError) as error:
            if runs_key in str(error):
                raise

    if aggregate_candidates:
        # Prefer the file with the largest seed set, then a non-seed aggregate.
        _, path, runs = max(
            aggregate_candidates,
            key=lambda item: (
                item[0],
                not bool(SEED_FILE_RE.search(item[1].name)),
                -len(item[1].parts),
            ),
        )
        return _summary_from_runs(runs, window, str(path))

    seed_files = [path for path in experiment.files if SEED_FILE_RE.search(path.name)]
    seed_runs: list[np.ndarray] = []
    for path in seed_files:
        try:
            with np.load(path, allow_pickle=False) as data:
                if metric in data:
                    run = np.asarray(data[metric], dtype=np.float64)
                    if run.ndim == 1 and run.size:
                        seed_runs.append(run)
        except (OSError, ValueError):
            continue
    if seed_runs:
        shortest = min(run.size for run in seed_runs)
        runs = np.stack([run[:shortest] for run in seed_runs])
        return _summary_from_runs(runs, window, f"{len(seed_runs)} file seed")

    # Some older archives saved only precomputed mean and variance.
    mean_key = f"{metric}_mean"
    variance_key = f"{metric}_variance"
    for path in experiment.files:
        try:
            with np.load(path, allow_pickle=False) as data:
                if mean_key in data and variance_key in data:
                    mean = np.asarray(data[mean_key], dtype=np.float64)
                    variance = np.asarray(data[variance_key], dtype=np.float64)
                    if mean.ndim == variance.ndim == 1 and mean.size == variance.size:
                        smoothed_mean = _centered_moving_average(
                            mean[np.newaxis, :], window
                        )[0]
                        smoothed_std = _centered_moving_average(
                            np.sqrt(np.maximum(variance, 0.0))[np.newaxis, :], window
                        )[0]
                        return TrainingSummary(
                            mean=smoothed_mean,
                            standard_deviation=smoothed_std,
                            run_count=0,
                            source=str(path),
                        )
        except (OSError, ValueError):
            continue

    for path in experiment.files:
        if SEED_FILE_RE.search(path.name):
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                if metric in data:
                    runs = _as_runs(data[metric], path, metric)
                    return _summary_from_runs(runs, window, str(path))
        except (OSError, ValueError):
            continue
    raise ValueError(
        f"Unable to load {metric!r} for experiment {experiment.name!r}"
    )


def default_output_path(results_dir: Path, metric: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir / "comparisons" / f"comparison_{metric}_{timestamp}.png"


def _png_output_path(output: Path) -> Path:
    output = output.expanduser().resolve()
    return output if output.suffix else output.with_suffix(".png")


def plot_comparison(
    experiments: Sequence[Experiment],
    metric: str,
    window: int,
    output: Path,
    *,
    show: bool,
) -> list[TrainingSummary]:
    """Plot mean trends and a +/- one-standard-deviation variance band."""
    if not experiments:
        raise ValueError("Select at least one experiment")

    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = [
        load_training_summary(experiment, metric, window)
        for experiment in experiments
    ]

    figure, axes = plt.subplots(figsize=(13, 7.5))
    colors = plt.get_cmap("tab10")
    for index, (experiment, summary) in enumerate(zip(experiments, summaries)):
        episodes = np.arange(1, summary.mean.size + 1)
        color = colors(index % 10)
        if summary.run_count:
            seed_label = "seed" if summary.run_count == 1 else "seeds"
            run_text = f"{summary.run_count} {seed_label}"
        else:
            run_text = "aggregate statistics"
        axes.plot(
            episodes,
            summary.mean,
            color=color,
            linewidth=2.2,
            label=f"{experiment.name} ({run_text})",
        )
        axes.fill_between(
            episodes,
            summary.mean - summary.standard_deviation,
            summary.mean + summary.standard_deviation,
            color=color,
            alpha=0.18,
        )

    metric_label = METRIC_LABELS[metric]
    axes.set_title(f"Experiment comparison — {metric_label}", fontweight="bold")
    axes.set_xlabel(f"Episode (centered moving average, window={window})")
    axes.set_ylabel(metric_label)
    axes.grid(True, linestyle="--", alpha=0.35)
    axes.legend(title="Mean and ±1σ band (variance across seeds)")
    figure.tight_layout()

    output = _png_output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return summaries


def launch_gui(experiments: Sequence[Experiment], results_dir: Path) -> None:
    """Open the experiment selector and create the requested comparison plot."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as error:
        raise RuntimeError(
            "Tkinter is not available. Install python3-tk or use --no-gui."
        ) from error

    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(
            "Unable to open a graphical window. Use --no-gui on a headless server."
        ) from error

    root.title("Training experiment comparison")
    root.minsize(720, 530)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    ttk.Label(
        root,
        text=f"Experiments found in: {results_dir}",
        padding=(12, 12, 12, 6),
    ).grid(row=0, column=0, sticky="ew")

    list_frame = ttk.Frame(root, padding=(12, 0))
    list_frame.grid(row=1, column=0, sticky="nsew")
    list_frame.columnconfigure(0, weight=1)
    list_frame.rowconfigure(0, weight=1)
    selector = tk.Listbox(
        list_frame,
        selectmode=tk.EXTENDED,
        exportselection=False,
        activestyle="dotbox",
    )
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=selector.yview)
    selector.configure(yscrollcommand=scrollbar.set)
    selector.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    for experiment in experiments:
        selector.insert(tk.END, experiment.name)

    select_buttons = ttk.Frame(root, padding=(12, 6))
    select_buttons.grid(row=2, column=0, sticky="ew")
    ttk.Button(
        select_buttons,
        text="Select all",
        command=lambda: selector.selection_set(0, tk.END),
    ).pack(side="left")
    ttk.Button(
        select_buttons,
        text="Clear selection",
        command=lambda: selector.selection_clear(0, tk.END),
    ).pack(side="left", padx=(8, 0))

    options = ttk.LabelFrame(root, text="Plot options", padding=12)
    options.grid(row=3, column=0, padx=12, pady=(0, 8), sticky="ew")
    options.columnconfigure(1, weight=1)

    metric_var = tk.StringVar(value=DEFAULT_METRIC)
    ttk.Label(options, text="Metric:").grid(row=0, column=0, sticky="w")
    metric_box = ttk.Combobox(
        options,
        state="readonly",
        textvariable=metric_var,
        values=list(METRIC_LABELS),
    )
    metric_box.grid(row=0, column=1, sticky="ew", padx=(10, 0))

    window_var = tk.IntVar(value=500)
    ttk.Label(options, text="Smoothing window:").grid(
        row=1, column=0, sticky="w", pady=(8, 0)
    )
    ttk.Spinbox(options, from_=1, to=100000, textvariable=window_var).grid(
        row=1, column=1, sticky="ew", padx=(10, 0), pady=(8, 0)
    )

    output_var = tk.StringVar(
        value=str(default_output_path(results_dir, DEFAULT_METRIC))
    )
    ttk.Label(options, text="PNG file:").grid(
        row=2, column=0, sticky="w", pady=(8, 0)
    )
    output_entry = ttk.Entry(options, textvariable=output_var)
    output_entry.grid(row=2, column=1, sticky="ew", padx=(10, 8), pady=(8, 0))

    def choose_output() -> None:
        selected = filedialog.asksaveasfilename(
            parent=root,
            title="Save comparison",
            defaultextension=".png",
            filetypes=(("PNG image", "*.png"), ("All files", "*.*")),
            initialdir=str(results_dir),
        )
        if selected:
            output_var.set(selected)

    ttk.Button(options, text="Browse…", command=choose_output).grid(
        row=2, column=2, pady=(8, 0)
    )

    status_var = tk.StringVar(value="Select at least two experiments to compare.")
    ttk.Label(root, textvariable=status_var, padding=(12, 2)).grid(
        row=4, column=0, sticky="w"
    )

    def generate() -> None:
        indices = selector.curselection()
        if len(indices) < 2:
            messagebox.showwarning(
                "Incomplete selection",
                "Select at least two experiments.",
                parent=root,
            )
            return
        try:
            window = int(window_var.get())
            selected_experiments = [experiments[index] for index in indices]
            missing = [
                experiment.name
                for experiment in selected_experiments
                if metric_var.get() not in experiment.metrics
            ]
            if missing:
                raise ValueError(
                    f"Metric {metric_var.get()!r} is missing from: {', '.join(missing)}"
                )
            output = _png_output_path(Path(output_var.get()))
            output_var.set(str(output))
            status_var.set("Generating plot…")
            root.update_idletasks()
            plot_comparison(
                selected_experiments,
                metric_var.get(),
                window,
                output,
                show=True,
            )
            status_var.set(f"Plot saved to {output}")
        except (OSError, ValueError, RuntimeError) as error:
            status_var.set("Unable to generate the plot.")
            messagebox.showerror("Error", str(error), parent=root)

    actions = ttk.Frame(root, padding=(12, 4, 12, 12))
    actions.grid(row=5, column=0, sticky="e")
    ttk.Button(actions, text="Close", command=root.destroy).pack(side="right")
    ttk.Button(actions, text="Generate comparison", command=generate).pack(
        side="right", padx=(0, 8)
    )

    root.mainloop()


def build_parser(framework_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare experiment training curves and display variance across "
            "seeds. With no options, open the graphical selector."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=framework_dir / "results",
        help="Directory containing experiments (default: framework results).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        metavar="NAME",
        help="Experiments to compare; omit in GUI mode.",
    )
    parser.add_argument(
        "--metric",
        choices=tuple(METRIC_LABELS),
        default=DEFAULT_METRIC,
    )
    parser.add_argument("--window", type=_positive_int, default=500)
    parser.add_argument("--output", type=Path, help="Path of the final PNG.")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Generate the PNG without opening windows (uses all experiments if omitted).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and metrics, then exit.",
    )
    return parser


def main(framework_dir: Path | None = None) -> int:
    framework_dir = (
        Path(framework_dir).resolve()
        if framework_dir is not None
        else Path(__file__).resolve().parent
    )
    args = build_parser(framework_dir).parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    experiments = discover_experiments(results_dir)

    if args.list:
        if not experiments:
            print(f"No experiments with NPZ data found in {results_dir}")
            return 0
        for experiment in experiments:
            print(f"{experiment.name}: {', '.join(sorted(experiment.metrics))}")
        return 0

    if not experiments:
        print(
            f"Error: no experiments with training data found in {results_dir}",
            file=sys.stderr,
        )
        return 2

    if args.no_gui or args.experiments:
        by_name = {experiment.name: experiment for experiment in experiments}
        names = args.experiments or list(by_name)
        unknown = [name for name in names if name not in by_name]
        if unknown:
            print(
                f"Error: experiments not found: {', '.join(unknown)}",
                file=sys.stderr,
            )
            return 2
        selected = [by_name[name] for name in names]
        output = _png_output_path(
            args.output or default_output_path(results_dir, args.metric)
        )
        try:
            summaries = plot_comparison(
                selected,
                args.metric,
                args.window,
                output,
                show=not args.no_gui,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
        print(f"Plot saved to: {output}")
        for experiment, summary in zip(selected, summaries):
            run_count = summary.run_count or "aggregate"
            run_label = "run" if summary.run_count == 1 else "runs"
            print(
                f"- {experiment.name}: {run_count} {run_label}, "
                f"{summary.mean.size} episodes"
            )
        return 0

    try:
        launch_gui(experiments, results_dir)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
