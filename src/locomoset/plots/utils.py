import json
import os
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def load_results(file_paths: list[str], n_samples=None) -> (list[dict], str):
    """Load results files from metric scans.

    Args:
        file_paths: Paths to results JSON files, or a single directory containing them.
        n_samples: If set, only load results for this many samples.

    Returns:
        (Loaded results JSON files, path to directory to save results)
    """
    # If the user passes a directory, glob for all JSON files in that directory.
    if len(file_paths) == 1 and os.path.isdir(file_paths[0]):
        save_dir = file_paths[0]
        file_paths = glob(os.path.join(file_paths[0], "*.json"))
    else:
        save_dir = "."

    results = []
    for rf in file_paths:
        with open(rf, "r") as f:
            this_file = json.load(f)
            if n_samples is not None and this_file["n_samples"] != n_samples:
                continue
            results.append(this_file)
    return results, save_dir


def parse_results(results: list[dict], actuals: dict | None = None) -> dict[str, dict]:
    """Parse results into a dict containing metric scores, the no. of images
    (samples) used to compute them, and optionally the model's actual fine-tuned
    performance for each metric present in the input results list.

    Args:
        results: Loaded results JSON files.
        actuals: Loaded mapping from model names to fine-tuned performance scores.

    Returns:
        Dict of extracted results for plotting.
    """
    parsed_results = {}
    keys_to_add = ["scores", "n_samples", "actuals"]

    for r in results:
        metric = r["metric"]
        model = r["model_name"]
        if metric not in parsed_results:
            parsed_results[metric] = {}
            for k in keys_to_add:
                parsed_results[metric][k] = {}
        if model not in parsed_results[metric]["scores"]:
            for k in keys_to_add:
                parsed_results[metric][k][model] = []

        parsed_results[metric]["scores"][model].append(r["result"]["score"])
        parsed_results[metric]["n_samples"][model].append(r["n_samples"])
        if actuals is not None:
            parsed_results[metric]["actuals"][model].append(actuals[model])

    return parsed_results


def plot_results(
    metric_scores: dict[str, list[float]],
    other_scores: dict[str, list[int]],
    metric_axis: str,
    metric_label: str,
    other_label: str,
    title: str,
    save_path: str,
    **scatter_args,
) -> Axes:
    """Make a plot of metric scores vs. another quantity.

    Args:
        metric_scores: Dict of model_name: list of metric scores.
        other_scores: Dict of model_name: other score to plot metric scores against.
        metric_axis: Whether to plot metric scores on the "x" or "y" axis.
        metric_label: Axis label for metric scores axies.
        other_label: Axis label for other scores axis.
        title: Plot title.
        save_path: Path where plot will be saved.
        **scatter_args: Additional keyword arguments to pass to plt.scatter.

    Returns:
        Plot axes.
    """
    fig, ax = plt.subplots(1, 1)
    for model in metric_scores:
        if metric_axis == "x":
            ax.scatter(
                metric_scores[model], other_scores[model], label=model, **scatter_args
            )
        elif metric_axis == "y":
            ax.scatter(
                other_scores[model], metric_scores[model], label=model, **scatter_args
            )
        else:
            raise ValueError(f"{metric_axis} must be 'x' or 'y'.")

    ax.legend()
    if metric_axis == "x":
        ax.set_xlabel(metric_label)
        ax.set_ylabel(other_label)
    else:
        ax.set_xlabel(other_label)
        ax.set_ylabel(metric_label)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path)
    print(f"Saved figure to {save_path}")

    return ax
