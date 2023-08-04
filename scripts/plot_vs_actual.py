import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from locomoset.models.scores import imagenet1k_scores


def load_results(file_paths: list[str], n_samples=None) -> list[dict]:
    """Load results files from metric scans.

    Args:
        file_paths: Paths to results JSON files.
        n_samples: If set, only load results for this many samples.

    Returns:
        Loaded results JSON files.
    """
    results = []
    for rf in file_paths:
        with open(rf, "r") as f:
            this_file = json.load(f)
            if n_samples is not None and this_file["n_samples"] != n_samples:
                continue
            results.append(this_file)
    return results


def get_scores_actuals(results: list[dict]) -> dict[str, dict]:
    """Parse results into a dict containing metric scores and actual fine-tuned
    performance for each metric present in the input results list.

    Args:
        results: Loaded results JSON files.

    Returns:
        Dict of extracted results for plotting.
    """
    parsed_results = {}

    for r in results:
        metric = r["metric"]
        model = r["model_name"]
        if metric not in parsed_results:
            parsed_results[metric] = {}
            parsed_results[metric]["scores"] = {}
            parsed_results[metric]["actuals"] = {}
        if model not in parsed_results[metric]["scores"]:
            parsed_results[metric]["scores"][model] = []
            parsed_results[metric]["actuals"][model] = []

        parsed_results[metric]["scores"][model].append(r["result"]["score"])
        parsed_results[metric]["actuals"][model].append(imagenet1k_scores[model])

    return parsed_results


def plot_results(
    metric_scores: dict[str, list[float]],
    actual_scores: dict[str, list[float]],
    metric: str,
    n_samples: int,
    ax: Axes,
    **scatter_args,
) -> Axes:
    """Make a metric score vs. model score plot for a result file saved from the
    run_renggli.py script.

    Args:
        metric_scores: Dict of model_name: list of metric scores
        actual_scores: Dict of model_name: list of actual model scores (same order as
            metric_scores).
        n_samples: No. images used to compute scores above.
        ax: Axes to plot to.

    Returns:
        Updated axes.
    """
    for model in metric_scores:
        ax.scatter(
            metric_scores[model], actual_scores[model], label=model, **scatter_args
        )
    ax.legend()
    ax.set_ylabel("ImageNet-1k Top-1 Accuracy")
    ax.set_xlabel(f"{metric} Score")
    ax.set_title(
        f"Fine-tuned Performance vs. {metric} Score "
        f"[ImageNet-1k, {n_samples} images]"
    )
    return ax


def main():
    parser = argparse.ArgumentParser(
        description="Make a plot of metric scores vs. actual fine-tuned performance"
    )
    parser.add_argument(
        "results_files",
        nargs="+",
        help="Paths to results files or a single directory containing results files.",
    )
    parser.add_argument(
        "--n_samples",
        required=False,
        default=None,
        type=int,
        help="If set, only plot results with this many samples.",
    )
    args = parser.parse_args()
    # If the user passes a directory, glob for all JSON files in that directory.
    if len(args.results_files) == 1 and os.path.isdir(args.results_files[0]):
        save_dir = args.results_files[0]
        args.results_files = glob(os.path.join(args.results_files[0], "*.json"))
    else:
        save_dir = "."

    results = load_results(args.results_files, args.n_samples)
    parsed_results = get_scores_actuals(results)
    for metric in parsed_results:
        fig, ax = plt.subplots(1, 1)
        _ = plot_results(
            parsed_results[metric]["scores"],
            parsed_results[metric]["actuals"],
            metric,
            args.n_samples,
            ax,
        )
        fig.tight_layout()
        fig.savefig(f"{save_dir}/actual_vs_{metric}.png")
        print("Saved figure to", f"{save_dir}/actual_vs_{metric}.png")


if __name__ == "__main__":
    main()
