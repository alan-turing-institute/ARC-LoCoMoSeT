import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def load_actual_scores() -> dict[str, float]:
    with open("results/model_scores_imagenet1k.json", "r") as f:
        actuals = json.load(f)
    return actuals


def load_results(file_paths: list[str], n_samples=None) -> list[dict]:
    """Load results files from metric scans.

    Args:
        file_paths: Paths to results JSON files.
        n_samples: If set, only load results for this many samples.

    Returns:
        list[dict]: _description_
    """
    results = []
    for rf in file_paths:
        with open(rf, "r") as f:
            this_file = json.load(f)
            if n_samples is not None and this_file["n_samples"] != n_samples:
                continue
            results.append(this_file)
    return results


def get_scores_actuals(
    results: list[dict], actuals: dict[str, float]
) -> dict[str, dict]:
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
        parsed_results[metric]["actuals"][model].append(actuals[model])

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
        description="Make a plot of metric scores vs. no. images"
    )
    parser.add_argument("results_files", nargs="+", help="Path(s) to results file(s)")
    parser.add_argument(
        "--n_samples",
        required=False,
        default=None,
        type=int,
        help="If set, only plot results with this many samples.",
    )
    args = parser.parse_args()
    results = load_results(args.results_files, args.n_samples)
    actuals = load_actual_scores()
    parsed_results = get_scores_actuals(results, actuals)
    for metric in parsed_results:
        _, ax = plt.subplots(1, 1)
        _ = plot_results(
            parsed_results[metric]["scores"],
            parsed_results[metric]["actuals"],
            metric,
            args.n_samples,
            ax,
        )
        plt.show()


if __name__ == "__main__":
    main()
