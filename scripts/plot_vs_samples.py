import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def load_results(file_paths: list[str]) -> list[dict]:
    results = []
    for rf in file_paths:
        with open(rf, "r") as f:
            results.append(json.load(f))
    return results


def get_scores_samples(results: list[dict]) -> dict[str, dict]:
    parsed_results = {}

    for r in results:
        metric = r["metric"]
        model = r["model_name"]
        if metric not in parsed_results:
            parsed_results[metric] = {}
            parsed_results[metric]["scores"] = {}
            parsed_results[metric]["n_samples"] = {}
        if model not in parsed_results[metric]["scores"]:
            parsed_results[metric]["scores"][model] = []
            parsed_results[metric]["n_samples"][model] = []

        parsed_results[metric]["scores"][model].append(r["result"]["score"])
        parsed_results[metric]["n_samples"][model].append(r["n_samples"])

    return parsed_results


def plot_results(
    metric_scores: dict[str, list[float]],
    n_samples: dict[str, list[int]],
    metric: str,
    ax: Axes,
    **scatter_args,
) -> Axes:
    """Make a metric score vs. no. images plot.

    Args:
        metric_scores: Dict of model_name: list of metric scores.
        n_samples: Dict of model_name: list of no. images used to compute scores above.
        ax: Axes to plot to.

    Returns:
        Updated axes.
    """
    for model in metric_scores:
        ax.scatter(n_samples[model], metric_scores[model], label=model, **scatter_args)
    ax.legend()
    ax.set_xlabel("No. Images")
    ax.set_ylabel(f"{metric} Score")
    return ax


def main():
    parser = argparse.ArgumentParser(
        description="Make a plot of metric scores vs. no. images"
    )
    parser.add_argument("results_files", nargs="+", help="Path(s) to results file(s)")
    args = parser.parse_args()
    results = load_results(args.results_files)
    parsed_results = get_scores_samples(results)
    for metric in parsed_results:
        _, ax = plt.subplots(1, 1)
        _ = plot_results(
            parsed_results[metric]["scores"],
            parsed_results[metric]["n_samples"],
            metric,
            ax,
        )
        plt.show()


if __name__ == "__main__":
    main()
