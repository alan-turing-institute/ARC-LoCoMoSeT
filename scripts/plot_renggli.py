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


def plot_result(result: dict, ax: Axes, **scatter_args) -> Axes:
    """Make a no. of images vs. Renggli score plot for a result file saved from the
    run_renggli.py script.

    Args:
        result: Loaded results JSON file.
        ax: Axes to plot to.

    Returns:
        Updated axes.
    """
    samples = [r["n_samples"] for r in result["results"]]
    scores = [r["score"] for r in result["results"]]
    ax.scatter(samples, scores, label=result["model_name"], **scatter_args)
    ax.legend()
    ax.set_xlabel("No. images")
    ax.set_ylabel("Renngli Score")
    return ax


def main():
    parser = argparse.ArgumentParser(description="Make a plot of Renggli runs")
    parser.add_argument("results_files", nargs="+", help="Path(s) to results file(s)")
    args = parser.parse_args()
    results = load_results(args.results_files)
    _, ax = plt.subplots(1, 1)
    for r in results:
        plot_result(r, ax)
    plt.show()


if __name__ == "__main__":
    main()
