import argparse

from matplotlib.axes import Axes

from locomoset.plots.utils import load_results, parse_results, plot_results


def plot_metric_vs_samples(
    metric_scores: dict[str, list[float]],
    n_samples: dict[str, list[int]],
    metric: str,
    save_path: str,
    **scatter_args,
) -> Axes:
    """Make a metric score vs. no. images plot.

    Args:
        metric_scores: Dict of model_name: list of metric scores.
        n_samples: Dict of model_name: list of no. images used to compute scores above.
        save_path: File path where plot will be saved.
        **scatter_args: Additional keyword arguments to pass to plt.scatter.

    Returns:
        Plot axes.
    """
    return plot_results(
        metric_scores,
        n_samples,
        "y",
        f"{metric} Score",
        "No. Images",
        f"{metric} Score vs. No. Images [ImageNet-1k]",
        save_path,
        **scatter_args,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a plot of metric scores vs. no. images"
    )
    parser.add_argument(
        "results_files",
        nargs="+",
        help="Paths to results files or a single directory containing results files.",
    )
    args = parser.parse_args()
    results, save_dir = load_results(args.results_files)
    parsed_results = parse_results(results)
    for metric in parsed_results:
        plot_metric_vs_samples(
            parsed_results[metric]["scores"],
            parsed_results[metric]["n_samples"],
            metric,
            f"{save_dir}/{metric}_vs_samples.png",
        )
