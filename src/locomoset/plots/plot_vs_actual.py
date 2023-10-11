import argparse

import yaml
from matplotlib.axes import Axes

from locomoset.plots.utils import load_results, parse_results, plot_results


def load_actuals(path: str) -> dict[str, float]:
    """Load actual model scores from a YAML file.

    Args:
        path (str): Path to a YAML file with mappings from model names to fine-tuned
            performance scores, e.g. one line of the file could be:
            microsoft/cvt-13: 81.6

    Returns:
        dict[str, float]: Loaded dictionary of model names to scores.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def plot_actual_vs_metric(
    metric_scores: dict[str, list[float]],
    actual_scores: dict[str, list[float]],
    metric: str,
    n_samples: int,
    save_path: str,
    **scatter_args,
) -> Axes:
    """Make a metric score vs. model score plot for a result file saved from the
    run_renggli.py script.

    Args:
        metric_scores: Dict of model_name: list of metric scores
        actual_scores: Dict of model_name: list of actual model scores (same order as
            metric_scores).
        n_samples: No. images used to compute scores above.
        save_path: File path where plot will be saved.
        **scatter_args: Additional keyword arguments to pass to plt.scatter.

    Returns:
        Updated axes.
    """
    return plot_results(
        metric_scores,
        actual_scores,
        "x",
        f"{metric} Score",
        "ImageNet-1k Top-1 Accuracy",
        f"Fine-tuned Performance vs. {metric} Score [ImageNet-1k, {n_samples} images]",
        save_path,
        **scatter_args,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a plot of metric scores vs. actual fine-tuned performance"
    )
    parser.add_argument(
        "results_files",
        nargs="+",
        help="Paths to results files or a single directory containing results files.",
    )
    parser.add_argument(
        "--scores_file",
        type=str,
        help=(
            "Path to YAML file containing mapping from model names to fine-tuned model "
            "scores."
        ),
    )
    parser.add_argument(
        "--n_samples",
        required=False,
        default=None,
        type=int,
        help="If set, only plot results with this many samples.",
    )
    args = parser.parse_args()
    results, save_dir = load_results(args.results_files, args.n_samples)
    actuals = load_actuals(args.scores_file)
    parsed_results = parse_results(results, actuals)
    for metric in parsed_results:
        plot_actual_vs_metric(
            parsed_results[metric]["scores"],
            parsed_results[metric]["actuals"],
            metric,
            args.n_samples,
            f"{save_dir}/actual_vs_{metric}.png",
        )
