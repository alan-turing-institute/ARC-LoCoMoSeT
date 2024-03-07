"""
    From a wandb group name plot collection of graphs of statistics.
"""

import argparse
import os

import numpy as np
from scipy.stats import spearmanr as spr

from locomoset.plots import utils


def plot_scores_vs_val(
    metric_scores: dict, val_acc: dict, metric_name: str, title: str, save_path: str
) -> None:
    """Plot metric scores vs. validation accuracy with and without outliers

    Args:
        metric_scores: metric scores dictionary
        val_acc: validation accuracy dictionary
        metric_name: metric name
        title: plot title
        save_path: path to directory for saving graphs
    """
    utils.plot_results(
        metric_scores=metric_scores,
        other_scores=val_acc,
        metric_axis="x",
        metric_label=f"{metric_name} score",
        other_label="validation accuracy",
        title=title,
        save_path=save_path + f"/{str(title).replace(' ', '-')}.png",
    )

    # remove outliers
    val_acc_no_outliers = {
        k: v
        for k, v in val_acc.items()
        if v > np.mean(list(val_acc.values())) - 3 * np.var(list(val_acc.values()))
    }
    metric_scores_no_outliers = {
        model: metric_scores[model] for model in val_acc_no_outliers.keys()
    }
    title_without_outliers = title + ", no outliers"
    utils.plot_results(
        metric_scores=metric_scores_no_outliers,
        other_scores=val_acc_no_outliers,
        metric_axis="x",
        metric_label=f"{metric_name} score",
        other_label="validation accuracy",
        title=title_without_outliers,
        save_path=save_path + f"/{str(title_without_outliers).replace(' ', '-')}.png",
    )


def plot_scores_vs_val_ranked(
    metric_scores: dict, val_acc: dict, metric_name: str, title: str, save_path: str
) -> None:
    """Plot ranked metric scores vs ranked validation accuracy with and without outliers

    Args:
        metric_scores: metric scores dictionaruy
        val_acc: validation accuracy dictionary
        metric_name: metric name
        title: plot title
        save_path: path for saving the plots
    """
    sorted_val_acc = {
        k: v for k, v in sorted(val_acc.items(), key=lambda item: item[1], reverse=True)
    }
    val_acc_rank = {k: idx + 1 for idx, k in enumerate(sorted_val_acc.keys())}
    sorted_metric_scores = {
        k: v
        for k, v in sorted(
            metric_scores.items(), key=lambda item: item[1], reverse=True
        )
    }
    metric_rank = {k: idx + 1 for idx, k in enumerate(sorted_metric_scores.keys())}
    ranked_title = title + ", ranked"
    utils.plot_results(
        metric_scores=metric_rank,
        other_scores=val_acc_rank,
        metric_axis="x",
        metric_label=f"ranked {metric_name} score",
        other_label="ranked validation accuracy",
        title=ranked_title,
        save_path=save_path + f"/{str(ranked_title).replace(' ', '-')}.png",
    )
    val_acc_no_outliers = {
        k: v
        for k, v in val_acc.items()
        if v > np.mean(list(val_acc.values())) - 3 * np.var(list(val_acc.values()))
    }
    sorted_val_acc_no_outliers = {
        k: v
        for k, v in sorted(
            val_acc_no_outliers.items(), key=lambda item: item[1], reverse=True
        )
    }
    val_acc_rank_no_outliers = {
        k: idx + 1 for idx, k in enumerate(sorted_val_acc_no_outliers.keys())
    }
    metric_scores_no_outliers = {
        model: metric_scores[model] for model in val_acc_no_outliers.keys()
    }
    sorted_metric_scores_no_outliers = {
        k: v
        for k, v in sorted(
            metric_scores_no_outliers.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    metric_rank_no_outliers = {
        k: idx + 1 for idx, k in enumerate(sorted_metric_scores_no_outliers)
    }
    ranked_no_outliers_title = title + ", without outliers, ranked"
    utils.plot_results(
        metric_scores=metric_rank_no_outliers,
        other_scores=val_acc_rank_no_outliers,
        metric_axis="x",
        metric_label=f"ranked {metric_name} score",
        other_label="ranked validation accuracy",
        title=ranked_no_outliers_title,
        save_path=save_path + f"/{str(ranked_no_outliers_title).replace(' ', '-')}.png",
    )


def plot_score_vs_samples(
    metric_scores: dict, val_acc: dict, metric_name: str, title: str, save_path: str
) -> None:
    """Plot metric score vs number of samples, for both raw scores for each metric and
    for correlation with the validation accuracy.

    Args:
        metric_scores: metric scores dictionary with different sample number results
        val_acc: validation accuracy
        metric_name: which metric is being used
        title: base title for the plot
        save_path: path to directory for saving the plots
    """
    n_samps = {}
    met_scores = {}
    for n_samp in metric_scores.keys():
        for model in metric_scores[n_samp]:
            if model not in met_scores.keys():
                met_scores[model] = []
                n_samps[model] = []
            met_scores[model].append(metric_scores[n_samp][model])
            n_samps[model].append(n_samp)

    utils.plot_results(
        metric_scores=met_scores,
        other_scores=n_samps,
        metric_axis="y",
        metric_label=f"{metric_name} score",
        other_label="number of samples",
        title=title,
        save_path=save_path + f"/{str(title).replace(' ', '-')}.png",
    )
    log_title = title + ", logged sample number"
    utils.plot_results(
        metric_scores=met_scores,
        other_scores=n_samps,
        metric_axis="y",
        metric_label=f"{metric_name} score",
        other_label="number of samples",
        title=log_title,
        save_path=save_path + f"/{str(log_title).replace(' ', '-')}.png",
        log_scale=True,
    )

    # Plot without outliers
    val_acc_no_outliers = {
        k: v
        for k, v in val_acc.items()
        if v > np.mean(list(val_acc.values())) - 3 * np.var(list(val_acc.values()))
    }
    n_samps = {}
    met_scores = {}
    for n_samp in metric_scores.keys():
        for model in val_acc_no_outliers.keys():
            if model not in met_scores.keys():
                met_scores[model] = []
                n_samps[model] = []
            met_scores[model].append(metric_scores[n_samp][model])
            n_samps[model].append(n_samp)

    title += ", no outliers"
    utils.plot_results(
        metric_scores=met_scores,
        other_scores=n_samps,
        metric_axis="y",
        metric_label=f"{metric_name} score",
        other_label="number of samples",
        title=title + ", no outliers",
        save_path=save_path + f"/{str(title).replace(' ', '-')}.png",
    )
    log_title = title + ", logged sample number"
    utils.plot_results(
        metric_scores=met_scores,
        other_scores=n_samps,
        metric_axis="y",
        metric_label=f"{metric_name} score",
        other_label="number of samples",
        title=log_title + ", no outliers",
        save_path=save_path + f"/{str(log_title).replace(' ', '-')}.png",
        log_scale=True,
    )


def plot_correlation_vs_samples(
    metric_scores: dict, val_acc: dict, metric_name: str, title: str, save_path: str
) -> None:
    """Plot the (spearmans rank) correlation between the validation accuracy and the

    Args:
        metric_scores: _description_
        val_acc: _description_
        metric_name: _description_
        title: _description_
        save_path: _description_
    """
    correlation_scores = {}
    n_samps = {}

    for idx, n_samp in enumerate(metric_scores.keys()):
        correlation_scores[idx] = spr(
            [metric_scores[n_samp][model] for model in val_acc.keys()],
            [val_acc[model] for model in val_acc.keys()],
        )[0]
        n_samps[idx] = n_samp

    utils.plot_results(
        metric_scores=correlation_scores,
        other_scores=n_samps,
        metric_axis="y",
        metric_label=f"corr({metric_name}, val_acc)",
        other_label="number of samples",
        title=title,
        save_path=save_path + f"/{str(title).replace(' ', '-')}.png",
    )

    log_title = title + ", log scale"
    utils.plot_results(
        metric_scores=correlation_scores,
        other_scores=n_samps,
        metric_axis="y",
        metric_label=f"corr({metric_name}, val_acc)",
        other_label="number of samples",
        title=log_title,
        save_path=save_path + f"/{str(log_title).replace(' ', '-')}.png",
        log_scale=True,
    )

    # remove outliers
    val_acc_no_outliers = {
        k: v
        for k, v in val_acc.items()
        if v > np.mean(list(val_acc.values())) - 3 * np.var(list(val_acc.values()))
    }
    correlation_scores = {}
    n_samps = {}

    for idx, n_samp in enumerate(metric_scores.keys()):
        correlation_scores[idx] = spr(
            [metric_scores[n_samp][model] for model in val_acc_no_outliers.keys()],
            [val_acc_no_outliers[model] for model in val_acc_no_outliers.keys()],
        )[0]
        n_samps[idx] = n_samp

    title += ", no outliers"
    utils.plot_results(
        metric_scores=correlation_scores,
        other_scores=n_samps,
        metric_axis="y",
        metric_label=f"corr({metric_name}, val_acc)",
        other_label="number of samples",
        title=title,
        save_path=save_path + f"/{str(title).replace(' ', '-')}.png",
    )

    log_title = title + ", log scale"
    utils.plot_results(
        metric_scores=correlation_scores,
        other_scores=n_samps,
        metric_axis="y",
        metric_label=f"corr({metric_name}, val_acc)",
        other_label="number of samples",
        title=log_title,
        save_path=save_path + f"/{str(log_title).replace(' ', '-')}.png",
        log_scale=True,
    )


def generate_plots(
    group_name: str, save_path: str, imagenet_scores_path: str | None = None
) -> None:
    """Generate the following plots for a set of results from weights and biases:

        - Metric scores vs. validation accuracy
        - (optional) Imagenet validation accuracy vs. validation accuracy
        - Ranked Metric scores vs. ranked validation accuracy
        - Ranked Imagenet validation accuracy vs. ranked validation accuracy
        - Metric scores vs. n samples
        - Correlation(metric scores, validation accuracy) vs. n samples
        - All above with outlier models (by validation accuracy score) remove

    Where applicable the metric scores will only use the max number of samples for
    plots.

    Args:
        group_name: weights and biases group name
        save_path: path to save the plots
        imagenet_scores_path (optional): path to imagenet scores, if given the plots
                                         will be made
    """
    metrics_df, train_df = utils.load_results_wandb(group_name)
    metric_res, train_res = utils.parse_results_dataframes(metrics_df, train_df)

    # Metric scores vs validation accuracy
    for metric in metric_res.keys():
        title = f"{metric} score vs validation accuracy"
        if metric == "n_pars":
            metric_scores = metric_res[metric]
        else:
            max_n_samples = max(metric_res[metric].keys())
            metric_scores = metric_res[metric][max_n_samples]
            title += f" n={max_n_samples}"

        plot_scores_vs_val(
            metric_scores=metric_scores,
            val_acc=train_res,
            metric_name=metric,
            title=title,
            save_path=save_path,
        )
        plot_scores_vs_val_ranked(
            metric_scores=metric_scores,
            val_acc=train_res,
            metric_name=metric,
            title=title,
            save_path=save_path,
        )

    # Image net graphs
    if imagenet_scores_path is not None:
        imagenet_scores = utils.load_imagenet_acc(imagenet_scores_path)
        title = "ImageNet validation accuracy vs validation accuracy"
        plot_scores_vs_val(
            imagenet_scores,
            train_res,
            metric_name="ImageNet_val_acc",
            title=title,
            save_path=save_path,
        )
        plot_scores_vs_val_ranked(
            imagenet_scores,
            train_res,
            metric_name="ImageNet_val_acc",
            title=title,
            save_path=save_path,
        )

    # Metric scores and correlation, vs n samples
    for metric in metric_res.keys():
        if metric == "n_pars":
            continue
        metric_scores = metric_res[metric]

        title = f"{metric} score vs number of samples"
        plot_score_vs_samples(
            metric_scores=metric_scores,
            val_acc=train_res,
            metric_name=metric,
            title=title,
            save_path=save_path,
        )

        title = f"Correlation of {metric} score and val acc vs number of samples"
        plot_correlation_vs_samples(
            metric_scores=metric_scores,
            val_acc=train_res,
            metric_name=metric,
            title=title,
            save_path=save_path,
        )


def main() -> None:
    desc1 = "Create a collection of plots of metric scores vs. other quantities from a"
    desc2 = " weights and biases group name"
    parser = argparse.ArgumentParser(description=desc1 + desc2)
    parser.add_argument("wandb_group_name", help="Weights and biases group name")
    parser.add_argument("imagenet_scores_path", help="Path to ImageNet accuracy scores")
    args = parser.parse_args()
    save_path = f"results/{str(args.wandb_group_name).replace('/','-')}"
    os.makedirs(save_path, exist_ok=True)
    generate_plots(args.wandb_group_name, save_path, args.imagenet_scores_path)
