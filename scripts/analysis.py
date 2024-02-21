import os

import numpy as np
import pandas as pd
import wandb
import yaml
from constants import DATASET_NAMES, MAX_SIZES
from scipy.stats import kendalltau
from utils import (
    bootstrap_corr,
    get_data,
    latexify,
    select_main_results,
    select_sample_results,
)


def compute_correlation(
    data,
    name,
    metric,
):
    x = data[metric].values
    x = [v["score"] for v in x]
    y = data["test_accuracy"].values

    mean, _, ci = bootstrap_corr(x, y, kendalltau)

    # corr = f"{round(mean, 2)}, ({round(ci[0], 2)}, {round(ci[1], 2)})"
    corr = [mean, ci]
    # corr = mean

    row = {
        "dataset": DATASET_NAMES[name[0]],
        "n_samples": name[1],
        "n_metric_samples": name[2],
        f"{metric}": corr,
    }

    return row


def get_correlations(
    group_data,
    metric,
):
    return pd.DataFrame(
        [compute_correlation(data, name, metric) for name, data in group_data]
    )


def compute_regret(
    data,
    name,
    metric,
    n,
):
    # Get best possible value
    oracle = data["test_accuracy"].max()

    # Get metric scores
    x = data[metric].values
    x = np.array([v["score"] for v in x])

    # Get metric recommendations
    sorted_indices = np.argsort(x)[::-1]
    indices = sorted_indices[:n]

    # Separate out to validate selection
    chosen = x[indices]
    not_chosen = x[sorted_indices[n:]]

    # If there are duplicate values
    if np.isin(not_chosen, chosen).any():
        # Find value to randomise
        smallest_chosen = chosen.min()

        # Separate out not smallest chosen
        chosen_not_smallest = indices[chosen != smallest_chosen]

        # Get all indices corresponding to smallest value
        chosen_smallest_inds = np.argwhere(x == smallest_chosen).reshape(
            -1,
        )

        # Compute regret over all values
        regret_list = []
        for ind in chosen_smallest_inds:
            selection = np.append(chosen_not_smallest, ind)
            outcome = data["test_accuracy"].iloc[selection].max()
            regret_list.append(oracle - outcome)

        # Compute regret as mean of list
        regret = np.mean(regret_list)

        # else, if no duplicate values to deal with (one can dream)
    else:
        outcome = data["test_accuracy"].iloc[indices].max()
        regret = oracle - outcome

    # Construct output
    row = {
        "dataset": DATASET_NAMES[name[0]],
        "n_samples": name[1],
        "n_metric_samples": name[2],
        metric: regret,
    }

    return row


def get_regrets(
    group_data,
    metric,
    n,
):
    return pd.DataFrame(
        [compute_regret(data, name, metric, n) for name, data in group_data]
    )


def select_results(
    select_fn,
    id_str,
    *args,
    **kwargs,
):
    out_list = [select_fn(arg, **kwargs) for arg in args]
    out = out_list[0]
    for i in range(1, len(out_list)):
        out = out.merge(out_list[i], on=["dataset", "n_samples", "n_metric_samples"])
    out = out.loc[
        :,
        out.columns.isin([id_str, "renggli", "LogME", "n_pars", "imagenet-validation"]),
    ]
    return out


def main():
    # File information
    path = "data"
    file = "results.pickle"
    filepath = os.path.join(path, file)

    # If data exists, read in from json instead of pulling. Else, pull from wandb
    if False:  # os.path.isfile(filepath): # ignore this idea for now because it's buggy
        data = pd.read_pickle(filepath)
    else:
        # Setup API
        api = wandb.Api()

        # Dataset names
        datasets = list(MAX_SIZES.keys())

        # Get data
        train_data = get_data(api, "train", datasets)
        metric_data = get_data(api, "metrics", datasets)

        # Merge
        data = metric_data.merge(
            train_data, on=["dataset_name", "model_name", "n_samples"]
        )

        # Read in imagenet validation accuracies
        with open("configs/scores_imagenet1k.yaml") as f:
            imagenet = yaml.safe_load(f)
        imagenet = [
            {"model_name": key, "imagenet-validation": {"score": imagenet[key]}}
            for key in imagenet.keys()
        ]
        imagenet = pd.DataFrame(imagenet)

        # Merge in imagenet validation accuracies
        data = data.merge(imagenet, on="model_name")

        # Multiply Test Accuracy by 100 for regret purposes
        data["test_accuracy"] = data["test_accuracy"] * 100

        # Create dir for future use
        if not os.path.exists(path):
            os.mkdir(path)

        # Write file
        data.to_pickle(filepath)

    # Create output dir if it doesn't already exist
    if not os.path.exists("tables"):
        os.mkdir("tables")

    # -------------------------------------------------------------------------------- #
    # Main Analysis: Within datasets ------------------------------------------------- #
    # -------------------------------------------------------------------------------- #

    # Group data by dataset, n_samples, n_metric_samples (vary model within group)
    group_data = data.groupby(["dataset_name", "n_samples", "n_metric_samples"])

    # Get corrs for each metric
    table1 = get_correlations(group_data, "renggli")
    table2 = get_correlations(group_data, "LogME")
    table3 = get_correlations(group_data, "n_pars")
    table4 = get_correlations(group_data, "imagenet-validation")

    # 1) Do the metrics work? -------------------------------------------------------- #

    # Table of main results
    main_results = select_results(
        select_main_results, "dataset", table1, table2, table3, table4
    )

    # Create latex file
    latexify(
        table=main_results,
        labels=["Dataset", "Renggli", "LogME", "N. Params", "ImageNet Acc."],
        t_name="Metrics Performance in Best Case Scenario",
        t_label="main-results",
        type="corr",
    )

    # 2) Varying Dataset Size? ------------------------------------------------------- #

    # Get tables for each dataset
    sample_results = [
        select_results(
            select_sample_results,
            "n_samples",
            table1,
            table2,
            table3,
            table4,
            dataset_name=dataset_name,
        )
        for dataset_name in DATASET_NAMES.values()
    ]

    # Create latex files
    for i, result in enumerate(sample_results):
        d_name = list(DATASET_NAMES.values())[i]
        latexify(
            table=result,
            labels=["N. Images", "Renggli", "LogME", "N. Params", "ImageNet Acc."],
            t_name="Metrics Performance Across Dataset Sizes: " + d_name,
            t_label="subsample-results-" + d_name,
            type="corr",
        )

    # 3) Allow subsampling? ---------------------------------------------------------- #

    # TODO

    # -------------------------------------------------------------------------------- #
    # Secondary analysis: how low cost are the metrics? ------------------------------ #
    # -------------------------------------------------------------------------------- #

    # TODO

    # -------------------------------------------------------------------------------- #
    # Robustness: Regret instead of rank correlation --------------------------------- #
    # -------------------------------------------------------------------------------- #

    # Compute Regrets
    regret_1 = get_regrets(group_data, "renggli", 1)
    regret_2 = get_regrets(group_data, "LogME", 1)
    regret_3 = get_regrets(group_data, "n_pars", 1)
    regret_4 = get_regrets(group_data, "imagenet-validation", 1)

    # Get regret results
    regret_results = select_results(
        select_main_results, "dataset", regret_1, regret_2, regret_3, regret_4
    )

    # Create latex file
    latexify(
        table=regret_results,
        labels=["Dataset", "Renggli", "LogME", "N. Params", "ImageNet Acc."],
        t_name="Metrics Performance Assessed by Regret",
        t_label="regret-results",
        type="regret",
    )

    # Get regret subsample results
    regret_subsample_results = [
        select_results(
            select_sample_results,
            "n_samples",
            regret_1,
            regret_2,
            regret_3,
            regret_4,
            dataset_name=dataset_name,
        )
        for dataset_name in DATASET_NAMES.values()
    ]

    # Create latex file
    for i, result in enumerate(regret_subsample_results):
        d_name = list(DATASET_NAMES.values())[i]
        latexify(
            table=result,
            labels=["N. Images", "Renggli", "LogME", "N. Params", "ImageNet Acc."],
            t_name="Metrics Regret Across Dataset Sizes: " + d_name,
            t_label="subsample-regret-" + d_name,
            type="regret",
        )

    # -------------------------------------------------------------------------------- #
    # Robustness: Within model instead of within dataset ----------------------------- #
    # -------------------------------------------------------------------------------- #

    # TODO

    # -------------------------------------------------------------------------------- #
    # Mean Results incl. slack questions --------------------------------------------- #
    # -------------------------------------------------------------------------------- #

    c_all = pd.concat(sample_results)
    r_all = pd.concat(regret_subsample_results)

    def make_mean_plus_ci(x):
        m = np.mean(x)
        se = np.std(x) / np.sqrt(len(x))
        # return f"{m:.4f}, ({m - 2*se:.4f} to {m + 2*se:.4f})"
        return {
            "mean": m,
            "lower": m - 2 * se,
            "upper": m + 2 * se,
        }

    a = (
        c_all[c_all.columns[1:5]]
        .apply(lambda x: [v[0] for v in x], axis=0)
        .apply(lambda x: make_mean_plus_ci(x))
    )
    b = r_all[r_all.columns[1:5]].apply(lambda x: make_mean_plus_ci(x))

    c = pd.concat([a, b], axis=1).transpose()
    d = pd.concat(
        [pd.DataFrame([{"Metric": "Kendall's Tau"}, {"Metric": "Regret"}]), c], axis=1
    )

    latexify(
        table=d,
        labels=["Metric", "Renggli", "LogME", "N. Params", "ImageNet Acc."],
        t_name="Mean Metric Performance According to Metric",
        t_label="mean-results",
        type="mean",
    )

    # main_results[main_results.columns[1:5]].apply(
    #     lambda x: [v[0] for v in x], axis=0
    #     ).mean()
    # regret_results[regret_results.columns[1:5]].mean()

    # main_results[main_results.columns[1:5]].apply(
    #     lambda x: [v[0] for v in x], axis=0
    #     ).apply(lambda x: [np.mean(x), np.std(x)/np.sqrt(len(x))])
    # regret_results[regret_results.columns[1:5]].apply(
    #     lambda x: [np.mean(x), np.std(x)/np.sqrt(len(x))]
    #     )

    # c_all[c_all.columns[1:5]].apply(lambda x: [v[0] for v in x], axis=0).mean()
    # r_all[r_all.columns[1:5]].mean()

    # c_all[c_all.columns[1:5]].apply(lambda x: [v[0] for v in x], axis=0).apply(
    #     lambda x: [np.mean(x), np.std(x)/np.sqrt(len(x))]
    #     )
    # r_all[r_all.columns[1:5]].apply(
    #     lambda x: [np.mean(x), np.std(x)/np.sqrt(len(x))]
    #     )

    # c_all[c_all.columns[1:5]].apply(
    #     lambda x: [v[0] for v in x], axis=0
    #     ).apply(lambda x: make_mean_plus_ci(x))
    # r_all[r_all.columns[1:5]].apply(lambda x: make_mean_plus_ci(x))

    # Placeholder
    print("Hello, world")


if __name__ == "__main__":
    main()
