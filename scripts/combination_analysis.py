"""
    Analysis for combining the metrics. General plan:

        1. Extract data (based on Phil's code)
        2. Scale the metric scores so they're on a similar scale.
        3. Create metric combinations.
        4. Compute correlations and regret for said combinations.
        5. Export LaTeX tables.
        6. Do some manual edits for the final tables.

    The plan is to get both the main (i.e. top level) and mean results for all
    combinations.
"""

import os
import pprint
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
import utils
import wandb
import yaml
from constants import DATASET_NAMES, MAX_SIZES, METRIC_NAMES
from scipy.stats import kendalltau, zscore


def scale_by_max(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Scale a metric score column by the max value in the column

    Args:
        data: overall dataframe
        metric: metric choice

    Returns:
        dataframe with rescaled metric score
    """
    data[metric] = data[metric].div(max(data[metric]))
    return data


def zscore_scale(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Z-score scale a metric score column.

    Args:
        data: overall dataframe
        metric: metric choice

    Returns:
        dataframe with rescaled metric score
    """
    data[metric] = zscore(data[metric])
    return data


def extract_metric_scores_and_scale(
    data: pd.DataFrame, scaling_fn: Callable
) -> pd.DataFrame:
    """Extract the metric scores and scale them by given scaling function.

    N.B. As with most of this analysis work this is hard coded to work with the current
    choices of metrics.

    Args:
        data: overall dataframe
        metric: scaling function choice

    Returns:
        dataframe with metric scores extracted and scaled to be on similar scale.
    """
    for metric in METRIC_NAMES:
        x = data[metric].values
        x = [v["score"] for v in x]
        data[metric] = x
        data = scaling_fn(data, metric)

    return data


def compute_combinations_correlation(
    data, name, metric, scaling_fn, *metric_combination
):
    scaled_data = extract_metric_scores_and_scale(data, scaling_fn)
    x = np.zeros_like(scaled_data[metric_combination[0]].values)
    if metric == "LogME":
        print(name, x)
    for met in metric_combination:
        x += scaled_data[met].values
    if metric == "LogME":
        print(name, x)
    y = scaled_data["test_accuracy"].values

    mean, _, ci = utils.bootstrap_corr(x, y, kendalltau)

    corr = [mean, ci]

    row = {
        "dataset": DATASET_NAMES[name[0]],
        "n_samples": name[1],
        "n_metric_samples": name[2],
        f"{metric}": corr,
    }

    return row


def get_correlations(group_data, metric, metric_combinations, scaling_fn):
    return pd.DataFrame(
        [
            compute_combinations_correlation(
                data, name, metric, scaling_fn, *metric_combinations
            )
            for name, data in group_data
        ]
    )


def compute_combinations_regret(data, name, metric, scaling_fn, n, *metric_combination):
    oracle = data["test_accuracy"].max()
    scaled_data = extract_metric_scores_and_scale(data, scaling_fn)
    x = np.zeros_like(scaled_data[metric_combination[0]].values)
    for met in metric_combination:
        x += scaled_data[met].values

    sorted_indices = np.argsort(x)[::-1]
    indices = sorted_indices[:n]

    chosen = x[indices]
    not_chosen = x[sorted_indices[:n]]

    if np.isin(not_chosen, chosen).any():
        smallest_chosen = chosen.min()
        chosen_not_smallest = indices[chosen != smallest_chosen]
        chosen_smallest_inds = np.argwhere(x == smallest_chosen).reshape(-1)
        regret_list = []
        for ind in chosen_smallest_inds:
            selection = np.append(chosen_not_smallest, ind)
            outcome = data["test_accuracy"].iloc[selection].max()
            regret_list.append(oracle - outcome)
        regret = np.mean(regret_list)
    else:
        outcome = data["test_accuracy"].iloc[indices].max()
        regret = oracle - outcome

    row = {
        "dataset": DATASET_NAMES[name[0]],
        "n_samples": name[1],
        "n_metric_samples": name[2],
        f"{metric}": regret,
    }

    return row


def get_regrets(group_data, metric, metric_combinations, n, scaling_fn):
    return pd.DataFrame(
        [
            compute_combinations_regret(
                data, name, metric, scaling_fn, n, *metric_combinations
            )
            for name, data in group_data
        ]
    )


def select_results(
    select_fn,
    id_str,
    selection_criteria,
    *args,
    **kwargs,
):
    selection_criteria.append(id_str)
    out_list = [select_fn(arg, **kwargs) for arg in args]
    out = out_list[0]
    for i in range(1, len(out_list)):
        out = out.merge(out_list[i], on=["dataset", "n_samples", "n_metric_samples"])
    out = out.loc[
        :,
        out.columns.isin(selection_criteria),
    ]
    return out


def main():
    api = wandb.Api()
    datasets = list(MAX_SIZES.keys())
    train_data = utils.get_data(api, "train", datasets)
    metric_data = utils.get_data(api, "metrics", datasets)
    data = metric_data.merge(train_data, on=["dataset_name", "model_name", "n_samples"])
    with open("configs/scores_imagenet1k.yaml") as f:
        imagenet = yaml.safe_load(f)
    imagenet = [
        {"model_name": key, "imagenet-validation": {"score": imagenet[key]}}
        for key in imagenet.keys()
    ]
    imagenet = pd.DataFrame(imagenet)
    data = data.merge(imagenet, on="model_name")
    data["test_accuracy"] = data["test_accuracy"] * 100

    print("Data collected and formatted")

    if not os.path.exists("tables"):
        os.mkdir("tables")

    # --------------------
    # Combination analysis
    # --------------------

    # Group data by dataset, n_samples, n_metric_samples
    group_data = data.groupby(["dataset_name", "n_samples", "n_metric_samples"])

    # Create combinations
    metric_names = {metric: [metric] for metric in METRIC_NAMES}
    for combo in combinations(METRIC_NAMES, 2):
        metric_names[combo[0] + "_" + combo[1]] = [combo[0], combo[1]]
    for combo in combinations(METRIC_NAMES, 3):
        metric_names[combo[0] + "_" + combo[1] + "_" + combo[2]] = [
            combo[0],
            combo[1],
            combo[2],
        ]
    metric_names[
        METRIC_NAMES[0]
        + "_"
        + METRIC_NAMES[1]
        + "_"
        + METRIC_NAMES[2]
        + "_"
        + METRIC_NAMES[3]
    ] = METRIC_NAMES

    print("Metric combinations collected")
    pprint.pprint(metric_names)

    # CORRELATIONS ---------------------------------------------------------
    corr_tables = []
    for metric in metric_names.keys():
        new_tab = get_correlations(
            group_data, metric, metric_names[metric], zscore_scale
        )
        # pprint.pprint(new_tab)
        corr_tables.append(new_tab)

    # Top level correlation results ---------------------
    main_results = select_results(
        utils.select_main_results, "dataset", list(metric_names.keys()), *corr_tables
    )

    # pivot and relabel for presentation
    main_results = main_results.T
    main_results.columns = main_results.iloc[0]
    main_results = main_results[1:]
    main_results = main_results.reset_index()
    main_results = main_results.rename(columns={"index": "metric_combos"})

    print("main results collected and formatted")
    pprint.pprint(main_results)

    # Create Latex File (some by hand edits in overleaf necessary)
    utils.latexify(
        table=main_results,
        labels=[
            "Metric Combinations",
            "VG Faucet",
            "VG Tree",
            "VG Watch",
            "RVL-CDIP",
            "WikiART",
            "Oxford Pets",
        ],
        t_name="Metric Combination Performance in Best Case Scenario",
        t_label="main-combo-results",
        type="corr",
    )

    print("table made for main results, starting regret")

    # REGRET -----------------------------------------------------------------
    regret_tables = []
    for metric in metric_names.keys():
        regret_tables.append(
            get_regrets(group_data, metric, metric_names[metric], 1, zscore_scale)
        )

    # Top level regret results
    regret_results = select_results(
        utils.select_main_results,
        "dataset",
        list(metric_names.keys()),
        *regret_tables,
    )

    # pivot and relable for presentation
    regret_results = regret_results.T
    regret_results.columns = regret_results.iloc[0]
    regret_results = regret_results[1:]
    regret_results = regret_results.reset_index()
    regret_results = regret_results.rename(columns={"index": "metric_combos"})

    print("regret results collected and formatted")
    pprint.pprint(regret_results)

    # Create latex File
    utils.latexify(
        table=regret_results,
        labels=[
            "Metric Combinations",
            "VG Faucet",
            "VG Tree",
            "VG Watch",
            "RVL-CDIP",
            "WikiART",
            "Oxford Pets",
        ],
        t_name="Metric Combinations Assessed by Regret",
        t_label="main-combo-regret",
        type="regret",
    )

    print("regret table saved, onto mean results")

    # MEAN RESULTS ---------------------------------------------------------
    c_all = pd.concat(
        [
            select_results(
                utils.select_sample_results,
                "n_samples",
                list(metric_names.keys()),
                *corr_tables,
                dataset_name=dataset_name,
            )
            for dataset_name in DATASET_NAMES.values()
        ]
    )

    r_all = pd.concat(
        [
            select_results(
                utils.select_sample_results,
                "n_samples",
                list(metric_names.keys()),
                *regret_tables,
                dataset_name=dataset_name,
            )
            for dataset_name in DATASET_NAMES.values()
        ]
    )

    def make_mean_plus_ci(x):
        m = np.mean(x)
        se = np.std(x) / np.sqrt(len(x))
        return {
            "mean": np.around(m, 2),
            "lower": np.around(m - 2 * se, 2),
            "upper": np.around(m + 2 * se, 2),
        }

    a = (
        c_all[c_all.columns[1:]]
        .apply(lambda x: [v[0] for v in x], axis=0)
        .apply(lambda x: make_mean_plus_ci(x))
    )
    b = r_all[r_all.columns[1:]].apply(lambda x: make_mean_plus_ci(x))

    c = pd.concat([a, b], axis=1).transpose()
    d = pd.concat(
        [pd.DataFrame([{"Metric": "Kendall's Tau"}, {"Metric": "Regret"}]), c], axis=1
    )

    # pivot and relable for presentation
    d = d.T
    d.columns = d.iloc[0]
    d = d[1:]
    d = d.reset_index()
    d = d.rename(columns={"index": "metric_combos"})

    d.to_csv("./tables/mean-combo-results.csv")

    utils.latexify(
        table=d,
        labels=[
            "Metric Combinations",
            "VG Faucet",
            "VG Tree",
            "VG Watch",
            "RVL-CDIP",
            "WikiART",
            "Oxford Pets",
        ],
        t_name="Mean Metric Combination Performance According to Metric",
        t_label="mean-combo-results",
        type="mean",
    )

    print("mean table made, all done!")


if __name__ == "__main__":
    main()
