from locomoset.metrics.logme import logme
from locomoset.metrics.metrics_for_metrics import (
    kendall_tau_correlation,
    pearsons_correlation,
    spearmans_rank_correlation,
)
from locomoset.metrics.nce import nce
from locomoset.metrics.parc import parc_class_function
from locomoset.metrics.renggli import renggli_class_function
from locomoset.metrics.task_agnostic import num_params_metric

METRIC_FOR_METRIC_FUNCTIONS = {
    "spearmans": spearmans_rank_correlation,
    "pearsons": pearsons_correlation,
    "kendall_tau": kendall_tau_correlation,
}

METRIC_FUNCTIONS = {
    "parc": parc_class_function,
    "renggli": renggli_class_function,
    "n_pars": num_params_metric,
    "LogME": logme,
    "NCE": nce,
}
