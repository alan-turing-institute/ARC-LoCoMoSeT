from locomoset.metrics.metrics_for_metrics import (
    kendall_tau_correlation,
    pearsons_correlation,
    spearmans_rank_correlation,
)
from locomoset.metrics.parc import parc
from locomoset.metrics.renggli import renggli
from locomoset.metrics.task_agnostic import num_params_metric

METRIC_FUNCTIONS = {"parc": parc, "renggli": renggli, "n_pars": num_params_metric}

METRIC_FOR_METRIC_FUNCTIONS = {
    "spearmans": spearmans_rank_correlation,
    "pearsons": pearsons_correlation,
    "kendall_tau": kendall_tau_correlation,
}
