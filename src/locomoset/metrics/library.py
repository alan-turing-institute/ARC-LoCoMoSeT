from locomoset.metrics.evaluate import (
    kendall_tau_correlation,
    pearsons_correlation,
    spearmans_rank_correlation,
)
from locomoset.metrics.logme import LogMEMetric
from locomoset.metrics.nce import nce
from locomoset.metrics.renggli import RenggliMetric
from locomoset.metrics.task_agnostic import NumParsMetric

METRIC_FOR_METRIC_FUNCTIONS = {
    "spearmans": spearmans_rank_correlation,
    "pearsons": pearsons_correlation,
    "kendall_tau": kendall_tau_correlation,
}

METRIC_FUNCTIONS = {
    "NCE": nce,
}

METRICS = {"LogME": LogMEMetric, "renggli": RenggliMetric, "n_pars": NumParsMetric}
