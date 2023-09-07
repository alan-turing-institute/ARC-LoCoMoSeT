from locomoset.metrics.parc import parc
from locomoset.metrics.renggli import renggli
from locomoset.metrics.task_agnostic import num_params_metric

METRIC_FUNCTIONS = {"parc": parc, "renggli": renggli, "n_pars": num_params_metric}
