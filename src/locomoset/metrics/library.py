from locomoset.metrics.logme import logme
from locomoset.metrics.nce import nce
from locomoset.metrics.parc import parc
from locomoset.metrics.renggli import renggli

METRIC_FUNCTIONS = {"parc": parc, "renggli": renggli, "LogME": logme, "NCE": nce, "n_pars": num_params_metric}
