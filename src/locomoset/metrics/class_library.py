from locomoset.metrics.metric_classes import (
    LogMEMetric,
    NumParsMetric,
    PARCMetric,
    RenggliMetric,
)

METRIC_CLASSES = {
    "n_pars": NumParsMetric,
    "renggli": RenggliMetric,
    "parc": PARCMetric,
    "LogME": LogMEMetric,
}
