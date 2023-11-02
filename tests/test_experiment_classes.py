"""
    Functions for testing the model experiment class
    (src/locomoset/models/model_experiment_classes.py)
"""

from transformers import PreTrainedModel

from locomoset.metrics.classes import MetricConfig, TopLevelMetricConfig
from locomoset.metrics.experiment import ModelMetricsExperiment
from locomoset.models.load import get_model_without_head


def test_model_exp_class_init(
    dummy_metric_config, dummy_model_name, dummy_dataset_name, test_seed
):
    """Test the class initialisation"""
    config = MetricConfig.from_dict(dummy_metric_config)
    model_experiment = ModelMetricsExperiment(config.to_dict)
    assert model_experiment.model_name == dummy_model_name
    assert model_experiment.dataset_name == dummy_dataset_name
    assert model_experiment.n_samples == 50
    assert model_experiment.random_state == test_seed
    assert list(model_experiment.metrics.keys()) == ["renggli"]
    assert model_experiment.inference_types == ["features"]


def test_features_inference(dummy_metric_config):
    """Test the features inference method"""
    config = MetricConfig.from_dict(dummy_metric_config)
    model_experiment = ModelMetricsExperiment(config.to_dict)
    features = model_experiment.features_inference()
    assert features.shape == (50, 5)


def test_perform_inference_task_specifc(dummy_metric_config):
    """Test the perform inference method for a task specific case"""
    config = MetricConfig.from_dict(dummy_metric_config)
    model_experiment = ModelMetricsExperiment(config.to_dict)
    inference = model_experiment.perform_inference(model_experiment.inference_types[0])
    assert inference[0].shape == (50, 5)


def test_perform_inference_task_agnostic(dummy_metric_config):
    """Test the perform inference method for a task agnostic case"""
    dummy_metric_config["metrics"] = ["n_pars"]
    config = MetricConfig.from_dict(dummy_metric_config)
    model_experiment = ModelMetricsExperiment(config.to_dict)
    assert list(model_experiment.metrics.keys())[0] == "n_pars"
    assert model_experiment.inference_types[0] == "model"
    inference = model_experiment.perform_inference(model_experiment.inference_types[0])
    assert isinstance(inference[0], PreTrainedModel)


def test_compute_metric_score(dummy_metric_config):
    """Test the compute metric score method"""
    dummy_metric_config["metrics"] = ["n_pars"]
    config = MetricConfig.from_dict(dummy_metric_config)
    model_experiment = ModelMetricsExperiment(config.to_dict)
    model_fn = get_model_without_head(model_experiment.model_name)
    metric_score = model_experiment.compute_metric_score(
        model_experiment.metrics["n_pars"], model_fn, None
    )
    assert metric_score[0] == 1686


def test_init_metric_config(dummy_metric_config, dummy_dataset_name, dummy_model_name):
    config = MetricConfig.from_dict(dummy_metric_config)
    assert config.run_name == f"{dummy_dataset_name}_{dummy_model_name}".replace(
        "/", "-"
    )
    assert config.use_wandb is True


def test_init_top_level_metric_config(dummy_top_level_config, dummy_config_gen_dtime):
    config = TopLevelMetricConfig.from_dict(dummy_top_level_config)
    config.config_type = "metrics"
    config.generate_sub_configs()
    assert config.config_type == "metrics"
    assert config.config_gen_dtime == dummy_config_gen_dtime
    assert isinstance(config.sub_configs[0], MetricConfig)


def test_top_level_metric_config_param_sweep(dummy_top_level_config, test_seed):
    config = TopLevelMetricConfig.from_dict(dummy_top_level_config)
    config.generate_sub_configs()
    assert len(config.sub_configs) == 2
    assert config.sub_configs[0].random_state == test_seed
    assert config.sub_configs[1].random_state == test_seed + 1
