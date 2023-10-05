"""
    Functions for testing the model experiment class
    (src/locomoset/models/model_experiment_classes.py)
"""

# import pytest

from locomoset.models.load import get_model_and_processor
from locomoset.models.model_experiment_classes import ModelExperiment


def test_model_exp_class_init(
    dummy_config, dummy_model_name, dummy_dataset_name, test_seed
):
    """Test the class initialisation"""
    model_experiment = ModelExperiment(dummy_config)
    assert model_experiment.model_name == dummy_model_name
    assert model_experiment.dataset_name == dummy_dataset_name
    assert model_experiment.n_samples == 50
    assert model_experiment.random_state == test_seed
    assert list(model_experiment.metrics.keys())[0] == "renggli"
    assert model_experiment.inference_types[0] == "features"


def test_features_inference(dummy_config):
    """Test the features inference method"""
    model_experiment = ModelExperiment(dummy_config)
    features = model_experiment.features_inference()
    assert features.shape == (50, 5)


def test_perform_inference_task_specifc(dummy_config):
    """Test the perform inference method for a task specific case"""
    model_experiment = ModelExperiment(dummy_config)
    inference = model_experiment.perform_inference(model_experiment.inference_types[0])
    assert inference[0].shape == (50, 5)


def test_perform_inference_task_agnostic(dummy_config):
    """Test the perform inference method for a task agnostic case"""
    dummy_config["metrics"] = ["n_pars"]
    model_experiment = ModelExperiment(dummy_config)
    assert list(model_experiment.metrics.keys())[0] == "n_pars"
    assert model_experiment.inference_types[0] == "None"
    inference = model_experiment.perform_inference(model_experiment.inference_types[0])
    assert inference[0] is None


def test_compute_metric_score(dummy_config):
    """Test the compute metric score method"""
    dummy_config["metrics"] = ["n_pars"]
    model_experiment = ModelExperiment(dummy_config)
    model_fn, _ = get_model_and_processor(model_experiment.model_name, num_labels=0)
    metric_score = model_experiment.compute_metric_score(
        model_experiment.metrics["n_pars"]["metric_fn"], model_fn, _, _
    )
    assert metric_score[0] == 1686
