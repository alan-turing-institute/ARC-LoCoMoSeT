"""
    Test functions for the model module classes (src/locomoset/models/classes)
"""

from transformers import TrainingArguments

from locomoset.models.classes import (
    FineTuningConfig,
    TopLevelFineTuningConfig,
    create_wandb_names,
)


def test_init_fine_tuning_config(
    dummy_fine_tuning_config, dummy_dataset_name, dummy_model_name
):
    config = FineTuningConfig.from_dict(dummy_fine_tuning_config)
    test_run_name = create_wandb_names(dummy_dataset_name, dummy_model_name)
    assert config.run_name == test_run_name
    assert isinstance(config.get_training_args(), TrainingArguments)
    assert config.use_wandb is True


def test_init_top_level_fine_tuning_config(
    dummy_top_level_config, dummy_config_gen_dtime
):
    config = TopLevelFineTuningConfig.from_dict(dummy_top_level_config)
    config.config_type = "train"
    config.generate_sub_configs()
    assert config.config_type == "train"
    assert config.config_gen_dtime == dummy_config_gen_dtime
    assert isinstance(config.sub_configs[0], FineTuningConfig)


def test_top_level_fine_tuning_config_param_sweep(dummy_top_level_config, test_seed):
    config = TopLevelFineTuningConfig.from_dict(dummy_top_level_config)
    config.generate_sub_configs()
    assert len(config.sub_configs) == 2
    assert config.sub_configs[0].random_state == test_seed
    assert config.sub_configs[1].random_state == test_seed + 1
