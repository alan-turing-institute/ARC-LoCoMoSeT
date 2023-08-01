"""
Save a small dummy model with random weights.
"""
from dummy_config import config
from transformers import ViTConfig, ViTForImageClassification

vit_config = ViTConfig(
    hidden_size=config["hidden_size"],
    num_hidden_layers=config["num_hidden_layers"],
    num_attention_heads=config["num_attention_heads"],
    intermediate_size=config["intermediate_size"],
    image_size=config["image_size"],
    patch_size=config["patch_size"],
    num_channels=config["num_channels"],
    encoder_stride=config["encoder_stride"],
)

model = ViTForImageClassification(vit_config)

model.save_pretrained("dummy_model")
