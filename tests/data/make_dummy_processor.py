"""
Save a dummy image processor that does nothing to use in tests.
"""
from dummy_config import config
from transformers import ViTImageProcessor

processor = ViTImageProcessor(
    do_resize=config["do_resize"],
    resample=config["resample"],
    do_rescale=config["do_rescale"],
    do_normalize=config["do_normalize"],
    size={"height": config["image_size"], "width": config["image_size"]},
)

processor.save_pretrained("dummy_model")
