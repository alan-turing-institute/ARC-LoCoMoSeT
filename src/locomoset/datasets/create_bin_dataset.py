"""
    From the visual genome object (https://huggingface.co/datasets/visual_genome)
    detection dataset create binary classification datasets based on a config with
    fields:

        - 'objects': list of object names from the visual genome dataset.
        - 'dataset_size': size of the dataset

    This will also make the dataset as balanced between images with and without the
    chosen object to the best degree possible, using all the available images if the
    total images with the objects in them is less than half the stated dataset size.
"""


import argparse
import os
from itertools import product

import numpy as np
import yaml
from datasets import Dataset, load_dataset


def get_balanced_id(binary_dataset: Dataset, dataset_size: int) -> list[int]:
    """Get a list of ids from the dataset such that there are an even number of labels 1
    and 0.

    Args:
        binary_dataset: binary classification dataset with field 'image_id'
        dataset_size: size of the dataset

    Returns:
        list of ids to filter by
    """
    one_number = dataset_size // 2
    zero_number = dataset_size - one_number
    one_count = 0
    zero_count = 0
    ids = []
    for data in binary_dataset:
        if one_count >= one_number and zero_count >= zero_number:
            break
        if data["label"] == 1:
            if one_count > one_number:
                continue
            else:
                one_count += 1
                ids.append(data["image_id"])
        elif data["label"] == 0:
            if zero_count > zero_number:
                continue
            else:
                zero_count += 1
                ids.append(data["image_id"])
    return ids


def drop_images_by_id(dataset: Dataset, ids_to_keep: list[int]) -> Dataset:
    """Drop the images by id

    Args:
        dataset: dataset on which from which to drop the images
        ids_to_keep: list of ids to keep

    Returns:
        Original dataset retaining all images with matching ids.
    """
    return dataset.filter(lambda sample: sample["image_id"] in ids_to_keep)


def convert_to_bin_classification(sample: dict, object: str) -> dict:
    """Convert a datapoint from a huggingface object classification dataset and turn it
    into a binary classification datapoint based on if the specified object in question
    appears in the image.

    Args:
        sample: datapoint sample with keys:
                - 'image': contains a PIL image.
                - 'objects': contains a list of dictionaries specifying the objects in
                             the image, with key 'names' giving the name of the image.
        object: name of object to create dataset from.

    Returns:
        new datapoint dictionary with keys:
            - 'image': contains the original PIL image
            - 'label': 1 if specified object is within the image and 0 otherwise.
    """
    sample["label"] = int(
        object in np.unique([obj["names"][0] for obj in sample["objects"]])
    )

    # Compute percent prominence of object:
    area_of_image = sample["height"] * sample["width"]
    object_areas = []
    for obj in sample["objects"]:
        if obj["names"][0] == object:
            object_areas.append(obj["w"] * obj["h"])
    sample["prominence"] = (sum(object_areas) / area_of_image) * 100
    return sample


def create_balanced_bin_dataset(
    dataset: Dataset, object_name: str, dataset_size: int | None = None, seed: int = 42
) -> Dataset:
    """Create a balanced binary classification dataset from an object detection
    dataset, based on a given object within that dataset. If the dataset_size is
    specified then the dataset will be of that size, otherwise it will select all the
    images containing the specified object, with a random selection of images that do
    not contain the object of the same number.

    Args:
        dataset: object detection dataset
        object_name: object name to create binary detection dataset from
        dataset_size: specified size of the new small dataset, defaults to none
        seed: random seed for shuffling the dataset

    Returns:
        balanced binary image classification datset
    """
    dataset = dataset.shuffle(seed=seed)
    bin_dataset = dataset.map(
        convert_to_bin_classification,
        batched=False,
        remove_columns=[
            "url",
            "width",
            "height",
            "coco_id",
            "flickr_id",
            "objects",
        ],
        fn_kwargs={"object": object_name},
    )
    if dataset_size is not None:
        return drop_images_by_id(
            bin_dataset, get_balanced_id(bin_dataset, dataset_size)
        )
    else:
        return drop_images_by_id(
            bin_dataset,
            get_balanced_id(bin_dataset, np.sum(bin_dataset["label"]) * 2),
        )


def create_top_config(
    dataset_name: str,
    object: str,
    seed: int,
    template_path: str,
    config_save_path: str,
) -> None:
    """Create a top level config from a template with the correct dataset name.

    Args:
        dataset_name: dataset name (location of the dataset)
        object: object selection for binary classification
        seed: seed used in dataset generation
        template_path: path to config template location
        config_save_path: path to save the config
    """
    with open(template_path, "r") as f:
        bin_config = yaml.safe_load(f)

    bin_config["dataset_names"] = dataset_name
    file_name = f"top_bin_config_{object}_{seed}.yaml"

    with open(f"{config_save_path}/{file_name}", "w") as f:
        yaml.safe_dump(bin_config, f)


def create_and_save_bin_datasets(
    dataset: Dataset,
    objects: list[str],
    dataset_size: int | None,
    seeds: list[int],
    save_path: str,
    config_template_path: str,
    config_save_path: str,
) -> None:
    """Given a list of objects from the visual genome dataset create balanced binary
    datasets of the specified size

    Args:
        dataset: top level object detection dataset (visual genome)
        objects: list of objects to create datasets from
        dataset_size: size of the dataset
        seeds: seeds to use for shuffling dataset
        config_template_path: path to the config template
        config_save_path: path for saving configs
    """
    for obj_seed in product(objects, seeds):
        bin_dataset = create_balanced_bin_dataset(
            dataset=dataset,
            object_name=obj_seed[0],
            dataset_size=dataset_size,
            seed=obj_seed[1],
        )
        path = f"{save_path}/bin_{obj_seed[0]}_seed_{obj_seed[1]}"
        os.makedirs(path)
        dataset_path = f"{path}/bin_{obj_seed[0]}_seed_{obj_seed[1]}.parquet"
        bin_dataset.to_parquet(dataset_path)
        create_top_config(
            dataset_name=path,
            object=obj_seed[0],
            seed=obj_seed[1],
            template_path=config_template_path,
            config_save_path=config_save_path,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Create binary classification datasets from given objects"
    )
    parser.add_argument("configfile", help="Path to config file")
    args = parser.parse_args()

    # read config yaml
    with open(args.configfile, "r") as f:
        config_dict = yaml.safe_load(f)

    # check for visual genome
    if config_dict["obj_detect_dataset"] != "visual_genome":
        raise ValueError("Only set up for visual genome currentl")
    vis_genome = load_dataset(
        "visual_genome",
        "objects_v1.2.0",
        cache_dir=config_dict["save_path"],
        split="train",
    )

    create_and_save_bin_datasets(
        dataset=vis_genome,
        objects=config_dict["objects"],
        dataset_size=config_dict["dataset_size"],
        seeds=config_dict["seeds"],
        save_path=config_dict["save_path"],
        config_template_path=config_dict["config_template_path"],
        config_save_path=config_dict["config_save_path"],
    )


if __name__ == "__main__":
    main()
