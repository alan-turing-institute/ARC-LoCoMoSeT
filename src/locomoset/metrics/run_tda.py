"""
Script for running the full pipeline of the TDA metric to produce experimental results
for varied hyperparameters.

NB: This is currently written with a streamed/iterable datset in mind for ease of
initial experiments.
"""

import argparse
import json
import os
from datetime import datetime
from time import time

import numpy as np
import yaml
from datasets import load_dataset
from gtda import diagrams as dgms
from gtda import homology as hom
from tqdm import tqdm

from locomoset.metrics.run import parameter_sweep_dicts
from locomoset.metrics.tda import (
    merge_diags,
    model_diags,
    tda_preprocess,
    tda_probe_set,
)

# from tdqm import tdqm


def run_tda_metric(**pars) -> dict:
    """Pipeline for TDA metric.

    This metric measure how topological invariants of the original data and ground truth
    labels relate to the same topological features in a combination of the original data
    and either predicted labels or features from the last layer of the network.
    Functionally this is achieved by the following pipeline (with preprocessing not
    included):

    1. Combine the pixel array for each image with either the ground truth label (either
    as a new row of [lab]*pixels or as a one hot vector that's been wrapped around and
    padded with zeros) or the predicted labels (similarly either as an integer value or
    one hot vector) or the features of the model (reduced via PCA to a divisor of the
    number of pixels on a side and repeated along the array to be of same size.)

    1. Compute the birth, death, homology dimension triples (a.k.a. homology diagrams)
    for each example image, with a set of diagrams for the ground truth and for each
    model considered.

    2. Combine all the diagrams for the ground truths and models together into a single
    array, padding diagram arrays (of shape [n, f, 3]) with either [0,0,0] or [0,0,1]
    (b,d,q) triples such that each f is the same and the balance of q=0 and q=1 is the
    same.

    3. Create persistence image from all diagrams (a mapping of the diagrams to various
    copies of R^2 with a binned Gaussian applied approximating the manifold). All
    diagrams for each model to be considered are required so the image considers the
    same area of R^2 for each.

    4. The persistence image will be of dimension (m, 2, 100, 100) for 2 homology
    dimensions and binned Gaussian of shape 100x100,
    m = (num_models + 1) * probe_set_size, i.e. there are n (probe_set_size) persistence
    images for the ground truth and for each model. Extract the array corresponding to
    the ground truths and each model, computing the Euclidean distance between the
    ground truth array and each model as the final metric.

    NB: The preprocessing requires these to be square, Grayscale images.

    NB: This is currently written with a streamed/iterable datset in mind for ease of
    initial experiments.

    Params:
        - dataset: dataset
        - n_examples: size of the probe set
        - models: list model names
        - pixel_square: one side of pixel length to reshape images to
        - one_hot_labels: use one hot labels vs. integer labels in padding
        - model_features: use the features or the classifier
        - random_state: seed for randomness
        - feature_reduction_level: divisor of pixel_square to reduce to
    """
    # TDA objects:
    cub_hom = hom.CubicalPersistence()
    per_img = dgms.PersistenceImage()
    print("loaded tda objects")

    # If using streamed/iterable dataset it needs to be shuffled for variation in images
    pars["dataset"] = pars["dataset"].shuffle(seed=pars["random_state"])

    # Start by creating the diagrams for the ground truths
    print("starting ground truth diagram computation")
    fn_kwargs = None
    if pars["pixel_square"] is not None:
        pixels = pars["pixel_square"]
        fn_kwargs = {"pixels": pixels}
    tda_imgs = pars["dataset"].map(
        tda_preprocess,
        remove_columns="image",
        batched=True,
        batch_size=1,
        fn_kwargs=fn_kwargs,
    )
    tda_imgs = tda_imgs.with_format("torch")
    print("tda image processing done!")

    print("starting computation of ground truth probe set")
    gtruth_probe_set = tda_probe_set(
        pars["n_examples"], tda_imgs, pars["one_hot_labels"], ground_truth=True
    )
    print("ground truth probe set completed")

    print("computing ground truth diagrams from probe set")
    diags = cub_hom.fit_transform(gtruth_probe_set)
    diags_shape = diags.shape
    print(f"ground truth diagrams computed, with shape {diags_shape}")

    # Create diagrams for each model and append to diags
    print("starting computation of model diagrams")
    model_diagrams = []
    for model in tqdm(pars["models"]):
        print(f"computing diagrams for {model}")
        model_diag = model_diags(model, tda_imgs, cub_hom, **pars)
        print(f"model diags shape {model_diag.shape}")
        # assert model_diags.shape == diags.shape, 'Diagram dimensions do not match!'
        # diags = np.concatenate((diags, model_diag), axis=0)
        model_diagrams.append(model_diag)
    mdiags = model_diagrams[0]
    if len(model_diagrams) > 1:
        for i in range(1, len(model_diagrams)):
            mdiags = merge_diags(mdiags, model_diagrams[i])
    print("done with model diag computations")

    print("combine diagrams")
    diags = merge_diags(diags, mdiags)
    print(f"final diagrams shape {diags.shape}")

    # Compute persistance image for entire set
    print("computing persistance image")
    perst_img = per_img.fit_transform(diags)

    # Compute metric from this
    print("computing metrics")
    results = {}
    base_img = perst_img[: pars["n_examples"], :, :, :]
    for idx, model in enumerate(pars["models"]):
        print(f"comp idxs {(idx+1)*pars['n_examples']}, {(idx+2)*pars['n_examples']}")
        results[model] = np.linalg.norm(
            base_img
            - perst_img[
                (idx + 1) * pars["n_examples"] : (idx + 2) * pars["n_examples"], :, :, :
            ]
        )
    return results


def run(config: dict):
    """Run the TDA metric for a given pair (dataset, (selection of models)) for a
    parameter sweep of the varying hyperparameters. Results saved to file path of form
    results/results_YYYYMMDD-HHMMSS.json by default.

    Args:
        config: loaded configuration dictionary including the following keys:
            - dataset: name of dataset
            - n_examples: list of different probe set sizes to consider
            - models: list model names
            - pixel_square: one side of pixel length to reshape images to
            - one_hot_labels: use one hot labels vs. integer labels in padding
            - model_features: use the features or the classifier
            - random_state: list of seeds for randomness
            - feat_red_dim: divisor of pixel_square to reduce to
            - (Optional) save_dir: Directory to save results, "results" if not set.
    """
    save_dir = config.get("save_dir", "results")
    os.makedirs(save_dir, exist_ok=True)

    # load the datset (streaming/iterable)
    dataset = load_dataset(
        config["dataset_name"],
        split=config["dataset_split"],
        streaming=True,
        token=True,
    )

    # creates all experiment variants
    print("creating config variants")
    config_variants = parameter_sweep_dicts(config, hold_constant="models")
    print(f"done with config variant creation, created {len(config_variants)} configs")

    for config_var in tqdm(config_variants):
        print("Starting computation for:")
        print(json.dumps(config_var, indent=4))
        config_var["dataset"] = dataset
        # config_var["models"] = config_var["models"][0]
        # config_var["n_examples"] = config_var["n_examples"][0]
        print(f"this should be a bool {config_var['model_features']}")

        date_str = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        save_path = f"{save_dir}/results_{date_str}.json"
        results = config_var
        metric_start = time()
        result = run_tda_metric(**config_var)
        results["result"] = {"results": result, "time": time() - metric_start}
        del results["dataset"]
        with open(save_path, "w") as f:
            json.dump(results, f, default=float)
        print(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute the TDA metric with parameter scans."
    )
    parser.add_argument("configfile", help="Path to config file")
    args = parser.parse_args()
    with open(args.configfile, "r") as f:
        config = yaml.safe_load(f)

    print("config file:")
    print(json.dumps(config, indent=4))

    run(config)


if __name__ == "__main__":
    main()
