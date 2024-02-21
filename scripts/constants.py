BIN_PATH = "/bask/projects/v/vjgo8416-locomoset/ARC-LoCoMoSeT/binary_datasets/"

MAX_SIZES = {
    "huggan/wikiart": 50000,
    BIN_PATH + "bin_tree": 26004,
    "pcuenq/oxford-pets": 4803,
    BIN_PATH + "bin_faucet": 2303,
    BIN_PATH + "bin_watch": 2825,
    "aharley/rvl_cdip": 50000,
}

DATASET_NAMES = {
    "huggan/wikiart": "WikiArt",
    BIN_PATH + "bin_tree": "VG Tree",
    "pcuenq/oxford-pets": "Oxford Pets",
    BIN_PATH + "bin_faucet": "VG Faucet",
    BIN_PATH + "bin_watch": "VG Watch",
    "aharley/rvl_cdip": "RVL-CDIP",
}

SAMPLES = [500, 1000, 5000, 10000]

METRIC_NAMES = ["n_pars", "renggli", "LogME", "imagenet-validation"]
