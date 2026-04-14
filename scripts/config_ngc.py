"""
Shared configuration for the NGC / DASGIB ICP pipeline.

NGC paths match the actual server layout under:
  /projects/users/people/mehuns_r/projects/neurovfm_private/
"""

import argparse

# Constants (identical to PhysioNet pipeline for consistency)
SEED         = 42
EMBED_DIM    = 768
NUM_CV_FOLDS = 5

# Base directory on NGC
_BASE = "/projects/users/people/mehuns_r/projects/neurovfm_private"

# Dataset
NGC_DATA_ROOT    = "/projects/users/data/UCPH/ICP/organized_data"
DEFAULT_CSV_PATH = f"{NGC_DATA_ROOT}/tables/icp_path_pair.csv"   # full dataset (feature extraction)
DEFAULT_DEV_CSV_PATH = f"{_BASE}/dev_labels.csv"                 # dev split only (ML training/CV)

# Model: weights are unpacked directly into hf_cache/
# (config.json and pytorch_model.bin sit at this path — no HF_HOME needed)
DEFAULT_MODEL_PATH = f"{_BASE}/hf_cache"

# Output directories
DEFAULT_FEATURES_DIR = f"{_BASE}/extracted_features"
DEFAULT_RESULTS_DIR  = f"{_BASE}/results"


def get_ngc_parser(description="", csv_default=None):
    if csv_default is None:
        csv_default = DEFAULT_CSV_PATH
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--csv-path", type=str, default=csv_default,
        help="Path to labels CSV (icp_path_pair.csv or dev_labels.csv)"
    )
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH,
        help="Local path to unpacked NeuroVFM weights (hf_cache/)"
    )
    parser.add_argument(
        "--features-dir", type=str, default=DEFAULT_FEATURES_DIR,
        help="Directory containing extracted .pt feature files"
    )
    parser.add_argument(
        "--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
        help="Directory for results, plots, and metrics JSON"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (cuda / cpu). Auto-detected if omitted."
    )
    return parser


def resolve_device(device_arg):
    if device_arg:
        return device_arg
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
