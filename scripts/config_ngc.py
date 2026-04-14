"""
Shared configuration for the NGC / DASGIB ICP pipeline.

NGC default paths assume the standard UCPH ICP project layout:
  /projects/users/data/UCPH/ICP/organized_data/

All paths are overridable via argparse so nothing is hardcoded
into job scripts.
"""

import argparse
from pathlib import Path

# Constants (same as PhysioNet pipeline for consistency)
SEED = 42
EMBED_DIM = 768
NUM_CV_FOLDS = 5

# NGC default paths
NGC_DATA_ROOT    = "/projects/users/data/UCPH/ICP/organized_data"
DEFAULT_CSV_PATH = f"{NGC_DATA_ROOT}/tables/icp_path_pair.csv"

# Model: points to the offline HuggingFace cache on NGC shared storage.
# On NGC, set HF_HOME to this directory (or override with --model-path).
DEFAULT_HF_HOME    = "/projects/users/people/mehuns_r/projects/neurovfm_private/hf_cache"
DEFAULT_MODEL_PATH = "mlinslab/neurovfm-encoder"   # resolved from HF_HOME

# Output directories (under the user's private project space on NGC)
DEFAULT_FEATURES_DIR = "/projects/users/people/mehuns_r/projects/neurovfm_private/extracted_features"
DEFAULT_RESULTS_DIR  = "/projects/users/people/mehuns_r/projects/neurovfm_private/results"


def get_ngc_parser(description=""):
    parser = argparse.ArgumentParser(description=description)

    # Dataset
    parser.add_argument(
        "--csv-path", type=str, default=DEFAULT_CSV_PATH,
        help="Path to icp_path_pair.csv (nii_file column contains full paths)"
    )

    # Model
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH,
        help="HuggingFace model ID (resolved from HF_HOME cache)"
    )
    parser.add_argument(
        "--hf-home", type=str, default=DEFAULT_HF_HOME,
        help="Path to offline HuggingFace cache (sets HF_HOME env var)"
    )

    # Outputs
    parser.add_argument(
        "--features-dir", type=str, default=DEFAULT_FEATURES_DIR,
        help="Directory for extracted .pt feature files"
    )
    parser.add_argument(
        "--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
        help="Directory for results, plots, and metrics JSON"
    )

    # Hardware
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
