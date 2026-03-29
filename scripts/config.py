"""
Shared configuration for the ICH detection pipeline.
All paths are configurable via argparse for NGC portability.
"""

import argparse
from pathlib import Path

# Constants
SEED = 42
EMBED_DIM = 768
NUM_CV_FOLDS = 5

# Default paths (local WSL setup)
DEFAULT_DATA_DIR = "/mnt/c/Users/mehme/Desktop/ct-scan-test"
DEFAULT_MODEL_PATH = "mlinslab/neurovfm-encoder"
DEFAULT_FEATURES_DIR = str(Path(__file__).resolve().parent.parent / "extracted_features")
DEFAULT_RESULTS_DIR = str(Path(__file__).resolve().parent.parent / "results")


def get_base_parser(description=""):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Path to PhysioNet ct-scan-test directory")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="HuggingFace model ID or local path to encoder weights")
    parser.add_argument("--features-dir", type=str, default=DEFAULT_FEATURES_DIR,
                        help="Directory for extracted .pt feature files")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="Directory for results, plots, metrics")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Auto-detected if omitted.")
    return parser


def resolve_device(device_arg):
    if device_arg:
        return device_arg
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
