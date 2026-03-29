"""
Feature Extraction Loop for PhysioNet CT-ICH dataset.
Iterates over all .nii files, runs them through NeuroVFM encoder,
and saves the embeddings as .pt files.
"""

import sys
import time
import torch
from pathlib import Path
from tqdm import tqdm

# Add parent dir to path so we can import from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_base_parser, resolve_device
from label_utils import get_available_patients


def main():
    parser = get_base_parser("Extract NeuroVFM features from CT scans")
    parser.add_argument("--single", type=int, default=None,
                        help="Process only this patient ID (for testing)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ct_dir = str(Path(args.data_dir) / "ct_scans")
    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"CT scans dir: {ct_dir}")
    print(f"Features output dir: {features_dir}")

    # Get patient list
    if args.single is not None:
        patient_ids = [args.single]
        print(f"Single patient mode: {args.single}")
    else:
        patient_ids = get_available_patients(args.data_dir)
        print(f"Found {len(patient_ids)} patients")

    # Load encoder (this downloads/caches weights from HuggingFace if needed)
    print("Loading NeuroVFM encoder...")
    from neurovfm.pipelines import load_encoder
    encoder, preprocessor = load_encoder(args.model_path)
    print("Encoder loaded.")

    processed = 0
    skipped = 0
    failed = []
    total_start = time.time()

    for pid in tqdm(patient_ids, desc="Extracting features"):
        output_path = features_dir / f"{pid:03d}.pt"

        # Skip if already processed
        if output_path.exists():
            tqdm.write(f"  [{pid:03d}] Already exists, skipping.")
            skipped += 1
            continue

        scan_path = str(Path(ct_dir) / f"{pid:03d}.nii")

        try:
            # Load and preprocess the scan
            # load_study handles: reorient, resample, crop, tokenize
            # For CT it automatically creates Brain/Blood/Bone windows
            # and sets correct path names (no manual BrainWindow hack needed)
            batch = preprocessor.load_study(scan_path, modality="ct")

            # Move tensors to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Extract features
            with torch.no_grad():
                embeddings = encoder.embed(batch)

            # Handle tuple return (some versions return tuple)
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]

            # Save to disk
            torch.save(embeddings.cpu(), output_path)

            tqdm.write(f"  [{pid:03d}] OK - shape: {embeddings.shape}")
            processed += 1

            # Free GPU memory
            del batch, embeddings
            torch.cuda.empty_cache()

        except Exception as e:
            tqdm.write(f"  [{pid:03d}] FAILED: {e}")
            failed.append((pid, str(e)))
            torch.cuda.empty_cache()

    elapsed = time.time() - total_start

    print('\n' + '='*50)
    print(f"Feature extraction complete in {elapsed:.1f}s")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed patients:")
        for pid, err in failed:
            print(f"    [{pid:03d}]: {err}")


if __name__ == "__main__":
    main()
