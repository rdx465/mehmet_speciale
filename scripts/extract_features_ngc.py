"""
Feature Extraction for DASGIB/NGC dataset.

Reads icp_path_pair.csv, applies all filters (axial-only, temporal,
quality, ICP>0), then runs each NIfTI through the NeuroVFM encoder
and saves embeddings as .pt files named by record_id.

Key differences from extract_features.py (PhysioNet version):
  - Patient IDs are strings (e.g. "1-154"), not zero-padded integers
  - NIfTI paths come directly from the CSV (no glob needed)
  - Model weights loaded directly from --model-path (no HF_HOME needed)

Usage (interactive / login node test):
    python extract_features_ngc.py --single 1-154

Usage (PBS job):
    python extract_features_ngc.py

Model weights are loaded directly from --model-path (no HF_HOME needed).
"""

import sys
import time
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_ngc import get_ngc_parser, resolve_device
from label_utils_ngc import get_aligned_arrays


def sanitize_filename(record_id):
    """
    Convert a record_id string to a safe filename stem.
    e.g. '1-154' → '1-154'  (dashes are valid in Linux filenames)
    """
    return str(record_id)


def main():
    parser = get_ngc_parser("Extract NeuroVFM features from DASGIB CT scans")
    parser.add_argument(
        "--single", type=str, default=None,
        help="Process only this record_id (e.g. '1-154') — for testing"
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device       : {device}")
    print(f"Features dir : {features_dir}")
    print(f"Model path   : {args.model_path}")

    # --- Load labels / patient list ---
    record_ids, nii_paths, labels = get_aligned_arrays(args.csv_path)

    # Build lookup: record_id → nii_path
    id_to_path = {rid: path for rid, path in zip(record_ids, nii_paths)}

    # Optionally restrict to a single patient for testing
    if args.single is not None:
        if args.single not in id_to_path:
            print(f"ERROR: record_id '{args.single}' not found after filtering.")
            sys.exit(1)
        process_ids = [args.single]
        process_paths = [id_to_path[args.single]]
    else:
        process_ids = list(record_ids)
        process_paths = [id_to_path[rid] for rid in process_ids]

    print(f"Patients to process: {len(process_ids)}")

    # --- Load NeuroVFM encoder ---
    print("\nLoading NeuroVFM encoder...")
    from neurovfm.pipelines import load_encoder
    encoder, preprocessor = load_encoder(args.model_path, device=device)
    print("Encoder loaded.\n")

    processed = 0
    skipped = 0
    failed = []
    total_start = time.time()

    for rid, nii_path in tqdm(
        zip(process_ids, process_paths),
        total=len(process_ids),
        desc="Extracting features"
    ):
        fname = sanitize_filename(rid)
        output_path = features_dir / f"{fname}.pt"

        # Skip if already processed
        if output_path.exists():
            tqdm.write(f"  [{rid}] Already exists, skipping.")
            skipped += 1
            continue

        if not Path(nii_path).exists():
            tqdm.write(f"  [{rid}] WARNING: NIfTI not found: {nii_path}")
            failed.append((rid, "file not found"))
            continue

        try:
            # load_study handles reorientation, resampling, windowing, tokenisation
            batch = preprocessor.load_study(nii_path, modality="ct")

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            with torch.no_grad():
                embeddings = encoder.embed(batch)

            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]

            torch.save(embeddings.cpu(), output_path)
            tqdm.write(f"  [{rid}] OK — shape: {embeddings.shape}")
            processed += 1

            del batch, embeddings
            torch.cuda.empty_cache()

        except Exception as e:
            tqdm.write(f"  [{rid}] FAILED: {e}")
            failed.append((rid, str(e)))
            torch.cuda.empty_cache()

    elapsed = time.time() - total_start

    print("\n" + "=" * 55)
    print(f"Feature extraction complete in {elapsed:.1f}s")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped}  (already done)")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print("  Failed patients:")
        for rid, err in failed:
            print(f"    [{rid}]: {err}")


if __name__ == "__main__":
    main()
