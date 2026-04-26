"""
Create stratified 70/30 development/test split for the DASGIB dataset.

IMPORTANT — run this script ONCE before any feature extraction or model
training.  The test split must be locked away and never used during
model development (no feature extraction, no CV, no hyperparameter tuning).

Split strategy
--------------
  Full filtered dataset  (N patients, after all label_utils_ngc filters)
  ├── 70%  →  dev_labels.csv    ← all CV work happens here
  │     └── 5-fold CV (each fold: ~80% train / ~20% val, handled automatically)
  └── 30%  →  test_labels.csv  ← LOCKED. Touch only for final evaluation.

Both CSVs are saved to the private project directory on NGC and contain
the same columns as the filtered label table so they are drop-in
replacements for icp_path_pair.csv in all downstream scripts.

Usage
-----
    python create_data_splits.py
    python create_data_splits.py --csv-path /path/to/icp_path_pair.csv
                                 --out-dir   /path/to/output/
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_utils_ngc import load_ngc_labels, ICP_THRESHOLD

# Output directory on NGC (same as the private project base)
DEFAULT_OUT_DIR = "/projects/users/people/mehuns_r/projects/neurovfm_private"
DEFAULT_CSV     = ("/projects/users/data/UCPH/ICP/organized_data"
                   "/tables/icp_path_pair.csv")

SEED      = 42
DEV_RATIO = 0.70   # 70% development, 30% final test


def save_csv(records, out_path):
    """Write list-of-dicts to CSV."""
    if not records:
        raise ValueError("No records to save.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_split_summary(name, records):
    labels  = [r["label"] for r in records]
    n_total = len(labels)
    n_pos   = sum(labels)
    n_neg   = n_total - n_pos
    pct_pos = 100 * n_pos / n_total if n_total > 0 else 0
    print(f"  {name:20s}: {n_total:3d} patients  "
          f"({n_pos} elevated ICP={ICP_THRESHOLD:.0f}+  "
          f"[{pct_pos:.1f}%]  |  {n_neg} normal)")


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified 70/30 dev/test split for DASGIB"
    )
    parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV,
                        help="Path to icp_path_pair.csv")
    parser.add_argument("--out-dir",  type=str, default=DEFAULT_OUT_DIR,
                        help="Directory to save dev_labels.csv and test_labels.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and filter labels ─────────────────────────────────────────────
    print("\nLoading and filtering icp_path_pair.csv ...")
    records, stats = load_ngc_labels(args.csv_path)

    print(f"\nFilter pipeline:")
    print(f"  Total CSV rows         : {stats['total_rows']}")
    print(f"  After icp==0 removed   : {stats['after_icp0_removed']}")
    print(f"  After temporal filter  : {stats['after_temporal_filter']}")
    print(f"  After quality filter   : {stats['after_quality_filter']}")
    print(f"  After deduplication    : {stats['after_dedup']}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Final patient count    : {stats['n_patients']}  "
          f"({stats['n_elevated']} elevated, {stats['n_normal']} normal)")

    if stats['n_patients'] < 10:
        print("\nERROR: Too few patients after filtering. Check CSV path and filters.")
        sys.exit(1)

    # ── 2. Stratified 70/30 split ─────────────────────────────────────────────
    labels = np.array([r["label"] for r in records])

    dev_idx, test_idx = train_test_split(
        np.arange(len(records)),
        test_size=1 - DEV_RATIO,
        stratify=labels,
        random_state=SEED,
    )

    dev_records  = [records[i] for i in dev_idx]
    test_records = [records[i] for i in test_idx]

    # ── 3. Save splits ────────────────────────────────────────────────────────
    dev_path  = out_dir / "dev_labels.csv"
    test_path = out_dir / "test_labels.csv"

    save_csv(dev_records,  dev_path)
    save_csv(test_records, test_path)

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Data split complete  (random_state={SEED})")
    print(f"{'='*60}")
    print_split_summary("Full dataset",    records)
    print_split_summary(f"Development (70%)", dev_records)
    print_split_summary(f"Test (30%)",        test_records)
    print(f"\n  Saved:")
    print(f"    {dev_path}")
    print(f"    {test_path}")
    print(f"\n{'='*60}")
    print(f"  NEXT STEP: point config_ngc.py at dev_labels.csv")
    print(f"  DEFAULT_CSV_PATH = \"{dev_path}\"")
    print(f"\n  WARNING: Do NOT use test_labels.csv until final evaluation.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
