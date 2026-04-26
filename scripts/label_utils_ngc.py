"""
Label utilities for the DASGIB/NGC dataset.

Reads icp_path_pair.csv and applies the following filters before
deriving binary ICP labels (matching Martin Zillmer's inclusion criteria):

  1. Remove rows with icp == 0.0  (invalid/missing measurement)
  2. Temporal filter              abs(days_from_ct) <= 1
  3. Quality filter               all ScanDimensions >= 50 pixels
  4. ICP threshold                icp > 15 mmHg  → label 1, else 0
  5. Deduplication                keep first scan per record_id

Returns a 1:1 mapping: record_id (str) → (nii_file path, label)
"""

import ast
import csv
import numpy as np
from pathlib import Path

ICP_THRESHOLD = 15.0          # mmHg — elevated ICP (Danish clinical standard)
MAX_DAYS_FROM_CT = 1          # ±1 day between CT and ICP measurement
MIN_SCAN_DIM = 50             # pixels — reject degenerate near-2D scans


def _parse_scan_dimensions(dim_str):
    """
    Parse ScanDimensions string like '(716, 512, 58)' into a tuple of ints.
    Returns None if parsing fails.
    """
    try:
        dims = ast.literal_eval(dim_str.strip())
        return tuple(int(d) for d in dims)
    except Exception:
        return None


def _is_axial(nii_file):
    """Return True if the filename contains 'ax' or 'tra' (axial orientation)."""
    name = Path(nii_file).name.lower()
    return "ax" in name or "tra" in name


def load_ngc_labels(csv_path):
    """
    Load and filter icp_path_pair.csv, returning a patient-level label table.

    The nii_file column already contains the full absolute path on NGC —
    no base directory is prepended.

    Args:
        csv_path : Path to icp_path_pair.csv

    Returns:
        records  : list of dicts with keys:
                     record_id (str), nii_path (str), icp (float), label (int)
        stats    : dict with row counts at each filter stage (for logging)
    """
    csv_path = Path(csv_path)

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    stats = {"total_rows": len(rows)}

    # --- 1. Remove icp == 0 ---
    rows = [r for r in rows if float(r["icp"]) != 0.0]
    stats["after_icp0_removed"] = len(rows)

    # --- 2. Temporal filter: abs(days_from_ct) <= 1 ---
    def _within_window(r):
        try:
            return abs(float(r["days_from_ct"])) <= MAX_DAYS_FROM_CT
        except (ValueError, TypeError):
            return False

    rows = [r for r in rows if _within_window(r)]
    stats["after_temporal_filter"] = len(rows)

    # --- 3. Quality filter: all ScanDimensions >= MIN_SCAN_DIM ---
    def _quality_ok(r):
        dims = _parse_scan_dimensions(r.get("ScanDimensions", ""))
        if dims is None:
            return False  # can't parse → reject
        return all(d >= MIN_SCAN_DIM for d in dims)

    rows = [r for r in rows if _quality_ok(r)]
    stats["after_quality_filter"] = len(rows)

    # --- 4. Deduplicate: keep first scan per record_id ---
    # CSV is assumed to be ordered; first occurrence = first scan in time.
    seen = set()
    unique_rows = []
    for r in rows:
        rid = r["record_id"]
        if rid not in seen:
            seen.add(rid)
            unique_rows.append(r)
    rows = unique_rows
    stats["after_dedup"] = len(rows)

    # --- 6. Build records with label ---
    records = []
    for r in rows:
        icp_val = float(r["icp"])
        label = 1 if icp_val > ICP_THRESHOLD else 0

        nii_path = r["nii_file"]  # CSV already contains the full absolute path

        records.append({
            "record_id": r["record_id"],
            "nii_path": nii_path,
            "icp": icp_val,
            "label": label,
        })

    n_pos = sum(rec["label"] for rec in records)
    n_neg = len(records) - n_pos
    stats["n_patients"] = len(records)
    stats["n_elevated"] = n_pos
    stats["n_normal"] = n_neg

    return records, stats


def load_split_labels(csv_path):
    """
    Read an already-filtered split CSV (dev_labels.csv or test_labels.csv).

    These files have 4 columns: record_id, nii_path, icp, label
    — produced by create_data_splits.py.  No filtering is applied here
    because the split was derived from an already-filtered dataset.

    Returns:
        record_ids : np.ndarray of str, shape [N]
        nii_paths  : np.ndarray of str, shape [N]
        labels     : np.ndarray of int, shape [N]  (1 = elevated ICP)
    """
    csv_path = Path(csv_path)
    records = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append({
                "record_id": row["record_id"],
                "nii_path":  row["nii_path"],
                "icp":       float(row["icp"]),
                "label":     int(row["label"]),
            })

    n_pos = sum(r["label"] for r in records)
    n_neg = len(records) - n_pos
    print(f"\nLoaded split CSV: {csv_path.name}")
    print(f"  => {len(records)} patients  "
          f"({n_pos} elevated ICP > {ICP_THRESHOLD} mmHg, {n_neg} normal)")

    record_ids = np.array([r["record_id"] for r in records], dtype=object)
    nii_paths  = np.array([r["nii_path"]  for r in records], dtype=object)
    labels     = np.array([r["label"]     for r in records], dtype=np.int64)

    return record_ids, nii_paths, labels


def get_aligned_arrays(csv_path):
    """
    Load labels from either the original icp_path_pair.csv (applies all
    filters) or an already-filtered split CSV (dev_labels / test_labels).

    Auto-detects format by checking for the 'nii_path' column header.

    Returns:
        record_ids : np.ndarray of str, shape [N]
        nii_paths  : np.ndarray of str, shape [N]
        labels     : np.ndarray of int, shape [N]  (1 = elevated ICP)
    """
    # Peek at the header to decide which reader to use
    with open(csv_path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))

    if "nii_path" in header:
        # Already-filtered split CSV — read directly
        return load_split_labels(csv_path)

    # Original icp_path_pair.csv — apply full filter pipeline
    records, stats = load_ngc_labels(csv_path)

    print("\nLabel loading — filter statistics:")
    print(f"  Total CSV rows        : {stats['total_rows']}")
    print(f"  After icp==0 removed  : {stats['after_icp0_removed']}")
    print(f"  After temporal filter : {stats['after_temporal_filter']}")
    print(f"  After quality filter  : {stats['after_quality_filter']}")
    print(f"  After deduplication   : {stats['after_dedup']}")
    print(f"  => {stats['n_patients']} patients  "
          f"({stats['n_elevated']} elevated ICP > {ICP_THRESHOLD} mmHg, "
          f"{stats['n_normal']} normal)")

    record_ids = np.array([r["record_id"] for r in records], dtype=object)
    nii_paths  = np.array([r["nii_path"]  for r in records], dtype=object)
    labels     = np.array([r["label"]     for r in records], dtype=np.int64)

    return record_ids, nii_paths, labels


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/projects/users/data/UCPH/ICP/organized_data/tables/icp_path_pair.csv"

    ids, paths, labels = get_aligned_arrays(csv_path)
    print(f"\nSample records:")
    for i in range(min(5, len(ids))):
        print(f"  {ids[i]:>10s}  label={labels[i]}  path={paths[i]}")
