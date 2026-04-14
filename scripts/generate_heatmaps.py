"""
Attention Heatmap Generator for ICH Detection QC.

For each patient, re-runs the CT preprocessing pipeline (no encoder/GPU needed)
to recover the patch coordinates, then maps MIL attention weights back to 3D
voxel space and saves visualisations of the top-N most-attended slices.

Output per patient:
    results/heatmaps/{pid:03d}_heatmap.png
        Three columns per slice row:
          1. CT (Brain window, preprocessed)
          2. CT + attention overlay (hot colormap)
          3. Attention heatmap only

Usage:
    # All patients that have attention weights
    python scripts/generate_heatmaps.py

    # Specific patients
    python scripts/generate_heatmaps.py --patients 49 73 105

    # Tune visualisation
    python scripts/generate_heatmaps.py --n-slices 7 --alpha 0.4
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_base_parser

# Must match the patch size used during feature extraction
PATCH_SIZE = (4, 16, 16)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_ct(nii_path):
    """
    Load a CT NIfTI, run the same preprocessing as extract_features.py, and
    return the per-window brain arrays and per-window patch coordinates.

    Returns
    -------
    brain_vol : np.ndarray [D, H, W]
        Brain-window CT volume after preprocessing (values in [0, 1]).
    volume_shape : tuple (D, H, W)
        Shape of the preprocessed volume (same for all three windows).
    window_coords : list of np.ndarray [N_i, 3]
        Patch-grid coordinates (d, h, w) for each foreground patch, one list
        entry per CT window (Brain → Blood → Bone).
    """
    from neurovfm.data.io import load_image
    from neurovfm.data.preprocess import prepare_for_inference, tokenize_volume

    img_sitk = load_image(str(nii_path), preprocess=True)
    if img_sitk is None:
        raise RuntimeError(f"load_image returned None for {nii_path}")

    result = prepare_for_inference(img_sitk, mode="ct")
    if result is None:
        raise RuntimeError(f"prepare_for_inference returned None for {nii_path}")

    img_arrs, background_mask, _ = result  # [brain_arr, blood_arr, bone_arr]

    window_coords = []
    for img_arr in img_arrs:
        _, coords, _ = tokenize_volume(
            img_arr,
            background_mask,
            patch_size=PATCH_SIZE,
            remove_background=True,  # identical to feature extraction
        )
        window_coords.append(coords)

    brain_vol = img_arrs[0]           # Brain window for background visualisation
    volume_shape = brain_vol.shape    # (D, H, W)

    return brain_vol, volume_shape, window_coords


# ─────────────────────────────────────────────────────────────────────────────
# Attention → 3D volume
# ─────────────────────────────────────────────────────────────────────────────

def build_attention_volume(window_coords, attn_weights, volume_shape):
    """
    Map per-patch attention weights back to a full 3D voxel volume.

    Patches from the three CT windows that share the same spatial position
    are averaged. The resulting patch-resolution grid is then upsampled to
    voxel resolution by nearest-neighbour repetition (np.repeat).

    Parameters
    ----------
    window_coords : list of np.ndarray [N_i, 3]
        Patch-grid coordinates per window (output of preprocess_ct).
    attn_weights : np.ndarray [N_total]
        Concatenated attention weights; N_total = sum of N_i.
    volume_shape : tuple (D, H, W)
        Shape of the preprocessed CT volume in voxels.

    Returns
    -------
    attn_vol : np.ndarray [D, H, W]  float32
        Attention values broadcast to voxel resolution.
    """
    p1, p2, p3 = PATCH_SIZE
    D, H, W = volume_shape
    n_d, n_h, n_w = D // p1, H // p2, W // p3

    attn_grid  = np.zeros((n_d, n_h, n_w), dtype=np.float64)
    count_grid = np.zeros((n_d, n_h, n_w), dtype=np.float64)

    offset = 0
    for coords in window_coords:
        n = len(coords)
        weights = attn_weights[offset : offset + n]
        # Vectorised scatter-add
        d_idx, h_idx, w_idx = coords[:, 0], coords[:, 1], coords[:, 2]
        # Clip to grid bounds (safety)
        valid = (d_idx < n_d) & (h_idx < n_h) & (w_idx < n_w)
        np.add.at(attn_grid,  (d_idx[valid], h_idx[valid], w_idx[valid]), weights[valid])
        np.add.at(count_grid, (d_idx[valid], h_idx[valid], w_idx[valid]), 1.0)
        offset += n

    # Average contributions where multiple windows overlap
    with np.errstate(invalid="ignore"):
        attn_grid = np.where(count_grid > 0, attn_grid / count_grid, 0.0)

    # Upsample patch-grid → voxel resolution (nearest-neighbour)
    attn_vol = np.repeat(np.repeat(np.repeat(
        attn_grid, p1, axis=0), p2, axis=1), p3, axis=2
    ).astype(np.float32)

    return attn_vol


# ─────────────────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(csv_path):
    """
    Load hemorrhage_diagnosis_raw_ct.csv and return a dict {patient_id: label}.
    label = 1 (SYG/ICH) if any slice has No_Hemorrhage == 0, else 0 (RASK).
    Returns empty dict if the CSV cannot be read.
    """
    import csv
    from collections import defaultdict

    labels = {}
    try:
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            slices = defaultdict(list)
            for row in reader:
                pid = int(row["PatientNumber"])
                slices[pid].append(int(row["No_Hemorrhage"]))
        for pid, vals in slices.items():
            labels[pid] = 0 if all(v == 1 for v in vals) else 1
    except Exception as e:
        print(f"  WARNING: could not load labels from {csv_path}: {e}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def top_slice_indices(attn_vol, n):
    """Return indices of the n slices with highest mean attention, sorted ascending."""
    scores = attn_vol.mean(axis=(1, 2))
    top = np.argsort(scores)[::-1][:n]
    return sorted(top.tolist())


def save_heatmap_figure(pid, brain_vol, attn_vol, out_dir,
                        n_slices=5, alpha=0.5, label=None):
    """
    Save a multi-row figure:  CT | Overlay | Attention-only
    One row per top-attended slice.

    Parameters
    ----------
    label : int or None
        Ground-truth label: 1 = SYG (ICH), 0 = RASK, None = ukendt.
    """
    slice_ids = top_slice_indices(attn_vol, n_slices)

    # ── Raw attention stats (shown in title for interpretability) ──────────
    raw_max  = float(attn_vol.max())
    raw_std  = float(attn_vol.std())
    # Concentrated attention (high max) → model is confident about a region.
    # Diffuse attention (low max, low std) → no clear focus → likely healthy.
    if raw_max >= 0.10:
        focus = "koncentreret (fokuseret region)"
    elif raw_max >= 0.03:
        focus = "moderat"
    else:
        focus = "diffus (ingen klar fokus)"

    # ── Ground-truth label string ──────────────────────────────────────────
    if label is None:
        label_str  = "Ukendt"
        label_col  = "gray"
    elif label == 1:
        label_str  = "SYG  (ICH positiv)"
        label_col  = "#c0392b"   # red
    else:
        label_str  = "RASK  (ingen blødning)"
        label_col  = "#27ae60"   # green

    # ── Normalise attention to [0, 1] for colourmap only ──────────────────
    a_min, a_max = attn_vol.min(), attn_vol.max()
    attn_norm = (attn_vol - a_min) / (a_max - a_min) if a_max > a_min else attn_vol.copy()

    fig, axes = plt.subplots(n_slices, 3, figsize=(13, 3.2 * n_slices),
                             squeeze=False)

    # ── Figure title ───────────────────────────────────────────────────────
    fig.suptitle(
        f"Patient {pid:03d}  —  Facit: {label_str}\n"
        f"Attention: max={raw_max:.4f}  std={raw_std:.4f}  →  {focus}\n"
        f"(top {n_slices} slices by mean attention)",
        fontsize=12, fontweight="bold", y=1.02,
        color=label_col,
    )

    col_titles = ["CT — Brain Window", "CT + Attention Overlay", "Attention Only"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight="bold")

    last_im = None
    for row, s in enumerate(slice_ids):
        ct_slice   = brain_vol[s]
        attn_slice = attn_norm[s]
        raw_slice_max = float(attn_vol[s].max())

        ax_ct, ax_ov, ax_heat = axes[row]

        ax_ct.imshow(ct_slice, cmap="gray", vmin=0, vmax=1, origin="lower")
        ax_ct.set_ylabel(f"slice {s}\n(raw max {raw_slice_max:.4f})", fontsize=8)
        ax_ct.axis("off")

        ax_ov.imshow(ct_slice, cmap="gray", vmin=0, vmax=1, origin="lower")
        last_im = ax_ov.imshow(attn_slice, cmap="hot", alpha=alpha,
                               vmin=0, vmax=1, origin="lower")
        ax_ov.axis("off")

        ax_heat.imshow(attn_slice, cmap="hot", vmin=0, vmax=1, origin="lower")
        ax_heat.axis("off")

    # Shared colourbar — note: shows normalised values, raw max in title
    fig.colorbar(last_im, ax=axes[:, 2].tolist(), shrink=0.7,
                 label=f"Attention (normaliseret 0→1,  rå max={raw_max:.4f})")

    fig.tight_layout()

    out_path = Path(out_dir) / f"{pid:03d}_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = get_base_parser("Generate MIL attention heatmaps for qualitative validation")
    parser.add_argument(
        "--patients", type=int, nargs="+", default=None,
        help="Patient IDs to process. Default: all with saved attention weights.",
    )
    parser.add_argument(
        "--n-slices", type=int, default=5,
        help="Number of top-attention slices to visualise per patient (default: 5).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Heatmap overlay opacity, 0=transparent 1=opaque (default: 0.5).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for heatmap PNGs (default: <results-dir>/heatmaps/).",
    )
    args = parser.parse_args()

    attn_dir = Path(args.results_dir) / "attention_weights"
    ct_dir   = Path(args.data_dir) / "ct_scans"
    out_dir  = Path(args.out_dir) if args.out_dir else Path(args.results_dir) / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ground-truth labels from CSV
    csv_path = Path(args.data_dir) / "hemorrhage_diagnosis_raw_ct.csv"
    labels   = load_labels(str(csv_path))
    if labels:
        n_pos = sum(v == 1 for v in labels.values())
        print(f"Labels indlæst: {len(labels)} patienter ({n_pos} SYG, {len(labels)-n_pos} RASK)")
    else:
        print("Ingen labels fundet — facit vises ikke i figurerne.")

    # Determine patient list
    if args.patients:
        patient_ids = args.patients
    else:
        patient_ids = sorted(
            int(p.stem.split("_")[0])
            for p in attn_dir.glob("*_attention.npy")
        )

    if not patient_ids:
        print(f"No attention weight files found in {attn_dir}")
        return

    print(f"Genererer heatmaps for {len(patient_ids)} patienter → {out_dir}")

    failed = []
    for pid in patient_ids:
        attn_path = attn_dir / f"{pid:03d}_attention.npy"
        nii_path  = ct_dir   / f"{pid:03d}.nii"

        if not attn_path.exists():
            print(f"  [{pid:03d}] Ingen attention weights — springer over.")
            continue
        if not nii_path.exists():
            print(f"  [{pid:03d}] NIfTI ikke fundet ({nii_path}) — springer over.")
            continue

        try:
            attn_weights = np.load(attn_path)                      # [N_total]
            brain_vol, volume_shape, window_coords = preprocess_ct(nii_path)

            total_patches = sum(len(c) for c in window_coords)
            if total_patches != len(attn_weights):
                n = min(total_patches, len(attn_weights))
                print(
                    f"  [{pid:03d}] Patch count mismatch: "
                    f"{total_patches} coords vs {len(attn_weights)} weights — "
                    f"bruger første {n}."
                )
                attn_weights = attn_weights[:n]
                trimmed, used = [], 0
                for c in window_coords:
                    take = min(len(c), n - used)
                    trimmed.append(c[:take])
                    used += take
                    if used >= n:
                        break
                window_coords = trimmed

            attn_vol = build_attention_volume(window_coords, attn_weights, volume_shape)
            label    = labels.get(pid, None)
            out_path = save_heatmap_figure(
                pid, brain_vol, attn_vol, out_dir,
                n_slices=args.n_slices, alpha=args.alpha, label=label,
            )
            label_str = {1: "SYG", 0: "RASK"}.get(label, "?")
            print(f"  [{pid:03d}] [{label_str}] Gemt → {out_path}")

        except Exception as exc:
            print(f"  [{pid:03d}] FAILED: {exc}")
            failed.append((pid, str(exc)))

    print(f"\nDone — {len(patient_ids) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        for pid, err in failed:
            print(f"  [{pid:03d}] {err}")


if __name__ == "__main__":
    main()
