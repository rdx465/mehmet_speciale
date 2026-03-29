"""
Feature Space Visualization for PhysioNet CT-ICH dataset.

Loads extracted .pt feature files, applies Mean+Max pooling,
reduces with PCA, then visualises with t-SNE (and UMAP if installed).

Outputs high-resolution PNG + PDF saved to results/.
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_base_parser
from label_utils import get_aligned_labels


# ── colour palette ────────────────────────────────────────────────────────────
COLOURS = {0: "#4C72B0", 1: "#DD8452"}   # blue = healthy, orange = ICH
LABELS  = {0: "No Hemorrhage (n=39)", 1: "Hemorrhage / ICH (n=36)"}


def load_features(features_dir, patient_ids, pooling="mean_max"):
    """
    Load per-patient .pt files and pool patches → fixed-size vectors.

    pooling options:
        "mean"      → [768]
        "max"       → [768]
        "mean_max"  → [1536]   ← best AUC in our ablation
    """
    vecs = []
    missing = []
    for pid in patient_ids:
        pt_path = Path(features_dir) / f"{pid:03d}.pt"
        if not pt_path.exists():
            missing.append(pid)
            vecs.append(None)
            continue

        patches = torch.load(pt_path, map_location="cpu")   # [N, 768]
        if isinstance(patches, tuple):
            patches = patches[0]
        patches = patches.float()

        if pooling == "mean":
            v = patches.mean(dim=0)
        elif pooling == "max":
            v = patches.max(dim=0).values
        elif pooling == "mean_max":
            v = torch.cat([patches.mean(dim=0), patches.max(dim=0).values])
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        vecs.append(v.numpy())

    if missing:
        print(f"  WARNING: missing .pt files for patients: {missing}")

    # Filter out None entries
    valid_mask = [v is not None for v in vecs]
    vecs   = np.array([v for v in vecs if v is not None], dtype=np.float32)
    return vecs, np.array(valid_mask)


def run_tsne(X, perplexity=20, random_state=42):
    """PCA → 50 dims → t-SNE → 2D."""
    print(f"  Running PCA (50 components) on shape {X.shape} ...")
    n_components = min(50, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA variance explained: {var_explained:.1f}%")

    print(f"  Running t-SNE (perplexity={perplexity}) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        n_iter=2000,
        random_state=random_state,
    )
    X_2d = tsne.fit_transform(X_pca)
    print(f"  t-SNE KL divergence: {tsne.kl_divergence_:.4f}")
    return X_2d, var_explained


def try_umap(X, random_state=42):
    """Attempt UMAP reduction (optional dependency)."""
    try:
        import umap
        print("  Running UMAP ...")
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(X)
    except ImportError:
        print("  UMAP not installed (pip install umap-learn). Skipping.")
        return None


def make_plot(X_2d, labels, title, results_dir, filename_stem, var_explained=None):
    """Render scatter plot and save as PNG + PDF."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for lbl in [0, 1]:
        mask = labels == lbl
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=COLOURS[lbl],
            label=LABELS[lbl],
            s=90,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

    # Annotate with AUC if available
    subtitle = ""
    if var_explained is not None:
        subtitle = f"(PCA pre-reduction: {var_explained:.0f}% variance explained)"

    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Component 1", fontsize=11)
    ax.set_ylabel("Component 2", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = out_dir / f"{filename_stem}.{ext}"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")

    plt.close(fig)


def main():
    parser = get_base_parser("Visualise NeuroVFM feature space via t-SNE / UMAP")
    parser.add_argument("--pooling", type=str, default="mean_max",
                        choices=["mean", "max", "mean_max"],
                        help="Pooling strategy (default: mean_max, best AUC)")
    parser.add_argument("--perplexity", type=float, default=20,
                        help="t-SNE perplexity (default 20; try 15-30 for n=75)")
    parser.add_argument("--umap", action="store_true",
                        help="Also generate UMAP plot (requires: pip install umap-learn)")
    args = parser.parse_args()

    results_dir  = Path(args.results_dir)
    features_dir = Path(args.features_dir)
    csv_path     = str(Path(args.data_dir) / "hemorrhage_diagnosis_raw_ct.csv")

    print("=" * 55)
    print("  NeuroVFM Feature Space Visualisation")
    print("=" * 55)

    # ── 1. Labels ─────────────────────────────────────────────────────────────
    patient_ids, labels = get_aligned_labels(csv_path, args.data_dir)

    # ── 2. Load & pool features ───────────────────────────────────────────────
    print(f"\nLoading features (pooling={args.pooling}) ...")
    X_raw, valid_mask = load_features(features_dir, patient_ids, pooling=args.pooling)
    labels_valid = labels[valid_mask]
    print(f"  Feature matrix: {X_raw.shape}")

    # ── 3. Standardise ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # ── 4. t-SNE ──────────────────────────────────────────────────────────────
    print("\nt-SNE:")
    X_tsne, var_exp = run_tsne(X, perplexity=args.perplexity)
    make_plot(
        X_tsne, labels_valid,
        title="t-SNE — NeuroVFM Feature Space (PhysioNet CT-ICH)",
        results_dir=results_dir,
        filename_stem="tsne_feature_space",
        var_explained=var_exp,
    )

    # ── 5. UMAP (optional) ────────────────────────────────────────────────────
    if args.umap:
        print("\nUMAP:")
        X_umap = try_umap(X)
        if X_umap is not None:
            make_plot(
                X_umap, labels_valid,
                title="UMAP — NeuroVFM Feature Space (PhysioNet CT-ICH)",
                results_dir=results_dir,
                filename_stem="umap_feature_space",
            )

    print("\nDone. Plots saved to:", results_dir)


if __name__ == "__main__":
    main()
