"""
Gated-Attention MIL Head for ICP detection on the DASGIB/NGC dataset.

Architecture: Gated Attention MIL (Ilse et al., 2018)
    "Attention-based Deep Multiple Instance Learning"
    https://arxiv.org/abs/1802.04712

Each patient is a "bag" of patch embeddings [N_patches, 768].
The attention network learns which patches matter for the ICP prediction.

Identical CV methodology to train_mil_head.py (PhysioNet) so results
are directly comparable to the Martin Zillmer baseline (~0.58 AUC).

Key difference: patient IDs are strings (e.g. '1-154'), feature files
are named {record_id}.pt accordingly.

Usage:
    python train_mil_head_ngc.py
    python train_mil_head_ngc.py --epochs 50 --lr 5e-4
"""

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_ngc import get_ngc_parser, SEED, NUM_CV_FOLDS
from label_utils_ngc import get_aligned_arrays


# ─────────────────────────────────────────────────────────────────────────────
# Model: Gated Attention MIL (Ilse et al. 2018)
# ─────────────────────────────────────────────────────────────────────────────

class GatedAttentionMIL(nn.Module):
    """
    Gated Attention MIL head.

    Input:  [N_patches, in_dim]  (variable N per patient/bag)
    Output: (logit [scalar], attention_weights [N_patches])

    Attention mechanism:
        a_k = softmax( W_a * (tanh(V*h_k) ⊙ sigmoid(U*h_k)) )
        z   = Σ_k  a_k * h_k
        y   = classifier(z)
    """

    def __init__(self, in_dim=768, hidden_dim=128, dropout=0.25):
        super().__init__()

        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        proj_dim = 512

        self.attention_V = nn.Linear(proj_dim, hidden_dim)   # tanh branch
        self.attention_U = nn.Linear(proj_dim, hidden_dim)   # sigmoid gate
        self.attention_W = nn.Linear(hidden_dim, 1)           # scalar weight

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [N_patches, in_dim]
        Returns:
            logit:   scalar tensor
            weights: [N_patches]  attention weights (sum to 1)
        """
        h = self.feature_proj(x)                        # [N, 512]

        A_V = torch.tanh(self.attention_V(h))           # [N, hidden_dim]
        A_U = torch.sigmoid(self.attention_U(h))        # [N, hidden_dim]
        A   = self.attention_W(A_V * A_U)               # [N, 1]
        A   = torch.softmax(A, dim=0)                   # [N, 1]

        z     = (A * h).sum(dim=0, keepdim=True)        # [1, 512]
        logit = self.classifier(z).squeeze()            # scalar
        weights = A.squeeze()                           # [N]

        return logit, weights


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_bags(features_dir, record_ids):
    """Load raw [N, 768] tensors for all patients. Keys are string record_ids."""
    bags, missing = {}, []
    for rid in record_ids:
        pt = Path(features_dir) / f"{rid}.pt"
        if not pt.exists():
            missing.append(rid)
            continue
        patches = torch.load(pt, map_location="cpu", weights_only=True)
        if isinstance(patches, tuple):
            patches = patches[0]
        bags[rid] = patches.float()
    if missing:
        print(f"  WARNING: missing .pt files for: {missing}")
    return bags


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, bags, ids, labels, device):
    model.train()
    total_loss = 0.0
    perm = np.random.permutation(len(ids))
    for i in perm:
        rid   = ids[i]
        label = torch.tensor(labels[i], dtype=torch.float32, device=device)
        x     = bags[rid].to(device)

        optimizer.zero_grad()
        logit, _ = model(x)
        loss = F.binary_cross_entropy_with_logits(logit, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(ids)


@torch.no_grad()
def evaluate(model, bags, ids, labels, device, save_weights=False):
    model.eval()
    probs, preds, attn_map = [], [], {}
    for rid, lbl in zip(ids, labels):
        x = bags[rid].to(device)
        logit, weights = model(x)
        prob = torch.sigmoid(logit).item()
        probs.append(prob)
        preds.append(int(prob >= 0.5))
        if save_weights:
            attn_map[rid] = weights.cpu().numpy()   # [N_patches]

    y_true = np.array(labels)
    y_prob = np.array(probs)
    y_pred = np.array(preds)

    auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    acc  = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = dict(auc=auc, acc=acc, sens=sens, spec=spec,
                   tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))
    return metrics, attn_map


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc(fold_results, results_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = []
    for fr in fold_results:
        fpr, tpr, _ = roc_curve(fr["y_true"], fr["y_prob"])
        ax.plot(fpr, tpr, alpha=0.35, color="steelblue", lw=1.2)
        aucs.append(fr["auc"])

    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.plot([], [], color="steelblue", lw=1.5,
            label=f"Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"ROC — NeuroVFM Gated-Attention MIL for ICP Detection (DASGIB)\n"
        f"Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        p = out / f"mil_roc_curve.{ext}"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = get_ngc_parser("Train Gated-Attention MIL head for ICP detection — DASGIB")
    parser.add_argument("--epochs",       type=int,   default=40,
                        help="Training epochs per fold (default 40)")
    parser.add_argument("--lr",           type=float, default=3e-4,
                        help="Learning rate (default 3e-4)")
    parser.add_argument("--hidden-dim",   type=int,   default=128,
                        help="Attention hidden dimension (default 128)")
    parser.add_argument("--dropout",      type=float, default=0.25,
                        help="Dropout rate (default 0.25)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay (default 1e-4)")
    parser.add_argument("--patience",     type=int,   default=15,
                        help="Early stopping patience in epochs (default 15)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = Path(args.results_dir)

    print("=" * 60)
    print("  NeuroVFM — Gated-Attention MIL Head Training (DASGIB)")
    print("=" * 60)
    print(f"  Device:   {device}")
    print(f"  Epochs:   {args.epochs}  |  LR: {args.lr}  |  Folds: {NUM_CV_FOLDS}")

    # ── 1. Labels & bags ──────────────────────────────────────────────────────
    record_ids, _, labels = get_aligned_arrays(args.csv_path)
    bags = load_all_bags(args.features_dir, record_ids)

    # Filter to patients with both label and feature file
    valid_mask = np.array([rid in bags for rid in record_ids])
    record_ids = record_ids[valid_mask]
    labels     = labels[valid_mask]
    print(f"\n  Loaded {len(bags)} patient bags")
    print(f"  Elevated ICP (label=1): {labels.sum()}  Normal (label=0): {(labels==0).sum()}\n")

    # ── 2. 5-Fold CV ──────────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=SEED)

    fold_metrics     = []
    fold_results     = []   # for ROC plotting
    all_attn_weights = {}   # rid → attention weights (from test appearance)
    t0 = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(record_ids, labels)):
        fold_num = fold_idx + 1
        print(f"── Fold {fold_num}/{NUM_CV_FOLDS} " + "─" * 40)

        train_ids    = record_ids[train_idx]
        train_labels = labels[train_idx]
        test_ids     = record_ids[test_idx]
        test_labels  = labels[test_idx]

        model = GatedAttentionMIL(
            in_dim=768,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

        best_auc    = 0.0
        best_state  = None
        patience_ct = 0

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, optimizer, bags, train_ids, train_labels, device)
            scheduler.step()

            metrics, _ = evaluate(model, bags, test_ids, test_labels, device)

            if metrics["auc"] > best_auc:
                best_auc    = metrics["auc"]
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ct = 0
            else:
                patience_ct += 1

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}  loss={loss:.4f}  "
                      f"val_AUC={metrics['auc']:.3f}  (best={best_auc:.3f})")

            if patience_ct >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        model.load_state_dict(best_state)

        metrics, attn = evaluate(model, bags, test_ids, test_labels, device,
                                 save_weights=True)
        all_attn_weights.update(attn)

        # Collect per-patient probs for ROC
        probs = []
        model.eval()
        with torch.no_grad():
            for rid in test_ids:
                logit, _ = model(bags[rid].to(device))
                probs.append(torch.sigmoid(logit).item())

        fold_results.append({
            "y_true": test_labels.tolist(),
            "y_prob": probs,
            "auc":    metrics["auc"],
        })
        fold_metrics.append(metrics)

        f1 = (2 * metrics["tp"] /
              (2 * metrics["tp"] + metrics["fp"] + metrics["fn"])
              if (2 * metrics["tp"] + metrics["fp"] + metrics["fn"]) > 0 else 0.0)
        print(f"  Fold {fold_num} result → "
              f"AUC={metrics['auc']:.3f}  Acc={metrics['acc']:.3f}  "
              f"Sens={metrics['sens']:.3f}  Spec={metrics['spec']:.3f}  F1={f1:.3f}")

    elapsed = time.time() - t0

    # ── 3. Summary ────────────────────────────────────────────────────────────
    def mean_std(key):
        vals = [m[key] for m in fold_metrics]
        return np.mean(vals), np.std(vals)

    print(f"\n{'='*60}")
    print(f"  MIL Head — {NUM_CV_FOLDS}-Fold CV Summary  ({elapsed:.0f}s total)")
    print(f"  (Compare to Martin baseline ~0.58 AUC)")
    print(f"{'='*60}")
    for key, label in [("auc","AUC"), ("acc","Accuracy"),
                        ("sens","Sensitivity"), ("spec","Specificity")]:
        mu, sd = mean_std(key)
        print(f"  {label:14s}: {mu:.3f} ± {sd:.3f}")

    # ── 4. Save attention weights ─────────────────────────────────────────────
    attn_dir = results_dir / "attention_weights"
    attn_dir.mkdir(parents=True, exist_ok=True)
    for rid, weights in all_attn_weights.items():
        # Sanitize record_id for filename (dashes are valid on Linux)
        np.save(attn_dir / f"{rid}_attention.npy", weights)
    print(f"\n  Attention weights saved for {len(all_attn_weights)} patients → {attn_dir}")

    # ── 5. Plots & JSON ───────────────────────────────────────────────────────
    plot_roc(fold_results, results_dir)

    summary = {
        "model": "Gated-Attention MIL (Ilse et al. 2018)",
        "dataset": "DASGIB",
        "icp_threshold_mmhg": 15,
        "cv_folds": NUM_CV_FOLDS,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "n_patients": int(len(labels)),
        "n_elevated": int(labels.sum()),
        "n_normal": int((labels == 0).sum()),
        "results": {
            key: {"mean": float(np.mean([m[key] for m in fold_metrics])),
                  "std":  float(np.std( [m[key] for m in fold_metrics]))}
            for key in ["auc", "acc", "sens", "spec"]
        },
        "per_fold": fold_metrics,
    }
    json_path = results_dir / "mil_metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Metrics saved → {json_path}")


if __name__ == "__main__":
    main()
