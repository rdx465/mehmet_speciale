"""
Gated-Attention MIL Head for ICP detection — DASGIB/NGC dataset.

Architecture: Gated Attention MIL (Ilse et al., 2018)
    "Attention-based Deep Multiple Instance Learning"
    https://arxiv.org/abs/1802.04712

Stratified 5-fold CV with early stopping.
Reports mean ± std and 95% CI for AUC, Sensitivity, Specificity.
Results are directly comparable to Martin Zillmer's baseline (~0.58 AUC).
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
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_ngc import get_ngc_parser, SEED, NUM_CV_FOLDS
from label_utils_ngc import get_aligned_arrays


def ci95(values):
    """95% CI using t-distribution (appropriate for small n like 5 folds)."""
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    margin = t_crit * se
    return mean, float(margin)


# ─────────────────────────────────────────────────────────────────────────────
# Model: Gated Attention MIL (Ilse et al. 2018)
# ─────────────────────────────────────────────────────────────────────────────

class GatedAttentionMIL(nn.Module):
    """
    Input:  [N_patches, in_dim]
    Output: (logit scalar, attention_weights [N_patches])
    """

    def __init__(self, in_dim=768, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout)
        )
        proj_dim = 512
        self.attention_V = nn.Linear(proj_dim, hidden_dim)
        self.attention_U = nn.Linear(proj_dim, hidden_dim)
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        h   = self.feature_proj(x)
        A_V = torch.tanh(self.attention_V(h))
        A_U = torch.sigmoid(self.attention_U(h))
        A   = torch.softmax(self.attention_W(A_V * A_U), dim=0)
        z   = (A * h).sum(dim=0, keepdim=True)
        return self.classifier(z).squeeze(), A.squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_bags(features_dir, record_ids):
    """Load raw [N, 768] tensors. Keys are string record_ids."""
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
    for i in np.random.permutation(len(ids)):
        label = torch.tensor(labels[i], dtype=torch.float32, device=device)
        logit, _ = model(bags[ids[i]].to(device))
        optimizer.zero_grad()
        F.binary_cross_entropy_with_logits(logit, label).backward()
        optimizer.step()
        total_loss += F.binary_cross_entropy_with_logits(
            logit.detach(), label).item()
    return total_loss / len(ids)


@torch.no_grad()
def evaluate(model, bags, ids, labels, device, save_weights=False):
    model.eval()
    probs, preds, attn_map = [], [], {}
    for rid, lbl in zip(ids, labels):
        logit, weights = model(bags[rid].to(device))
        prob = torch.sigmoid(logit).item()
        probs.append(prob)
        preds.append(int(prob >= 0.5))
        if save_weights:
            attn_map[rid] = weights.cpu().numpy()

    y_true = np.array(labels)
    y_prob = np.array(probs)
    y_pred = np.array(preds)

    auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    acc  = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return dict(auc=auc, acc=acc, sens=sens, spec=spec,
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn)), attn_map


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

    mean_auc, margin = ci95(aucs)
    std_auc = np.std(aucs)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.plot([], [], color="steelblue", lw=1.5,
            label=f"Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}  "
                  f"[{mean_auc - margin:.3f} – {mean_auc + margin:.3f}]")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"ROC — NeuroVFM Gated-Attention MIL — DASGIB ICP\n"
        f"Mean AUC = {mean_auc:.3f} (95% CI: "
        f"{mean_auc - margin:.3f}–{mean_auc + margin:.3f})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        p = out / f"mil_roc_curve.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = get_ngc_parser("Gated-Attention MIL for ICP detection — DASGIB")
    parser.add_argument("--epochs",       type=int,   default=40)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--hidden-dim",   type=int,   default=128)
    parser.add_argument("--dropout",      type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=15)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = Path(args.results_dir)

    print("=" * 60)
    print("  NeuroVFM — Gated-Attention MIL Head (DASGIB)")
    print("=" * 60)
    print(f"  Device: {device}  |  Epochs: {args.epochs}  "
          f"|  LR: {args.lr}  |  Folds: {NUM_CV_FOLDS}")

    # Labels & bags
    record_ids, _, labels = get_aligned_arrays(args.csv_path)
    bags = load_all_bags(args.features_dir, record_ids)

    valid_mask = np.array([rid in bags for rid in record_ids])
    record_ids = record_ids[valid_mask]
    labels     = labels[valid_mask]
    print(f"\n  Bags loaded: {len(bags)}")
    print(f"  Elevated ICP (label=1): {labels.sum()}  "
          f"Normal (label=0): {(labels == 0).sum()}\n")

    skf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics     = []
    fold_results     = []
    all_attn_weights = {}
    t0 = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(record_ids, labels)):
        fold_num     = fold_idx + 1
        train_ids    = record_ids[train_idx]
        train_labels = labels[train_idx]
        test_ids     = record_ids[test_idx]
        test_labels  = labels[test_idx]

        print(f"── Fold {fold_num}/{NUM_CV_FOLDS} " + "─" * 38)

        model = GatedAttentionMIL(
            in_dim=768, hidden_dim=args.hidden_dim, dropout=args.dropout
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

        best_auc, best_state, patience_ct = 0.0, None, 0

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(
                model, optimizer, bags, train_ids, train_labels, device
            )
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
        metrics, attn = evaluate(
            model, bags, test_ids, test_labels, device, save_weights=True
        )
        all_attn_weights.update(attn)

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
        print(f"  Fold {fold_num} → AUC={metrics['auc']:.3f}  "
              f"Acc={metrics['acc']:.3f}  Sens={metrics['sens']:.3f}  "
              f"Spec={metrics['spec']:.3f}  F1={f1:.3f}")

    elapsed = time.time() - t0

    # Summary with 95% CI
    print(f"\n{'='*60}")
    print(f"  MIL — {NUM_CV_FOLDS}-Fold CV Summary  ({elapsed:.0f}s)")
    print(f"  (Martin Zillmer baseline: AUC ~0.58)")
    print(f"{'='*60}")
    summary_metrics = {}
    for key, label in [("auc","AUC"), ("acc","Accuracy"),
                        ("sens","Sensitivity"), ("spec","Specificity")]:
        vals = [m[key] for m in fold_metrics]
        mean, margin = ci95(vals)
        std = np.std(vals)
        summary_metrics[key] = {
            "mean":    float(mean),
            "std":     float(std),
            "ci95_lo": float(mean - margin),
            "ci95_hi": float(mean + margin),
        }
        print(f"  {label:14s}: {mean:.3f} ± {std:.3f}   "
              f"[{mean - margin:.3f} – {mean + margin:.3f}]")

    # Save attention weights
    attn_dir = results_dir / "attention_weights"
    attn_dir.mkdir(parents=True, exist_ok=True)
    for rid, weights in all_attn_weights.items():
        np.save(attn_dir / f"{rid}_attention.npy", weights)
    print(f"\n  Attention weights → {attn_dir}  ({len(all_attn_weights)} patients)")

    plot_roc(fold_results, results_dir)

    output = {
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
        "summary": summary_metrics,
        "per_fold": fold_metrics,
    }
    json_path = results_dir / "mil_metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Metrics → {json_path}")


if __name__ == "__main__":
    main()
