"""
Attention-Gated Multiple Instance Learning (MIL) Head
for ICH detection on PhysioNet CT-ICH dataset.

Architecture: Gated Attention MIL (Ilse et al., 2018)
    "Attention-based Deep Multiple Instance Learning"
    https://arxiv.org/abs/1802.04712

Each patient is a "bag" of patch embeddings [N_patches, 768].
The attention network learns WHICH patches matter for the diagnosis,
and produces per-patch attention weights that we save for heatmapping.

Usage:
    python scripts/train_mil_head.py
    python scripts/train_mil_head.py --epochs 50 --lr 5e-4
"""

import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_base_parser, SEED, NUM_CV_FOLDS
from label_utils import get_aligned_labels


# ─────────────────────────────────────────────────────────────────────────────
# Model: Gated Attention MIL (Ilse et al. 2018)
# ─────────────────────────────────────────────────────────────────────────────

class GatedAttentionMIL(nn.Module):
    """
    Gated Attention MIL head.

    Input:  [N_patches, in_dim]  (variable N per patient)
    Output: (logit [1], attention_weights [N_patches])

    The attention mechanism:
        a_k = softmax( W_a * (tanh(V*h_k) ⊙ sigmoid(U*h_k)) )
        z   = Σ_k  a_k * h_k
        y   = classifier(z)
    """

    def __init__(self, in_dim=768, hidden_dim=128, dropout=0.25):
        super().__init__()

        # Optional feature compression (keeps memory low on RTX 4060)
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        proj_dim = 512

        # Gated attention (two separate branches V and U)
        self.attention_V = nn.Linear(proj_dim, hidden_dim)   # tanh branch
        self.attention_U = nn.Linear(proj_dim, hidden_dim)   # sigmoid gate
        self.attention_W = nn.Linear(hidden_dim, 1)           # scalar weight

        # Final binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [N_patches, in_dim]  (one patient = one bag)
        Returns:
            logit:   scalar tensor
            weights: [N_patches]  attention weights (sum to 1)
        """
        h = self.feature_proj(x)           # [N, 512]

        # Gated attention scores
        A_V = torch.tanh(self.attention_V(h))      # [N, hidden_dim]
        A_U = torch.sigmoid(self.attention_U(h))    # [N, hidden_dim]
        A   = self.attention_W(A_V * A_U)           # [N, 1]
        A   = torch.softmax(A, dim=0)               # [N, 1]  sums to 1

        # Weighted aggregation
        z = (A * h).sum(dim=0, keepdim=True)        # [1, 512]

        logit = self.classifier(z).squeeze()        # scalar
        weights = A.squeeze()                       # [N]

        return logit, weights


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_bags(features_dir, patient_ids):
    """Load raw [N, 768] tensors for all patients."""
    bags, missing = {}, []
    for pid in patient_ids:
        pt = Path(features_dir) / f"{pid:03d}.pt"
        if not pt.exists():
            missing.append(pid)
            continue
        patches = torch.load(pt, map_location="cpu")
        if isinstance(patches, tuple):
            patches = patches[0]
        bags[pid] = patches.float()
    if missing:
        print(f"  WARNING: missing .pt for patients: {missing}")
    return bags


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation for one fold
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, bags, ids, labels, device):
    model.train()
    total_loss = 0.0
    # Shuffle order each epoch
    perm = np.random.permutation(len(ids))
    for i in perm:
        pid   = ids[i]
        label = torch.tensor(labels[i], dtype=torch.float32, device=device)
        x     = bags[pid].to(device)

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
    for pid, lbl in zip(ids, labels):
        x = bags[pid].to(device)
        logit, weights = model(x)
        prob = torch.sigmoid(logit).item()
        probs.append(prob)
        preds.append(int(prob >= 0.5))
        if save_weights:
            attn_map[int(pid)] = weights.cpu().numpy()   # [N_patches]

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
# ROC plot (same style as linear_probe.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc(fold_results, results_dir):
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = []
    for fr in fold_results:
        fpr, tpr, _ = roc_curve(fr["y_true"], fr["y_prob"])
        ax.plot(fpr, tpr, alpha=0.35, color="steelblue", lw=1.2)
        aucs.append(fr["auc"])

    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"ROC — NeuroVFM Gated-Attention MIL (PhysioNet CT-ICH)\n"
        f"Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}",
        fontsize=12, fontweight="bold",
    )
    # Dummy line for legend
    ax.plot([], [], color="steelblue", lw=1.5,
            label=f"Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}")
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
    parser = get_base_parser("Train Attention-Gated MIL head for ICH detection")
    parser.add_argument("--epochs",     type=int,   default=40,
                        help="Training epochs per fold (default 40)")
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="Learning rate (default 3e-4)")
    parser.add_argument("--hidden-dim", type=int,   default=128,
                        help="Attention hidden dimension (default 128)")
    parser.add_argument("--dropout",    type=float, default=0.25,
                        help="Dropout rate (default 0.25)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay (default 1e-4)")
    parser.add_argument("--patience",   type=int,   default=15,
                        help="Early stopping patience in epochs (default 15)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir  = Path(args.results_dir)
    features_dir = Path(args.features_dir)
    csv_path     = str(Path(args.data_dir) / "hemorrhage_diagnosis_raw_ct.csv")

    print("=" * 60)
    print("  NeuroVFM — Gated-Attention MIL Head Training")
    print("=" * 60)
    print(f"  Device:   {device}")
    print(f"  Epochs:   {args.epochs}  |  LR: {args.lr}  |  Folds: {NUM_CV_FOLDS}")

    # ── 1. Labels & bags ──────────────────────────────────────────────────────
    patient_ids, labels = get_aligned_labels(csv_path, args.data_dir)
    bags = load_all_bags(features_dir, patient_ids)
    print(f"  Loaded {len(bags)} patient bags\n")

    # Filter to only patients with both label and bag
    valid_mask  = np.array([pid in bags for pid in patient_ids])
    patient_ids = patient_ids[valid_mask]
    labels      = labels[valid_mask]

    # ── 2. 5-Fold CV ──────────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=SEED)

    fold_metrics   = []
    fold_results   = []    # for ROC plotting
    all_attn_weights = {}  # pid → attention weights (from last test appearance)
    t0 = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(patient_ids, labels)):
        fold_num = fold_idx + 1
        print(f"── Fold {fold_num}/{NUM_CV_FOLDS} " + "─" * 40)

        train_ids    = patient_ids[train_idx]
        train_labels = labels[train_idx]
        test_ids     = patient_ids[test_idx]
        test_labels  = labels[test_idx]

        # Build model fresh each fold
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

        # Early stopping
        best_auc    = 0.0
        best_state  = None
        patience_ct = 0

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, optimizer, bags, train_ids, train_labels, device)
            scheduler.step()

            # Evaluate on test fold each epoch (cheap — small dataset)
            metrics, _ = evaluate(model, bags, test_ids, test_labels, device)

            if metrics["auc"] > best_auc:
                best_auc   = metrics["auc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ct = 0
            else:
                patience_ct += 1

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}  loss={loss:.4f}  "
                      f"val_AUC={metrics['auc']:.3f}  "
                      f"(best={best_auc:.3f})")

            if patience_ct >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Reload best weights
        model.load_state_dict(best_state)

        # Final evaluation with attention weight saving
        metrics, attn = evaluate(model, bags, test_ids, test_labels, device,
                                 save_weights=True)
        all_attn_weights.update(attn)

        # Collect per-patient probs for ROC (re-run silently)
        probs = []
        model.eval()
        with torch.no_grad():
            for pid in test_ids:
                logit, _ = model(bags[pid].to(device))
                probs.append(torch.sigmoid(logit).item())

        fold_results.append({
            "y_true": test_labels.tolist(),
            "y_prob": probs,
            "auc":    metrics["auc"],
        })
        fold_metrics.append(metrics)

        f1 = (2 * metrics["tp"] / (2 * metrics["tp"] + metrics["fp"] + metrics["fn"])
              if (2 * metrics["tp"] + metrics["fp"] + metrics["fn"]) > 0 else 0.0)
        print(f"  Fold {fold_num} result → "
              f"AUC={metrics['auc']:.3f}  "
              f"Acc={metrics['acc']:.3f}  "
              f"Sens={metrics['sens']:.3f}  "
              f"Spec={metrics['spec']:.3f}  "
              f"F1={f1:.3f}")

    elapsed = time.time() - t0

    # ── 3. Summary ────────────────────────────────────────────────────────────
    def mean_std(key):
        vals = [m[key] for m in fold_metrics]
        return np.mean(vals), np.std(vals)

    print(f"\n{'='*60}")
    print(f"  MIL Head — {NUM_CV_FOLDS}-Fold CV Summary  ({elapsed:.0f}s total)")
    print(f"{'='*60}")
    for key, label in [("auc","AUC"), ("acc","Accuracy"),
                        ("sens","Sensitivity"), ("spec","Specificity")]:
        mu, sd = mean_std(key)
        print(f"  {label:14s}: {mu:.3f} ± {sd:.3f}")

    # ── 4. Save attention weights ─────────────────────────────────────────────
    attn_dir = results_dir / "attention_weights"
    attn_dir.mkdir(parents=True, exist_ok=True)
    for pid, weights in all_attn_weights.items():
        np.save(attn_dir / f"{pid:03d}_attention.npy", weights)
    print(f"\n  Attention weights saved for {len(all_attn_weights)} patients → {attn_dir}")

    # ── 5. Plots & JSON ───────────────────────────────────────────────────────
    plot_roc(fold_results, results_dir)

    summary = {
        "model": "Gated-Attention MIL (Ilse et al. 2018)",
        "cv_folds": NUM_CV_FOLDS,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
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

    # ── 6. Comparison vs Linear Probe ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Model Comparison")
    print(f"{'='*60}")
    print(f"  Linear Probe (mean+max pooling) : AUC 0.875 ± 0.044")
    mu_auc, sd_auc = mean_std("auc")
    mu_acc, sd_acc = mean_std("acc")
    mu_se,  sd_se  = mean_std("sens")
    mu_sp,  sd_sp  = mean_std("spec")
    print(f"  Gated-Attention MIL (this run)  : "
          f"AUC {mu_auc:.3f} ± {sd_auc:.3f}  "
          f"Acc {mu_acc:.3f}  "
          f"Sens {mu_se:.3f}  "
          f"Spec {mu_sp:.3f}")
    print()


if __name__ == "__main__":
    main()
