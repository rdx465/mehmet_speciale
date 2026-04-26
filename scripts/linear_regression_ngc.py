"""
Linear Regression baseline for ICP detection — DASGIB/NGC dataset.

Uses sklearn LinearRegression, thresholds predictions at 0.5 for
binary classification. Simpler baseline than logistic regression.

Stratified 5-fold CV. Reports mean ± std and 95% CI for AUC,
Sensitivity, Specificity.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_ngc import get_ngc_parser, SEED, NUM_CV_FOLDS, DEFAULT_DEV_CSV_PATH
from label_utils_ngc import get_aligned_arrays


def ci95(values):
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, float(t_crit * se)


def load_features(features_dir, record_ids, pooling="mean"):
    features_dir = Path(features_dir)
    X, valid_mask = [], []

    for rid in record_ids:
        pt_path = features_dir / f"{rid}.pt"
        if not pt_path.exists():
            print(f"  WARNING: {pt_path} not found, skipping {rid}")
            valid_mask.append(False)
            continue

        raw = torch.load(pt_path, map_location="cpu", weights_only=True)
        if isinstance(raw, tuple):
            raw = raw[0]

        if pooling == "mean":
            pooled = raw.mean(dim=0)
        elif pooling == "max":
            pooled = raw.max(dim=0).values
        elif pooling == "mean_max":
            pooled = torch.cat([raw.mean(dim=0), raw.max(dim=0).values])
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        X.append(pooled.numpy())
        valid_mask.append(True)

    return np.stack(X), np.array(valid_mask)


def run_cv(X, y, n_splits=NUM_CV_FOLDS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_metrics = []
    all_y_true, all_y_prob, all_y_pred = [], [], []
    fold_roc_data = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = LinearRegression()
        clf.fit(X_train, y_train)

        y_prob = clf.predict(X_test)
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        fold_metrics.append({
            "fold": fold_idx + 1,
            "auc": auc, "accuracy": acc,
            "sensitivity": sens, "specificity": spec, "f1": f1,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })
        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())
        all_y_pred.extend(y_pred.tolist())
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fold_roc_data.append((fpr, tpr, auc))

        print(f"  Fold {fold_idx+1}: AUC={auc:.3f}  Acc={acc:.3f}  "
              f"Sens={sens:.3f}  Spec={spec:.3f}")

    summary = {}
    for m in ["auc", "accuracy", "sensitivity", "specificity", "f1"]:
        vals = [f[m] for f in fold_metrics]
        mean, margin = ci95(vals)
        summary[m] = {
            "mean":    float(mean),
            "std":     float(np.std(vals)),
            "ci95_lo": float(mean - margin),
            "ci95_hi": float(mean + margin),
        }

    return {
        "folds": fold_metrics, "summary": summary,
        "roc_data": fold_roc_data,
        "all_y_true": all_y_true,
        "all_y_prob": all_y_prob,
        "all_y_pred": all_y_pred,
    }


def plot_roc(roc_data, results_dir, pooling):
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for fpr, tpr, auc in roc_data:
        ax.plot(fpr, tpr, alpha=0.3, lw=1)
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean([d[2] for d in roc_data])
    std_auc  = np.std([d[2] for d in roc_data])

    ax.plot(mean_fpr, mean_tpr, color="b", lw=2,
            label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr,
                    np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1),
                    alpha=0.2, color="b")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — Linear Regression (pooling={pooling}) — DASGIB ICP")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    out_path = Path(results_dir) / f"linreg_{pooling}_roc_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC plot saved: {out_path}")


def main():
    parser = get_ngc_parser("Linear Regression baseline for ICP — DASGIB",
                            csv_default=DEFAULT_DEV_CSV_PATH)
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "mean_max"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    record_ids, _, labels = get_aligned_arrays(args.csv_path)

    print(f"\nLoading features (pooling={args.pooling})...")
    X, valid_mask = load_features(args.features_dir, record_ids, pooling=args.pooling)
    record_ids = record_ids[valid_mask]
    labels     = labels[valid_mask]
    print(f"Loaded {len(X)} feature vectors, shape: {X.shape}")
    print(f"Elevated ICP (label=1): {labels.sum()}  "
          f"Normal (label=0): {(labels == 0).sum()}")

    print(f"\n{'='*60}")
    print(f"Stratified {NUM_CV_FOLDS}-Fold CV — Linear Regression "
          f"(pooling={args.pooling})")
    print(f"{'='*60}")
    results = run_cv(X, labels)

    print(f"\nSummary (Martin Zillmer baseline: AUC ~0.58):")
    print(f"  {'Metric':>12s}   {'Mean':>6s} ± {'Std':>6s}   95% CI")
    print(f"  {'-'*50}")
    for m in ["auc", "accuracy", "sensitivity", "specificity", "f1"]:
        s = results["summary"][m]
        print(f"  {m:>12s}   {s['mean']:.3f} ± {s['std']:.3f}   "
              f"[{s['ci95_lo']:.3f} – {s['ci95_hi']:.3f}]")

    plot_roc(results["roc_data"], results_dir, args.pooling)

    output = {
        "model": "Linear Regression",
        "dataset": "DASGIB",
        "pooling": args.pooling,
        "icp_threshold_mmhg": 15,
        "n_patients": int(len(labels)),
        "n_elevated": int(labels.sum()),
        "n_normal": int((labels == 0).sum()),
        "cv_folds": NUM_CV_FOLDS,
        "per_fold": results["folds"],
        "summary": results["summary"],
    }
    metrics_path = results_dir / f"linreg_{args.pooling}_metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
