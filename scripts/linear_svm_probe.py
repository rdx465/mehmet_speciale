"""
Linear SVM Probe for ICH Detection using NeuroVFM features.

Unlike logistic regression (which uses a sigmoid nonlinearity), LinearSVC
fits a pure linear decision boundary (hyperplane) — making this a true
linear probe in the representation learning sense.

AUC is computed from the raw SVM decision function scores (distance to
hyperplane), not from calibrated probabilities.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_base_parser, SEED, EMBED_DIM, NUM_CV_FOLDS


def load_features(features_dir, patient_ids, pooling="mean"):
    """Load .pt feature files and aggregate per patient."""
    features_dir = Path(features_dir)
    X = []
    valid_mask = []

    for pid in patient_ids:
        pt_path = features_dir / f"{pid:03d}.pt"
        if not pt_path.exists():
            print(f"  WARNING: {pt_path} not found, skipping patient {pid}")
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


def run_cv(X, y, n_splits=NUM_CV_FOLDS, C_values=None):
    """
    Stratified k-fold CV with nested C-tuning for LinearSVC.
    AUC uses decision_function scores (signed distance to hyperplane).
    """
    if C_values is None:
        C_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_metrics = []
    all_y_true = []
    all_y_score = []
    all_y_pred = []
    fold_roc_data = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Inner CV to select best C
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        best_C = C_values[0]
        best_inner_auc = -1

        for C in C_values:
            inner_aucs = []
            for inner_train, inner_val in inner_skf.split(X_train, y_train):
                clf = LinearSVC(C=C, max_iter=10000, random_state=SEED)
                clf.fit(X_train[inner_train], y_train[inner_train])
                scores = clf.decision_function(X_train[inner_val])
                try:
                    auc = roc_auc_score(y_train[inner_val], scores)
                except ValueError:
                    auc = 0.5
                inner_aucs.append(auc)

            mean_auc = np.mean(inner_aucs)
            if mean_auc > best_inner_auc:
                best_inner_auc = mean_auc
                best_C = C

        # Train final model with best C
        clf = LinearSVC(C=best_C, max_iter=10000, random_state=SEED)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_score = clf.decision_function(X_test)  # raw linear scores, no sigmoid

        try:
            auc = roc_auc_score(y_test, y_score)
        except ValueError:
            auc = 0.5
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        fold_metrics.append({
            "fold": fold_idx + 1, "C": best_C,
            "auc": auc, "accuracy": acc,
            "sensitivity": sens, "specificity": spec, "f1": f1,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })

        all_y_true.extend(y_test.tolist())
        all_y_score.extend(y_score.tolist())
        all_y_pred.extend(y_pred.tolist())

        fpr, tpr, _ = roc_curve(y_test, y_score)
        fold_roc_data.append((fpr, tpr, auc))

        print(f"  Fold {fold_idx+1}: AUC={auc:.3f}  Acc={acc:.3f}  "
              f"Sens={sens:.3f}  Spec={spec:.3f}  C={best_C}")

    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {}
    for m in metric_names:
        vals = [f[m] for f in fold_metrics]
        summary[f"{m}_mean"] = float(np.mean(vals))
        summary[f"{m}_std"] = float(np.std(vals))

    return {
        "folds": fold_metrics, "summary": summary,
        "roc_data": fold_roc_data,
        "all_y_true": all_y_true,
        "all_y_score": all_y_score,
        "all_y_pred": all_y_pred,
    }


def run_loo(X, y):
    """Leave-One-Out CV as secondary evaluation."""
    loo = LeaveOneOut()
    y_true_all, y_score_all, y_pred_all = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LinearSVC(C=1.0, max_iter=10000, random_state=SEED)
        clf.fit(X_train, y_train)

        y_pred_all.append(clf.predict(X_test)[0])
        y_score_all.append(clf.decision_function(X_test)[0])
        y_true_all.append(y_test[0])

    y_true_all = np.array(y_true_all)
    y_score_all = np.array(y_score_all)
    y_pred_all = np.array(y_pred_all)

    auc = roc_auc_score(y_true_all, y_score_all)
    acc = accuracy_score(y_true_all, y_pred_all)
    print(f"  LOO-CV: AUC={auc:.3f}  Acc={acc:.3f}")
    return {"auc": float(auc), "accuracy": float(acc)}


def plot_roc(roc_data, results_dir):
    """Plot per-fold ROC curves with mean curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for fpr, tpr, auc in roc_data:
        ax.plot(fpr, tpr, alpha=0.3, lw=1)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean([d[2] for d in roc_data])
    std_auc = np.std([d[2] for d in roc_data])

    ax.plot(mean_fpr, mean_tpr, color="b", lw=2,
            label=f"Mean ROC (AUC = {mean_auc:.3f} +/- {std_auc:.3f})")
    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1), alpha=0.2, color="b")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC - NeuroVFM Linear SVM Probe for ICH Detection")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    out_path = Path(results_dir) / "svm_roc_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC plot saved: {out_path}")


def plot_confusion_matrix(y_true, y_pred, results_dir):
    """Plot aggregated confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix - Linear SVM (aggregated across folds)")
    fig.colorbar(im)
    classes = ["No Hemorrhage", "Hemorrhage"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(classes)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=16)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    out_path = Path(results_dir) / "svm_confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved: {out_path}")


def main():
    parser = get_base_parser("Linear SVM Probe for ICH detection")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "mean_max"],
                        help="Feature aggregation strategy")
    parser.add_argument("--run-loo", action="store_true",
                        help="Also run Leave-One-Out CV")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    from label_utils import get_aligned_labels
    csv_path = str(Path(args.data_dir) / "hemorrhage_diagnosis_raw_ct.csv")
    patient_ids, labels = get_aligned_labels(csv_path, args.data_dir)

    print(f"\nLoading features (pooling={args.pooling})...")
    X, valid_mask = load_features(args.features_dir, patient_ids, pooling=args.pooling)
    patient_ids = patient_ids[valid_mask]
    labels = labels[valid_mask]
    print(f"Loaded {len(X)} feature vectors, shape: {X.shape}")
    print(f"Positive (ICH): {labels.sum()}  Negative: {(labels == 0).sum()}")

    print(f"\n{'='*55}")
    print(f"Stratified {NUM_CV_FOLDS}-Fold CV  —  Linear SVM  (pooling={args.pooling})")
    print(f"AUC via decision_function scores (no sigmoid)")
    print(f"{'='*55}")
    results = run_cv(X, labels)

    print(f"\nSummary:")
    for metric in ["auc", "accuracy", "sensitivity", "specificity", "f1"]:
        mean = results["summary"][f"{metric}_mean"]
        std = results["summary"][f"{metric}_std"]
        print(f"  {metric:>12s}: {mean:.3f} +/- {std:.3f}")

    print(f"\nGenerating plots...")
    plot_roc(results["roc_data"], results_dir)
    plot_confusion_matrix(results["all_y_true"], results["all_y_pred"], results_dir)

    loo_results = None
    if args.run_loo:
        print(f"\n{'='*55}")
        print("Leave-One-Out Cross-Validation  —  Linear SVM")
        print(f"{'='*55}")
        loo_results = run_loo(X, labels)

    output = {
        "model": "Linear SVM (LinearSVC, decision_function scores for AUC)",
        "pooling": args.pooling,
        "n_patients": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "cv_folds": NUM_CV_FOLDS,
        "cv_results": results["folds"],
        "summary": results["summary"],
    }
    if loo_results:
        output["loo_cv"] = loo_results

    metrics_path = results_dir / "svm_metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
