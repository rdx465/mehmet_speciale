"""
Final test-set evaluation for ICP detection — DASGIB/NGC dataset.

For each model:
  1. Tune hyperparameters via 3-fold inner CV on dev_labels.csv
  2. Refit on the FULL dev split
  3. Predict on the held-out test_labels.csv
  4. Report AUC, Acc, Sens, Spec, F1 with bootstrap 95% CIs

Models covered:
  - Linear Regression      (no hyperparameter)
  - Logistic Regression    (tune C)
  - Linear SVM             (tune C)
  - Gated-Attention MIL    (CV-ensemble: average sigmoid over 5 dev folds)

Pooling for non-MIL models: mean | max | mean_max (default: mean).

Output:
  results_dir/test_metrics_summary.json
  results_dir/test_roc_curves.png
"""

import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_ngc import (
    SEED, NUM_CV_FOLDS,
    DEFAULT_DEV_CSV_PATH, DEFAULT_FEATURES_DIR, DEFAULT_RESULTS_DIR,
)
from label_utils_ngc import get_aligned_arrays
from train_mil_head_ngc import GatedAttentionMIL, load_all_bags, train_one_epoch

# Default location of the held-out test split (created by create_data_splits.py)
DEFAULT_TEST_CSV_PATH = str(
    Path(DEFAULT_DEV_CSV_PATH).parent / "test_labels.csv"
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature loading (mirrors linear_probe_ngc.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_pooled_features(features_dir, record_ids, pooling="mean"):
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


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap 95% CI on a single test set
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_metrics(y_true, y_score, y_pred, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs, accs, senss, specs, f1s = [], [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        yp = y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        senss.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    def _ci(vals):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(np.mean(vals)), float(lo), float(hi)

    return {
        "auc":         _ci(aucs),
        "accuracy":    _ci(accs),
        "sensitivity": _ci(senss),
        "specificity": _ci(specs),
        "f1":          _ci(f1s),
    }


def point_metrics(y_true, y_score, y_pred):
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.5
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc": float(auc), "accuracy": float(acc), "f1": float(f1),
        "sensitivity": float(sens), "specificity": float(spec),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter selection on dev (inner 3-fold CV)
# ─────────────────────────────────────────────────────────────────────────────

def select_best_C(X_dev, y_dev, model_cls, decision_attr,
                  C_values=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Generic inner 3-fold CV. model_cls(C=...).fit(...) → AUC via decision_attr."""
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_C, best_auc = C_values[0], -1.0
    for C in C_values:
        aucs = []
        for tr, va in inner.split(X_dev, y_dev):
            clf = model_cls(C=C)
            clf.fit(X_dev[tr], y_dev[tr])
            scores = getattr(clf, decision_attr)(X_dev[va])
            if scores.ndim == 2:                # predict_proba → take pos col
                scores = scores[:, 1]
            try:
                aucs.append(roc_auc_score(y_dev[va], scores))
            except ValueError:
                aucs.append(0.5)
        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc, best_C = mean_auc, C
    return best_C, best_auc


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn model wrappers
# ─────────────────────────────────────────────────────────────────────────────

def fit_eval_linreg(X_dev, y_dev, X_test, y_test):
    clf = LinearRegression().fit(X_dev, y_dev)
    y_score = clf.predict(X_test)
    y_pred  = (y_score >= 0.5).astype(int)
    return y_score, y_pred, {}


def fit_eval_logreg(X_dev, y_dev, X_test, y_test):
    def _ctor(C): return LogisticRegression(
        C=C, max_iter=10000, random_state=SEED, solver="lbfgs"
    )
    best_C, _ = select_best_C(X_dev, y_dev, _ctor, "predict_proba")
    clf = _ctor(best_C).fit(X_dev, y_dev)
    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred  = clf.predict(X_test)
    return y_score, y_pred, {"best_C": best_C}


def fit_eval_svm(X_dev, y_dev, X_test, y_test):
    def _ctor(C): return LinearSVC(C=C, max_iter=10000, random_state=SEED)
    best_C, _ = select_best_C(X_dev, y_dev, _ctor, "decision_function")
    clf = _ctor(best_C).fit(X_dev, y_dev)
    y_score = clf.decision_function(X_test)
    y_pred  = clf.predict(X_test)
    return y_score, y_pred, {"best_C": best_C}


# ─────────────────────────────────────────────────────────────────────────────
# MIL CV-ensemble: train 5 models (one per dev fold) and average on test
# ─────────────────────────────────────────────────────────────────────────────

def fit_eval_mil(dev_ids, dev_labels, dev_bags,
                 test_ids, test_labels, test_bags,
                 device, epochs=40, lr=3e-4, hidden_dim=128,
                 dropout=0.25, weight_decay=1e-4, patience=15):
    skf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=SEED)
    probs_per_fold = []

    for fold_idx, (tr, va) in enumerate(skf.split(dev_ids, dev_labels)):
        train_ids   = dev_ids[tr]
        train_lbls  = dev_labels[tr]
        val_ids     = dev_ids[va]
        val_lbls    = dev_labels[va]

        model = GatedAttentionMIL(in_dim=768, hidden_dim=hidden_dim,
                                  dropout=dropout).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=1e-6
        )

        best_auc, best_state, stalled = 0.0, None, 0
        for epoch in range(1, epochs + 1):
            train_one_epoch(model, opt, dev_bags, train_ids, train_lbls, device)
            sched.step()
            with torch.no_grad():
                model.eval()
                val_probs = []
                for rid in val_ids:
                    logit, _ = model(dev_bags[rid].to(device))
                    val_probs.append(torch.sigmoid(logit).item())
                try:
                    val_auc = roc_auc_score(val_lbls, val_probs)
                except ValueError:
                    val_auc = 0.5
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                stalled = 0
            else:
                stalled += 1
            if stalled >= patience:
                break
        model.load_state_dict(best_state)

        # Predict on test
        model.eval()
        fold_probs = []
        with torch.no_grad():
            for rid in test_ids:
                logit, _ = model(test_bags[rid].to(device))
                fold_probs.append(torch.sigmoid(logit).item())
        probs_per_fold.append(np.array(fold_probs))
        print(f"  MIL fold {fold_idx+1}/{NUM_CV_FOLDS}  inner val AUC = {best_auc:.3f}")

    y_score = np.mean(probs_per_fold, axis=0)   # ensemble average
    y_pred  = (y_score >= 0.5).astype(int)
    return y_score, y_pred, {"ensemble_size": NUM_CV_FOLDS}


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc(curves, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_score, auc) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Test-set ROC curves — DASGIB ICP")
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC plot saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Held-out test evaluation for ICP detection — DASGIB"
    )
    p.add_argument("--dev-csv",  type=str, default=DEFAULT_DEV_CSV_PATH)
    p.add_argument("--test-csv", type=str, default=DEFAULT_TEST_CSV_PATH)
    p.add_argument("--features-dir", type=str, default=DEFAULT_FEATURES_DIR)
    p.add_argument("--results-dir",  type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--pooling", type=str, default="mean",
                   choices=["mean", "max", "mean_max"])
    p.add_argument("--skip-mil", action="store_true",
                   help="Skip the (slow) MIL ensemble — only sklearn models.")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load labels ---
    print(f"\nDev split:")
    dev_ids,  _, dev_labels  = get_aligned_arrays(args.dev_csv)
    print(f"\nTest split:")
    test_ids, _, test_labels = get_aligned_arrays(args.test_csv)

    # --- Load pooled features for sklearn models ---
    print(f"\nLoading pooled features (pooling={args.pooling})...")
    X_dev,  m_dev  = load_pooled_features(args.features_dir, dev_ids,  args.pooling)
    X_test, m_test = load_pooled_features(args.features_dir, test_ids, args.pooling)
    dev_ids,  dev_labels  = dev_ids[m_dev],   dev_labels[m_dev]
    test_ids, test_labels = test_ids[m_test], test_labels[m_test]
    print(f"  Dev : {X_dev.shape}   ({dev_labels.sum()} elevated / {(dev_labels==0).sum()} normal)")
    print(f"  Test: {X_test.shape}  ({test_labels.sum()} elevated / {(test_labels==0).sum()} normal)")

    scaler = StandardScaler().fit(X_dev)
    X_dev_s  = scaler.transform(X_dev)
    X_test_s = scaler.transform(X_test)

    results = {}
    roc_curves = {}

    # --- 1. Linear Regression ---
    print("\n[1/4] Linear Regression ...")
    y_score, y_pred, info = fit_eval_linreg(X_dev_s, dev_labels, X_test_s, test_labels)
    pm = point_metrics(test_labels, y_score, y_pred)
    bm = bootstrap_metrics(test_labels, y_score, y_pred, n_boot=args.n_boot)
    results["linear_regression"] = {"point": pm, "bootstrap_95ci": bm, "params": info}
    roc_curves["Linear Regression"] = (test_labels, y_score, pm["auc"])
    print(f"   AUC = {pm['auc']:.3f}")

    # --- 2. Logistic Regression ---
    print("\n[2/4] Logistic Regression ...")
    y_score, y_pred, info = fit_eval_logreg(X_dev_s, dev_labels, X_test_s, test_labels)
    pm = point_metrics(test_labels, y_score, y_pred)
    bm = bootstrap_metrics(test_labels, y_score, y_pred, n_boot=args.n_boot)
    results["logistic_regression"] = {"point": pm, "bootstrap_95ci": bm, "params": info}
    roc_curves["Logistic Regression"] = (test_labels, y_score, pm["auc"])
    print(f"   AUC = {pm['auc']:.3f}   best C = {info['best_C']}")

    # --- 3. Linear SVM ---
    print("\n[3/4] Linear SVM ...")
    y_score, y_pred, info = fit_eval_svm(X_dev_s, dev_labels, X_test_s, test_labels)
    pm = point_metrics(test_labels, y_score, y_pred)
    bm = bootstrap_metrics(test_labels, y_score, y_pred, n_boot=args.n_boot)
    results["linear_svm"] = {"point": pm, "bootstrap_95ci": bm, "params": info}
    roc_curves["Linear SVM"] = (test_labels, y_score, pm["auc"])
    print(f"   AUC = {pm['auc']:.3f}   best C = {info['best_C']}")

    # --- 4. Gated-Attention MIL ---
    if not args.skip_mil:
        print("\n[4/4] Gated-Attention MIL (5-fold dev ensemble) ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dev_bags  = load_all_bags(args.features_dir, dev_ids)
        test_bags = load_all_bags(args.features_dir, test_ids)
        m_dev_b   = np.array([rid in dev_bags  for rid in dev_ids])
        m_test_b  = np.array([rid in test_bags for rid in test_ids])
        dev_ids_b,  dev_labels_b  = dev_ids[m_dev_b],   dev_labels[m_dev_b]
        test_ids_b, test_labels_b = test_ids[m_test_b], test_labels[m_test_b]

        y_score, y_pred, info = fit_eval_mil(
            dev_ids_b, dev_labels_b, dev_bags,
            test_ids_b, test_labels_b, test_bags, device,
        )
        pm = point_metrics(test_labels_b, y_score, y_pred)
        bm = bootstrap_metrics(test_labels_b, y_score, y_pred, n_boot=args.n_boot)
        results["mil_gated_attention"] = {
            "point": pm, "bootstrap_95ci": bm, "params": info
        }
        roc_curves["Gated-Attention MIL"] = (test_labels_b, y_score, pm["auc"])
        print(f"   AUC = {pm['auc']:.3f}")
    else:
        print("\n[4/4] MIL skipped (--skip-mil).")

    # --- Save ---
    plot_roc(roc_curves, results_dir / "test_roc_curves.png")

    output = {
        "dataset": "DASGIB",
        "pooling": args.pooling,
        "icp_threshold_mmhg": 15,
        "n_dev":  int(len(dev_labels)),
        "n_test": int(len(test_labels)),
        "test_elevated": int(test_labels.sum()),
        "test_normal":   int((test_labels == 0).sum()),
        "bootstrap_n":   args.n_boot,
        "models": results,
    }
    out_path = results_dir / "test_metrics_summary.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved: {out_path}")

    # --- Pretty print ---
    print(f"\n{'='*70}")
    print(f"  Held-out TEST results (Martin Zillmer baseline: AUC ~0.58)")
    print(f"{'='*70}")
    print(f"  {'Model':30s}  AUC    Sens   Spec   Acc    F1")
    for name, r in results.items():
        pm = r["point"]
        print(f"  {name:30s}  {pm['auc']:.3f}  {pm['sensitivity']:.3f}  "
              f"{pm['specificity']:.3f}  {pm['accuracy']:.3f}  {pm['f1']:.3f}")


if __name__ == "__main__":
    main()
