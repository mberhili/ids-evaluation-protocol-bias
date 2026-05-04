"""
Cross-dataset experiments with semantically aligned features (Pivot B design).

Response to Reviewer #2 M5: replaces index-based truncation with semantic
feature alignment using UNSW-NB15 as pivot between paradigms.

For each pair:
  1. In-domain UNSW (aligned features only) - isolates truncation effect
  2. In-domain Other (aligned features only) - isolates truncation effect
  3. Cross-dataset UNSW -> Other (aligned)
  4. Cross-dataset Other -> UNSW (aligned)
  5. AWE-IDS weight + threshold optimization on cross-dataset directions
"""

import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

from feature_alignment import load_aligned_pair, UNSW_CIC_MAPPING, UNSW_NSL_MAPPING

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_models(random_state=42):
    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=random_state, n_jobs=-1, eval_metric="logloss",
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=random_state, n_jobs=-1, verbosity=-1,
        ),
    }


def metrics(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            out["auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            out["auc"] = float("nan")
    return out


def fit_eval(X_train, y_train, X_test, y_test, apply_smote=True, random_state=42):
    """Fit XGB + LGBM, return per-model metrics and probas on X_test."""
    if apply_smote:
        sm = SMOTE(random_state=random_state)
        X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    else:
        X_train_sm, y_train_sm = X_train, y_train

    results = {}
    probas = {}
    fitted_models = {}
    for name, model in make_models(random_state).items():
        t0 = time.time()
        model.fit(X_train_sm, y_train_sm)
        train_t = time.time() - t0
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        m = metrics(y_test, y_pred, y_proba)
        m["train_time"] = train_t
        results[name] = m
        probas[name] = y_proba
        fitted_models[name] = model
    return results, probas, fitted_models


def predict_probas(models, X):
    """Get probas[:,1] for each fitted model on X."""
    return {name: m.predict_proba(X)[:, 1] for name, m in models.items()}


def apply_awe_calibration(probas_dict, calibration, y_target):
    """Apply a (weights, threshold) calibration to compute target accuracy."""
    names = sorted(probas_dict.keys())
    p0 = probas_dict[names[0]]
    p1 = probas_dict[names[1]]
    w = calibration["weights"]
    hybrid = w[names[0]] * p0 + w[names[1]] * p1
    y_pred = (hybrid > calibration["threshold"]).astype(int)
    return {
        "accuracy": accuracy_score(y_target, y_pred),
        "f1": f1_score(y_target, y_pred, zero_division=0),
    }


def awe_ids_search(probas_dict, y_target, weight_step=0.05,
                   threshold_lo=0.30, threshold_hi=0.75, threshold_step=0.005):
    """Grid search over (weight, threshold) for ensemble probability fusion."""
    names = sorted(probas_dict.keys())
    if len(names) != 2:
        return None
    p0 = probas_dict[names[0]]
    p1 = probas_dict[names[1]]

    best = {"accuracy": -1.0, "f1": 0.0, "weights": None, "threshold": 0.5}
    for w0 in np.arange(0, 1.0 + weight_step / 2, weight_step):
        w1 = 1.0 - w0
        hybrid = w0 * p0 + w1 * p1
        for t in np.arange(threshold_lo, threshold_hi + threshold_step / 2, threshold_step):
            y_pred = (hybrid > t).astype(int)
            acc = accuracy_score(y_target, y_pred)
            if acc > best["accuracy"]:
                f1 = f1_score(y_target, y_pred, zero_division=0)
                best = {
                    "accuracy": acc, "f1": f1,
                    "weights": {names[0]: round(float(w0), 3), names[1]: round(float(w1), 3)},
                    "threshold": round(float(t), 4),
                }
    return best


def run_cross_direction(label, source_X_train, source_y_train,
                        target_X_test, target_y_test, n_feat, pair,
                        source_name, target_name, random_state=42):
    """Cross-dataset evaluation with both deployable and oracle AWE-IDS calibration.

    Deployable: split source_train 80/20, calibrate weights+threshold on the held-out
    source_calib portion (no target labels seen).
    Oracle: calibrate directly on target_test (upper bound, not deployable).
    """
    print(f"\n{label}")
    rows = []

    # Hold out 20% of source train for AWE-IDS validation calibration
    src_tr, src_calib, y_tr, y_calib = train_test_split(
        source_X_train, source_y_train,
        test_size=0.2, random_state=random_state, stratify=source_y_train,
    )

    # Train on 80% of source, evaluate on target test
    res_target, probas_target, fitted = fit_eval(
        src_tr, y_tr, target_X_test, target_y_test, random_state=random_state)
    for m, met in res_target.items():
        rows.append({"pair": pair, "experiment": label,
                     "source": source_name, "target": target_name,
                     "model": m, "calibration": "n/a", "n_features": n_feat, **met})
        print(f"  {m}: acc={met['accuracy']:.4f}, f1={met['f1']:.4f}")

    # Get probas on the source calibration set (no target labels involved)
    probas_calib = predict_probas(fitted, src_calib)

    # Deployable AWE-IDS: optimize on source_calib, apply to target_test
    deploy_cal = awe_ids_search(probas_calib, y_calib)
    if deploy_cal:
        deploy_eval = apply_awe_calibration(probas_target, deploy_cal, target_y_test)
        rows.append({"pair": pair, "experiment": label,
                     "source": source_name, "target": target_name,
                     "model": "AWE-IDS", "calibration": "deployable",
                     "n_features": n_feat,
                     "accuracy": deploy_eval["accuracy"], "f1": deploy_eval["f1"],
                     "auc": float("nan"), "train_time": 0.0,
                     "weights": str(deploy_cal["weights"]),
                     "threshold": deploy_cal["threshold"]})
        print(f"  AWE-IDS (deployable, calib on source): "
              f"acc={deploy_eval['accuracy']:.4f}, weights={deploy_cal['weights']}, "
              f"t={deploy_cal['threshold']}")

    # Oracle AWE-IDS: optimize directly on target_test (upper bound)
    oracle_cal = awe_ids_search(probas_target, target_y_test)
    if oracle_cal:
        rows.append({"pair": pair, "experiment": label,
                     "source": source_name, "target": target_name,
                     "model": "AWE-IDS", "calibration": "oracle",
                     "n_features": n_feat,
                     "accuracy": oracle_cal["accuracy"], "f1": oracle_cal["f1"],
                     "auc": float("nan"), "train_time": 0.0,
                     "weights": str(oracle_cal["weights"]),
                     "threshold": oracle_cal["threshold"]})
        print(f"  AWE-IDS (oracle, calib on target):    "
              f"acc={oracle_cal['accuracy']:.4f}, weights={oracle_cal['weights']}, "
              f"t={oracle_cal['threshold']}")
        if deploy_cal:
            gap = oracle_cal["accuracy"] - deploy_eval["accuracy"]
            print(f"  Calibration gap (oracle - deployable): {gap:+.4f}")

    return rows


def run_indomain(label, X_train, y_train, X_test, y_test, n_feat, pair,
                 dataset_name, random_state=42):
    """In-domain evaluation with both deployable and oracle AWE-IDS calibration.

    Deployable: hold out 20% of train for calibration; evaluate on official test.
    Oracle: calibrate on test set directly.
    """
    print(f"\n{label} on {dataset_name} ({n_feat} features)")
    rows = []

    src_tr, src_calib, y_tr, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train,
    )

    res, probas_test, fitted = fit_eval(src_tr, y_tr, X_test, y_test,
                                        random_state=random_state)
    for m, met in res.items():
        rows.append({"pair": pair, "experiment": label,
                     "source": dataset_name, "target": dataset_name,
                     "model": m, "calibration": "n/a", "n_features": n_feat, **met})
        print(f"  {m}: acc={met['accuracy']:.4f}, f1={met['f1']:.4f}")

    probas_calib = predict_probas(fitted, src_calib)

    deploy_cal = awe_ids_search(probas_calib, y_calib)
    if deploy_cal:
        deploy_eval = apply_awe_calibration(probas_test, deploy_cal, y_test)
        rows.append({"pair": pair, "experiment": label,
                     "source": dataset_name, "target": dataset_name,
                     "model": "AWE-IDS", "calibration": "deployable",
                     "n_features": n_feat,
                     "accuracy": deploy_eval["accuracy"], "f1": deploy_eval["f1"],
                     "auc": float("nan"), "train_time": 0.0,
                     "weights": str(deploy_cal["weights"]),
                     "threshold": deploy_cal["threshold"]})
        print(f"  AWE-IDS (deployable): acc={deploy_eval['accuracy']:.4f}, "
              f"weights={deploy_cal['weights']}, t={deploy_cal['threshold']}")

    oracle_cal = awe_ids_search(probas_test, y_test)
    if oracle_cal:
        rows.append({"pair": pair, "experiment": label,
                     "source": dataset_name, "target": dataset_name,
                     "model": "AWE-IDS", "calibration": "oracle",
                     "n_features": n_feat,
                     "accuracy": oracle_cal["accuracy"], "f1": oracle_cal["f1"],
                     "auc": float("nan"), "train_time": 0.0,
                     "weights": str(oracle_cal["weights"]),
                     "threshold": oracle_cal["threshold"]})
        print(f"  AWE-IDS (oracle):     acc={oracle_cal['accuracy']:.4f}, "
              f"weights={oracle_cal['weights']}, t={oracle_cal['threshold']}")
        if deploy_cal:
            gap = oracle_cal["accuracy"] - deploy_eval["accuracy"]
            print(f"  Calibration gap (oracle - deployable): {gap:+.4f}")

    return rows


def run_pair(pair, random_state=42):
    """Run all experiments for one pair (unsw_nsl or unsw_cic)."""
    print(f"\n{'='*78}\nPAIR: {pair}\n{'='*78}")
    d = load_aligned_pair(pair, random_state=random_state)
    other = d["other_name"]
    n_feat = len(d["feature_names"])
    print(f"Features (aligned): {n_feat}")
    print(f"UNSW: train={d['unsw_X_train'].shape}, test={d['unsw_X_test'].shape}")
    print(f"{other}: train={d['other_X_train'].shape}, test={d['other_X_test'].shape}")

    rows = []
    rows += run_indomain("indomain_unsw_aligned",
                         d["unsw_X_train"], d["unsw_y_train"],
                         d["unsw_X_test"], d["unsw_y_test"],
                         n_feat, pair, "UNSW-NB15", random_state=random_state)
    rows += run_indomain("indomain_other_aligned",
                         d["other_X_train"], d["other_y_train"],
                         d["other_X_test"], d["other_y_test"],
                         n_feat, pair, other, random_state=random_state)
    rows += run_cross_direction("cross_unsw_to_other",
                                d["unsw_X_train"], d["unsw_y_train"],
                                d["other_X_test"], d["other_y_test"],
                                n_feat, pair, "UNSW-NB15", other,
                                random_state=random_state)
    rows += run_cross_direction("cross_other_to_unsw",
                                d["other_X_train"], d["other_y_train"],
                                d["unsw_X_test"], d["unsw_y_test"],
                                n_feat, pair, other, "UNSW-NB15",
                                random_state=random_state)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=["unsw_nsl", "unsw_cic", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = ["unsw_nsl", "unsw_cic"] if args.pair == "both" else [args.pair]

    all_rows = []
    for pair in pairs:
        all_rows.extend(run_pair(pair, random_state=args.seed))

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "cross_dataset_aligned_dual.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")
    print(f"\nSummary:")
    cols = ["pair", "experiment", "source", "target", "model", "calibration", "accuracy", "f1"]
    print(df[cols].to_string(index=False))
