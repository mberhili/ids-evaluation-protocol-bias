"""
AWE-IDS grid sensitivity analysis for Reviewer #1, comment 4.

Trains XGBoost and LightGBM on each dataset (in-domain, full features),
then runs AWE-IDS weight + threshold grid search at three resolutions:
  - coarse: weight step 0.10, threshold step 0.01
  - used:   weight step 0.05, threshold step 0.005  (paper baseline)
  - fine:   weight step 0.01, threshold step 0.001

The grid search operates on precomputed probabilities, so the cost is
dominated by training the two component models, not by the grid.
"""

import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

from preprocessing import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GRIDS = {
    "coarse": {"weight_step": 0.10, "threshold_step": 0.010},
    "used":   {"weight_step": 0.05, "threshold_step": 0.005},
    "fine":   {"weight_step": 0.01, "threshold_step": 0.001},
}

THRESHOLD_LO = 0.30
THRESHOLD_HI = 0.75


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


def grid_search(p0, p1, y_target, weight_step, threshold_step):
    """Exhaustive grid search over (weight, threshold), return best (acc, w, t)."""
    best = {"accuracy": -1.0, "f1": 0.0, "w_xgb": None, "threshold": None}
    weights = np.arange(0, 1.0 + weight_step / 2, weight_step)
    thresholds = np.arange(THRESHOLD_LO, THRESHOLD_HI + threshold_step / 2, threshold_step)
    for w0 in weights:
        w1 = 1.0 - w0
        hybrid = w0 * p0 + w1 * p1
        for t in thresholds:
            y_pred = (hybrid > t).astype(int)
            acc = accuracy_score(y_target, y_pred)
            if acc > best["accuracy"]:
                best = {
                    "accuracy": acc,
                    "f1": f1_score(y_target, y_pred, zero_division=0),
                    "w_xgb": round(float(w0), 3),
                    "threshold": round(float(t), 4),
                }
    n_combos = len(weights) * len(thresholds)
    return best, n_combos


def run_one_dataset(name, random_state=42):
    print(f"\n{'='*70}\nDATASET: {name}\n{'='*70}")
    d = load_dataset(name)
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]

    # Hold out 20% of train for deployable calibration
    X_tr, X_calib, y_tr, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train,
    )
    sm = SMOTE(random_state=random_state)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)

    models = make_models(random_state)
    fitted = {}
    for mname, model in models.items():
        t0 = time.time()
        model.fit(X_tr_sm, y_tr_sm)
        print(f"  trained {mname} in {time.time()-t0:.1f}s")
        fitted[mname] = model

    p_xgb_calib = fitted["XGBoost"].predict_proba(X_calib)[:, 1]
    p_lgb_calib = fitted["LightGBM"].predict_proba(X_calib)[:, 1]
    p_xgb_test = fitted["XGBoost"].predict_proba(X_test)[:, 1]
    p_lgb_test = fitted["LightGBM"].predict_proba(X_test)[:, 1]

    rows = []
    for grid_name, params in GRIDS.items():
        # Deployable: optimize on calib set
        deploy_best, n_combos = grid_search(
            p_xgb_calib, p_lgb_calib, y_calib,
            params["weight_step"], params["threshold_step"])
        # Apply deployable calibration to test probas
        hybrid_test = deploy_best["w_xgb"] * p_xgb_test + (1 - deploy_best["w_xgb"]) * p_lgb_test
        y_pred_deploy = (hybrid_test > deploy_best["threshold"]).astype(int)
        deploy_test_acc = accuracy_score(y_test, y_pred_deploy)
        deploy_test_f1 = f1_score(y_test, y_pred_deploy, zero_division=0)

        # Oracle: optimize directly on test set
        oracle_best, _ = grid_search(
            p_xgb_test, p_lgb_test, y_test,
            params["weight_step"], params["threshold_step"])

        rows.append({
            "dataset": name,
            "grid": grid_name,
            "weight_step": params["weight_step"],
            "threshold_step": params["threshold_step"],
            "n_combinations": n_combos,
            "deploy_test_acc": deploy_test_acc,
            "deploy_test_f1": deploy_test_f1,
            "deploy_w_xgb": deploy_best["w_xgb"],
            "deploy_threshold": deploy_best["threshold"],
            "oracle_test_acc": oracle_best["accuracy"],
            "oracle_test_f1": oracle_best["f1"],
            "oracle_w_xgb": oracle_best["w_xgb"],
            "oracle_threshold": oracle_best["threshold"],
        })
        print(f"  {grid_name:6s} ({n_combos:5d} combos): "
              f"deploy={deploy_test_acc:.4f}, oracle={oracle_best['accuracy']:.4f}, "
              f"gap={oracle_best['accuracy']-deploy_test_acc:+.4f}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["unsw-nb15", "nsl-kdd", "cicids2017"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_rows = []
    for name in args.datasets:
        try:
            all_rows.extend(run_one_dataset(name, random_state=args.seed))
        except Exception as e:
            print(f"\nERROR on {name}: {e}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "grid_sensitivity.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")
    print(f"\nSummary table:")
    cols = ["dataset", "grid", "n_combinations",
            "deploy_test_acc", "oracle_test_acc"]
    print(df[cols].to_string(index=False))

    # Stability per dataset: range of deploy_test_acc across grids
    print(f"\nGrid-induced variation (max - min across grids):")
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        deploy_range = sub["deploy_test_acc"].max() - sub["deploy_test_acc"].min()
        oracle_range = sub["oracle_test_acc"].max() - sub["oracle_test_acc"].min()
        print(f"  {ds:15s}: deploy range={deploy_range:.4f}, oracle range={oracle_range:.4f}")
