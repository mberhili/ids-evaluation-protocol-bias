"""
CNN-BiLSTM hyperparameter tuning sweep for Reviewer #1, comment 3.

The reviewer asked whether CNN-BiLSTM's low weight (<5%) in AWE-IDS reflects
under-tuning or architectural unsuitability for tabular IDS data. This script
tests several CNN-BiLSTM configurations on UNSW-NB15 (the dataset where AWE-IDS
weights matter most) and reports the weight assigned to CNN-BiLSTM in each case.

Configurations swept:
  1. Default (baseline from paper)
  2. Lower learning rate (0.0001)
  3. Higher learning rate (0.005)
  4. Larger LSTM units (128)
  5. Higher dropout (0.5)
  6. More capacity (LSTM=128 + Conv filters=128/64)

For each config, we report AWE-IDS weights under both calibration modes.
"""

import os
import time
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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Bidirectional, LSTM,
    Dropout, Dense, BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


CNN_CONFIGS = [
    {"name": "default",      "lr": 0.001,  "lstm": 64,  "conv1": 64, "conv2": 32, "dropout": 0.3, "epochs": 30, "batch": 64},
    {"name": "lower_lr",     "lr": 0.0001, "lstm": 64,  "conv1": 64, "conv2": 32, "dropout": 0.3, "epochs": 50, "batch": 64},
    {"name": "higher_lr",    "lr": 0.005,  "lstm": 64,  "conv1": 64, "conv2": 32, "dropout": 0.3, "epochs": 30, "batch": 64},
    {"name": "larger_lstm",  "lr": 0.001,  "lstm": 128, "conv1": 64, "conv2": 32, "dropout": 0.3, "epochs": 30, "batch": 64},
    {"name": "higher_drop",  "lr": 0.001,  "lstm": 64,  "conv1": 64, "conv2": 32, "dropout": 0.5, "epochs": 30, "batch": 64},
    {"name": "more_capacity","lr": 0.001,  "lstm": 128, "conv1": 128,"conv2": 64, "dropout": 0.3, "epochs": 30, "batch": 128},
]


def build_cnn_bilstm(n_features, cfg, random_state=42):
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    model = Sequential([
        Conv1D(cfg["conv1"], 3, activation="relu", padding="same",
               input_shape=(n_features, 1)),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(cfg["conv2"], 3, activation="relu", padding="same"),
        MaxPooling1D(2),
        Bidirectional(LSTM(cfg["lstm"], return_sequences=False)),
        Dropout(cfg["dropout"]),
        Dense(64, activation="relu"),
        Dropout(cfg["dropout"] * 2 / 3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(learning_rate=cfg["lr"]),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def search_3way(p_xgb, p_lgb, p_cnn, y_target, weight_step=0.05,
                threshold_step=0.005, threshold_lo=0.30, threshold_hi=0.75):
    """Search over (w_xgb, w_lgb, w_cnn) on simplex with threshold."""
    best = {"accuracy": -1.0, "f1": 0.0, "w_xgb": None, "w_lgb": None,
            "w_cnn": None, "threshold": None}
    weights = np.arange(0, 1.0 + weight_step / 2, weight_step)
    thresholds = np.arange(threshold_lo, threshold_hi + threshold_step / 2,
                           threshold_step)
    for w0 in weights:
        for w1 in weights:
            if w0 + w1 > 1.0 + 1e-6:
                continue
            w2 = 1.0 - w0 - w1
            if w2 < -1e-6:
                continue
            w2 = max(0.0, w2)
            hybrid = w0 * p_xgb + w1 * p_lgb + w2 * p_cnn
            for t in thresholds:
                y_pred = (hybrid > t).astype(int)
                acc = accuracy_score(y_target, y_pred)
                if acc > best["accuracy"]:
                    best = {
                        "accuracy": acc,
                        "f1": f1_score(y_target, y_pred, zero_division=0),
                        "w_xgb": round(float(w0), 3),
                        "w_lgb": round(float(w1), 3),
                        "w_cnn": round(float(w2), 3),
                        "threshold": round(float(t), 4),
                    }
    return best


def run_one_config(cfg, X_tr_sm, y_tr_sm, X_calib, y_calib,
                   X_test, y_test, p_xgb_calib, p_lgb_calib,
                   p_xgb_test, p_lgb_test, n_features, random_state=42):
    print(f"\n--- Config: {cfg['name']} ---")
    print(f"    lr={cfg['lr']}, lstm={cfg['lstm']}, conv={cfg['conv1']}/{cfg['conv2']}, "
          f"dropout={cfg['dropout']}, epochs={cfg['epochs']}, batch={cfg['batch']}")

    t0 = time.time()
    cnn = build_cnn_bilstm(n_features, cfg, random_state=random_state)
    X_tr_3d = X_tr_sm.reshape(-1, n_features, 1)
    X_calib_3d = X_calib.reshape(-1, n_features, 1)
    X_test_3d = X_test.reshape(-1, n_features, 1)

    early = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    cnn.fit(X_tr_3d, y_tr_sm, epochs=cfg["epochs"],
            batch_size=cfg["batch"], verbose=0, callbacks=[early])
    train_t = time.time() - t0

    p_cnn_calib = cnn.predict(X_calib_3d, batch_size=512, verbose=0).ravel()
    p_cnn_test = cnn.predict(X_test_3d, batch_size=512, verbose=0).ravel()

    cnn_acc_test = accuracy_score(y_test, (p_cnn_test > 0.5).astype(int))

    # AWE-IDS deployable: optimize on calib, apply to test
    deploy_cal = search_3way(p_xgb_calib, p_lgb_calib, p_cnn_calib, y_calib)
    hybrid_test = (deploy_cal["w_xgb"] * p_xgb_test
                   + deploy_cal["w_lgb"] * p_lgb_test
                   + deploy_cal["w_cnn"] * p_cnn_test)
    y_pred_deploy = (hybrid_test > deploy_cal["threshold"]).astype(int)
    deploy_test_acc = accuracy_score(y_test, y_pred_deploy)

    # AWE-IDS oracle: optimize on test
    oracle_cal = search_3way(p_xgb_test, p_lgb_test, p_cnn_test, y_test)

    print(f"    CNN-BiLSTM standalone test acc: {cnn_acc_test:.4f}  (train: {train_t:.1f}s)")
    print(f"    AWE deployable: acc={deploy_test_acc:.4f}, "
          f"weights=(XGB={deploy_cal['w_xgb']}, LGBM={deploy_cal['w_lgb']}, "
          f"CNN={deploy_cal['w_cnn']})")
    print(f"    AWE oracle:     acc={oracle_cal['accuracy']:.4f}, "
          f"weights=(XGB={oracle_cal['w_xgb']}, LGBM={oracle_cal['w_lgb']}, "
          f"CNN={oracle_cal['w_cnn']})")

    return {
        "config": cfg["name"],
        "lr": cfg["lr"],
        "lstm": cfg["lstm"],
        "conv1": cfg["conv1"],
        "conv2": cfg["conv2"],
        "dropout": cfg["dropout"],
        "epochs": cfg["epochs"],
        "batch": cfg["batch"],
        "cnn_standalone_acc": cnn_acc_test,
        "cnn_train_time_sec": train_t,
        "deploy_acc": deploy_test_acc,
        "deploy_w_xgb": deploy_cal["w_xgb"],
        "deploy_w_lgb": deploy_cal["w_lgb"],
        "deploy_w_cnn": deploy_cal["w_cnn"],
        "deploy_threshold": deploy_cal["threshold"],
        "oracle_acc": oracle_cal["accuracy"],
        "oracle_w_xgb": oracle_cal["w_xgb"],
        "oracle_w_lgb": oracle_cal["w_lgb"],
        "oracle_w_cnn": oracle_cal["w_cnn"],
        "oracle_threshold": oracle_cal["threshold"],
    }


def main(dataset_name="unsw-nb15", random_state=42):
    print(f"\n{'='*70}\nDATASET: {dataset_name}\n{'='*70}")
    d = load_dataset(dataset_name)
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]
    n_features = d["n_features"]
    print(f"  features={n_features}, train={len(y_train)}, test={len(y_test)}")

    X_tr, X_calib, y_tr, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train,
    )
    sm = SMOTE(random_state=random_state)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)

    # Train tree-based once (deterministic given seed)
    print("\n--- Training tree-based components ---")
    xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                random_state=random_state, n_jobs=-1,
                                eval_metric="logloss", verbosity=0)
    t0 = time.time(); xgb_clf.fit(X_tr_sm, y_tr_sm); print(f"  XGB:  {time.time()-t0:.1f}s")
    lgb_clf = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 random_state=random_state, n_jobs=-1, verbosity=-1)
    t0 = time.time(); lgb_clf.fit(X_tr_sm, y_tr_sm); print(f"  LGBM: {time.time()-t0:.1f}s")

    p_xgb_calib = xgb_clf.predict_proba(X_calib)[:, 1]
    p_lgb_calib = lgb_clf.predict_proba(X_calib)[:, 1]
    p_xgb_test = xgb_clf.predict_proba(X_test)[:, 1]
    p_lgb_test = lgb_clf.predict_proba(X_test)[:, 1]

    rows = []
    for cfg in CNN_CONFIGS:
        try:
            row = run_one_config(cfg, X_tr_sm, y_tr_sm, X_calib, y_calib,
                                 X_test, y_test, p_xgb_calib, p_lgb_calib,
                                 p_xgb_test, p_lgb_test, n_features,
                                 random_state=random_state)
            row["dataset"] = dataset_name
            rows.append(row)
        except Exception as e:
            print(f"    ERROR on config {cfg['name']}: {e}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, f"cnn_bilstm_tuning_{dataset_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")

    print(f"\nCNN-BiLSTM weight share across configurations:")
    print(df[["config", "deploy_w_cnn", "oracle_w_cnn",
              "cnn_standalone_acc"]].to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="unsw-nb15")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.dataset, random_state=args.seed)
