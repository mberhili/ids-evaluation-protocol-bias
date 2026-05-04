"""
Methodological Critique Experiments.
A1: Biased validation_split vs proper evaluation (3 datasets)
A2: Seed sensitivity analysis (100 seeds, 3 datasets)
A3: Data leakage in preprocessing
"""

import numpy as np
import pandas as pd
import time
import os
import argparse
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from preprocessing import load_dataset, apply_smote
from models import create_model
from evaluation import compute_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# A1: Biased validation_split vs Proper Evaluation
# ============================================================

def experiment_a1_bias(dataset_name, n_seeds=10):
    """Compare Keras-style validation_split (biased) vs proper stratified holdout.

    The bias: Keras validation_split takes the LAST portion of the training data,
    which is often not shuffled or not representative. Many papers report results
    on this biased validation set instead of a proper test set.
    """
    print(f"\n{'='*70}")
    print(f"A1: Biased vs Proper Evaluation — {dataset_name.upper()}")
    print(f"{'='*70}")

    data = load_dataset(dataset_name)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    results_biased = []
    results_proper = []

    for seed in range(n_seeds):
        # --- Biased approach: Keras-like validation_split ---
        # Take last 10% of training data as validation (no shuffle)
        n_val = int(0.1 * len(y_train))
        X_train_biased = X_train[:-n_val]
        y_train_biased = y_train[:-n_val]
        X_val_biased = X_train[-n_val:]
        y_val_biased = y_train[-n_val:]

        # Apply SMOTE on biased train
        X_tr_sm, y_tr_sm = apply_smote(X_train_biased, y_train_biased, random_state=seed)

        model_biased = XGBClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            random_state=seed, n_jobs=-1, eval_metric='logloss'
        )
        model_biased.fit(X_tr_sm, y_tr_sm)

        # Evaluate on biased validation set
        y_pred_biased = model_biased.predict(X_val_biased)
        acc_biased = accuracy_score(y_val_biased, y_pred_biased)
        f1_biased = f1_score(y_val_biased, y_pred_biased, zero_division=0)

        # Evaluate same model on proper test set
        y_pred_test = model_biased.predict(X_test)
        acc_test_from_biased = accuracy_score(y_test, y_pred_test)
        f1_test_from_biased = f1_score(y_test, y_pred_test, zero_division=0)

        results_biased.append({
            'seed': seed,
            'val_acc': acc_biased,
            'val_f1': f1_biased,
            'test_acc': acc_test_from_biased,
            'test_f1': f1_test_from_biased,
        })

        # --- Proper approach: Stratified split ---
        X_tr_proper, X_val_proper, y_tr_proper, y_val_proper = train_test_split(
            X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
        )
        X_tr_sm2, y_tr_sm2 = apply_smote(X_tr_proper, y_tr_proper, random_state=seed)

        model_proper = XGBClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            random_state=seed, n_jobs=-1, eval_metric='logloss'
        )
        model_proper.fit(X_tr_sm2, y_tr_sm2)

        y_pred_proper = model_proper.predict(X_val_proper)
        acc_proper = accuracy_score(y_val_proper, y_pred_proper)
        f1_proper = f1_score(y_val_proper, y_pred_proper, zero_division=0)

        y_pred_test2 = model_proper.predict(X_test)
        acc_test_proper = accuracy_score(y_test, y_pred_test2)
        f1_test_proper = f1_score(y_test, y_pred_test2, zero_division=0)

        results_proper.append({
            'seed': seed,
            'val_acc': acc_proper,
            'val_f1': f1_proper,
            'test_acc': acc_test_proper,
            'test_f1': f1_test_proper,
        })

        print(f"  Seed {seed:>3d}: Biased val={acc_biased:.4f} test={acc_test_from_biased:.4f} | "
              f"Proper val={acc_proper:.4f} test={acc_test_proper:.4f}")

    df_biased = pd.DataFrame(results_biased)
    df_proper = pd.DataFrame(results_proper)

    print(f"\n  Summary ({dataset_name}):")
    print(f"  {'Method':<20} {'Val Acc (mean±std)':>25} {'Test Acc (mean±std)':>25} {'Gap':>10}")
    print(f"  {'-'*80}")

    b_val = f"{df_biased['val_acc'].mean():.4f}±{df_biased['val_acc'].std():.4f}"
    b_test = f"{df_biased['test_acc'].mean():.4f}±{df_biased['test_acc'].std():.4f}"
    b_gap = df_biased['val_acc'].mean() - df_biased['test_acc'].mean()
    print(f"  {'Biased (val_split)':<20} {b_val:>25} {b_test:>25} {b_gap:>+10.4f}")

    p_val = f"{df_proper['val_acc'].mean():.4f}±{df_proper['val_acc'].std():.4f}"
    p_test = f"{df_proper['test_acc'].mean():.4f}±{df_proper['test_acc'].std():.4f}"
    p_gap = df_proper['val_acc'].mean() - df_proper['test_acc'].mean()
    print(f"  {'Proper (stratified)':<20} {p_val:>25} {p_test:>25} {p_gap:>+10.4f}")

    return {
        'dataset': dataset_name,
        'biased': df_biased,
        'proper': df_proper,
        'biased_val_mean': df_biased['val_acc'].mean(),
        'biased_test_mean': df_biased['test_acc'].mean(),
        'proper_val_mean': df_proper['val_acc'].mean(),
        'proper_test_mean': df_proper['test_acc'].mean(),
        'bias_gap': b_gap,
        'proper_gap': p_gap,
    }


# ============================================================
# A2: Seed Sensitivity Analysis
# ============================================================

def experiment_a2_seed_sensitivity(dataset_name, n_seeds=100, model_name='XGBoost'):
    """Show how random seed affects reported results.

    Tests n_seeds different random seeds for train/test splitting
    and reports the distribution of accuracy scores.
    """
    print(f"\n{'='*70}")
    print(f"A2: Seed Sensitivity — {dataset_name.upper()} ({model_name})")
    print(f"{'='*70}")

    # Check for existing UNSW-NB15 seed results
    existing_csv = os.path.join(BASE_DIR, 'experiment_seed_search_results.csv')
    if dataset_name.lower().replace('-', '_') in ['unsw_nb15', 'unsw-nb15'] and os.path.exists(existing_csv):
        print(f"  Loading existing seed results from {existing_csv}")
        df_existing = pd.read_csv(existing_csv)
        if len(df_existing) >= n_seeds:
            df = df_existing.head(n_seeds).copy()
            df.columns = ['seed', 'accuracy', 'threshold']
            print(f"  Loaded {len(df)} existing results")
            print(f"  Accuracy: mean={df['accuracy'].mean():.4f}, std={df['accuracy'].std():.4f}")
            print(f"  Range: [{df['accuracy'].min():.4f}, {df['accuracy'].max():.4f}]")
            print(f"  Best seed: {df.loc[df['accuracy'].idxmax(), 'seed']:.0f} "
                  f"({df['accuracy'].max():.4f})")
            return {
                'dataset': dataset_name,
                'model': model_name,
                'results': df,
                'mean': df['accuracy'].mean(),
                'std': df['accuracy'].std(),
                'min': df['accuracy'].min(),
                'max': df['accuracy'].max(),
                'range': df['accuracy'].max() - df['accuracy'].min(),
            }

    data = load_dataset(dataset_name)
    X_all = np.vstack([data['X_train'], data['X_test']])
    y_all = np.concatenate([data['y_train'], data['y_test']])

    results = []
    n_features = data['n_features']

    for seed in range(n_seeds):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all
        )
        X_tr_sm, y_tr_sm = apply_smote(X_tr, y_tr, random_state=42)

        model = create_model(model_name, n_features=n_features, random_state=42)
        model.fit(X_tr_sm, y_tr_sm)

        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, zero_division=0)

        results.append({'seed': seed, 'accuracy': acc, 'f1': f1})

        if (seed + 1) % 10 == 0:
            print(f"  Seed {seed+1}/{n_seeds}: acc={acc:.4f}")

    df = pd.DataFrame(results)

    print(f"\n  Seed Sensitivity Summary ({dataset_name}, {model_name}):")
    print(f"  Accuracy: mean={df['accuracy'].mean():.4f}, std={df['accuracy'].std():.4f}")
    print(f"  Range: [{df['accuracy'].min():.4f}, {df['accuracy'].max():.4f}]")
    print(f"  Spread: {(df['accuracy'].max() - df['accuracy'].min())*100:.2f}%")
    print(f"  Best seed: {df.loc[df['accuracy'].idxmax(), 'seed']} ({df['accuracy'].max():.4f})")
    print(f"  Worst seed: {df.loc[df['accuracy'].idxmin(), 'seed']} ({df['accuracy'].min():.4f})")

    return {
        'dataset': dataset_name,
        'model': model_name,
        'results': df,
        'mean': df['accuracy'].mean(),
        'std': df['accuracy'].std(),
        'min': df['accuracy'].min(),
        'max': df['accuracy'].max(),
        'range': df['accuracy'].max() - df['accuracy'].min(),
    }


# ============================================================
# A3: Data Leakage in Preprocessing
# ============================================================

def experiment_a3_data_leakage(dataset_name):
    """Demonstrate how preprocessing data leakage inflates results.

    Compares:
    - WRONG: fit scaler on ALL data, then split → leakage
    - CORRECT: split first, fit scaler on train only → no leakage
    """
    print(f"\n{'='*70}")
    print(f"A3: Data Leakage — {dataset_name.upper()}")
    print(f"{'='*70}")

    # Load raw data (before scaling)
    if dataset_name.lower().replace('-', '_') in ['unsw_nb15', 'unsw-nb15']:
        train_df = pd.read_csv(os.path.join(BASE_DIR, 'UNSW_NB15_training-set.csv'))
        test_df = pd.read_csv(os.path.join(BASE_DIR, 'UNSW_NB15_testing-set.csv'))
        drop_cols = ['id', 'attack_cat']
        train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
        test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

        y_train = train_df['label'].values
        y_test = test_df['label'].values
        X_train_raw = train_df.drop('label', axis=1).copy()
        X_test_raw = test_df.drop('label', axis=1).copy()

        # Encode categoricals
        for col in X_train_raw.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            combined = pd.concat([X_train_raw[col], X_test_raw[col]]).astype(str)
            le.fit(combined)
            X_train_raw[col] = le.transform(X_train_raw[col].astype(str))
            X_test_raw[col] = le.transform(X_test_raw[col].astype(str))

        for col in X_train_raw.select_dtypes(include=['int64', 'float64']).columns:
            X_train_raw[col] = X_train_raw[col].fillna(X_train_raw[col].median())
            X_test_raw[col] = X_test_raw[col].fillna(X_train_raw[col].median())

        X_train_raw = X_train_raw.replace([np.inf, -np.inf], 0)
        X_test_raw = X_test_raw.replace([np.inf, -np.inf], 0)
    else:
        # For other datasets, use preprocessed data
        data = load_dataset(dataset_name)
        print("  Note: Using pre-processed data (leakage effect may be smaller)")
        X_train_raw = pd.DataFrame(data['X_train'])
        X_test_raw = pd.DataFrame(data['X_test'])
        y_train = data['y_train']
        y_test = data['y_test']

    results_leaky = []
    results_correct = []

    for seed in range(10):
        # --- WRONG: Fit scaler on ALL data (leakage) ---
        X_all = pd.concat([X_train_raw, X_test_raw], ignore_index=True)
        scaler_leaky = MinMaxScaler()
        X_all_scaled = scaler_leaky.fit_transform(X_all)
        X_train_leaky = X_all_scaled[:len(y_train)]
        X_test_leaky = X_all_scaled[len(y_train):]

        X_tr_sm, y_tr_sm = apply_smote(X_train_leaky, y_train, random_state=seed)
        model_leaky = XGBClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            random_state=seed, n_jobs=-1, eval_metric='logloss'
        )
        model_leaky.fit(X_tr_sm, y_tr_sm)
        acc_leaky = accuracy_score(y_test, model_leaky.predict(X_test_leaky))

        # --- CORRECT: Fit scaler on train only ---
        scaler_correct = MinMaxScaler()
        X_train_correct = scaler_correct.fit_transform(X_train_raw)
        X_test_correct = scaler_correct.transform(X_test_raw)

        X_tr_sm2, y_tr_sm2 = apply_smote(X_train_correct, y_train, random_state=seed)
        model_correct = XGBClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            random_state=seed, n_jobs=-1, eval_metric='logloss'
        )
        model_correct.fit(X_tr_sm2, y_tr_sm2)
        acc_correct = accuracy_score(y_test, model_correct.predict(X_test_correct))

        results_leaky.append(acc_leaky)
        results_correct.append(acc_correct)

        print(f"  Seed {seed}: Leaky={acc_leaky:.4f}, Correct={acc_correct:.4f}, "
              f"Diff={acc_leaky - acc_correct:+.4f}")

    mean_leaky = np.mean(results_leaky)
    mean_correct = np.mean(results_correct)

    print(f"\n  Summary ({dataset_name}):")
    print(f"  Leaky:   {mean_leaky:.4f}±{np.std(results_leaky):.4f}")
    print(f"  Correct: {mean_correct:.4f}±{np.std(results_correct):.4f}")
    print(f"  Inflation: {(mean_leaky - mean_correct)*100:+.2f}%")

    return {
        'dataset': dataset_name,
        'leaky_mean': mean_leaky,
        'leaky_std': np.std(results_leaky),
        'correct_mean': mean_correct,
        'correct_std': np.std(results_correct),
        'inflation': mean_leaky - mean_correct,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bias Analysis Experiments')
    parser.add_argument('--experiments', nargs='+', default=['a1', 'a2', 'a3'],
                        choices=['a1', 'a2', 'a3'],
                        help='Experiments to run')
    parser.add_argument('--datasets', nargs='+', default=['unsw-nb15', 'nsl-kdd', 'cicids2017'])
    parser.add_argument('--n-seeds-a2', type=int, default=100,
                        help='Number of seeds for A2 experiment')
    args = parser.parse_args()

    all_a1, all_a2, all_a3 = [], [], []

    for ds_name in args.datasets:
        try:
            if 'a1' in args.experiments:
                result = experiment_a1_bias(ds_name)
                all_a1.append(result)

            if 'a2' in args.experiments:
                result = experiment_a2_seed_sensitivity(ds_name, n_seeds=args.n_seeds_a2)
                all_a2.append(result)

            if 'a3' in args.experiments:
                result = experiment_a3_data_leakage(ds_name)
                all_a3.append(result)
        except FileNotFoundError as e:
            print(f"\nSKIPPED {ds_name}: {e}")

    # Save A1 results
    if all_a1:
        rows = []
        for r in all_a1:
            rows.append({
                'dataset': r['dataset'],
                'biased_val_acc': r['biased_val_mean'],
                'biased_test_acc': r['biased_test_mean'],
                'proper_val_acc': r['proper_val_mean'],
                'proper_test_acc': r['proper_test_mean'],
                'bias_gap': r['bias_gap'],
                'proper_gap': r['proper_gap'],
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(RESULTS_DIR, 'a1_bias_results.csv'), index=False)
        print(f"\nA1 results saved.")

    # Save A2 results
    if all_a2:
        rows = []
        for r in all_a2:
            rows.append({
                'dataset': r['dataset'],
                'model': r['model'],
                'mean_acc': r['mean'],
                'std_acc': r['std'],
                'min_acc': r['min'],
                'max_acc': r['max'],
                'range_pct': r['range'] * 100,
            })
            # Save per-seed results
            r['results'].to_csv(
                os.path.join(RESULTS_DIR, f"a2_seed_sensitivity_{r['dataset']}.csv"),
                index=False
            )
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(RESULTS_DIR, 'a2_seed_summary.csv'), index=False)
        print(f"\nA2 results saved.")

    # Save A3 results
    if all_a3:
        rows = []
        for r in all_a3:
            rows.append({
                'dataset': r['dataset'],
                'leaky_acc': r['leaky_mean'],
                'correct_acc': r['correct_mean'],
                'inflation_pct': r['inflation'] * 100,
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(RESULTS_DIR, 'a3_leakage_results.csv'), index=False)
        print(f"\nA3 results saved.")

    print("\nAll bias analysis experiments complete.")
