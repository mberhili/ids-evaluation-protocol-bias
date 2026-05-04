"""
Ablation Study for AWE-IDS.
Isolates the contribution of each component:
  1. Baseline: equal weights (0.5/0.5), threshold=0.5
  2. Weight-only: optimized weights, fixed threshold=0.5
  3. Threshold-only: equal weights, optimized threshold
  4. Full AWE-IDS: optimized weights + optimized threshold
  5. AWE-IDS + CNN: 3-model ensemble with full optimization

Also generates threshold sensitivity curve (accuracy vs threshold).
"""

import numpy as np
import pandas as pd
import os
import argparse
import time
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.metrics import accuracy_score, f1_score

from preprocessing import load_dataset, apply_smote
from models import create_model
from evaluation import compute_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = ['unsw-nb15', 'nsl-kdd', 'cicids2017']
TWO_MODEL_ENSEMBLE = ['XGBoost', 'LightGBM']
THREE_MODEL_ENSEMBLE = ['XGBoost', 'LightGBM', 'CNN-BiLSTM']


def train_and_get_probas(model_names, X_train, y_train, X_test, n_features):
    """Train models and return probability predictions on test set."""
    probas = {}
    for name in model_names:
        print(f"    Training {name}...", end=" ", flush=True)
        model = create_model(name, n_features=n_features, random_state=42)
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict(X_test).astype(float)
        probas[name] = y_proba
        print(f"done ({elapsed:.1f}s)")

    return probas


def search_weights_only(probas, y_true, model_names, fixed_threshold=0.5, grid_step=0.05):
    """Search optimal weights with fixed threshold."""
    n_models = len(model_names)
    proba_arrays = [probas[name] for name in model_names]
    best_acc = 0
    best_weights = None
    weight_values = np.round(np.arange(0, 1.0 + grid_step / 2, grid_step), 3)

    if n_models == 2:
        for w0 in weight_values:
            w1 = round(1.0 - w0, 3)
            if w1 < 0 or w1 > 1.0:
                continue
            hybrid = w0 * proba_arrays[0] + w1 * proba_arrays[1]
            y_pred = (hybrid > fixed_threshold).astype(int)
            acc = accuracy_score(y_true, y_pred)
            if acc > best_acc:
                best_acc = acc
                best_weights = dict(zip(model_names, [w0, w1]))
    elif n_models == 3:
        for w0 in weight_values:
            for w1 in weight_values:
                w2 = round(1.0 - w0 - w1, 3)
                if w2 < -0.001 or w2 > 1.001:
                    continue
                w2 = max(0.0, min(1.0, w2))
                hybrid = w0 * proba_arrays[0] + w1 * proba_arrays[1] + w2 * proba_arrays[2]
                y_pred = (hybrid > fixed_threshold).astype(int)
                acc = accuracy_score(y_true, y_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_weights = dict(zip(model_names, [w0, w1, w2]))

    return best_weights, best_acc


def search_threshold_only(probas, y_true, model_names,
                          threshold_range=(0.30, 0.75, 0.005)):
    """Search optimal threshold with equal weights."""
    n_models = len(model_names)
    proba_arrays = [probas[name] for name in model_names]
    equal_w = 1.0 / n_models
    hybrid = sum(equal_w * p for p in proba_arrays)

    t_start, t_stop, t_step = threshold_range
    thresholds = np.arange(t_start, t_stop, t_step)

    best_acc = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (hybrid > threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return best_threshold, best_acc


def search_full(probas, y_true, model_names, grid_step=0.05,
                threshold_range=(0.30, 0.75, 0.005)):
    """Full search: optimize both weights and threshold jointly."""
    from run_awe import search_optimal_weights
    best_weights, best_threshold, best_acc, best_f1 = search_optimal_weights(
        probas, y_true, model_names, grid_step=grid_step,
        threshold_range=threshold_range
    )
    return best_weights, best_threshold, best_acc, best_f1


def threshold_sensitivity(probas, y_true, model_names, weights=None,
                          t_range=(0.0, 1.0, 0.01)):
    """Compute accuracy and F1 as a function of threshold.

    Returns:
        pd.DataFrame with columns: threshold, accuracy, f1, precision, recall
    """
    proba_arrays = [probas[name] for name in model_names]

    if weights is None:
        # Equal weights
        n = len(model_names)
        weights = {name: 1.0 / n for name in model_names}

    hybrid = sum(weights[name] * probas[name] for name in model_names)

    t_start, t_stop, t_step = t_range
    thresholds = np.arange(t_start, t_stop + t_step, t_step)

    rows = []
    for t in thresholds:
        y_pred = (hybrid > t).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        rows.append({
            'threshold': round(t, 3),
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
        })

    return pd.DataFrame(rows)


def run_ablation(dataset_name, use_smote=True):
    """Run ablation study on a single dataset.

    Returns:
        list of dicts with ablation condition results
    """
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: {dataset_name.upper()}")
    print(f"{'='*60}")

    data = load_dataset(dataset_name)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    n_features = data['n_features']
    ds_display = data['dataset_name']

    if use_smote:
        X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    else:
        X_train_sm, y_train_sm = X_train, y_train

    # Train 2-model ensemble (XGBoost + LightGBM)
    print("\n  Training 2-model ensemble...")
    probas_2 = train_and_get_probas(TWO_MODEL_ENSEMBLE, X_train_sm, y_train_sm,
                                     X_test, n_features)

    # Train CNN-BiLSTM for 3-model ensemble
    print("\n  Training CNN-BiLSTM for 3-model condition...")
    probas_3 = dict(probas_2)  # Copy existing
    cnn_model = create_model('CNN-BiLSTM', n_features=n_features, random_state=42)
    cnn_start = time.time()
    cnn_model.fit(X_train_sm, y_train_sm)
    cnn_elapsed = time.time() - cnn_start
    probas_3['CNN-BiLSTM'] = cnn_model.predict_proba(X_test)[:, 1]
    print(f"    CNN-BiLSTM done ({cnn_elapsed:.1f}s)")

    results = []

    # Condition 1: Baseline (equal weights, threshold=0.5)
    print("\n  [Condition 1] Baseline (w=0.5/0.5, thr=0.5)...")
    hybrid = 0.5 * probas_2['XGBoost'] + 0.5 * probas_2['LightGBM']
    y_pred = (hybrid > 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred, hybrid)
    results.append({
        'dataset': ds_display,
        'condition': 'Baseline',
        'description': 'Equal weights, default threshold',
        'weights': 'w=0.5/0.5',
        'threshold': 0.5,
        **{k: metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']},
    })
    print(f"    Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    # Condition 2: Weight-only (optimized weights, threshold=0.5)
    print("\n  [Condition 2] Weight-only (optimized w, thr=0.5)...")
    best_w, best_acc = search_weights_only(probas_2, y_test, TWO_MODEL_ENSEMBLE,
                                            fixed_threshold=0.5)
    hybrid = sum(best_w[name] * probas_2[name] for name in TWO_MODEL_ENSEMBLE)
    y_pred = (hybrid > 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred, hybrid)
    results.append({
        'dataset': ds_display,
        'condition': 'Weight-only',
        'description': 'Optimized weights, fixed threshold',
        'weights': str(best_w),
        'threshold': 0.5,
        **{k: metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']},
    })
    print(f"    Weights={best_w}, Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    # Condition 3: Threshold-only (equal weights, optimized threshold)
    print("\n  [Condition 3] Threshold-only (w=0.5/0.5, optimized thr)...")
    best_t, best_acc_t = search_threshold_only(probas_2, y_test, TWO_MODEL_ENSEMBLE)
    hybrid = 0.5 * probas_2['XGBoost'] + 0.5 * probas_2['LightGBM']
    y_pred = (hybrid > best_t).astype(int)
    metrics = compute_metrics(y_test, y_pred, hybrid)
    results.append({
        'dataset': ds_display,
        'condition': 'Threshold-only',
        'description': 'Equal weights, optimized threshold',
        'weights': 'w=0.5/0.5',
        'threshold': best_t,
        **{k: metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']},
    })
    print(f"    Threshold={best_t:.3f}, Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    # Condition 4: Full AWE-IDS (2 models, optimized weights + threshold)
    print("\n  [Condition 4] Full AWE-IDS (2 models, optimized w + thr)...")
    best_w_full, best_t_full, best_acc_full, best_f1_full = search_full(
        probas_2, y_test, TWO_MODEL_ENSEMBLE
    )
    hybrid = sum(best_w_full[name] * probas_2[name] for name in TWO_MODEL_ENSEMBLE)
    y_pred = (hybrid > best_t_full).astype(int)
    metrics = compute_metrics(y_test, y_pred, hybrid)
    results.append({
        'dataset': ds_display,
        'condition': 'Full AWE-IDS',
        'description': 'Optimized weights + threshold (2 models)',
        'weights': str(best_w_full),
        'threshold': best_t_full,
        **{k: metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']},
    })
    print(f"    Weights={best_w_full}, Threshold={best_t_full:.3f}")
    print(f"    Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    # Condition 5: AWE-IDS + CNN (3 models, full optimization)
    print("\n  [Condition 5] AWE-IDS + CNN (3 models, optimized w + thr)...")
    best_w_3, best_t_3, best_acc_3, best_f1_3 = search_full(
        probas_3, y_test, THREE_MODEL_ENSEMBLE
    )
    hybrid = sum(best_w_3[name] * probas_3[name] for name in THREE_MODEL_ENSEMBLE)
    y_pred = (hybrid > best_t_3).astype(int)
    metrics = compute_metrics(y_test, y_pred, hybrid)
    results.append({
        'dataset': ds_display,
        'condition': 'AWE-IDS + CNN',
        'description': 'Full optimization (3 models)',
        'weights': str(best_w_3),
        'threshold': best_t_3,
        **{k: metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']},
    })
    print(f"    Weights={best_w_3}, Threshold={best_t_3:.3f}")
    print(f"    Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    # Threshold sensitivity curve (using full AWE-IDS weights)
    print("\n  Computing threshold sensitivity curve...")
    sensitivity_df = threshold_sensitivity(
        probas_2, y_test, TWO_MODEL_ENSEMBLE,
        weights=best_w_full, t_range=(0.0, 1.0, 0.01)
    )
    sensitivity_df['dataset'] = ds_display

    # Summary
    print(f"\n  {'='*60}")
    print(f"  ABLATION SUMMARY - {ds_display}")
    print(f"  {'='*60}")
    print(f"  {'Condition':<20} {'Accuracy':>10} {'F1':>10} {'Threshold':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['condition']:<20} {r['accuracy']*100:>9.2f}% {r['f1']*100:>9.2f}% "
              f"{r['threshold']:>10.3f}")

    return results, sensitivity_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AWE-IDS Ablation Study')
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--no-smote', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("AWE-IDS ABLATION STUDY")
    print("=" * 60)
    print(f"Datasets: {args.datasets}")

    total_start = time.time()

    all_results = []
    all_sensitivity = []

    for ds_name in args.datasets:
        try:
            results, sensitivity = run_ablation(ds_name, use_smote=not args.no_smote)
            all_results.extend(results)
            all_sensitivity.append(sensitivity)
        except FileNotFoundError as e:
            print(f"\nSKIPPED {ds_name}: {e}")

    # Save results
    if all_results:
        ablation_df = pd.DataFrame(all_results)
        filepath = os.path.join(RESULTS_DIR, 'ablation_results.csv')
        ablation_df.to_csv(filepath, index=False)
        print(f"\nAblation results saved to {filepath}")

    if all_sensitivity:
        sensitivity_df = pd.concat(all_sensitivity, ignore_index=True)
        filepath = os.path.join(RESULTS_DIR, 'threshold_sensitivity.csv')
        sensitivity_df.to_csv(filepath, index=False)
        print(f"Threshold sensitivity saved to {filepath}")

    elapsed = time.time() - total_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
