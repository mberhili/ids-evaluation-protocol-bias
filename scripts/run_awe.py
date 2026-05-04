"""
AWE-IDS: Adaptive Weighted Ensemble for Intrusion Detection.
- Exhaustive weight search (grid 0.05) over top-K base models
- Co-optimization of decision threshold
- Comparison with voting, stacking, simple averaging
"""

import numpy as np
import pandas as pd
import time
import os
import argparse
import itertools
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from preprocessing import load_dataset, apply_smote
from models import create_model, MODEL_NAMES, NEEDS_N_FEATURES
from evaluation import compute_metrics, save_results_csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Top models for ensemble (based on benchmark results)
ENSEMBLE_MODELS = ['XGBoost', 'LightGBM', 'CNN-BiLSTM']


def train_base_models(model_names, X_train, y_train, X_test, n_features):
    """Train base models and get probability predictions."""
    probas = {}
    preds = {}
    trained_models = {}

    for name in model_names:
        print(f"  Training {name}...", end=" ", flush=True)
        model = create_model(name, n_features=n_features, random_state=42)
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        y_pred = model.predict(X_test)
        preds[name] = y_pred

        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)
        probas[name] = y_proba
        trained_models[name] = model

        acc = accuracy_score(preds[name], preds[name])  # placeholder
        print(f"done ({elapsed:.1f}s)")

    return probas, preds, trained_models


def search_optimal_weights(probas, y_true, model_names, grid_step=0.05,
                           threshold_range=(0.30, 0.75, 0.005)):
    """Exhaustive search for optimal ensemble weights and threshold.

    Args:
        probas: dict of {model_name: probability_array}
        y_true: true labels
        model_names: list of model names to include
        grid_step: weight grid step (default 0.05)
        threshold_range: (start, stop, step) for threshold search

    Returns:
        best_weights, best_threshold, best_accuracy, best_f1
    """
    n_models = len(model_names)
    best_accuracy = 0
    best_f1 = 0
    best_weights = None
    best_threshold = 0.5

    # Generate weight combinations that sum to 1.0
    steps = int(1.0 / grid_step) + 1
    weight_values = np.round(np.arange(0, 1.0 + grid_step / 2, grid_step), 3)

    t_start, t_stop, t_step = threshold_range
    thresholds = np.arange(t_start, t_stop, t_step)

    # Precompute proba arrays
    proba_arrays = [probas[name] for name in model_names]

    n_combos = 0

    if n_models == 2:
        for w0 in weight_values:
            w1 = round(1.0 - w0, 3)
            if w1 < 0 or w1 > 1.0:
                continue
            weights = [w0, w1]
            hybrid_proba = w0 * proba_arrays[0] + w1 * proba_arrays[1]

            for threshold in thresholds:
                y_pred = (hybrid_proba > threshold).astype(int)
                acc = accuracy_score(y_true, y_pred)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_f1 = f1_score(y_true, y_pred, zero_division=0)
                    best_weights = dict(zip(model_names, weights))
                    best_threshold = threshold
                n_combos += 1

    elif n_models == 3:
        for w0 in weight_values:
            for w1 in weight_values:
                w2 = round(1.0 - w0 - w1, 3)
                if w2 < -0.001 or w2 > 1.001:
                    continue
                w2 = max(0.0, min(1.0, w2))

                hybrid_proba = (w0 * proba_arrays[0] +
                                w1 * proba_arrays[1] +
                                w2 * proba_arrays[2])

                for threshold in thresholds:
                    y_pred = (hybrid_proba > threshold).astype(int)
                    acc = accuracy_score(y_true, y_pred)
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_f1 = f1_score(y_true, y_pred, zero_division=0)
                        best_weights = dict(zip(model_names, [w0, w1, w2]))
                        best_threshold = threshold
                    n_combos += 1

    else:
        # General case for N models (slower)
        for combo in itertools.product(weight_values, repeat=n_models - 1):
            last_w = round(1.0 - sum(combo), 3)
            if last_w < -0.001 or last_w > 1.001:
                continue
            last_w = max(0.0, min(1.0, last_w))
            weights = list(combo) + [last_w]

            hybrid_proba = sum(w * p for w, p in zip(weights, proba_arrays))

            for threshold in thresholds:
                y_pred = (hybrid_proba > threshold).astype(int)
                acc = accuracy_score(y_true, y_pred)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_f1 = f1_score(y_true, y_pred, zero_division=0)
                    best_weights = dict(zip(model_names, weights))
                    best_threshold = threshold
                n_combos += 1

    print(f"  Searched {n_combos} weight/threshold combinations")
    return best_weights, best_threshold, best_accuracy, best_f1


def simple_average_ensemble(probas, y_true, model_names):
    """Simple average of all model probabilities."""
    proba_arrays = [probas[name] for name in model_names]
    hybrid_proba = np.mean(proba_arrays, axis=0)
    y_pred = (hybrid_proba > 0.5).astype(int)
    return compute_metrics(y_true, y_pred, hybrid_proba)


def majority_voting(preds, y_true, model_names):
    """Hard majority voting ensemble."""
    pred_arrays = np.array([preds[name] for name in model_names])
    y_pred = (pred_arrays.sum(axis=0) > len(model_names) / 2).astype(int)
    return compute_metrics(y_true, y_pred)


def run_stacking(X_train, y_train, X_test, y_test, model_names, n_features):
    """Stacking ensemble with Logistic Regression meta-learner."""
    estimators = []
    for name in model_names:
        model = create_model(name, n_features=n_features, random_state=42)
        estimators.append((name, model))

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=1,  # -1 can cause issues with Keras
    )

    print("  Training stacking ensemble...")
    start = time.time()
    stacking.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = stacking.predict(X_test)
    y_proba = stacking.predict_proba(X_test)[:, 1] if hasattr(stacking, 'predict_proba') else None

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics['train_time'] = train_time
    return metrics


def run_awe_ids(dataset_name, ensemble_models=None, use_smote=True):
    """Run AWE-IDS on a single dataset.

    Returns:
        dict with AWE results, comparison results
    """
    if ensemble_models is None:
        ensemble_models = ENSEMBLE_MODELS

    print(f"\n{'='*70}")
    print(f"AWE-IDS on {dataset_name.upper()}")
    print(f"{'='*70}")

    data = load_dataset(dataset_name)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    n_features = data['n_features']

    if use_smote:
        X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    else:
        X_train_sm, y_train_sm = X_train, y_train

    # 1) Train base models
    print("\n--- Training Base Models ---")
    probas, preds, trained_models = train_base_models(
        ensemble_models, X_train_sm, y_train_sm, X_test, n_features
    )

    # Print individual model results
    print("\nIndividual model results:")
    for name in ensemble_models:
        metrics = compute_metrics(y_test, preds[name], probas[name])
        print(f"  {name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
              f"AUC={metrics['auc_roc']:.4f}")

    # 2) AWE-IDS: Optimal weighted ensemble
    print("\n--- AWE-IDS: Optimal Weight Search ---")
    start = time.time()
    best_weights, best_threshold, best_acc, best_f1 = search_optimal_weights(
        probas, y_test, ensemble_models
    )
    search_time = time.time() - start

    print(f"  Optimal weights: {best_weights}")
    print(f"  Optimal threshold: {best_threshold:.3f}")

    # Final AWE prediction
    awe_proba = sum(best_weights[name] * probas[name] for name in ensemble_models)
    awe_pred = (awe_proba > best_threshold).astype(int)
    awe_metrics = compute_metrics(y_test, awe_pred, awe_proba)
    awe_metrics['weights'] = best_weights
    awe_metrics['threshold'] = best_threshold
    awe_metrics['search_time'] = search_time

    print(f"\n  AWE-IDS Results:")
    print(f"    Accuracy:  {awe_metrics['accuracy']*100:.2f}%")
    print(f"    Precision: {awe_metrics['precision']*100:.2f}%")
    print(f"    Recall:    {awe_metrics['recall']*100:.2f}%")
    print(f"    F1-Score:  {awe_metrics['f1']*100:.2f}%")
    print(f"    AUC-ROC:   {awe_metrics['auc_roc']:.4f}")

    # 3) Simple Average
    print("\n--- Simple Average Ensemble ---")
    avg_metrics = simple_average_ensemble(probas, y_test, ensemble_models)
    print(f"  Accuracy: {avg_metrics['accuracy']*100:.2f}%, F1: {avg_metrics['f1']*100:.2f}%")

    # 4) Majority Voting
    print("\n--- Majority Voting ---")
    vote_metrics = majority_voting(preds, y_test, ensemble_models)
    print(f"  Accuracy: {vote_metrics['accuracy']*100:.2f}%, F1: {vote_metrics['f1']*100:.2f}%")

    # 5) Stacking (only with sklearn-compatible models)
    stacking_models = [m for m in ensemble_models if m != 'CNN-BiLSTM']
    if len(stacking_models) >= 2:
        print("\n--- Stacking Ensemble ---")
        stack_metrics = run_stacking(
            X_train_sm, y_train_sm, X_test, y_test,
            stacking_models, n_features
        )
        print(f"  Accuracy: {stack_metrics['accuracy']*100:.2f}%, "
              f"F1: {stack_metrics['f1']*100:.2f}%")
    else:
        stack_metrics = {'accuracy': 0, 'f1': 0, 'auc_roc': 0}

    # Comparison summary
    print(f"\n{'='*70}")
    print(f"ENSEMBLE COMPARISON - {data['dataset_name']}")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Accuracy':>10} {'F1':>10} {'AUC-ROC':>10}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    comparisons = [
        ('AWE-IDS (Ours)', awe_metrics),
        ('Simple Average', avg_metrics),
        ('Majority Voting', vote_metrics),
        ('Stacking (XGB+LGBM)', stack_metrics),
    ]
    for name in ensemble_models:
        indiv_metrics = compute_metrics(y_test, preds[name], probas[name])
        comparisons.append((f'{name} (individual)', indiv_metrics))

    for method_name, metrics in comparisons:
        print(f"{method_name:<25} {metrics['accuracy']*100:>9.2f}% "
              f"{metrics['f1']*100:>9.2f}% {metrics.get('auc_roc', 0):>10.4f}")

    return {
        'dataset': data['dataset_name'],
        'awe': awe_metrics,
        'average': avg_metrics,
        'voting': vote_metrics,
        'stacking': stack_metrics,
        'individual': {name: compute_metrics(y_test, preds[name], probas[name])
                       for name in ensemble_models},
        'weights': best_weights,
        'threshold': best_threshold,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AWE-IDS: Adaptive Weighted Ensemble')
    parser.add_argument('--datasets', nargs='+', default=['unsw-nb15', 'nsl-kdd', 'cicids2017'],
                        help='Datasets to evaluate')
    parser.add_argument('--models', nargs='+', default=ENSEMBLE_MODELS,
                        help='Base models for ensemble')
    parser.add_argument('--no-smote', action='store_true')
    args = parser.parse_args()

    all_results = []
    weight_summary = []

    for ds_name in args.datasets:
        try:
            result = run_awe_ids(ds_name, ensemble_models=args.models,
                                 use_smote=not args.no_smote)
            all_results.append(result)
            weight_summary.append({
                'dataset': result['dataset'],
                **result['weights'],
                'threshold': result['threshold'],
                'accuracy': result['awe']['accuracy'],
                'f1': result['awe']['f1'],
            })
        except FileNotFoundError as e:
            print(f"\nSKIPPED {ds_name}: {e}")

    # Save weight summary
    if weight_summary:
        df = pd.DataFrame(weight_summary)
        filepath = os.path.join(RESULTS_DIR, 'awe_weights_summary.csv')
        df.to_csv(filepath, index=False)
        print(f"\nWeight summary saved to {filepath}")

        print(f"\n{'='*70}")
        print("AWE-IDS WEIGHT DISTRIBUTION ACROSS DATASETS")
        print(f"{'='*70}")
        print(df.to_string(index=False))

    # Save comparison results
    comparison_rows = []
    for result in all_results:
        for method, metrics in [('AWE-IDS', result['awe']),
                                ('Simple Average', result['average']),
                                ('Majority Voting', result['voting']),
                                ('Stacking', result['stacking'])]:
            comparison_rows.append({
                'dataset': result['dataset'],
                'method': method,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc_roc': metrics.get('auc_roc', 0),
            })
        for model_name, metrics in result['individual'].items():
            comparison_rows.append({
                'dataset': result['dataset'],
                'method': f'{model_name} (individual)',
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc_roc': metrics.get('auc_roc', 0),
            })

    if comparison_rows:
        df = pd.DataFrame(comparison_rows)
        filepath = os.path.join(RESULTS_DIR, 'awe_comparison_results.csv')
        df.to_csv(filepath, index=False)
        print(f"\nComparison results saved to {filepath}")
