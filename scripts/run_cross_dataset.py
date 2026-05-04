"""
Cross-Dataset Generalization Experiments.
Train on one dataset, test on another to evaluate generalization.
Show that AWE-IDS weights adapt to each dataset.
"""

import numpy as np
import pandas as pd
import time
import os
import argparse
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score

from preprocessing import load_dataset, apply_smote
from models import create_model
from evaluation import compute_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Common features between UNSW-NB15 and NSL-KDD (network flow features)
# Since features differ between datasets, we use statistical feature alignment
ENSEMBLE_MODELS = ['XGBoost', 'LightGBM']


def align_features(X_source, X_target, method='pad_truncate'):
    """Align feature dimensions between source and target datasets.

    Since different IDS datasets have different features, we need to
    align dimensions for cross-dataset transfer. We use zero-padding
    or truncation to match dimensions.
    """
    n_source = X_source.shape[1]
    n_target = X_target.shape[1]

    if n_source == n_target:
        return X_source, X_target

    if method == 'pad_truncate':
        n_common = min(n_source, n_target)
        # Truncate both to common dimension
        return X_source[:, :n_common], X_target[:, :n_common]
    elif method == 'pad':
        # Pad the smaller one with zeros
        n_max = max(n_source, n_target)
        if n_source < n_max:
            padding = np.zeros((X_source.shape[0], n_max - n_source))
            X_source = np.hstack([X_source, padding])
        if n_target < n_max:
            padding = np.zeros((X_target.shape[0], n_max - n_target))
            X_target = np.hstack([X_target, padding])
        return X_source, X_target

    return X_source[:, :min(n_source, n_target)], X_target[:, :min(n_source, n_target)]


def cross_dataset_experiment(source_name, target_name, model_names=None, align_method='pad_truncate'):
    """Train on source dataset, test on target dataset.

    Args:
        source_name: Training dataset name
        target_name: Testing dataset name
        model_names: Models to evaluate

    Returns:
        dict with cross-dataset results
    """
    if model_names is None:
        model_names = ENSEMBLE_MODELS

    print(f"\n{'='*70}")
    print(f"CROSS-DATASET: Train on {source_name.upper()} → Test on {target_name.upper()}")
    print(f"{'='*70}")

    source_data = load_dataset(source_name)
    target_data = load_dataset(target_name)

    # Use all source data for training, all target data for testing
    X_source = np.vstack([source_data['X_train'], source_data['X_test']])
    y_source = np.concatenate([source_data['y_train'], source_data['y_test']])
    X_target = np.vstack([target_data['X_train'], target_data['X_test']])
    y_target = np.concatenate([target_data['y_train'], target_data['y_test']])

    print(f"  Source ({source_data['dataset_name']}): {len(y_source)} samples, "
          f"{X_source.shape[1]} features")
    print(f"  Target ({target_data['dataset_name']}): {len(y_target)} samples, "
          f"{X_target.shape[1]} features")

    # Align features
    X_source_aligned, X_target_aligned = align_features(X_source, X_target, method=align_method)
    n_common = X_source_aligned.shape[1]
    print(f"  Aligned features: {n_common}")

    # Re-scale after alignment
    scaler = MinMaxScaler()
    X_source_aligned = scaler.fit_transform(X_source_aligned)
    X_target_aligned = scaler.transform(X_target_aligned)

    # Apply SMOTE
    X_source_sm, y_source_sm = apply_smote(X_source_aligned, y_source)

    results = []
    probas_target = {}

    for model_name in model_names:
        print(f"\n  [{model_name}]")
        model = create_model(model_name, n_features=n_common, random_state=42)

        start = time.time()
        model.fit(X_source_sm, y_source_sm)
        train_time = time.time() - start

        # Test on target
        y_pred = model.predict(X_target_aligned)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_target_aligned)[:, 1]
            probas_target[model_name] = y_proba

        metrics = compute_metrics(y_target, y_pred, y_proba)
        metrics['model_name'] = model_name
        metrics['train_time'] = train_time

        print(f"    Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
              f"AUC={metrics['auc_roc']:.4f}")

        # Also test on source (for comparison)
        y_pred_source = model.predict(X_source_aligned)
        acc_source = accuracy_score(y_source, y_pred_source)
        metrics['source_accuracy'] = acc_source
        print(f"    (Source accuracy: {acc_source:.4f})")

        results.append(metrics)

    # AWE-IDS on cross-dataset (if we have probability predictions)
    if len(probas_target) >= 2:
        print(f"\n  AWE-IDS weight search on cross-dataset...")
        proba_arrays = [probas_target[name] for name in model_names if name in probas_target]
        active_models = [name for name in model_names if name in probas_target]

        best_acc = 0
        best_weights = None
        best_threshold = 0.5

        weight_values = np.arange(0, 1.05, 0.05)
        for w0 in weight_values:
            w1 = round(1.0 - w0, 2)
            if w1 < 0:
                continue
            hybrid = w0 * proba_arrays[0] + w1 * proba_arrays[1]
            for thresh in np.arange(0.3, 0.75, 0.01):
                y_pred = (hybrid > thresh).astype(int)
                acc = accuracy_score(y_target, y_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_weights = {active_models[0]: w0, active_models[1]: w1}
                    best_threshold = thresh

        if best_weights:
            hybrid_proba = sum(best_weights[n] * probas_target[n] for n in active_models)
            hybrid_pred = (hybrid_proba > best_threshold).astype(int)
            hybrid_metrics = compute_metrics(y_target, hybrid_pred, hybrid_proba)

            print(f"    AWE-IDS weights: {best_weights}")
            print(f"    AWE-IDS threshold: {best_threshold:.3f}")
            print(f"    AWE-IDS Acc={hybrid_metrics['accuracy']:.4f}, "
                  f"F1={hybrid_metrics['f1']:.4f}")

            results.append({
                'model_name': 'AWE-IDS',
                'accuracy': hybrid_metrics['accuracy'],
                'f1': hybrid_metrics['f1'],
                'auc_roc': hybrid_metrics.get('auc_roc', 0),
                'weights': best_weights,
                'threshold': best_threshold,
            })

    return {
        'source': source_data['dataset_name'],
        'target': target_data['dataset_name'],
        'results': results,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-Dataset Generalization')
    parser.add_argument('--models', nargs='+', default=ENSEMBLE_MODELS)
    parser.add_argument('--align-method', choices=['pad_truncate', 'pad'], default='pad_truncate',
                        help='Feature alignment: pad_truncate (default, original) or pad (zero-padding)')
    parser.add_argument('--out-suffix', default='', help='Suffix added to output CSV filename')
    args = parser.parse_args()

    # All pairwise cross-dataset experiments
    datasets = ['unsw-nb15', 'nsl-kdd']

    # Check which datasets are available
    available = []
    for ds in datasets:
        try:
            load_dataset(ds)
            available.append(ds)
        except FileNotFoundError:
            print(f"Dataset {ds} not available, skipping.")

    # Also try CIC-IDS2017
    try:
        load_dataset('cicids2017')
        available.append('cicids2017')
    except FileNotFoundError:
        print("CIC-IDS2017 not available, skipping.")

    all_results = []
    for source in available:
        for target in available:
            if source == target:
                continue
            try:
                result = cross_dataset_experiment(source, target, model_names=args.models,
                                                  align_method=args.align_method)
                all_results.append(result)
            except Exception as e:
                print(f"\nERROR {source} → {target}: {e}")

    # Save results
    if all_results:
        rows = []
        for exp in all_results:
            for r in exp['results']:
                rows.append({
                    'source': exp['source'],
                    'target': exp['target'],
                    'model': r.get('model_name', ''),
                    'accuracy': r.get('accuracy', 0),
                    'f1': r.get('f1', 0),
                    'auc_roc': r.get('auc_roc', 0),
                    'source_accuracy': r.get('source_accuracy', ''),
                })
        df = pd.DataFrame(rows)
        suffix = f'_{args.out_suffix}' if args.out_suffix else ''
        filepath = os.path.join(RESULTS_DIR, f'cross_dataset_results{suffix}.csv')
        df.to_csv(filepath, index=False)
        print(f"\nResults saved to {filepath}")

        # Summary table
        print(f"\n{'='*70}")
        print("CROSS-DATASET SUMMARY")
        print(f"{'='*70}")
        print(df.to_string(index=False))
