"""
Unified Evaluation Pipeline for IDS Models.
Stratified 5-Fold CV, comprehensive metrics, CSV export.
"""

import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

from models import create_model, NEEDS_N_FEATURES
from preprocessing import apply_smote

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute all classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc_roc'] = 0.0
    else:
        metrics['auc_roc'] = 0.0

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0

    return metrics


def evaluate_model_cv(model_name, X, y, n_splits=5, random_state=42,
                      use_smote=True, verbose=True):
    """Evaluate a model using Stratified K-Fold Cross-Validation.

    Returns:
        dict with mean and std of all metrics, plus per-fold details
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    n_features = X.shape[1]
    total_time = 0

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        if use_smote:
            X_train_fold, y_train_fold = apply_smote(X_train_fold, y_train_fold, random_state=42)

        model = create_model(model_name, n_features=n_features, random_state=random_state)

        start_time = time.time()
        model.fit(X_train_fold, y_train_fold)
        train_time = time.time() - start_time

        start_time = time.time()
        y_pred = model.predict(X_test_fold)
        predict_time = time.time() - start_time

        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_fold)[:, 1]

        metrics = compute_metrics(y_test_fold, y_pred, y_proba)
        metrics['train_time'] = train_time
        metrics['predict_time'] = predict_time
        metrics['fold'] = fold_idx + 1
        total_time += train_time + predict_time

        fold_results.append(metrics)
        all_y_true.extend(y_test_fold)
        all_y_pred.extend(y_pred)
        if y_proba is not None:
            all_y_proba.extend(y_proba)

        if verbose:
            print(f"    Fold {fold_idx+1}/{n_splits}: "
                  f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
                  f"AUC={metrics['auc_roc']:.4f}, Time={train_time:.1f}s")

    # Aggregate results
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    summary = {}
    for key in metric_keys:
        values = [r[key] for r in fold_results]
        summary[f'{key}_mean'] = np.mean(values)
        summary[f'{key}_std'] = np.std(values)

    summary['total_time'] = total_time
    summary['avg_train_time'] = np.mean([r['train_time'] for r in fold_results])
    summary['avg_predict_time'] = np.mean([r['predict_time'] for r in fold_results])
    summary['model_name'] = model_name
    summary['n_folds'] = n_splits
    summary['fold_results'] = fold_results

    # Overall confusion matrix
    overall_metrics = compute_metrics(
        np.array(all_y_true), np.array(all_y_pred),
        np.array(all_y_proba) if all_y_proba else None
    )
    summary['overall_cm'] = {
        'tn': overall_metrics.get('tn', 0),
        'fp': overall_metrics.get('fp', 0),
        'fn': overall_metrics.get('fn', 0),
        'tp': overall_metrics.get('tp', 0),
    }

    return summary


def evaluate_model_holdout(model_name, X_train, y_train, X_test, y_test,
                           use_smote=True, verbose=True):
    """Evaluate a model using train/test holdout split.

    Returns:
        dict with metrics
    """
    n_features = X_train.shape[1]

    if use_smote:
        X_train_sm, y_train_sm = apply_smote(X_train, y_train, random_state=42)
    else:
        X_train_sm, y_train_sm = X_train, y_train

    model = create_model(model_name, n_features=n_features, random_state=42)

    start_time = time.time()
    model.fit(X_train_sm, y_train_sm)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics['train_time'] = train_time
    metrics['predict_time'] = predict_time
    metrics['model_name'] = model_name
    metrics['y_pred'] = y_pred
    metrics['y_proba'] = y_proba

    if verbose:
        print(f"  {model_name}: Acc={metrics['accuracy']:.4f}, "
              f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, "
              f"F1={metrics['f1']:.4f}, AUC={metrics['auc_roc']:.4f}, "
              f"Time={train_time:.1f}s")

    return metrics


def save_results_csv(results, filename, extra_cols=None):
    """Save benchmark results to CSV.

    Args:
        results: list of dicts with metrics
        filename: output filename (will be saved in results/)
        extra_cols: additional columns to include
    """
    rows = []
    for r in results:
        row = {}
        if extra_cols:
            row.update(extra_cols)
        row['model'] = r.get('model_name', '')
        row['dataset'] = r.get('dataset_name', '')

        for key in ['accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std',
                     'recall_mean', 'recall_std', 'f1_mean', 'f1_std',
                     'auc_roc_mean', 'auc_roc_std', 'total_time',
                     'avg_train_time', 'avg_predict_time']:
            if key in r:
                row[key] = r[key]

        # Also handle holdout results
        for key in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc',
                     'train_time', 'predict_time']:
            if key in r and f'{key}_mean' not in r:
                row[key] = r[key]

        rows.append(row)

    df = pd.DataFrame(rows)
    filepath = os.path.join(RESULTS_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    return df


def format_results_table(results, dataset_name=''):
    """Format results as a readable table string."""
    header = f"\n{'='*90}\n"
    if dataset_name:
        header += f"Results for {dataset_name}\n"
    header += f"{'='*90}\n"
    header += f"{'Model':<15} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1':>12} {'AUC-ROC':>12}\n"
    header += f"{'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}\n"

    lines = [header]
    for r in results:
        if 'accuracy_mean' in r:
            line = (f"{r['model_name']:<15} "
                    f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f} "
                    f"{r['precision_mean']:.4f}±{r['precision_std']:.4f} "
                    f"{r['recall_mean']:.4f}±{r['recall_std']:.4f} "
                    f"{r['f1_mean']:.4f}±{r['f1_std']:.4f} "
                    f"{r['auc_roc_mean']:.4f}±{r['auc_roc_std']:.4f}")
        else:
            line = (f"{r.get('model_name', 'N/A'):<15} "
                    f"{r.get('accuracy', 0):.4f}        "
                    f"{r.get('precision', 0):.4f}        "
                    f"{r.get('recall', 0):.4f}        "
                    f"{r.get('f1', 0):.4f}        "
                    f"{r.get('auc_roc', 0):.4f}")
        lines.append(line)

    return '\n'.join(lines)
