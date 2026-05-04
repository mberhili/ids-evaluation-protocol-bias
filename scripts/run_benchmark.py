"""
Benchmark: 7 Models × 3 Datasets.
Stratified 5-Fold CV, reports mean ± std for all metrics.
Produces Tables 4, 5, 6 of the paper.
"""

import numpy as np
import time
import sys
import os
import argparse

from preprocessing import load_dataset
from models import get_model_list, SLOW_MODELS
from evaluation import (
    evaluate_model_cv, evaluate_model_holdout,
    save_results_csv, format_results_table
)

DATASETS = ['unsw-nb15', 'nsl-kdd', 'cicids2017']


def run_benchmark_cv(datasets=None, models=None, n_splits=5, use_smote=True):
    """Run full benchmark with cross-validation."""
    if datasets is None:
        datasets = DATASETS
    if models is None:
        models = get_model_list(include_slow=True)

    all_results = []

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name.upper()}")
        print(f"{'='*70}")

        try:
            data = load_dataset(ds_name)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        X = np.vstack([data['X_train'], data['X_test']])
        y = np.concatenate([data['y_train'], data['y_test']])

        print(f"  Total samples: {len(y)}, Features: {data['n_features']}")
        print(f"  Normal: {sum(y==0)}, Attack: {sum(y==1)}")

        n_samples = len(y)

        for model_name in models:
            # Skip slow models on large datasets
            if model_name in SLOW_MODELS and n_samples > 200000:
                print(f"\n  [{model_name}] Skipped (too slow for {n_samples} samples)")
                # Use holdout instead of CV for large dataset + slow model
                print(f"  [{model_name}] Using holdout evaluation instead...")
                result = evaluate_model_holdout(
                    model_name,
                    data['X_train'], data['y_train'],
                    data['X_test'], data['y_test'],
                    use_smote=use_smote
                )
                result['dataset_name'] = data['dataset_name']
                result['evaluation'] = 'holdout'
                all_results.append(result)
                continue

            print(f"\n  [{model_name}] {n_splits}-Fold CV...")
            result = evaluate_model_cv(
                model_name, X, y,
                n_splits=n_splits,
                use_smote=use_smote,
                verbose=True
            )
            result['dataset_name'] = data['dataset_name']
            result['evaluation'] = f'{n_splits}-fold-cv'
            all_results.append(result)

        # Print summary table for this dataset
        dataset_results = [r for r in all_results if r.get('dataset_name') == data['dataset_name']]
        print(format_results_table(dataset_results, data['dataset_name']))

    return all_results


def run_benchmark_holdout(datasets=None, models=None, use_smote=True):
    """Run benchmark with train/test holdout (faster, for initial testing)."""
    if datasets is None:
        datasets = DATASETS
    if models is None:
        models = get_model_list(include_slow=True)

    all_results = []

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name.upper()}")
        print(f"{'='*70}")

        try:
            data = load_dataset(ds_name)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        print(f"  Train: {data['n_train']}, Test: {data['n_test']}, Features: {data['n_features']}")

        for model_name in models:
            print(f"\n  [{model_name}]")
            result = evaluate_model_holdout(
                model_name,
                data['X_train'], data['y_train'],
                data['X_test'], data['y_test'],
                use_smote=use_smote
            )
            result['dataset_name'] = data['dataset_name']
            all_results.append(result)

        dataset_results = [r for r in all_results if r.get('dataset_name') == data['dataset_name']]
        print(format_results_table(dataset_results, data['dataset_name']))

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark 7 Models × 3 Datasets')
    parser.add_argument('--mode', choices=['cv', 'holdout'], default='holdout',
                        help='Evaluation mode: cv (5-fold) or holdout (faster)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to use (default: all available)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to benchmark (default: all)')
    parser.add_argument('--no-smote', action='store_true',
                        help='Disable SMOTE oversampling')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    args = parser.parse_args()

    datasets = args.datasets or DATASETS
    models = args.models or get_model_list(include_slow=True)

    print("=" * 70)
    print("BENCHMARK: 7 MODELS × 3 DATASETS")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"SMOTE: {not args.no_smote}")

    start = time.time()

    if args.mode == 'cv':
        results = run_benchmark_cv(
            datasets=datasets, models=models,
            n_splits=args.folds, use_smote=not args.no_smote
        )
    else:
        results = run_benchmark_holdout(
            datasets=datasets, models=models,
            use_smote=not args.no_smote
        )

    # Save results
    save_results_csv(results, f'benchmark_{args.mode}_results.csv')

    elapsed = time.time() - start
    print(f"\nTotal benchmark time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
