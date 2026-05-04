"""
Statistical Significance Tests for AWE-IDS.
- Stratified 5-Fold CV for 7 models x 3 datasets -> per-fold scores
- Friedman test on the score matrix
- Nemenyi post-hoc test if Friedman is significant
- Wilcoxon signed-rank pairwise with Bonferroni correction
"""

import numpy as np
import pandas as pd
import os
import argparse
import time
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from scipy import stats
from itertools import combinations

from preprocessing import load_dataset, apply_smote
from models import create_model, MODEL_NAMES, SLOW_MODELS, NEEDS_N_FEATURES
from evaluation import compute_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = ['unsw-nb15', 'nsl-kdd', 'cicids2017']


def collect_fold_scores(datasets=None, models=None, n_splits=5, use_smote=True,
                        cic_max_samples=None):
    """Run stratified 5-fold CV for all models x datasets, return per-fold accuracy.

    Args:
        cic_max_samples: if set, subsample CIC-IDS2017 to this size before CV.
                         Useful to include CNN-BiLSTM (default: None = full dataset).

    Returns:
        pd.DataFrame with columns: dataset, model, fold, accuracy, f1, auc_roc
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils import resample

    if datasets is None:
        datasets = DATASETS
    if models is None:
        models = [m for m in MODEL_NAMES if m not in SLOW_MODELS]

    rows = []

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {ds_name.upper()}")
        print(f"{'='*60}")

        try:
            data = load_dataset(ds_name)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        X = np.vstack([data['X_train'], data['X_test']])
        y = np.concatenate([data['y_train'], data['y_test']])
        n_features = data['n_features']
        ds_display = data['dataset_name']

        # Optionally subsample CIC-IDS2017 for CNN-BiLSTM tractability
        is_cic = 'cic' in ds_name.lower() or 'cicids' in ds_name.lower()
        if is_cic and cic_max_samples is not None and len(y) > cic_max_samples:
            X, y = resample(X, y, n_samples=cic_max_samples,
                            stratify=y, random_state=42, replace=False)
            print(f"  CIC subsampled to {len(y):,} samples for CNN tractability")

        print(f"  Samples: {len(y)}, Features: {n_features}")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for model_name in models:
            # Skip slow models on large datasets (unless already subsampled)
            if model_name in SLOW_MODELS and len(y) > 200000:
                print(f"  [{model_name}] Skipped (too slow for {len(y)} samples)")
                continue

            print(f"  [{model_name}] {n_splits}-Fold CV...", end=" ", flush=True)
            start = time.time()

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if use_smote:
                    X_train, y_train = apply_smote(X_train, y_train, random_state=42)

                model = create_model(model_name, n_features=n_features, random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]

                metrics = compute_metrics(y_test, y_pred, y_proba)

                rows.append({
                    'dataset': ds_display,
                    'model': model_name,
                    'fold': fold_idx + 1,
                    'accuracy': metrics['accuracy'],
                    'f1': metrics['f1'],
                    'auc_roc': metrics['auc_roc'],
                })

            elapsed = time.time() - start
            fold_accs = [r['accuracy'] for r in rows
                         if r['model'] == model_name and r['dataset'] == ds_display]
            print(f"done ({elapsed:.1f}s) "
                  f"Acc={np.mean(fold_accs):.4f}+-{np.std(fold_accs):.4f}")

    return pd.DataFrame(rows)


def friedman_test(fold_df, metric='accuracy'):
    """Run Friedman test on the per-fold score matrix.

    Args:
        fold_df: DataFrame with columns [dataset, model, fold, accuracy, ...]
        metric: Which metric to test

    Returns:
        dict with statistic, p_value, significant
    """
    models = sorted(fold_df['model'].unique())
    # Build score matrix: each row is a (dataset, fold) observation
    groups = fold_df.groupby(['dataset', 'fold'])

    score_matrix = []
    for (ds, fold), group in groups:
        row = {}
        for _, r in group.iterrows():
            row[r['model']] = r[metric]
        if len(row) == len(models):
            score_matrix.append([row[m] for m in models])

    score_matrix = np.array(score_matrix)
    print(f"\nFriedman test on {score_matrix.shape[0]} observations x {len(models)} models")

    # Friedman test
    stat, p_value = stats.friedmanchisquare(*score_matrix.T)

    result = {
        'test': 'Friedman',
        'metric': metric,
        'n_observations': score_matrix.shape[0],
        'n_models': len(models),
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'models': models,
    }

    print(f"  Chi-square statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant (p<0.05): {result['significant']}")

    return result, score_matrix, models


def nemenyi_test(score_matrix, models):
    """Run Nemenyi post-hoc test after significant Friedman.

    Returns:
        pd.DataFrame pairwise comparison matrix
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        print("  ERROR: scikit-posthocs not installed. pip install scikit-posthocs")
        return None

    # Use the score_matrix directly (non-melted format)
    # Convert to DataFrame: rows = observations, columns = models
    score_df = pd.DataFrame(score_matrix, columns=models)

    # Nemenyi test (non-melted: rows=blocks, columns=groups)
    nemenyi_result = sp.posthoc_nemenyi_friedman(score_df)

    print("\nNemenyi post-hoc test (p-values):")
    print(nemenyi_result.to_string())

    return nemenyi_result


def wilcoxon_pairwise(fold_df, metric='accuracy'):
    """Run Wilcoxon signed-rank tests for all model pairs with Bonferroni correction.

    Returns:
        pd.DataFrame with columns: model_a, model_b, statistic, p_value, p_corrected, significant
    """
    models = sorted(fold_df['model'].unique())
    n_comparisons = len(list(combinations(models, 2)))

    results = []
    for model_a, model_b in combinations(models, 2):
        # Get paired observations (same dataset + fold)
        df_a = fold_df[fold_df['model'] == model_a].set_index(['dataset', 'fold'])
        df_b = fold_df[fold_df['model'] == model_b].set_index(['dataset', 'fold'])

        common_idx = df_a.index.intersection(df_b.index)
        if len(common_idx) < 5:
            continue

        scores_a = df_a.loc[common_idx, metric].values
        scores_b = df_b.loc[common_idx, metric].values

        # Wilcoxon signed-rank test
        try:
            stat, p_value = stats.wilcoxon(scores_a, scores_b, alternative='two-sided')
        except ValueError:
            # All differences are zero
            stat, p_value = 0, 1.0

        p_corrected = min(p_value * n_comparisons, 1.0)  # Bonferroni

        results.append({
            'model_a': model_a,
            'model_b': model_b,
            'n_pairs': len(common_idx),
            'mean_a': np.mean(scores_a),
            'mean_b': np.mean(scores_b),
            'diff': np.mean(scores_a) - np.mean(scores_b),
            'statistic': stat,
            'p_value': p_value,
            'p_corrected': p_corrected,
            'significant': p_corrected < 0.05,
        })

    return pd.DataFrame(results)


def compute_average_ranks(score_matrix, models):
    """Compute average ranks for Critical Difference diagram."""
    n_obs, n_models = score_matrix.shape
    ranks = np.zeros_like(score_matrix)

    for i in range(n_obs):
        # Rank from best (1) to worst (n_models) - higher score is better
        ranks[i] = stats.rankdata(-score_matrix[i])

    avg_ranks = ranks.mean(axis=0)
    rank_df = pd.DataFrame({
        'model': models,
        'avg_rank': avg_ranks,
    }).sort_values('avg_rank')

    return rank_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistical Tests for AWE-IDS')
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--no-smote', action='store_true')
    parser.add_argument('--skip-cv', action='store_true',
                        help='Skip CV and use existing per_fold_scores.csv')
    parser.add_argument('--cic-max-samples', type=int, default=None,
                        help='Subsample CIC-IDS2017 to N samples (e.g. 50000) '
                             'to enable CNN-BiLSTM in statistical tests')
    args = parser.parse_args()

    models = args.models
    if models is None:
        models = [m for m in MODEL_NAMES if m not in SLOW_MODELS]

    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)
    print(f"Datasets: {args.datasets}")
    print(f"Models: {models}")
    print(f"Folds: {args.folds}")

    total_start = time.time()

    # Step 1: Collect per-fold scores
    fold_scores_path = os.path.join(RESULTS_DIR, 'per_fold_scores.csv')

    if args.skip_cv and os.path.exists(fold_scores_path):
        print(f"\nLoading existing fold scores from {fold_scores_path}")
        fold_df = pd.read_csv(fold_scores_path)
    else:
        print("\n--- Step 1: Collecting per-fold scores via 5-Fold CV ---")
        fold_df = collect_fold_scores(
            datasets=args.datasets, models=models,
            n_splits=args.folds, use_smote=not args.no_smote,
            cic_max_samples=args.cic_max_samples
        )
        fold_df.to_csv(fold_scores_path, index=False)
        print(f"\nPer-fold scores saved to {fold_scores_path}")

    print(f"\nCollected {len(fold_df)} fold results")
    print(fold_df.groupby(['dataset', 'model'])['accuracy'].agg(['mean', 'std']).to_string())

    # Step 2: Friedman test
    print("\n--- Step 2: Friedman Test ---")
    friedman_result, score_matrix, model_list = friedman_test(fold_df, metric='accuracy')

    friedman_df = pd.DataFrame([{
        'test': 'Friedman',
        'statistic': friedman_result['statistic'],
        'p_value': friedman_result['p_value'],
        'significant': friedman_result['significant'],
        'n_observations': friedman_result['n_observations'],
        'n_models': friedman_result['n_models'],
    }])
    friedman_df.to_csv(os.path.join(RESULTS_DIR, 'statistical_friedman.csv'), index=False)

    # Step 3: Nemenyi post-hoc (if Friedman significant)
    if friedman_result['significant']:
        print("\n--- Step 3: Nemenyi Post-hoc Test ---")
        nemenyi_df = nemenyi_test(score_matrix, model_list)
        if nemenyi_df is not None:
            nemenyi_df.to_csv(os.path.join(RESULTS_DIR, 'statistical_nemenyi.csv'))
            print(f"Nemenyi results saved.")

        # Average ranks for CD diagram
        rank_df = compute_average_ranks(score_matrix, model_list)
        rank_df.to_csv(os.path.join(RESULTS_DIR, 'statistical_ranks.csv'), index=False)
        print(f"\nAverage Ranks:")
        print(rank_df.to_string(index=False))
    else:
        print("\nFriedman not significant - skipping post-hoc tests.")

    # Step 4: Wilcoxon pairwise
    print("\n--- Step 4: Wilcoxon Signed-Rank Tests (Bonferroni corrected) ---")
    wilcoxon_df = wilcoxon_pairwise(fold_df, metric='accuracy')
    wilcoxon_df.to_csv(os.path.join(RESULTS_DIR, 'statistical_wilcoxon.csv'), index=False)

    sig_pairs = wilcoxon_df[wilcoxon_df['significant']]
    print(f"\nSignificant pairs ({len(sig_pairs)}/{len(wilcoxon_df)}):")
    if len(sig_pairs) > 0:
        print(sig_pairs[['model_a', 'model_b', 'diff', 'p_corrected']].to_string(index=False))

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Results saved to {RESULTS_DIR}/statistical_*.csv")
