# Reproducibility Package

## Paper

**How Evaluation Protocol Choices Inflate Intrusion Detection Accuracy: A Three-Dataset Systematic Study**
*Accepted for publication in Computers & Security (Elsevier), 2026.*

**Authors:** Mohammed Berhili, Omar Chaieb, Mohammed Benabdellah
**Affiliation:** SIOSI Research Team (Intelligent Systems, Optimisation and Computer Security), LIPIO Laboratory (Interdisciplinary Laboratory of Physics, Computer Science and Oncology), Faculty of Sciences, Mohammed First University, Oujda 60000, Morocco
**Corresponding author:** Mohammed Berhili — berhili.mohammed@ump.ac.ma

This repository contains all code, scripts, train/test indices, random seeds, and result files needed to reproduce every experiment reported in the manuscript.

---

## Contents

.
├── README.md                # this file
├── requirements.txt         # Python dependencies (pinned)
├── data/                    # raw dataset placement (NOT redistributed; see download.md)
├── scripts/                 # all experiment code
│   ├── preprocessing.py             # unified loaders for UNSW-NB15, NSL-KDD, CIC-IDS2017
│   ├── feature_alignment.py         # semantic alignment for cross-dataset
│   ├── models.py                    # XGBoost, LightGBM, CNN-BiLSTM, RF, SVM, MLP, KNN
│   ├── evaluation.py                # metrics helpers (accuracy, F1, AUC, etc.)
│   ├── run_benchmark.py             # 7-model benchmark across 3 datasets
│   ├── run_awe.py                   # AWE-IDS weight + threshold optimization
│   ├── run_bias_analysis.py         # validation_split, seed sensitivity, preprocessing leakage (A1/A2/A3)
│   ├── run_cross_dataset.py         # cross-dataset transfer (original index-based truncation)
│   ├── run_cross_dataset_aligned.py # cross-dataset with semantic alignment + dual calibration
│   ├── run_grid_sensitivity.py      # AWE-IDS grid resolution sensitivity
│   ├── run_cnn_bilstm_tuning.py     # CNN-BiLSTM hyperparameter sweep
│   ├── run_statistical_tests.py     # Friedman + Wilcoxon + Nemenyi
│   └── run_ablation.py              # 5-condition ablation
├── results/                 # CSV results for every experiment in the paper
└── docs/                    # supplementary documentation

## Reproduction

### 1. Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Tested with Python 3.10 and 3.14 on macOS Darwin.

2. Datasets

The three datasets are not redistributed (license / size). Place them as follows

┌─────────────┬───────────────────────────────────────────────────────┬───────────────────────────────┐
│   Dataset   │                         Files                         │                         Source                          │
├─────────────┼───────────────────────────────────────────────────────┼───────────────────────────────┤
│ UNSW-NB15   │ UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv │ https://research.unsw.edu.au/projects/unsw-nb15-dataset │
├─────────────┼───────────────────────────────────────────────────────┼───────────────────────────────┤
│ NSL-KDD     │ data/nsl-kdd/KDDTrain+.txt, KDDTest+.txt              │ https://www.unb.ca/cic/datasets/nsl.html                │
├─────────────┼───────────────────────────────────────────────────────┼───────────────────────────────┤
│ CIC-IDS2017 │ data/cicids2017/*.pcap_ISCX.csv (8 daily CSVs)        │ https://www.unb.ca/cic/datasets/ids-2017.html           │
└─────────────┴───────────────────────────────────────────────────────┴───────────────────────────────┘

UNSW-NB15 official train/test split is used as-is. NSL-KDD KDDTrain+ / KDDTest+ C-IDS2017 is subsampled stratified to 500{,}000 flows with random_state=42; theexact subsampling is reproduced inside preprocessing.py:load_cicids2017.

3. Running experiments

All experiments are deterministic at random_state=42 unless noted. Wall-clock estimates are for an Apple M1 / 16 GB.

# Benchmark (7 models on 3 datasets) -- ~90 min
python scripts/run_benchmark.py

# AWE-IDS optimization (paper baseline numbers) -- ~30 min
python scripts/run_awe.py

# Bias analysis (A1: validation_split, A2: seed, A3: preprocessing leakage) -- ~20 min
python scripts/run_bias_analysis.py

# Statistical tests (Friedman, Wilcoxon, Nemenyi) -- ~15 min
python scripts/run_statistical_tests.py

# Ablation (5 conditions) -- ~15 min
python scripts/run_ablation.py

# --- Cross-dataset and sensitivity experiments ---

# Cross-dataset with semantic alignment + dual calibration -- ~45 min
python scripts/run_cross_dataset_aligned.py --pair both

# Cross-dataset original truncation (for comparison) -- ~30 min
python scripts/run_cross_dataset.py --align-method pad_truncate --out-suffix tru

# Cross-dataset with zero-padding alignment (sensitivity to alignment method) --
python scripts/run_cross_dataset.py --align-method pad --out-suffix padding

# AWE-IDS grid resolution sensitivity -- ~5 min
python scripts/run_grid_sensitivity.py

# CNN-BiLSTM hyperparameter sweep -- ~30 min
python scripts/run_cnn_bilstm_tuning.py --dataset unsw-nb15

All scripts write CSVs into results/. Pre-computed results from our runs are already present there for direct comparison.

4. Results layout

┌────────────────────────────────────────────────────────────────────────────┬──────────────────────────┬─────────────────────────────────────────────────────────────────────┐
│                                    File                                    │                              Description                             │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ benchmark_holdout_results.csv                                              │ S7 models × 3 datasets, single train/test split                      │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ awe_comparison_results.csv                                                 │ SAWE-IDS vs simple/voting/stacking, in-domain                        │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ a1_bias_results.csv                                                        │ Svalidation_split inflation                                          │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ a2_seed_sensitivity_*.csv                                                  │ Sseed sensitivity per dataset                                        │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ a3_leakage_results.csv                                                     │ Spreprocessing leakage (negligible effect)                           │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ ablation_results.csv                                                       │ S5 conditions                                                        │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ cross_dataset_results.csv                                                  │ SOriginal truncation-based (for historical comparison)               │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ cross_dataset_results_padding.csv                                          │ SZero-padding alternative                                            │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ cross_dataset_aligned.csv                                                  │ SSemantic alignment, single-mode AWE                                 │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ cross_dataset_aligned_dual.csv                                             │ SSemantic alignment + dual AWE-IDS calibration (deployable + oracle) │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ grid_sensitivity.csv                                                       │ SAWE-IDS optimum stability across grid resolutions                   │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ cnn_bilstm_tuning_unsw-nb15.csv                                            │ SCNN-BiLSTM weight share under hyperparameter sweep                  │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ statistical_friedman.csv, _wilcoxon.csv, _nemenyi.csv, per_fold_scores.csv │ SFriedman χ²=50.08, p<10⁻⁶                                           │
├────────────────────────────────────────────────────────────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────┤
│ table_T*.csv                                                               │ TOne CSV per published table                                         │
└────────────────────────────────────────────────────────────────────────────┴──────────────────────────┴─────────────────────────────────────────────────────────────────────┘

5. Determinism notes

- Random seed 42 throughout.
- CIC-IDS2017 subsampling is stratified train_test_split(stratify=label, random_00-flow subset is reproducible from the raw 2.8M-flow concatenation.
- train_test_split for the deployable AWE-IDS calibration split (80/20) uses the same seed.
- SMOTE uses random_state=42.
- TensorFlow/Keras runs (CNN-BiLSTM) use tf.random.set_seed(42) and np.random.seed(42). Note: TF/CUDA non-determinism is bounded but not zero.

6. Hardware notes

CIC-IDS2017 full pipeline (subsampling + SMOTE + 5-fold) requires ~16 GB RAM. CNN-BiLSTM training is the slowest component (~2 minutes per epoch on UNSW with default config; ~5 minutes on
CIC subsample).

---

License

Code: MIT (see LICENSE).
Result CSVs: CC-BY 4.0.
Datasets are NOT included; obtain from their respective official sources.
