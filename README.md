# Reproducibility Package

## Paper

**How Evaluation Protocol Choices Inflate Intrusion Detection Accuracy: A Three-Dataset Systematic Study**

*Accepted for publication in Computers & Security (Elsevier), 2026.*

**Authors:** Mohammed Berhili, Omar Chaieb, Mohammed Benabdellah

**Affiliation:** SIOSI Research Team (Intelligent Systems, Optimisation and Computer Security), LIPIO Laboratory (Interdisciplinary Laboratory of Physics, Computer Science and Oncology), Faculty of Sciences, Mohammed First University, Oujda 60000, Morocco

**Corresponding author:** Mohammed Berhili - berhili.mohammed@ump.ac.ma

This repository contains all code, scripts, train/test indices, random seeds, and result files needed to reproduce every experiment reported in the manuscript.

---

## Contents

    .
    ├── README.md                        # this file
    ├── requirements.txt                 # Python dependencies (pinned)
    ├── data/                            # raw dataset placement (NOT redistribu
    ├── scripts/                         # all experiment code
    │   ├── preprocessing.py             # unified loaders for UNSW-NB15, NSL-KD
    │   ├── feature_alignment.py         # semantic alignment for cross-dataset
    │   ├── models.py                    # XGBoost, LightGBM, CNN-BiLSTM, RF, SV
    │   ├── evaluation.py                # metrics helpers (accuracy, F1, AUC, etc.)
    │   ├── run_benchmark.py             # 7-model benchmark across 3 datasets
    │   ├── run_awe.py                   # AWE-IDS weight + threshold optimization
    │   ├── run_bias_analysis.py         # validation_split, seed, leakage (A1/A
    │   ├── run_cross_dataset.py         # cross-dataset transfer (index-based truncation)
    │   ├── run_cross_dataset_aligned.py # cross-dataset: semantic alignment + d
    │   ├── run_grid_sensitivity.py      # AWE-IDS grid resolution sensitivity
    │   ├── run_cnn_bilstm_tuning.py     # CNN-BiLSTM hyperparameter sweep
    │   ├── run_statistical_tests.py     # Friedman + Wilcoxon + Nemenyi
    │   └── run_ablation.py              # 5-condition ablation
    ├── results/                         # CSV results for every experiment in the paper
    └── docs/                            # supplementary documentation

## Reproduction

### 1. Environment

    python3.10 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Tested with Python 3.10 and 3.14 on macOS Darwin.

### 2. Datasets

The three datasets are not redistributed (license / size). Place them as follows

| Dataset     | Files                                                     | Sour                         |
|-------------|----------------------------------------------------------|--------------------------------------------------------|
| UNSW-NB15   | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` | httprojects/unsw-nb15-dataset |
| NSL-KDD     | `KDDTrain+.txt`, `KDDTest+.txt`                           | https://www.unb.ca/cic/datasets/nsl.html               |
| CIC-IDS2017 | `*.pcap_ISCX.csv` (8 daily CSVs)                          | https/ids-2017.html          |

UNSW-NB15 official train/test split is used as-is. NSL-KDD `KDDTrain+` / `KDDTes. CIC-IDS2017 is subsampled stratified to 500,000 flows with `random_state=42`;the exact subsampling is reproduced inside `preprocessing.py:load_cicids2017`.

### 3. Running experiments

All experiments are deterministic at `random_state=42` unless noted. Wall-clock estimates are for an Apple M1 / 16 GB.

    # Benchmark (7 models on 3 datasets) -- ~90 min
    python scripts/run_benchmark.py

    # AWE-IDS optimization (paper baseline numbers) -- ~30 min
    python scripts/run_awe.py

    # Bias analysis (A1: validation_split, A2: seed, A3: leakage) -- ~20 min
    python scripts/run_bias_analysis.py

    # Statistical tests (Friedman, Wilcoxon, Nemenyi) -- ~15 min
    python scripts/run_statistical_tests.py

    # Ablation (5 conditions) -- ~15 min
    python scripts/run_ablation.py

    # Cross-dataset: semantic alignment + dual calibration -- ~45 min
    python scripts/run_cross_dataset_aligned.py --pair both

    # Cross-dataset: original truncation (for comparison) -- ~30 min
    python scripts/run_cross_dataset.py --align-method pad_truncate --out-suffix

    # Cross-dataset: zero-padding alignment (sensitivity) -- ~30 min
    python scripts/run_cross_dataset.py --align-method pad --out-suffix padding

    # AWE-IDS grid resolution sensitivity -- ~5 min
    python scripts/run_grid_sensitivity.py

    # CNN-BiLSTM hyperparameter sweep -- ~30 min
    python scripts/run_cnn_bilstm_tuning.py --dataset unsw-nb15

All scripts write CSVs into `results/`. Pre-computed results from our runs are already present there for direct comparison.

### 4. Results layout

| File                                | Paper section          | Description                                              |
|-------------------------------------|------------------------|--------------------------------|
| `benchmark_holdout_results.csv`     | Sec. 5 (Benchmark)     | 7 models x 3 datasets, single train/test split           |
| `awe_comparison_results.csv`        | Sec. 5 (AWE-IDS)       | AWE-IDS vs simpain             |
| `a1_bias_results.csv`               | Sec. 5 (A1)            | validation_split inflation                               |
| `a2_seed_sensitivity_*.csv`         | Sec. 5 (A2)            | seed sensitivit                |
| `a3_leakage_results.csv`            | Sec. 5 (A3)            | preprocessing leakage (negligible effect)               |
| `ablation_results.csv`              | Sec. 5 (Ablation)      | 5 conditions                   |
| `cross_dataset_results.csv`         | Sec. 5 (Cross-dataset) | Original truncation-based                                |
| `cross_dataset_results_padding.csv` | Sec. 5 (sensitivity)   | Zero-padding al                |
| `cross_dataset_aligned.csv`         | Sec. 5 (alignment)     | Semantic alignment, single-mode AWE                      |
| `cross_dataset_aligned_dual.csv`    | Sec. 5 (alignment)     | Semantic alignmtion            |
| `grid_sensitivity.csv`              | Sec. 5 (grid)          | AWE-IDS optimum stability across grid resolutions        |
| `cnn_bilstm_tuning_unsw-nb15.csv`   | Sec. 5 (CNN-BiLSTM)    | CNN-BiLSTM weigter sweep       |
| `statistical_*.csv`                 | Sec. 5 (Stats)         | Friedman chi2=50.08, p<1e-6; Wilcoxon; Nemenyi; per-fold |
| `table_T*.csv`                      | Tables 1-9             | One CSV per pub                |

### 5. Determinism notes

- Random seed 42 throughout.
- CIC-IDS2017 subsampling is stratified `train_test_split(stratify=label, random_state=42)`; the exact 500,000-flow subset is reproducible from the raw 2.8M-flow concatenation.
- The deployable AWE-IDS calibration split (80/20) uses the same seed.
- SMOTE uses `random_state=42`.
- TensorFlow/Keras runs (CNN-BiLSTM) use `tf.random.set_seed(42)` and `np.randomterminism is bounded but not zero.

### 6. Hardware notes

CIC-IDS2017 full pipeline (subsampling + SMOTE + 5-fold) requires ~16 GB RAM. CNlowest component (~2 min/epoch on UNSW; ~5 min on CIC subsample).

---

## License

Code: MIT (see LICENSE). Result CSVs: CC-BY 4.0. Datasets are NOT included; obtain from their respective official sources.
