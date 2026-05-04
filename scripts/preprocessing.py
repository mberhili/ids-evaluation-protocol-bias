"""
Unified Preprocessing Module for IDS Datasets.
Supports: UNSW-NB15, NSL-KDD, CIC-IDS2017.
"""

import numpy as np
import pandas as pd
import warnings
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# NSL-KDD attack type mapping to binary
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
]


def _clean_dataframe(X, y):
    """Common cleaning: handle NaN, Inf, encode categoricals, scale."""
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    X = X.replace([np.inf, -np.inf], 0)

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    scaler = MinMaxScaler()
    feature_names = X.columns.tolist()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    return X_scaled.values, y, feature_names, scaler, label_encoders


def load_unsw_nb15(use_all_features=True):
    """Load and preprocess UNSW-NB15 dataset.

    Returns:
        dict with keys: X_train, y_train, X_test, y_test, feature_names, n_features, dataset_name
    """
    train_path = os.path.join(BASE_DIR, 'UNSW_NB15_training-set.csv')
    test_path = os.path.join(BASE_DIR, 'UNSW_NB15_testing-set.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Drop id and attack_cat (keep label)
    drop_cols = ['id', 'attack_cat']
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    y_train = train_df['label'].values
    y_test = test_df['label'].values
    X_train = train_df.drop('label', axis=1).copy()
    X_test = test_df.drop('label', axis=1).copy()

    # Clean training data
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in numerical_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    for col in categorical_cols:
        mode_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)

    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str).apply(
            lambda x: x if x in le.classes_ else le.classes_[0]
        ))
        label_encoders[col] = le

    scaler = MinMaxScaler()
    feature_names = X_train.columns.tolist()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'feature_names': feature_names,
        'n_features': X_train_scaled.shape[1],
        'dataset_name': 'UNSW-NB15',
        'n_train': len(y_train),
        'n_test': len(y_test),
    }


def load_nsl_kdd():
    """Load and preprocess NSL-KDD dataset.

    Returns:
        dict with keys: X_train, y_train, X_test, y_test, feature_names, n_features, dataset_name
    """
    train_path = os.path.join(BASE_DIR, 'datasets', 'nsl-kdd', 'KDDTrain+.txt')
    test_path = os.path.join(BASE_DIR, 'datasets', 'nsl-kdd', 'KDDTest+.txt')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"NSL-KDD not found. Run download_datasets.py first.\n{train_path}")

    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test_df = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)

    # Binary label: normal=0, attack=1
    train_df['label'] = (train_df['attack_type'] != 'normal').astype(int)
    test_df['label'] = (test_df['attack_type'] != 'normal').astype(int)

    # Drop attack_type and difficulty_level
    drop_cols = ['attack_type', 'difficulty_level']
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    y_train = train_df['label'].values
    y_test = test_df['label'].values
    X_train = train_df.drop('label', axis=1).copy()
    X_test = test_df.drop('label', axis=1).copy()

    # Clean
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in numerical_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    for col in categorical_cols:
        mode_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)

    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

    scaler = MinMaxScaler()
    feature_names = X_train.columns.tolist()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'feature_names': feature_names,
        'n_features': X_train_scaled.shape[1],
        'dataset_name': 'NSL-KDD',
        'n_train': len(y_train),
        'n_test': len(y_test),
    }


def load_cicids2017(max_samples=500000):
    """Load and preprocess CIC-IDS2017 dataset.

    Args:
        max_samples: Maximum samples to use (dataset is ~2.8M, we subsample for tractability).

    Returns:
        dict with keys: X_train, y_train, X_test, y_test, feature_names, n_features, dataset_name
    """
    data_dir = os.path.join(BASE_DIR, 'datasets', 'cicids2017')

    csv_files = [
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv',
    ]

    dfs = []
    for fname in csv_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, encoding='utf-8', low_memory=False)
            df.columns = df.columns.str.strip()
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"CIC-IDS2017 not found. Place CSV files in {data_dir}\n"
            "Download from: https://www.kaggle.com/datasets/cicdataset/cicids2017"
        )

    data = pd.concat(dfs, ignore_index=True)

    # Binary label
    label_col = 'Label'
    if label_col not in data.columns:
        for c in data.columns:
            if 'label' in c.lower():
                label_col = c
                break

    data['label'] = (data[label_col].str.strip() != 'BENIGN').astype(int)
    data = data.drop(columns=[label_col])

    # Remove problematic columns
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP',
                 'Destination Port', 'Timestamp']
    data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')

    # Handle NaN/Inf
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=1, thresh=int(0.5 * len(data)))  # Drop cols with >50% NaN
    data = data.dropna()

    # Subsample if needed (stratified)
    if len(data) > max_samples:
        from sklearn.model_selection import train_test_split
        data, _ = train_test_split(data, train_size=max_samples,
                                   stratify=data['label'], random_state=42)

    y = data['label'].values
    X = data.drop('label', axis=1).copy()

    # Encode categoricals
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Handle remaining NaN
    X = X.fillna(0)

    # 80/20 stratified split
    from sklearn.model_selection import train_test_split
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'feature_names': feature_names,
        'n_features': X_train_scaled.shape[1],
        'dataset_name': 'CIC-IDS2017',
        'n_train': len(y_train),
        'n_test': len(y_test),
    }


def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE oversampling to balance classes."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def get_stratified_kfold(n_splits=5, random_state=42):
    """Get a StratifiedKFold splitter with fixed seed."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def load_dataset(name, **kwargs):
    """Universal dataset loader.

    Args:
        name: One of 'unsw-nb15', 'nsl-kdd', 'cicids2017'

    Returns:
        dict with dataset info
    """
    loaders = {
        'unsw-nb15': load_unsw_nb15,
        'nsl-kdd': load_nsl_kdd,
        'cicids2017': load_cicids2017,
    }
    name_lower = name.lower().replace('_', '-')
    if name_lower not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")
    return loaders[name_lower](**kwargs)


if __name__ == '__main__':
    # Test loading
    print("Testing dataset loaders...")

    for name in ['unsw-nb15', 'nsl-kdd', 'cicids2017']:
        try:
            data = load_dataset(name)
            print(f"\n{data['dataset_name']}:")
            print(f"  Train: {data['n_train']} samples, {data['n_features']} features")
            print(f"  Test:  {data['n_test']} samples")
            print(f"  Train class distribution: Normal={sum(data['y_train']==0)}, Attack={sum(data['y_train']==1)}")
            print(f"  Test class distribution:  Normal={sum(data['y_test']==0)}, Attack={sum(data['y_test']==1)}")
        except FileNotFoundError as e:
            print(f"\n{name}: {e}")
