"""
Semantic feature alignment for cross-dataset IDS experiments.

Implements the Pivot B design (UNSW-NB15 as bridge between paradigms):
- UNSW <-> CIC-IDS2017: 14 features (flow-based paradigm match)
- UNSW <-> NSL-KDD: 10 features (mixed paradigm match)
- NSL <-> CIC: not reported (only 3 strictly alignable, paradigms incompatible)

Each mapping is (source_name, target_name, match_type):
- "strict": same physical quantity, same direction
- "quasi-strict": same concept, slight definitional difference
- "strict-categorical": same conceptual category but different vocabularies
- "approximate": same concept but different windowing or metric type
"""

UNSW_CIC_MAPPING = [
    ("dur",              "Flow Duration",                "strict"),
    ("spkts",            "Total Fwd Packets",            "strict"),
    ("dpkts",            "Total Backward Packets",       "strict"),
    ("sbytes",           "Total Length of Fwd Packets",  "strict"),
    ("dbytes",           "Total Length of Bwd Packets",  "strict"),
    ("rate",             "Flow Packets/s",               "strict"),
    ("sinpkt",           "Fwd IAT Mean",                 "strict"),
    ("dinpkt",           "Bwd IAT Mean",                 "strict"),
    ("sjit",             "Fwd IAT Std",                  "quasi-strict"),
    ("djit",             "Bwd IAT Std",                  "quasi-strict"),
    ("swin",             "Init_Win_bytes_forward",       "strict"),
    ("dwin",             "Init_Win_bytes_backward",      "strict"),
    ("smean",            "Fwd Packet Length Mean",       "strict"),
    ("dmean",            "Bwd Packet Length Mean",       "strict"),
]

UNSW_NSL_MAPPING = [
    ("dur",              "duration",                     "strict"),
    ("proto",            "protocol_type",                "strict-categorical"),
    ("service",          "service",                      "strict-categorical"),
    ("state",            "flag",                         "strict-categorical"),
    ("sbytes",           "src_bytes",                    "strict"),
    ("dbytes",           "dst_bytes",                    "strict"),
    ("is_sm_ips_ports",  "land",                         "strict"),
    ("ct_dst_ltm",       "dst_host_count",               "approximate"),
    ("ct_srv_dst",       "dst_host_srv_count",           "approximate"),
    ("ct_src_dport_ltm", "dst_host_same_src_port_rate",  "approximate"),
]

# Unified protocol vocabulary (used for both UNSW and NSL-KDD).
# Values not in the unified set are mapped to "other".
PROTO_UNIFIED = {
    "tcp": 0,
    "udp": 1,
    "icmp": 2,
    "other": 3,
}


def get_unsw_indices(feature_names, pair):
    """Return column indices in UNSW feature_names for the aligned subset.

    Args:
        feature_names: list of column names in UNSW X array
        pair: 'unsw_cic' or 'unsw_nsl'

    Returns:
        list of indices in the same order as the mapping definition
    """
    mapping = UNSW_CIC_MAPPING if pair == "unsw_cic" else UNSW_NSL_MAPPING
    indices = []
    for unsw_name, _, _ in mapping:
        if unsw_name not in feature_names:
            raise KeyError(f"UNSW feature '{unsw_name}' missing in dataset")
        indices.append(feature_names.index(unsw_name))
    return indices


def get_target_indices(feature_names, pair):
    """Return column indices in the target dataset (NSL or CIC) for aligned subset."""
    mapping = UNSW_CIC_MAPPING if pair == "unsw_cic" else UNSW_NSL_MAPPING
    indices = []
    for _, target_name, _ in mapping:
        if target_name not in feature_names:
            raise KeyError(f"Target feature '{target_name}' missing in dataset")
        indices.append(feature_names.index(target_name))
    return indices


def aligned_feature_names(pair):
    """Human-readable names for the aligned subset (used in result CSVs)."""
    mapping = UNSW_CIC_MAPPING if pair == "unsw_cic" else UNSW_NSL_MAPPING
    return [f"{src}<->{tgt}" for src, tgt, _ in mapping]


def encode_proto_unified(values):
    """Map protocol strings to unified integer codes.

    Accepts a pandas Series or array of strings; returns an int array.
    """
    import numpy as np
    return np.array([PROTO_UNIFIED.get(str(v).lower().strip(), PROTO_UNIFIED["other"])
                     for v in values], dtype=int)


def _load_unsw_raw():
    """Load UNSW-NB15 raw with categoricals as strings (no encoding/scaling yet)."""
    import os
    import numpy as np
    import pandas as pd
    base = os.path.dirname(os.path.abspath(__file__))
    train_df = pd.read_csv(os.path.join(base, "UNSW_NB15_training-set.csv"))
    test_df = pd.read_csv(os.path.join(base, "UNSW_NB15_testing-set.csv"))
    drop_cols = ["id", "attack_cat"]
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    X_train = train_df.drop("label", axis=1).copy()
    X_test = test_df.drop("label", axis=1).copy()
    for col in X_train.select_dtypes(include=["int64", "float64"]).columns:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    return X_train, X_test, y_train, y_test


def _load_nsl_raw():
    """Load NSL-KDD raw with categoricals as strings."""
    import os
    import numpy as np
    import pandas as pd
    base = os.path.dirname(os.path.abspath(__file__))
    cols = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type", "difficulty_level",
    ]
    train_df = pd.read_csv(os.path.join(base, "datasets/nsl-kdd/KDDTrain+.txt"),
                           header=None, names=cols)
    test_df = pd.read_csv(os.path.join(base, "datasets/nsl-kdd/KDDTest+.txt"),
                          header=None, names=cols)
    train_df["label"] = (train_df["attack_type"] != "normal").astype(int)
    test_df["label"] = (test_df["attack_type"] != "normal").astype(int)
    train_df = train_df.drop(columns=["attack_type", "difficulty_level"])
    test_df = test_df.drop(columns=["attack_type", "difficulty_level"])
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    X_train = train_df.drop("label", axis=1).copy()
    X_test = test_df.drop("label", axis=1).copy()
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    return X_train, X_test, y_train, y_test


def _load_cic_raw(max_samples=500000, random_state=42):
    """Load CIC-IDS2017 raw, subsampled."""
    import os
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "datasets/cicids2017")
    files = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
    ]
    dfs = []
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, encoding="utf-8", low_memory=False)
            df.columns = df.columns.str.strip()
            dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    label_col = "Label"
    data["label"] = (data[label_col].str.strip() != "BENIGN").astype(int)
    data = data.drop(columns=[label_col])
    drop_cols = ["Flow ID", "Source IP", "Source Port", "Destination IP",
                 "Destination Port", "Timestamp"]
    data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=1, thresh=int(0.5 * len(data)))
    data = data.dropna()
    if len(data) > max_samples:
        data, _ = train_test_split(data, train_size=max_samples,
                                   stratify=data["label"], random_state=random_state)
    y = data["label"].values
    X = data.drop("label", axis=1).copy()
    X = X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y)
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train, y_test


def load_aligned_pair(pair, random_state=42):
    """Load a dataset pair with semantically aligned features.

    For UNSW<->NSL pair, categorical proto/protocol_type uses unified encoding.
    Other categoricals (service, state/flag) are encoded per-dataset (vocabularies
    fundamentally differ, this reflects realistic deployment).

    Args:
        pair: 'unsw_cic' or 'unsw_nsl'

    Returns:
        dict with X/y arrays for UNSW (train/test) and the other dataset (train/test),
        all using the aligned feature subset, MinMaxScaled.
    """
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder

    if pair == "unsw_cic":
        unsw_X_tr, unsw_X_te, unsw_y_tr, unsw_y_te = _load_unsw_raw()
        cic_X_tr, cic_X_te, cic_y_tr, cic_y_te = _load_cic_raw(random_state=random_state)
        unsw_cols = [m[0] for m in UNSW_CIC_MAPPING]
        cic_cols = [m[1] for m in UNSW_CIC_MAPPING]
        unsw_X_tr = unsw_X_tr[unsw_cols].copy()
        unsw_X_te = unsw_X_te[unsw_cols].copy()
        cic_X_tr = cic_X_tr[cic_cols].copy()
        cic_X_te = cic_X_te[cic_cols].copy()
        # No categoricals in this pair (all 14 are numeric)
        scaler_u = MinMaxScaler().fit(unsw_X_tr.values)
        scaler_c = MinMaxScaler().fit(cic_X_tr.values)
        return {
            "pair": pair,
            "unsw_X_train": scaler_u.transform(unsw_X_tr.values),
            "unsw_X_test":  scaler_u.transform(unsw_X_te.values),
            "unsw_y_train": unsw_y_tr,
            "unsw_y_test":  unsw_y_te,
            "other_name":   "CIC-IDS2017",
            "other_X_train": scaler_c.transform(cic_X_tr.values),
            "other_X_test":  scaler_c.transform(cic_X_te.values),
            "other_y_train": cic_y_tr,
            "other_y_test":  cic_y_te,
            "feature_names": aligned_feature_names(pair),
        }

    elif pair == "unsw_nsl":
        unsw_X_tr, unsw_X_te, unsw_y_tr, unsw_y_te = _load_unsw_raw()
        nsl_X_tr, nsl_X_te, nsl_y_tr, nsl_y_te = _load_nsl_raw()
        unsw_cols = [m[0] for m in UNSW_NSL_MAPPING]
        nsl_cols = [m[1] for m in UNSW_NSL_MAPPING]
        unsw_X_tr = unsw_X_tr[unsw_cols].copy()
        unsw_X_te = unsw_X_te[unsw_cols].copy()
        nsl_X_tr = nsl_X_tr[nsl_cols].copy()
        nsl_X_te = nsl_X_te[nsl_cols].copy()

        # proto: unified encoding (matches across datasets)
        unsw_X_tr["proto"] = encode_proto_unified(unsw_X_tr["proto"])
        unsw_X_te["proto"] = encode_proto_unified(unsw_X_te["proto"])
        nsl_X_tr["protocol_type"] = encode_proto_unified(nsl_X_tr["protocol_type"])
        nsl_X_te["protocol_type"] = encode_proto_unified(nsl_X_te["protocol_type"])

        # service, state/flag: per-dataset LabelEncoder (vocabularies differ)
        for col in ["service", "state"]:
            le = LabelEncoder()
            combined = list(unsw_X_tr[col].astype(str)) + list(unsw_X_te[col].astype(str))
            le.fit(combined)
            unsw_X_tr[col] = le.transform(unsw_X_tr[col].astype(str))
            unsw_X_te[col] = le.transform(unsw_X_te[col].astype(str))

        for col in ["service", "flag"]:
            le = LabelEncoder()
            combined = list(nsl_X_tr[col].astype(str)) + list(nsl_X_te[col].astype(str))
            le.fit(combined)
            nsl_X_tr[col] = le.transform(nsl_X_tr[col].astype(str))
            nsl_X_te[col] = le.transform(nsl_X_te[col].astype(str))

        scaler_u = MinMaxScaler().fit(unsw_X_tr.values.astype(float))
        scaler_n = MinMaxScaler().fit(nsl_X_tr.values.astype(float))
        return {
            "pair": pair,
            "unsw_X_train": scaler_u.transform(unsw_X_tr.values.astype(float)),
            "unsw_X_test":  scaler_u.transform(unsw_X_te.values.astype(float)),
            "unsw_y_train": unsw_y_tr,
            "unsw_y_test":  unsw_y_te,
            "other_name":   "NSL-KDD",
            "other_X_train": scaler_n.transform(nsl_X_tr.values.astype(float)),
            "other_X_test":  scaler_n.transform(nsl_X_te.values.astype(float)),
            "other_y_train": nsl_y_tr,
            "other_y_test":  nsl_y_te,
            "feature_names": aligned_feature_names(pair),
        }

    else:
        raise ValueError(f"Unknown pair: {pair}")


if __name__ == "__main__":
    print("UNSW <-> CIC mapping:")
    for src, tgt, kind in UNSW_CIC_MAPPING:
        print(f"  [{kind:14s}] {src:20s} <-> {tgt}")
    print(f"  Total: {len(UNSW_CIC_MAPPING)} features")

    print("\nUNSW <-> NSL-KDD mapping:")
    for src, tgt, kind in UNSW_NSL_MAPPING:
        print(f"  [{kind:18s}] {src:20s} <-> {tgt}")
    print(f"  Total: {len(UNSW_NSL_MAPPING)} features")
