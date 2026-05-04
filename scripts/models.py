"""
Model Factory for IDS Classification.
7 models: XGBoost, LightGBM, CNN-BiLSTM, Random Forest, SVM, MLP, KNN.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Bidirectional, LSTM,
    Dropout, Dense, BatchNormalization, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_xgboost(random_state=42):
    return XGBClassifier(
        n_estimators=2000,
        max_depth=15,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss',
    )


def create_lightgbm(random_state=42):
    return LGBMClassifier(
        n_estimators=2000,
        max_depth=18,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def create_random_forest(random_state=42):
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )


def create_svm(random_state=42):
    return SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        random_state=random_state,
    )


def create_mlp(random_state=42):
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
    )


def create_knn():
    return KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1,
    )


# ============================================================
# CNN-BiLSTM (Keras)
# ============================================================

class CNNBiLSTMWrapper:
    """Sklearn-compatible wrapper for CNN-BiLSTM."""

    def __init__(self, n_features, epochs=30, batch_size=64, random_state=42):
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.classes_ = np.array([0, 1])

    def _build_model(self):
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        model = Sequential([
            Conv1D(64, 3, activation='relu', padding='same',
                   input_shape=(self.n_features, 1)),
            BatchNormalization(),
            MaxPooling1D(2),
            Conv1D(32, 3, activation='relu', padding='same'),
            MaxPooling1D(2),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X, y):
        self.model = self._build_model()
        X_3d = X.reshape(-1, self.n_features, 1)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ]

        self.model.fit(
            X_3d, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict(self, X):
        X_3d = X.reshape(-1, self.n_features, 1)
        proba = self.model.predict(X_3d, verbose=0).flatten()
        return (proba > 0.5).astype(int)

    def predict_proba(self, X):
        X_3d = X.reshape(-1, self.n_features, 1)
        proba_pos = self.model.predict(X_3d, verbose=0).flatten()
        return np.column_stack([1 - proba_pos, proba_pos])


def create_cnn_bilstm(n_features, random_state=42):
    return CNNBiLSTMWrapper(n_features=n_features, random_state=random_state)


# ============================================================
# MODEL FACTORY
# ============================================================

MODEL_NAMES = ['XGBoost', 'LightGBM', 'CNN-BiLSTM', 'RandomForest', 'SVM', 'MLP', 'KNN']

# Models that need n_features at creation time
NEEDS_N_FEATURES = {'CNN-BiLSTM'}

# Models that are slow on large datasets (>100K samples)
SLOW_MODELS = {'SVM', 'KNN'}


def create_model(name, n_features=None, random_state=42):
    """Factory function to create a model by name.

    Args:
        name: One of MODEL_NAMES
        n_features: Required for CNN-BiLSTM
        random_state: Random seed

    Returns:
        model instance (sklearn-compatible)
    """
    factories = {
        'XGBoost': lambda: create_xgboost(random_state),
        'LightGBM': lambda: create_lightgbm(random_state),
        'CNN-BiLSTM': lambda: create_cnn_bilstm(n_features, random_state),
        'RandomForest': lambda: create_random_forest(random_state),
        'SVM': lambda: create_svm(random_state),
        'MLP': lambda: create_mlp(random_state),
        'KNN': lambda: create_knn(),
    }

    if name not in factories:
        raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}")

    if name in NEEDS_N_FEATURES and n_features is None:
        raise ValueError(f"{name} requires n_features parameter")

    return factories[name]()


def get_model_list(include_slow=True):
    """Get list of models to benchmark.

    Args:
        include_slow: If False, exclude SVM and KNN (slow on large datasets)
    """
    if include_slow:
        return MODEL_NAMES.copy()
    return [m for m in MODEL_NAMES if m not in SLOW_MODELS]
