"""Training utilities for CNN-LSTM-Attention malware classifier experiments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


logger = logging.getLogger(__name__)


MetricsDict = Dict[str, float]


@dataclass
class TrainingResult:
    """Container for trained model artifacts."""

    model: tf.keras.Model
    scaler: StandardScaler
    history: tf.keras.callbacks.History | None = None
    history_path: Path | None = None


def get_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 4,
    random_state: int = 42,
) -> StratifiedKFold:
    """Create a stratified K-fold splitter with shuffling."""

    logger.debug(
        "Creating StratifiedKFold with %d splits and random_state=%d",
        n_splits,
        random_state,
    )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _prepare_checkpoint(experiment: str, fold: int) -> Path:
    checkpoint_dir = Path("results") / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{experiment}_{fold}.h5"


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsDict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_one_fold(
    model_fn: Callable[..., tf.keras.Model],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    patience: int = 3,
) -> Tuple[MetricsDict, tf.keras.callbacks.History, StandardScaler]:
    """Train a single fold with scaling, callbacks, and metric computation.

    The calling function may set ``train_one_fold._experiment_name`` and
    ``train_one_fold._fold_index`` attributes to control checkpoint naming.
    Defaults are ``experiment`` and ``0`` respectively if unset.
    """

    experiment = getattr(train_one_fold, "_experiment_name", "experiment")
    fold_index = getattr(train_one_fold, "_fold_index", 0)
    logger.info("Training fold %d for experiment '%s'", fold_index, experiment)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = model_fn(input_dim=X_train.shape[1])

    checkpoint_path = _prepare_checkpoint(experiment, fold_index)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=2,
    )

    val_pred_prob = model.predict(X_val_scaled, batch_size=batch_size, verbose=0)
    val_pred = (val_pred_prob >= 0.5).astype(int).ravel()
    metrics = _compute_metrics(y_val, val_pred)
    logger.info("Fold %d metrics: %s", fold_index, json.dumps(metrics, indent=2))
    return metrics, history, scaler


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable[..., tf.keras.Model],
    n_splits: int = 4,
    epochs: int = 20,
    batch_size: int = 256,
) -> Tuple[List[MetricsDict], MetricsDict]:
    """Run stratified cross-validation and aggregate metrics."""

    skf = get_kfold_splits(X, y, n_splits=n_splits)
    fold_metrics: List[MetricsDict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        train_one_fold._experiment_name = "cv"
        train_one_fold._fold_index = fold_idx
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        metrics, _, _ = train_one_fold(
            model_fn,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
        )
        fold_metrics.append(metrics)

    aggregated_metrics = {
        key: float(np.mean([fold[key] for fold in fold_metrics])) for key in fold_metrics[0]
    }
    logger.info("Aggregated CV metrics: %s", json.dumps(aggregated_metrics, indent=2))
    return fold_metrics, aggregated_metrics


def train_final_model(
    model_fn: Callable[..., tf.keras.Model],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    patience: int = 3,
    experiment_name: str = "final",
) -> TrainingResult:
    """Train model on the full training set with optional validation."""

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = model_fn(input_dim=X_train.shape[1])

    callbacks: List[tf.keras.callbacks.Callback] = []
    checkpoint_path = _prepare_checkpoint(experiment_name, 0)
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss" if X_val is not None else "loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    )

    if X_val is not None and y_val is not None:
        X_val_scaled = scaler.transform(X_val)
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
        validation_data: tuple[np.ndarray, np.ndarray] | None = (X_val_scaled, y_val)
    else:
        validation_data = None

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    history_path = Path("results") / f"{experiment_name}_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump(history.history, fp, indent=2)
    logger.info("Saved training history to %s", history_path)

    return TrainingResult(model=model, scaler=scaler, history=history, history_path=history_path)


def evaluate_model(
    model: tf.keras.Model,
    scaler: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
) -> MetricsDict:
    """Evaluate a trained model with a provided scaler."""

    X_scaled = scaler.transform(X)
    y_pred_prob = model.predict(X_scaled, batch_size=batch_size, verbose=0)
    y_pred = (y_pred_prob >= 0.5).astype(int).ravel()
    metrics = _compute_metrics(y, y_pred)
    logger.debug("Evaluation metrics: %s", json.dumps(metrics, indent=2))
    return metrics


__all__ = [
    "TrainingResult",
    "get_kfold_splits",
    "train_one_fold",
    "run_cross_validation",
    "train_final_model",
    "evaluate_model",
]
