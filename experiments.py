"""Experiment workflows for the CNN-LSTM-Attention malware classifier."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model_def import build_cnn_lstm_attention_model
from train_utils import (
    MetricsDict,
    evaluate_model,
    run_cross_validation,
    train_final_model,
)


np.random.seed(42)
tf.random.set_seed(42)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def load_ember_vectorized(ember_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load vectorized EMBER dataset features.

    Parameters
    ----------
    ember_root:
        Path to the root directory containing the EMBER dataset.

    Returns
    -------
    Tuple containing ``(X_train, y_train, X_test, y_test)`` as ``np.float32`` / ``np.int32``.

    Raises
    ------
    RuntimeError
        If the ``ember`` package is not installed or vectorized features cannot be created.
    """

    try:
        import ember
    except ImportError as exc:  # pragma: no cover - guidance for users
        raise RuntimeError(
            "The 'ember' package is required. Install via 'pip install git+https://github.com/elastic/ember.git' "
            "or clone the repository and run 'python setup.py install'."
        ) from exc

    ember_path = Path(ember_root)
    if not ember_path.exists():
        raise RuntimeError(f"EMBER root '{ember_root}' does not exist.")

    try:
        X_train, y_train, X_test, y_test = ember.read_vectorized_features(str(ember_path))
    except FileNotFoundError:
        logger.warning("Vectorized features not found. Creating them (this may take a while)...")
        try:
            ember.create_vectorized_features(str(ember_path))
            X_train, y_train, X_test, y_test = ember.read_vectorized_features(str(ember_path))
        except Exception as exc:  # pragma: no cover - creation failure is environment specific
            raise RuntimeError(
                "Failed to create vectorized features. Ensure EMBER raw data is available and that LIEF==0.9.0 "
                "is installed."
            ) from exc

    return (
        X_train.astype(np.float32),
        y_train.astype(np.int32),
        X_test.astype(np.float32),
        y_test.astype(np.int32),
    )


def _save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
    logger.info("Saved JSON to %s", path)


def run_cv_experiment(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 4,
    epochs: int = 20,
    batch_size: int = 256,
) -> Tuple[list[MetricsDict], MetricsDict]:
    """Execute cross-validation experiment."""

    logger.info("Starting %d-fold cross-validation", n_splits)
    fold_metrics, aggregated_metrics = run_cross_validation(
        X,
        y,
        build_cnn_lstm_attention_model,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
    )

    _save_json({"folds": fold_metrics, "aggregate": aggregated_metrics}, Path("results/cv_metrics.json"))
    return fold_metrics, aggregated_metrics


def run_internal_test_experiment(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.1,
    epochs: int = 30,
    batch_size: int = 256,
) -> Dict[str, MetricsDict | str]:
    """Train final model with internal holdout evaluation."""

    logger.info("Splitting data with test_size=%s", test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=42,
    )
    result = train_final_model(
        build_cnn_lstm_attention_model,
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        experiment_name="internal",
    )
    metrics = evaluate_model(result.model, result.scaler, X_val, y_val, batch_size=batch_size)

    model_dir = Path("results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / "internal_final_weights.h5"
    scaler_path = model_dir / "internal_scaler.joblib"
    result.model.save_weights(weights_path)
    joblib.dump(result.scaler, scaler_path)
    logger.info("Saved model weights to %s and scaler to %s", weights_path, scaler_path)

    payload = {
        "metrics": metrics,
        "weights_path": str(weights_path),
        "scaler_path": str(scaler_path),
    }
    _save_json(payload, Path("results/internal_metrics.json"))
    return payload


def run_external_generalization_experiment(
    X_2018: np.ndarray,
    y_2018: np.ndarray,
    X_2024: np.ndarray,
    y_2024: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
) -> Dict[str, MetricsDict | str]:
    """Train on EMBER 2018 and evaluate on EMBER 2024 without refitting the scaler."""

    logger.info("Running external generalization experiment")
    X_train, X_val, y_train, y_val = train_test_split(
        X_2018,
        y_2018,
        test_size=0.1,
        stratify=y_2018,
        random_state=42,
    )
    result = train_final_model(
        build_cnn_lstm_attention_model,
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        experiment_name="external",
    )

    metrics_2018_val = evaluate_model(result.model, result.scaler, X_val, y_val, batch_size=batch_size)
    metrics_2024 = evaluate_model(result.model, result.scaler, X_2024, y_2024, batch_size=batch_size)

    model_dir = Path("results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / "external_final_weights.h5"
    scaler_path = model_dir / "external_scaler.joblib"
    result.model.save_weights(weights_path)
    joblib.dump(result.scaler, scaler_path)
    logger.info("Saved model weights to %s and scaler to %s", weights_path, scaler_path)

    payload: Dict[str, Any] = {
        "metrics_2018_val": metrics_2018_val,
        "metrics_2024": metrics_2024,
        "weights_path": str(weights_path),
        "scaler_path": str(scaler_path),
    }
    _save_json(payload, Path("results/external_metrics.json"))
    return payload


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices={"cv", "internal", "external"}, required=True)
    parser.add_argument("--ember_root", type=str, help="Path to EMBER dataset root", required=False)
    parser.add_argument("--ember_root_2024", type=str, help="Path to EMBER 2024 root for external test")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_splits", type=int, default=4)
    parser.add_argument("--test_size", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = _parse_args(argv)

    if args.run == "cv":
        if not args.ember_root:
            raise SystemExit("--ember_root is required for CV experiment")
        X_train, y_train, *_ = load_ember_vectorized(args.ember_root)
        run_cv_experiment(
            X_train,
            y_train,
            n_splits=args.n_splits,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.run == "internal":
        if not args.ember_root:
            raise SystemExit("--ember_root is required for internal experiment")
        X_train, y_train, *_ = load_ember_vectorized(args.ember_root)
        run_internal_test_experiment(
            X_train,
            y_train,
            test_size=args.test_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.run == "external":
        if not args.ember_root or not args.ember_root_2024:
            raise SystemExit("Both --ember_root and --ember_root_2024 are required for external experiment")
        X_train, y_train, *_ = load_ember_vectorized(args.ember_root)
        X_2024_train, y_2024_train, X_2024_test, y_2024_test = load_ember_vectorized(args.ember_root_2024)
        X_2024 = np.concatenate([X_2024_train, X_2024_test], axis=0)
        y_2024 = np.concatenate([y_2024_train, y_2024_test], axis=0)
        run_external_generalization_experiment(
            X_train,
            y_train,
            X_2024,
            y_2024,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:  # pragma: no cover - argparse guarantees choices
        raise SystemExit(f"Unknown run type: {args.run}")


if __name__ == "__main__":
    """Example usage when running as a script.

    To run 4-fold cross-validation using EMBER 2018 vectorized features:
        python experiments.py --run cv --ember_root /data/ember2018 --epochs 20

    To train a final model with an internal 90/10 split for evaluation:
        python experiments.py --run internal --ember_root /data/ember2018 --epochs 30 --test_size 0.1

    To evaluate generalization on EMBER 2024 (using scaler from 2018):
        python experiments.py --run external --ember_root /data/ember2018 \
            --ember_root_2024 /data/ember2024 --epochs 30

    Results and artifacts (metrics JSON, model weights, scalers) are stored under the
    ``results/`` directory. Uncomment the desired ``run_*`` call below when using as a
    library instead of CLI.
    """

    # Example programmatic usage:
    # ember_root_2018 = "/data/ember2018"
    # ember_root_2024 = "/data/ember2024"
    # X_train_2018, y_train_2018, X_test_2018, y_test_2018 = load_ember_vectorized(ember_root_2018)
    # run_cv_experiment(X_train_2018, y_train_2018)
    # run_internal_test_experiment(np.concatenate([X_train_2018, X_test_2018]),
    #                              np.concatenate([y_train_2018, y_test_2018]))
    # X_2024_train, y_2024_train, X_2024_test, y_2024_test = load_ember_vectorized(ember_root_2024)
    # run_external_generalization_experiment(
    #     X_train_2018,
    #     y_train_2018,
    #     np.concatenate([X_2024_train, X_2024_test]),
    #     np.concatenate([y_2024_train, y_2024_test]),
    # )

    main(sys.argv[1:])
