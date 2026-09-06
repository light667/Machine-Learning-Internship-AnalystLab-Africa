"""
Baseline model integration for the HealthConnect no-show pipeline.

This is intentionally lightweight: the goal for Week 5 is a working,
versioned integration structure (train -> save -> load -> predict), not the
best-performing model. A local, file-based stand-in for a model registry is
used here in place of dedicated MLOps tooling, per the Week 4 design.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from src import config

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = [
    "appointment_id", "patient_id", "booking_date", "appointment_date",
    "appointment_outcome", "target",
]


class ModelError(Exception):
    """Raised on model training, saving, or loading failures."""


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except identifiers, raw dates, and the target itself."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def train_baseline_model(train_df: pd.DataFrame) -> tuple[LogisticRegression, list[str]]:
    """
    Train a Logistic Regression baseline: fast, interpretable, and a
    reasonable first checkpoint before more complex models are considered
    in a later iteration.
    """
    if "target" not in train_df.columns:
        raise ModelError("train_df is missing the 'target' column -- run preprocessing.clean() first.")

    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    if y_train.nunique() < 2:
        raise ModelError(
            f"Training target has only {y_train.nunique()} class(es); "
            "cannot train a classifier. Check the temporal split date."
        )

    model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    model.fit(X_train, y_train)
    logger.info("Trained baseline LogisticRegression on %d rows, %d features", len(X_train), len(feature_cols))
    return model, feature_cols


def evaluate_model(model, test_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Evaluate a trained model on the held-out test set."""
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": cm,
        "n_test_rows": len(test_df),
    }
    logger.info("Baseline evaluation: %s", metrics)
    return metrics


def save_model(model, feature_cols: list[str], metrics: dict, version: str | None = None) -> Path:
    """
    Save the model artifact plus a metadata sidecar file (features, metrics,
    timestamp, version) -- a minimal, file-based stand-in for a model
    registry entry, so any prediction can later be traced back to exactly
    which model and metrics produced it.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = version or datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")

    model_path = config.MODELS_DIR / f"baseline_logreg_{version}.joblib"
    metadata_path = config.MODELS_DIR / f"baseline_logreg_{version}.json"

    joblib.dump(model, model_path)
    metadata = {
        "version": version,
        "model_type": "LogisticRegression",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_cols,
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved model to %s (metadata: %s)", model_path, metadata_path)
    return model_path


def load_model(model_path: str | Path):
    """Load a previously saved model artifact."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
