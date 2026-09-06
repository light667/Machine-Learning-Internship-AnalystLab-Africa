"""
Preprocessing / cleaning components for the HealthConnect appointment dataset.

Each function does one clearly-named transformation so the pipeline in
pipeline.py reads as a readable sequence of steps, and each step can be
unit-tested in isolation (see tests/test_preprocessing.py).
"""

import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def filter_to_modelling_scope(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict the dataset to the binary modelling scope defined in Week 4:
    No-Show vs Attended. Cancelled appointments are a distinct, already-known
    event and are excluded (see docs/PIPELINE.md for the rationale).
    """
    before = len(df)
    out = df[~df[config.TARGET_COLUMN].isin(config.EXCLUDED_OUTCOME_CLASSES)].copy()
    logger.info(
        "filter_to_modelling_scope: kept %d/%d rows (excluded classes: %s)",
        len(out), before, config.EXCLUDED_OUTCOME_CLASSES,
    )
    return out


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns identified as data leakage risks (see config.LEAKAGE_COLUMNS)."""
    present = [c for c in config.LEAKAGE_COLUMNS if c in df.columns]
    out = df.drop(columns=present)
    if present:
        logger.info("drop_leakage_columns: removed %s", present)
    return out


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are simple derivations of another retained column."""
    present = [c for c in config.DERIVED_REDUNDANT_COLUMNS if c in df.columns]
    out = df.drop(columns=present)
    if present:
        logger.info("drop_redundant_columns: removed %s", present)
    return out


def encode_reminder_channel_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode reminder_channel's missing values as an explicit "Not Sent" category
    rather than imputing them -- the missingness is logical (see validation.py),
    not a gap to fill in.
    """
    out = df.copy()
    out["reminder_channel"] = out["reminder_channel"].fillna("Not Sent")
    return out


def impute_genuine_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute the small amount of genuinely random missingness in
    distance_to_clinic_km using the median, and add a missingness indicator
    flag so the model can still learn from the fact that a value was missing.
    """
    out = df.copy()
    if "distance_to_clinic_km" in out.columns:
        was_missing = out["distance_to_clinic_km"].isna()
        median_value = out["distance_to_clinic_km"].median()
        out["distance_to_clinic_km"] = out["distance_to_clinic_km"].fillna(median_value)
        out["distance_to_clinic_km_was_missing"] = was_missing.astype(int)
        logger.info(
            "impute_genuine_missing_values: filled %d missing distance_to_clinic_km "
            "values with median=%.2f",
            int(was_missing.sum()), median_value,
        )
    return out


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the target as a binary integer column: 1 = No-Show, 0 = Attended."""
    out = df.copy()
    out["target"] = (out[config.TARGET_COLUMN] == config.POSITIVE_CLASS).astype(int)
    return out


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal categorical columns present in the dataframe."""
    cols_present = [c for c in config.CATEGORICAL_COLUMNS if c in df.columns]
    out = pd.get_dummies(df, columns=cols_present, drop_first=True)
    bool_cols = out.select_dtypes(include="bool").columns
    out[bool_cols] = out[bool_cols].astype(int)
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning sequence (order matters: scope filter before target encode)."""
    df = filter_to_modelling_scope(df)
    df = drop_leakage_columns(df)
    df = drop_redundant_columns(df)
    df = encode_reminder_channel_missingness(df)
    df = impute_genuine_missing_values(df)
    df = encode_target(df)
    return df
