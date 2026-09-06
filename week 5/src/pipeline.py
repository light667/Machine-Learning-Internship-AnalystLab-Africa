"""
End-to-end data pipeline orchestration for the HealthConnect no-show model.

Running this module executes the full sequence:
    ingest -> validate -> clean -> engineer features -> temporal split -> save

Each stage is wrapped so a failure at any point raises a clear, contextualized
error rather than an opaque traceback from deep inside pandas.
"""

import logging
import sys

import pandas as pd

from src import config, data_ingestion, feature_engineering, preprocessing, validation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when a pipeline stage fails in a way that should halt execution."""


def temporal_train_test_split(
    df: pd.DataFrame, split_date: str = config.SPLIT_DATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically: appointments before split_date -> train,
    on/after split_date -> test. See config.SPLIT_DATE for rationale.
    """
    dates = pd.to_datetime(df["appointment_date"])
    cutoff = pd.Timestamp(split_date)
    train = df[dates < cutoff].copy()
    test = df[dates >= cutoff].copy()

    if len(train) == 0 or len(test) == 0:
        raise PipelineError(
            f"Temporal split at {split_date} produced an empty train or test set "
            f"(train={len(train)}, test={len(test)}). Check config.SPLIT_DATE against "
            "the actual date range in the data."
        )
    return train, test


def run_pipeline(raw_path=config.RAW_DATA_PATH, save_outputs: bool = True) -> dict:
    """
    Run the full pipeline and return a dict with the processed train/test
    frames plus the data quality check reports, for use in the demo notebook
    and in tests.
    """
    try:
        df = data_ingestion.load_appointment_data(raw_path)
    except data_ingestion.DataIngestionError as exc:
        logger.error("Pipeline halted at ingestion: %s", exc)
        raise PipelineError(f"Ingestion failed: {exc}") from exc

    quality_reports = validation.run_all_checks(df)
    failed_checks = [r for r in quality_reports if not r["passed"]]
    if failed_checks:
        # Consistency checks failing would indicate a structural data problem;
        # the known/expected leakage-risk check is allowed to "fail" (flag)
        # without halting the pipeline, since it's a documented, handled risk.
        hard_failures = [
            r for r in failed_checks if r["check"] != "waiting_time_leakage_risk"
        ]
        if hard_failures:
            raise PipelineError(
                f"Pipeline halted: data quality check(s) failed unexpectedly: "
                f"{[r['check'] for r in hard_failures]}. Inspect validation reports "
                "before proceeding."
            )
        logger.warning(
            "Known, handled data risk flagged (waiting_time_minutes leakage) -- "
            "continuing, since this column is dropped in preprocessing."
        )

    try:
        df_clean = preprocessing.clean(df)
        df_features = feature_engineering.engineer_features(df_clean)
        df_final = preprocessing.encode_categoricals(df_features)
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline halted during preprocessing/feature engineering: %s", exc)
        raise PipelineError(f"Preprocessing failed: {exc}") from exc

    train_df, test_df = temporal_train_test_split(df_final)

    if save_outputs:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train_path = config.PROCESSED_DIR / "train.csv"
        test_path = config.PROCESSED_DIR / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info("Saved processed train/test sets to %s", config.PROCESSED_DIR)

    logger.info(
        "Pipeline complete: train=%d rows, test=%d rows, %d features",
        len(train_df), len(test_df), df_final.shape[1],
    )

    return {
        "raw": df,
        "train": train_df,
        "test": test_df,
        "quality_reports": quality_reports,
    }


if __name__ == "__main__":
    try:
        run_pipeline()
    except PipelineError as exc:
        logger.error("Pipeline run failed: %s", exc)
        sys.exit(1)
