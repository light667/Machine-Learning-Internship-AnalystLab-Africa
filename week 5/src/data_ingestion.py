# Data ingestion for the HealthConnect appointment dataset.

import logging
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class DataIngestionError(Exception):
    """Raised when the raw dataset cannot be loaded or fails schema checks."""


def load_appointment_data(path: str | Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw HealthConnect appointment CSV and validate its schema.

    Parameters
    ----------
    path : str or Path
        Location of the raw CSV file. Defaults to the configured raw data path.

    Returns
    -------
    pd.DataFrame
        The raw appointment data, unmodified.

    Raises
    ------
    DataIngestionError
        If the file does not exist, cannot be parsed, or is missing any
        required column.
    """
    path = Path(path)

    if not path.exists():
        raise DataIngestionError(
            f"Raw data file not found at '{path}'. "
            "Check that the file has been placed under data/raw/ and has not "
            "been renamed or moved."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc: 
        raise DataIngestionError(f"Failed to read '{path}' as CSV: {exc}") from exc

    if df.empty:
        raise DataIngestionError(f"'{path}' was read successfully but contains no rows.")

    missing_columns = set(config.REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise DataIngestionError(
            "Raw dataset is missing required column(s): "
            f"{sorted(missing_columns)}. Expected schema: {config.REQUIRED_COLUMNS}"
        )

    logger.info("Loaded %d rows / %d columns from %s", len(df), df.shape[1], path)
    return df
