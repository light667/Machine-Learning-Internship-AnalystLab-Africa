"""
Feature engineering for the HealthConnect no-show prediction problem.

The main engineered feature -- each patient's historical no-show rate -- is
the trickiest piece of this pipeline to get right: it must only use
appointments strictly *before* the one being scored. Computing it naively
(e.g. a per-patient average over the whole dataset) would leak future
information into earlier rows and silently inflate offline model performance
without providing any real predictive power in production, where the future
outcome is obviously not yet known.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_historical_no_show_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `historical_no_show_rate` and `historical_appointment_count`: for each
    appointment, the patient's no-show rate and appointment count considering
    only their *prior* appointments (strictly earlier appointment_date).

    Patients with no prior appointments get historical_appointment_count = 0
    and historical_no_show_rate = 0 (treated as "no evidence of risk yet"
    rather than missing, since a model needs a numeric value either way).

    Why this feature: previous_no_shows / previous_appointments already exist
    in the raw data, but they are static, pre-computed "as of dataset export"
    counters -- they do not necessarily reflect the state of history strictly
    before *this specific* appointment_date. Recomputing it explicitly, row
    by row, in date order, guarantees no forward-looking information leaks in,
    which matters most for patients with several appointments spread across
    the dataset's time range.
    """
    out = df.copy()
    out["appointment_date"] = pd.to_datetime(out["appointment_date"])
    out = out.sort_values(["patient_id", "appointment_date"]).reset_index(drop=True)

    out["_is_no_show"] = (out["target"] == 1).astype(int) if "target" in out.columns else 0

    grouped = out.groupby("patient_id")["_is_no_show"]
    # cumulative count/sum of *prior* rows only -> shift(1) before cumulating
    prior_count = grouped.cumcount()
    prior_no_shows = grouped.cumsum() - out["_is_no_show"]

    out["historical_appointment_count"] = prior_count
    out["historical_no_show_rate"] = (prior_no_shows / prior_count.replace(0, pd.NA)).fillna(0.0)

    out = out.drop(columns=["_is_no_show"])
    logger.info(
        "add_historical_no_show_rate: computed for %d rows across %d patients",
        len(out), out["patient_id"].nunique(),
    )
    return out


def add_reminder_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a simple binary flag for whether any reminder was sent, in addition
    to the more granular reminder_channel category -- gives the model an
    easy, low-noise linear signal alongside the more granular one-hot columns.
    """
    out = df.copy()
    out["reminder_sent_flag"] = (out["reminder_sent"] == "Yes").astype(int)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering sequence."""
    df = add_historical_no_show_rate(df)
    df = add_reminder_flag(df)
    return df
