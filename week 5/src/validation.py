"""
Automated data quality checks for the HealthConnect appointment dataset.

These checks turn the ad hoc exploration performed during Week 4 into
repeatable, testable code that runs against every new data batch, so that
data quality issues are caught automatically rather than rediscovered by hand
each time.

Each check function returns a small dict report rather than raising, so that
`run_all_checks` can produce one consolidated data quality summary. Only
genuinely broken input (handled in data_ingestion.py) raises an exception --
these checks report on data *quality*, not structural validity.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def check_no_show_history_consistency(df: pd.DataFrame) -> dict:
    """previous_no_shows should never exceed previous_appointments."""
    violations = df[df["previous_no_shows"] > df["previous_appointments"]]
    return {
        "check": "no_show_history_consistency",
        "passed": len(violations) == 0,
        "n_violations": len(violations),
        "violation_ids": violations["appointment_id"].tolist(),
    }


def check_booking_lead_days_consistency(df: pd.DataFrame) -> dict:
    """booking_lead_days should equal (appointment_date - booking_date).days."""
    booking = pd.to_datetime(df["booking_date"])
    appointment = pd.to_datetime(df["appointment_date"])
    computed = (appointment - booking).dt.days
    mismatches = df[computed != df["booking_lead_days"]]
    return {
        "check": "booking_lead_days_consistency",
        "passed": len(mismatches) == 0,
        "n_violations": len(mismatches),
        "violation_ids": mismatches["appointment_id"].tolist(),
    }


def check_appointment_day_consistency(df: pd.DataFrame) -> dict:
    """appointment_day should equal the actual weekday name of appointment_date."""
    computed_day = pd.to_datetime(df["appointment_date"]).dt.day_name()
    mismatches = df[computed_day != df["appointment_day"]]
    return {
        "check": "appointment_day_consistency",
        "passed": len(mismatches) == 0,
        "n_violations": len(mismatches),
        "violation_ids": mismatches["appointment_id"].tolist(),
    }


def check_reminder_channel_missingness_is_logical(df: pd.DataFrame) -> dict:
    """
    reminder_channel should be missing if and only if reminder_sent == 'No'.

    This confirms the Week 4 finding that this missingness is expected
    (not a data defect) -- if that relationship ever breaks on new data,
    this check will flag it instead of the pipeline silently mis-encoding
    the column.
    """
    missing_channel = df["reminder_channel"].isna()
    no_reminder = df["reminder_sent"] == "No"
    mismatches = df[missing_channel != no_reminder]
    return {
        "check": "reminder_channel_missingness_is_logical",
        "passed": len(mismatches) == 0,
        "n_violations": len(mismatches),
        "violation_ids": mismatches["appointment_id"].tolist(),
    }


def check_waiting_time_leakage_risk(df: pd.DataFrame) -> dict:
    """
    Flag (not fail) if waiting_time_minutes is populated for No-Show
    appointments -- this is the leakage risk identified in Week 4. This
    check exists to make the risk visible on every run, and to detect if
    the anomaly is ever fixed upstream (in which case the column could be
    safely reconsidered as a feature).
    """
    no_show_rows = df[df["appointment_outcome"] == "No-Show"]
    if len(no_show_rows) == 0:
        pct_populated = 0.0
    else:
        pct_populated = no_show_rows["waiting_time_minutes"].notna().mean()
    return {
        "check": "waiting_time_leakage_risk",
        "passed": bool(pct_populated == 0.0),  # "passed" = no leakage risk present
        "pct_no_show_rows_with_waiting_time": round(float(pct_populated), 4),
        "note": (
            "waiting_time_minutes is populated for the majority of No-Show rows, "
            "which is logically inconsistent and confirms this column must be "
            "excluded from model features (see config.LEAKAGE_COLUMNS)."
            if pct_populated > 0
            else "No leakage signal detected on this run."
        ),
    }


CHECKS = [
    check_no_show_history_consistency,
    check_booking_lead_days_consistency,
    check_appointment_day_consistency,
    check_reminder_channel_missingness_is_logical,
    check_waiting_time_leakage_risk,
]


def run_all_checks(df: pd.DataFrame) -> list[dict]:
    """Run every registered data quality check and return their reports."""
    reports = [check(df) for check in CHECKS]
    for report in reports:
        level = logging.INFO if report["passed"] else logging.WARNING
        logger.log(level, "Data quality check '%s': %s", report["check"], report)
    return reports
