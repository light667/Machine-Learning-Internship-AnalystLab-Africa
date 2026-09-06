import pandas as pd

from src import validation


def _base_row(**overrides):
    row = {
        "appointment_id": "A1",
        "booking_date": "2025-01-01",
        "appointment_date": "2025-01-10",
        "booking_lead_days": 9,
        "appointment_day": "Friday",
        "previous_appointments": 3,
        "previous_no_shows": 1,
        "reminder_sent": "Yes",
        "reminder_channel": "SMS",
        "appointment_outcome": "Attended",
        "waiting_time_minutes": 20,
    }
    row.update(overrides)
    return row


def test_no_show_history_consistency_passes_on_valid_data():
    df = pd.DataFrame([_base_row()])
    report = validation.check_no_show_history_consistency(df)
    assert report["passed"] is True
    assert report["n_violations"] == 0


def test_no_show_history_consistency_catches_violation():
    df = pd.DataFrame([_base_row(previous_appointments=1, previous_no_shows=5)])
    report = validation.check_no_show_history_consistency(df)
    assert report["passed"] is False
    assert report["n_violations"] == 1
    assert "A1" in report["violation_ids"]


def test_booking_lead_days_consistency_passes():
    df = pd.DataFrame([_base_row()])
    report = validation.check_booking_lead_days_consistency(df)
    assert report["passed"] is True


def test_booking_lead_days_consistency_catches_mismatch():
    df = pd.DataFrame([_base_row(booking_lead_days=999)])
    report = validation.check_booking_lead_days_consistency(df)
    assert report["passed"] is False
    assert report["n_violations"] == 1


def test_appointment_day_consistency_catches_wrong_day():
    # 2025-01-10 is actually a Friday, so "Monday" here is wrong
    df = pd.DataFrame([_base_row(appointment_day="Monday")])
    report = validation.check_appointment_day_consistency(df)
    assert report["passed"] is False


def test_reminder_channel_missingness_is_logical_passes():
    df = pd.DataFrame([
        _base_row(reminder_sent="Yes", reminder_channel="SMS"),
        _base_row(appointment_id="A2", reminder_sent="No", reminder_channel=None),
    ])
    report = validation.check_reminder_channel_missingness_is_logical(df)
    assert report["passed"] is True


def test_reminder_channel_missingness_catches_illogical_case():
    # reminder_sent = No but channel is still populated -> should be flagged
    df = pd.DataFrame([_base_row(reminder_sent="No", reminder_channel="SMS")])
    report = validation.check_reminder_channel_missingness_is_logical(df)
    assert report["passed"] is False


def test_waiting_time_leakage_risk_flags_when_present():
    df = pd.DataFrame([
        _base_row(appointment_outcome="No-Show", waiting_time_minutes=15),
    ])
    report = validation.check_waiting_time_leakage_risk(df)
    assert report["passed"] is False
    assert report["pct_no_show_rows_with_waiting_time"] == 1.0


def test_waiting_time_leakage_risk_ok_when_absent():
    df = pd.DataFrame([
        _base_row(appointment_outcome="No-Show", waiting_time_minutes=None),
    ])
    report = validation.check_waiting_time_leakage_risk(df)
    assert report["passed"] is True


def test_run_all_checks_returns_one_report_per_check():
    df = pd.DataFrame([_base_row()])
    reports = validation.run_all_checks(df)
    assert len(reports) == len(validation.CHECKS)
