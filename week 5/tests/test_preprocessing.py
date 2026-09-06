import numpy as np
import pandas as pd

from src import preprocessing


def _sample_df():
    return pd.DataFrame([
        {"appointment_id": "A1", "appointment_outcome": "Attended",
         "waiting_time_minutes": 20, "appointment_day": "Monday",
         "reminder_channel": "SMS", "distance_to_clinic_km": 5.0},
        {"appointment_id": "A2", "appointment_outcome": "No-Show",
         "waiting_time_minutes": 18, "appointment_day": "Tuesday",
         "reminder_channel": None, "distance_to_clinic_km": np.nan},
        {"appointment_id": "A3", "appointment_outcome": "Cancelled",
         "waiting_time_minutes": 5, "appointment_day": "Wednesday",
         "reminder_channel": "Email", "distance_to_clinic_km": 3.0},
    ])


def test_filter_to_modelling_scope_removes_cancelled():
    df = _sample_df()
    out = preprocessing.filter_to_modelling_scope(df)
    assert "Cancelled" not in out["appointment_outcome"].values
    assert len(out) == 2


def test_drop_leakage_columns_removes_waiting_time():
    df = _sample_df()
    out = preprocessing.drop_leakage_columns(df)
    assert "waiting_time_minutes" not in out.columns


def test_drop_redundant_columns_removes_appointment_day():
    df = _sample_df()
    out = preprocessing.drop_redundant_columns(df)
    assert "appointment_day" not in out.columns


def test_encode_reminder_channel_missingness_fills_not_sent():
    df = _sample_df()
    out = preprocessing.encode_reminder_channel_missingness(df)
    assert out.loc[out["appointment_id"] == "A2", "reminder_channel"].iloc[0] == "Not Sent"
    assert out["reminder_channel"].isna().sum() == 0


def test_impute_genuine_missing_values_fills_and_flags():
    df = _sample_df()
    out = preprocessing.impute_genuine_missing_values(df)
    assert out["distance_to_clinic_km"].isna().sum() == 0
    assert "distance_to_clinic_km_was_missing" in out.columns
    assert out.loc[out["appointment_id"] == "A2", "distance_to_clinic_km_was_missing"].iloc[0] == 1
    assert out.loc[out["appointment_id"] == "A1", "distance_to_clinic_km_was_missing"].iloc[0] == 0


def test_encode_target_maps_no_show_to_one():
    df = _sample_df()
    out = preprocessing.encode_target(df)
    mapping = dict(zip(out["appointment_id"], out["target"]))
    assert mapping["A1"] == 0  # Attended
    assert mapping["A2"] == 1  # No-Show
    assert mapping["A3"] == 0  # Cancelled (not the positive class; excluded later anyway)


def test_clean_end_to_end_produces_binary_target_and_no_leakage_column():
    df = _sample_df()
    out = preprocessing.clean(df)
    # Cancelled row removed by scope filter
    assert len(out) == 2
    assert "waiting_time_minutes" not in out.columns
    assert set(out["target"].unique()) <= {0, 1}
