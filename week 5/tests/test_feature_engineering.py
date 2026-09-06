import pandas as pd

from src import feature_engineering


def test_historical_no_show_rate_uses_only_prior_appointments():
    """
    Controlled example with a known-by-hand expected result, to make sure the
    feature never looks at the current or a future row -- this is the single
    most important correctness property of this pipeline.

    Patient P1: No-Show, Attended, No-Show (in date order)
      -> row 1: 0 prior appointments        -> rate = 0.0
      -> row 2: 1 prior appointment (1 NS)  -> rate = 1.0
      -> row 3: 2 prior appointments (1 NS) -> rate = 0.5
    Patient P2: single appointment -> 0 prior appointments -> rate = 0.0
    """
    df = pd.DataFrame([
        {"patient_id": "P1", "appointment_date": "2025-01-01", "target": 1},
        {"patient_id": "P1", "appointment_date": "2025-02-01", "target": 0},
        {"patient_id": "P1", "appointment_date": "2025-03-01", "target": 1},
        {"patient_id": "P2", "appointment_date": "2025-01-15", "target": 0},
    ])

    out = feature_engineering.add_historical_no_show_rate(df)
    out = out.sort_values(["patient_id", "appointment_date"]).reset_index(drop=True)

    p1 = out[out["patient_id"] == "P1"].reset_index(drop=True)
    assert p1.loc[0, "historical_appointment_count"] == 0
    assert p1.loc[0, "historical_no_show_rate"] == 0.0
    assert p1.loc[1, "historical_appointment_count"] == 1
    assert p1.loc[1, "historical_no_show_rate"] == 1.0
    assert p1.loc[2, "historical_appointment_count"] == 2
    assert p1.loc[2, "historical_no_show_rate"] == 0.5

    p2 = out[out["patient_id"] == "P2"].reset_index(drop=True)
    assert p2.loc[0, "historical_appointment_count"] == 0
    assert p2.loc[0, "historical_no_show_rate"] == 0.0


def test_historical_no_show_rate_is_order_independent_in_input():
    """Feeding rows out of chronological order should not change the result,
    since the function sorts internally before computing."""
    df_in_order = pd.DataFrame([
        {"patient_id": "P1", "appointment_date": "2025-01-01", "target": 1},
        {"patient_id": "P1", "appointment_date": "2025-02-01", "target": 0},
    ])
    df_reversed = df_in_order.iloc[::-1].reset_index(drop=True)

    out1 = feature_engineering.add_historical_no_show_rate(df_in_order)
    out2 = feature_engineering.add_historical_no_show_rate(df_reversed)

    out1 = out1.sort_values("appointment_date").reset_index(drop=True)
    out2 = out2.sort_values("appointment_date").reset_index(drop=True)

    pd.testing.assert_series_equal(
        out1["historical_no_show_rate"], out2["historical_no_show_rate"]
    )


def test_add_reminder_flag_maps_yes_no_to_binary():
    df = pd.DataFrame([
        {"reminder_sent": "Yes"},
        {"reminder_sent": "No"},
    ])
    out = feature_engineering.add_reminder_flag(df)
    assert out["reminder_sent_flag"].tolist() == [1, 0]
