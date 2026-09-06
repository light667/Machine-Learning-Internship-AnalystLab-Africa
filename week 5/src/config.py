"""
Central configuration for the HealthConnect no-show prediction pipeline.

Keeping these values in one place means every module (ingestion, preprocessing,
feature engineering, modelling) agrees on file locations, column names, and key
decisions -- instead of "magic values" scattered across the codebase.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "HealthConnect_Appointment_Data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# --- Schema --------------------------------------------------------------
# Columns the raw dataset must contain. Ingestion fails fast if any are missing,
# rather than letting a silently-broken schema flow downstream.
REQUIRED_COLUMNS = [
    "appointment_id", "patient_id", "age", "age_group", "gender",
    "appointment_type", "booking_date", "appointment_date", "booking_lead_days",
    "appointment_day", "appointment_time", "previous_appointments",
    "previous_no_shows", "reminder_sent", "reminder_channel",
    "distance_to_clinic_km", "waiting_time_minutes", "appointment_outcome",
]

TARGET_COLUMN = "appointment_outcome"

# --- Modelling scope decisions (documented in the Week 4 design doc) -----
# Cancelled appointments are a distinct, already-known-in-advance event and
# are excluded from the binary No-Show vs Attended model.
EXCLUDED_OUTCOME_CLASSES = ["Cancelled"]
POSITIVE_CLASS = "No-Show"
NEGATIVE_CLASS = "Attended"

# waiting_time_minutes is populated even for No-Show/Cancelled appointments,
# which is logically inconsistent with its meaning -- treated as a data
# leakage risk and dropped before modelling (see docs/PIPELINE.md).
LEAKAGE_COLUMNS = ["waiting_time_minutes"]

# Identifier columns kept for traceability/feature engineering but never
# passed directly into the model as predictive features.
ID_COLUMNS = ["appointment_id", "patient_id"]

# Columns that are simple text-encoded duplicates of another column
# (kept for interpretability upstream, dropped right before modelling
# to avoid redundant/collinear inputs).
DERIVED_REDUNDANT_COLUMNS = ["appointment_day"]  # derivable from appointment_date

CATEGORICAL_COLUMNS = [
    "gender", "age_group", "appointment_type", "appointment_time",
    "reminder_sent", "reminder_channel",
]

NUMERIC_COLUMNS = [
    "age", "booking_lead_days", "previous_appointments",
    "previous_no_shows", "distance_to_clinic_km",
]

# --- Train/test strategy --------------------------------------------------
# A temporal split (not random) is used: the model must predict future
# appointments from past ones in production, so evaluation should reflect
# that. Appointments on/after SPLIT_DATE go to the test set.
SPLIT_DATE = "2026-02-01"  # ~72/28 train/test split on this dataset

RANDOM_SEED = 42
