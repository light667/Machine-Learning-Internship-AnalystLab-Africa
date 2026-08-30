# HealthConnect Experience Lab — Week 4: ML System Design

**AnalystLab Africa — Experience Lab Internship Programme**
Track: Machine Learning Engineering
Project: HealthConnect Clinic — Appointment No-Show Prediction

## Overview

Week 4 marks the start of the HealthConnect Experience Lab, a shared project across all
internship tracks. This submission covers the Machine Learning Engineering track's Week 4
deliverable: an initial system design for a reproducible ML pipeline that predicts appointment
no-shows, based on review of the provided appointment dataset and data dictionary.

No model is trained this week — per the assignment scope, Week 4 focuses on problem
understanding, data review, and system design only.

## Contents

| File | Description |
|---|---|
| `ML_System_Design_Document.docx` | Full design document: ML problem overview, input/output definition, system architecture diagram, ML workflow, dependencies, reproducibility approach, assumptions/limitations/risks |
| `Week4_Project_Summary.docx` | Concise summary: problem, resources, key observations, approach, considerations, Week 5 focus |
| `fig_architecture.png` | High-level system architecture diagram (also embedded in the design document) |
| `HealthConnect_Appointment_Data.csv` | Original provided dataset (unmodified, per resource-handling instructions) |
| `HealthConnect_Data_Dictionary.xlsx` | Original provided data dictionary (unmodified) |

## Key Findings from Week 4 Data Review

- **Target scoping:** `appointment_outcome` has 3 classes (Attended 46.3%, No-Show 48.5%,
  Cancelled 5.3%). Cancelled is excluded from the v1 binary model as a distinct, already-known
  event.
- **Data leakage risk identified:** `waiting_time_minutes` is populated for the large majority
  of No-Show and Cancelled appointments, which is logically inconsistent with its stated meaning
  — flagged and excluded from candidate inputs pending clarification with the data team.
- **Logical vs genuine missingness:** `reminder_channel`'s missing values align exactly with
  `reminder_sent = No` (expected, not a defect). `distance_to_clinic_km` and
  `waiting_time_minutes` have small amounts of genuinely random missingness requiring an
  imputation decision.
- **Strong early signal:** no-show rate rises from 43.5% (no prior no-shows) to 68.8% (3+ prior
  no-shows) — `previous_no_shows` is a promising predictor.
- **Consistency checks passed:** `booking_lead_days`, `appointment_day`, and `age_group` are all
  correctly derivable from their source columns; `previous_no_shows` never exceeds
  `previous_appointments`.

## Proposed Architecture (Summary)

Batch scoring pipeline: Data Ingestion → Preprocessing/Feature Engineering → Model Training →
Model Registry → Batch Prediction Service → Clinic Ops Dashboard/Reminder System → Outcome
Logging → Monitoring & Drift Detection → Retraining Trigger (feedback loop).

See `ML_System_Design_Document.docx` Section 3 for the full diagram and rationale.

## Next Steps (Week 5)

Build the reproducible ingestion/validation/feature-engineering pipeline, implement a temporal
train/test split, resolve the `waiting_time_minutes` and missing-value questions with the data
team, and produce a first documented baseline model.

## Tags

`#AnalystLabAfrica` `#MachineLearning` `#MLEngineering` `#HealthConnect`
