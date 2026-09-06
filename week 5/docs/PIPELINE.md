# HealthConnect No-Show Pipeline — Documentation

**Track:** Machine Learning Engineering
**Week 5 status:** Initial implementation (not deployed)

## 1. Week 4 → Week 5 Recap

Week 4 produced an `ML System Design Document` defining the problem (binary
No-Show vs Attended classification), the input/output contract, a high-level
architecture, and a set of assumptions/risks — most notably a suspected data
leakage issue in `waiting_time_minutes`. Week 5 implements the first working
version of that design: a modular, tested data pipeline plus a versioned
baseline model integration. No deployment or serving layer is built at this
stage, per the assignment scope.

## 2. Repository Structure

```
healthconnect_ml_pipeline/
├── data/
│   ├── raw/               # original, untouched project resources
│   └── processed/         # pipeline output (train.csv, test.csv)
├── models/                 # versioned model artifacts + JSON metadata sidecars
├── src/
│   ├── config.py            # all shared paths, schema, and scope decisions
│   ├── data_ingestion.py     # load + schema validation
│   ├── validation.py         # automated data quality checks (from Week 4 findings)
│   ├── preprocessing.py      # cleaning, scoping, encoding
│   ├── feature_engineering.py# historical no-show rate, reminder flag
│   ├── pipeline.py           # orchestrates the full sequence, with error handling
│   └── model.py              # baseline training, evaluation, versioned save/load
├── tests/                   # pytest unit + integration tests (23 tests, all passing)
├── notebooks/
│   └── pipeline_demo.ipynb  # executed end-to-end walkthrough
├── docs/
│   ├── PIPELINE.md          # this file
│   └── test_evidence.txt    # saved output of a full passing test run
├── requirements.txt
└── README.md
```

## 3. Pipeline Workflow

`src/pipeline.run_pipeline()` executes, in order:

1. **Ingestion** (`data_ingestion.py`) — loads the raw CSV, raises
   `DataIngestionError` if the file is missing, unreadable, empty, or missing
   any required column.
2. **Validation** (`validation.py`) — runs five automated data quality
   checks (see Section 4). A genuine structural failure halts the pipeline;
   the known `waiting_time_minutes` leakage flag is logged as a warning and
   the pipeline continues, since it is a documented, handled risk.
3. **Cleaning** (`preprocessing.clean()`) — scopes to No-Show/Attended,
   drops the leakage column, drops the redundant `appointment_day` column,
   encodes `reminder_channel`'s logical missingness, imputes the small
   genuine gap in `distance_to_clinic_km` (median + missingness flag),
   encodes the binary target.
4. **Feature engineering** (`feature_engineering.engineer_features()`) —
   adds `historical_no_show_rate` / `historical_appointment_count` (leak-safe,
   computed only from strictly prior appointments) and a `reminder_sent_flag`.
5. **Encoding** (`preprocessing.encode_categoricals()`) — one-hot encodes
   remaining nominal categorical columns.
6. **Temporal train/test split** (`pipeline.temporal_train_test_split()`) —
   splits chronologically at `config.SPLIT_DATE` rather than randomly, since
   the model must always predict future appointments from past data in
   production; a random split would overstate offline performance.
7. **Save** — writes `train.csv` / `test.csv` to `data/processed/` (the
   original raw file is never modified).

`src/model.py` then provides a separate, reusable train → evaluate → save →
load cycle, decoupled from the data pipeline so either can be run or tested
independently.

## 4. Automated Data Quality Checks

| Check | What it verifies | Behavior if it fails |
|---|---|---|
| `no_show_history_consistency` | `previous_no_shows` never exceeds `previous_appointments` | Halts pipeline (structural issue) |
| `booking_lead_days_consistency` | `booking_lead_days` matches the computed date difference | Halts pipeline |
| `appointment_day_consistency` | `appointment_day` matches the actual weekday of `appointment_date` | Halts pipeline |
| `reminder_channel_missingness_is_logical` | `reminder_channel` is missing if and only if `reminder_sent == "No"` | Halts pipeline |
| `waiting_time_leakage_risk` | Flags if `waiting_time_minutes` is populated for No-Show rows (it shouldn't logically be) | **Does not halt** — known, documented risk; column is dropped downstream regardless |

On the current dataset, the first four checks pass; the fifth reports the
known leakage risk (98.5% of No-Show rows have a populated
`waiting_time_minutes`), confirming the Week 4 finding automatically rather
than requiring manual re-inspection.

## 5. Error Handling

- `DataIngestionError` — missing file, unreadable CSV, empty file, or missing
  required column(s); raised with a specific, actionable message.
- `PipelineError` — wraps ingestion failures, unexpected data-quality
  failures, preprocessing/feature-engineering exceptions, and a degenerate
  temporal split (empty train or test set), so a failure anywhere in the
  sequence surfaces with context about *which* stage failed.
- `ModelError` — missing target column, a training set with only one class,
  or a missing model file on load.

## 6. Configuration & Dependencies

All shared constants (file paths, required schema, the modelling-scope
decisions from Week 4, the temporal split date, the random seed) live in
`src/config.py` — a single source of truth so every module agrees on them.
Python dependencies are pinned at a minimum version in `requirements.txt`.

## 7. Model Versioning (Lightweight Registry)

`model.save_model()` writes two files per trained model:
`baseline_logreg_{version}.joblib` (the artifact) and a matching `.json`
metadata sidecar recording the model type, training timestamp, exact feature
list, and evaluation metrics. This is a minimal, file-based stand-in for a
dedicated model registry (e.g. MLflow), consistent with the Week 4 design's
"reproducibility and model management" requirement, without adding
infrastructure that isn't yet needed.

## 8. Testing Evidence

23 tests across `tests/test_validation.py`, `tests/test_preprocessing.py`,
`tests/test_feature_engineering.py`, and `tests/test_pipeline_integration.py`
— all passing (see `docs/test_evidence.txt` for the saved run). Coverage
includes: each validation check against both valid and deliberately-broken
inputs, each preprocessing step in isolation, a hand-verified correctness
test for the historical no-show rate logic, and an end-to-end smoke test
that runs the real pipeline and trains/evaluates a model.

Run locally with:
```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## 9. Known Issues & Limitations

- `waiting_time_minutes` remains unexplained — dropped rather than
  investigated further, since resolving *why* it's populated for no-shows
  requires input from whoever generated/owns the source data.
- The baseline model (Logistic Regression, ROC-AUC ≈ 0.66 on this split) is
  intentionally not tuned — Week 5's goal was a working, tested integration
  structure, not best performance.
- No real-time serving, monitoring dashboard, or reminder-system integration
  exists yet — out of scope for Week 5 per the assignment ("the focus is not
  deployment at this stage").
- The model registry is a local folder of files, not a dedicated tool; this
  is a reasonable placeholder for this project's current scale, flagged here
  in case scale later requires revisiting it.
