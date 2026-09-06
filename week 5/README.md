# HealthConnect No-Show Prediction — ML Pipeline

**AnalystLab Africa — Experience Lab Internship Programme**
Track: Machine Learning Engineering
Week 5: ML Pipeline Development & Initial Implementation

## What This Is

An initial, tested implementation of the ML pipeline designed in Week 4: a
modular Python package that ingests the HealthConnect appointment data, runs
automated data quality checks, cleans and feature-engineers it, splits it
temporally into train/test sets, and trains/evaluates/saves a versioned
baseline classification model. No deployment or serving layer is built at
this stage — see `docs/PIPELINE.md` for the full design and workflow
documentation.

## Quick Start

```bash
pip install -r requirements.txt

# Run the full pipeline (ingestion -> validation -> cleaning -> features -> split -> save)
python -m src.pipeline

# Run the test suite (23 tests)
python -m pytest tests/ -v

# Explore interactively
jupyter notebook notebooks/pipeline_demo.ipynb
```

## Repository Structure

See `docs/PIPELINE.md` Section 2 for the full annotated structure.

## Key Results (Week 5 Baseline)

| Metric | Value |
|---|---|
| Train / Test rows | 3,406 / 1,331 (temporal split at 2026-02-01) |
| Model | Logistic Regression (baseline) |
| Accuracy | 0.615 |
| Precision | 0.633 |
| Recall | 0.598 |
| F1 | 0.615 |
| ROC-AUC | 0.664 |

The goal at this stage is a working, versioned, tested integration — not
best-model performance (see `docs/PIPELINE.md` Section 9 for limitations).

## Key Finding Carried Forward From Week 4

Automated validation confirms `waiting_time_minutes` is populated for 98.5%
of No-Show appointments — logically inconsistent with a "time spent waiting"
metric, and a strong data leakage signal. The pipeline drops this column
before modelling; see `src/validation.py::check_waiting_time_leakage_risk`.

## Cross-Track Note

This pipeline implements the binary scoping decision (No-Show vs Attended,
excluding Cancelled) originally proposed in the Machine Learning Engineering
track's own Week 4 design document. In the absence of a live multi-intern
team for this exercise, the natural Data-Science-track hand-off is simulated:
`data/processed/train.csv` and `test.csv` are structured as the clean,
feature-engineered dataset a Data Science track would expect to receive for
further model experimentation in Week 6.

## Next Steps (Week 6)

- Resolve the `waiting_time_minutes` anomaly with the data-owning team.
- Compare the baseline against an ensemble model under the same temporal split.
- Add basic monitoring/logging hooks ahead of any real serving integration.

## Tags

`#AnalystLabAfrica` `#MachineLearning` `#MLEngineering` `#HealthConnect`
