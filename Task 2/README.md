# Customer Churn Prediction — Week 2: Data Preprocessing & Feature Engineering

**AnalystLab Africa — Machine Learning Internship Programme**
Client: ABC Communications Ltd · Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Overview

This repository contains the Week 2 deliverables: a full, documented preprocessing pipeline
that transforms the raw Telco Customer Churn dataset (7,043 customers, 21 columns) into a
clean, fully numeric, machine-learning-ready dataset (26 columns, 0 missing values).

## Contents

| File | Description |
|---|---|
| `week2_preprocessing.ipynb` | End-to-end, executed Jupyter notebook: inspection → cleaning → feature engineering → encoding → scaling → outlier treatment → feature selection → final export |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Original raw dataset |
| `telco_ml_ready.csv` | Final processed, ML-ready dataset |
| `Business_Understanding_Report.docx` | Business objective, target variable, problem framing, evaluation metrics, candidate algorithms |
| `Data_Preprocessing_Report.docx` | Data quality issues, cleaning decisions, feature engineering, encoding/scaling, outlier treatment, feature selection |

## Pipeline Summary

1. **Data quality issues found:** `TotalCharges` mistyped as text, hiding 11 missing values
   (blank strings) tied to `tenure = 0`; redundant category labels across 7 columns.
2. **Cleaning:** fixed `TotalCharges` dtype and imputed missing values with 0; standardized
   redundant labels to `"No"`.
3. **Feature engineering:** added `TenureGroup`, `NumServicesSubscribed`, `AvgChargePerService`;
   dropped `customerID`.
4. **Encoding:** Label Encoding (binaries), Ordinal Encoding (`Contract`, `TenureGroup`),
   One-Hot Encoding (remaining nominal categories).
5. **Outliers:** detected via IQR + Z-score; only the engineered `AvgChargePerService` had
   outliers (~3.3%), treated by IQR capping.
6. **Feature selection:** correlation analysis, variance threshold, Random Forest importance —
   `Contract`, `tenure`, billing amounts, and the engineered features rank highest.
7. **Scaling:** `StandardScaler` applied to all numeric columns.

Final dataset: **7,043 rows × 26 columns, 0 missing values, fully numeric — ready for model
training.**

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook week2_preprocessing.ipynb
```

## Tags

`#AnalystLabAfrica` `#MachineLearning` `#DataPreprocessing` `#CustomerChurn`
