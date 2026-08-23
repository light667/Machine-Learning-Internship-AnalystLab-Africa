# Customer Churn Prediction — Week 3: Model Development & Performance Evaluation

**AnalystLab Africa — Machine Learning Internship Programme**
Client: ABC Communications Ltd · Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Overview

This repository contains the Week 3 deliverables: five supervised classification models
trained and evaluated on the Week 2 preprocessed dataset to predict customer churn, with a
full performance comparison, feature importance analysis, and a deployment recommendation.

## Contents

| File | Description |
|---|---|
| `week3_modeling.ipynb` | End-to-end, executed notebook: data prep → train/test split → SMOTE → 5 models trained → evaluation → feature importance → learning curve → recommendation |
| `telco_train_final.csv` | Final SMOTE-balanced training set used for modelling |
| `telco_test_final.csv` | Final held-out test set (original class distribution) |
| `model_results.csv` | Metrics table for all 5 models |
| `feature_importance.csv` | Random Forest feature importances |
| `Business_Report.docx` | Business-facing report: approach, results, comparison, recommendations, next steps |
| `Model_Evaluation_Report.docx` | Technical report: full metrics, confusion matrices, ROC/PR curves, learning curve, feature importance |
| `requirements.txt` | Python dependencies to reproduce the notebook |

## Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.761 | 0.536 | **0.743** | 0.623 | **0.836** |
| Logistic Regression | 0.764 | 0.543 | 0.711 | 0.616 | 0.832 |
| Gradient Boosting | 0.770 | 0.551 | 0.717 | 0.623 | 0.830 |
| XGBoost | 0.764 | 0.547 | 0.634 | 0.587 | 0.814 |
| Decision Tree | 0.738 | 0.505 | 0.727 | 0.596 | 0.783 |

**Recommended model: Random Forest** — best ROC-AUC and best recall among the top models,
which matters most here since missing an at-risk customer is costlier than a false alarm.

**Top predictors:** Contract type, tenure, monthly/total charges, and the engineered
`AvgChargePerService` and `TenureGroup` features.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook week3_modeling.ipynb
```

## Tags

`#AnalystLabAfrica` `#MachineLearning` `#DataScience` `#CustomerChurn`
