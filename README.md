# Telecom-Customer-Churn-Prediction-Python-Scikit-learn-CatBoost-LightGBM-2025-
Full Machine Learning pipeline on a 5,000-record imbalanced telecom dataset, from data discovery and preprocessing through model training, evaluation, and business interpretation.

## Overview
Predicting customer churn for a telecom company using gradient boosting models on an imbalanced dataset (26% churn rate). Full ML pipeline from EDA through model selection and business interpretation.

## Results
| Model | Recall | ROC-AUC | F1 |
|-------|--------|---------|-----|
| CatBoost (weighted) | 0.90 | 0.75 | 0.61 |
| LightGBM (weighted) | 0.72 | 0.84 | 0.63 |

**Selected model:** CatBoost — prioritized recall to minimize missed churners as the highest business-cost error.

## Key Findings
- Contract type is the strongest churn predictor (SHAP)
- Fiber optic customers churn at 41% vs 7% for non-internet customers
- Month-to-month contracts churn at 42% vs 2% for two-year contracts
- Class imbalance handled via scale_pos_weight tuning

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, CatBoost, LightGBM, SHAP, Plotly, Seaborn, Matplotlib

## Dataset
Telco Customer Churn — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)
