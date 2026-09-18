# Patient No-Show Prediction

MSc Data Analytics dissertation, Edge Hill University (2026).

Comparative evaluation of four machine learning algorithms — Logistic
Regression, Random Forest, XGBoost and an Artificial Neural Network,
for predicting patient non-attendance at outpatient appointments, with
SHAP explainability and a Power BI decision support dashboard.

**Dataset:** Kaggle Medical Appointment No-Show (110,526 records,
20.19% no-show rate)

**Best model:** XGBoost — F1 0.4504, AUC-ROC 0.7327, MCC 0.2809

**Key finding:** appointment lead time is the dominant predictor
(mean |SHAP| = 0.82), more than four times stronger than the
second-ranked feature.

## Files
- `chapter4.ipynb` — full pipeline: cleaning, feature engineering,
  SMOTE, model training, evaluation, SHAP, bias analysis
- `Evidence_Notebook_Colab.py` — script export of the notebook

## Licence
MIT
