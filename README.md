# Fraud Detection — Adey Innovations Inc.

End-to-end machine learning pipeline for detecting fraudulent transactions in e-commerce data, covering data cleaning, feature engineering, class-imbalance handling (SMOTE), model training/comparison, and SHAP-based explainability.

## Project Overview

This project was built for the 10 Academy KAIM Week 5&6 challenge. It walks through the full fraud-detection workflow on `Fraud_Data.csv` (e-commerce transactions):

1. **Data cleaning & EDA** — handle missing values/duplicates, fix IP-address formats, and explore class imbalance, univariate and bivariate patterns (`eda-fraud-data.ipynb`, `eda-creditcard.ipynb`).
2. **Feature engineering** — IP-to-country geolocation, `time_since_signup`, `hour_of_day`/`day_of_week`, transaction velocity, encoding/scaling, and SMOTE applied to the training set only (`feature-engineering.ipynb`).
3. **Modeling** — train and compare a Logistic Regression baseline against a Random Forest ensemble (with cross-validation), and select the best model with a documented justification (`modeling.ipynb`).
4. **Explainability** — use SHAP to explain the selected model's predictions globally and for individual True Positive / False Positive / False Negative cases, translated into business recommendations (`shap-explainability.ipynb`).

A full write-up of the analysis, results, and recommendations is in [`reports/final-report.md`](reports/final-report.md).

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/          # Original datasets (gitignored)
│   └── processed/    # Cleaned & feature-engineered data (gitignored)
├── notebooks/        # Jupyter notebooks (EDA → feature engineering → modeling → SHAP)
├── src/              # Reusable Python modules
├── tests/            # Unit tests
├── models/           # Saved model artifacts (.joblib, gitignored)
├── scripts/          # Standalone scripts
├── reports/          # Final report + figures (gitignored)
└── requirements.txt
```

## Datasets

| File | Description |
|------|-------------|
| `Fraud_Data.csv` | E-commerce transactions with user/device/behavioral features |
| `IpAddress_to_Country.csv` | IP range → country mapping |
| `creditcard.csv` | Anonymized bank credit card transactions (PCA features V1–V28) |

Place all raw files in `data/raw/`.

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd fraud-detection

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

## Datasets continued

Place all raw files in `data/raw/` before running the notebooks. Cleaned/processed CSVs and trained models are written to `data/processed/` and `models/` respectively (both gitignored).

## Notebooks (run in order)

| Notebook | Task |
|----------|------|
| `eda-fraud-data.ipynb` | EDA for e-commerce transactions |
| `eda-creditcard.ipynb` | EDA for bank credit card data |
| `feature-engineering.ipynb` | IP geolocation, time features, velocity, encoding/scaling, SMOTE |
| `modeling.ipynb` | Logistic Regression baseline + Random Forest ensemble, cross-validation, model selection |
| `shap-explainability.ipynb` | SHAP summary + force plots + business recommendations |

## Key Decisions

- **SMOTE** applied on the training set only to handle severe class imbalance (test set kept in its real-world distribution).
- **AUC-PR and F1-Score** used as primary metrics (accuracy is misleading on imbalanced data).
- **Random Forest selected as the final model** over XGBoost/LightGBM — see [Results](#results) and `reports/final-report.md` for the full justification.

## Results

Final model comparison on the held-out e-commerce test set (`Fraud_Data.csv`):

| Model | AUC-PR | F1-Score | Precision (Fraud) | Recall (Fraud) |
|-------|--------|----------|--------------------|-----------------|
| Logistic Regression (baseline) | 0.404 | 0.338 | 0.23 | 0.65 |
| **Random Forest (selected)** | **0.638** | **0.706** | **0.99** | 0.55 |

Random Forest was selected for its substantially higher AUC-PR and F1, near-perfect fraud precision, and full SHAP `TreeExplainer` compatibility. See `reports/final-report.md` for the comparison against XGBoost/LightGBM and SHAP-based feature insights.

## Author

Kalkidan Asfaw — 10 Academy AI Mastery, Week 5&6
