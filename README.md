# Fraud Detection — Adey Innovations Inc.

End-to-end machine learning pipeline for detecting fraudulent transactions across e-commerce and bank credit card datasets.

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/          # Original datasets (gitignored)
│   └── processed/    # Cleaned & feature-engineered data (gitignored)
├── notebooks/        # Jupyter notebooks (EDA → modeling → SHAP)
├── src/              # Reusable Python modules
├── tests/            # Unit tests
├── models/           # Saved model artifacts
├── scripts/          # Standalone scripts
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
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

## Notebooks (run in order)

| Notebook | Task |
|----------|------|
| `eda-fraud-data.ipynb` | EDA for e-commerce transactions |
| `eda-creditcard.ipynb` | EDA for bank credit card data |
| `feature-engineering.ipynb` | IP geolocation, time features, velocity |
| `modeling.ipynb` | Logistic Regression + XGBoost, evaluation |
| `shap-explainability.ipynb` | SHAP summary + force plots + recommendations |

## Key Decisions

- **SMOTE** applied on training set only to handle severe class imbalance.
- **AUC-PR and F1-Score** used as primary metrics (accuracy is misleading on imbalanced data).
- **Two separate pipelines** — e-commerce and credit card are treated as independent problems.

## Results

| Dataset | Model | AUC-PR | F1-Score |
|---------|-------|--------|----------|
| E-commerce | XGBoost | TBD | TBD |
| Credit Card | XGBoost | TBD | TBD |

## Author

Kalkidan Fantu — 10 Academy AI Mastery, Week 5&6
