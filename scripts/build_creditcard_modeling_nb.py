"""Appends a credit-card modeling section ("Part 2") to notebooks/modeling.ipynb.

The e-commerce stream (Part 1) already exists in the notebook with pre-split,
SMOTE'd, encoded data. The credit-card stream has only creditcard_cleaned.csv,
so this section runs the full pipeline (split -> scale -> SMOTE -> train ->
evaluate) and reuses the evaluate_model() helper defined earlier in Part 1.

Idempotent: running it again strips any previously appended Part 2 first.
"""
import nbformat as nbf

PATH = "notebooks/modeling.ipynb"
MARKER = "# Part 2 — Credit Card Stream (`creditcard.csv`)"

nb = nbf.read(PATH, as_version=4)

# Drop a previously appended Part 2 so re-running is idempotent.
cut = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown" and "".join(c["source"]).lstrip().startswith(MARKER):
        cut = i
        break
if cut is not None:
    nb["cells"] = nb["cells"][:cut]

new_cells = []


def md(text):
    new_cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    new_cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# Re-title the notebook's intro so it clearly covers both streams.
intro = nb["cells"][0]
if intro["cell_type"] == "markdown":
    intro["source"] = (
        "# Model Building & Evaluation\n\n"
        "Both transaction streams are modeled as independent problems:\n\n"
        "- **Part 1 — E-commerce stream** (`Fraud_Data.csv`)\n"
        "- **Part 2 — Credit card stream** (`creditcard.csv`)\n\n"
        "Each part covers a Logistic Regression baseline, a Random Forest ensemble, "
        "stratified K-Fold cross-validation, a comparison table (AUC-PR, F1, "
        "Confusion Matrix), and a model-selection justification."
    )

md(
    """
---

# Part 2 — Credit Card Stream (`creditcard.csv`)

The bank credit-card stream is modeled here as an independent problem. It is far
more imbalanced than the e-commerce stream (fraud ≈ 0.17% vs ≈ 9.4%), so the same
imbalance-aware approach is applied end to end. Unlike Part 1 — which loaded
pre-split, pre-encoded data — this stream starts from the cleaned file and runs
the full pipeline: stratified split → scale `Amount`/`Time` → SMOTE on the
training set only → train → evaluate. The `evaluate_model()` helper defined in
Part 1 is reused.
"""
)

md("## 8. Load Data — Credit Card")

code(
    """
df_cc = pd.read_csv('data/processed/creditcard_cleaned.csv')

X_cc = df_cc.drop(columns=['Class'])
y_cc = df_cc['Class']

print(f'Dataset: {df_cc.shape}')
print('Class distribution:', y_cc.value_counts().to_dict())
print(f'Fraud rate: {100 * y_cc.mean():.4f}%')
"""
)

md(
    """
## 9. Stratified Train/Test Split

An 80/20 stratified split preserves the extreme class ratio in both partitions.
The test set is left untouched (no resampling) so evaluation reflects the
real-world distribution.
"""
)

code(
    """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

X_train_cc, X_test_cc, y_train_cc, y_test_cc = train_test_split(
    X_cc, y_cc, test_size=0.2, stratify=y_cc, random_state=RANDOM_STATE
)
print(f'Train: {X_train_cc.shape}, Test: {X_test_cc.shape}')
print('Train fraud:', int(y_train_cc.sum()), '| Test fraud:', int(y_test_cc.sum()))
"""
)

md(
    """
## 10. Scale `Amount` and `Time`

V1–V28 are PCA components already on a comparable scale, so only the two
raw-scale columns need standardizing. The scaler is fit on the training set only
and applied to the test set, to avoid leakage.
"""
)

code(
    """
scale_cols = ['Time', 'Amount']
scaler_cc = StandardScaler()

X_train_cc = X_train_cc.copy()
X_test_cc = X_test_cc.copy()
X_train_cc[scale_cols] = scaler_cc.fit_transform(X_train_cc[scale_cols])
X_test_cc[scale_cols] = scaler_cc.transform(X_test_cc[scale_cols])

joblib.dump(scaler_cc, 'models/scaler_creditcard.joblib')
print('Scaled columns:', scale_cols)
"""
)

md(
    """
## 11. SMOTE on Training Set Only

SMOTE synthesizes minority-class examples to balance the training set. As in
Part 1, it is applied **after** the split, to the training data only — the test
set keeps its real 0.17% fraud rate.
"""
)

code(
    """
smote_cc = SMOTE(random_state=RANDOM_STATE)
X_train_cc_res, y_train_cc_res = smote_cc.fit_resample(X_train_cc, y_train_cc)
print('Before SMOTE:', y_train_cc.value_counts().to_dict())
print('After  SMOTE:', y_train_cc_res.value_counts().to_dict())
"""
)

md("## 12. Baseline — Logistic Regression (Credit Card)")

code(
    """
lr_cc = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
lr_cc.fit(X_train_cc_res, y_train_cc_res)
lr_cc_results = evaluate_model('Logistic Regression (CC)', lr_cc, X_test_cc, y_test_cc)

joblib.dump(lr_cc, 'models/logistic_regression_creditcard.joblib')
"""
)

md("## 13. Ensemble — Random Forest (Credit Card)")

code(
    """
rf_cc = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_cc.fit(X_train_cc_res, y_train_cc_res)
rf_cc_results = evaluate_model('Random Forest (CC)', rf_cc, X_test_cc, y_test_cc)

joblib.dump(rf_cc, 'models/random_forest_creditcard.joblib')
"""
)

md(
    """
## 14. Cross-Validation (Stratified K-Fold) — Credit Card

5-fold stratified CV on the held-out test set confirms the ranking is stable
rather than an artifact of a single split.
"""
)

code(
    """
for name, model in [('Logistic Regression (CC)', lr_cc), ('Random Forest (CC)', rf_cc)]:
    scores = cross_val_score(model, X_test_cc, y_test_cc, cv=cv,
                             scoring='average_precision', n_jobs=-1)
    print(f'{name:28s} CV AUC-PR: {scores.mean():.4f} ± {scores.std():.4f}')
"""
)

md("## 15. Comparison Table — Credit Card")

code(
    """
results_cc = pd.DataFrame([lr_cc_results, rf_cc_results])
results_cc = results_cc.sort_values('AUC-PR', ascending=False).reset_index(drop=True)
print(results_cc.to_string(index=False))
"""
)

md(
    """
## 16. Model Selection Justification — Credit Card

As with the e-commerce stream, **AUC-PR and F1** drive the decision rather than
accuracy, which is meaningless when fraud is only 0.17% of transactions. The
comparison table above reports both models on the untouched test set; the
higher-AUC-PR / higher-F1 model is selected and saved to `models/`.

**Explainability note:** because V1–V28 are anonymized PCA components, the
feature-level interpretability available for the e-commerce model (e.g.
`time_since_signup`) does not transfer here — SHAP on this model could only point
to opaque components. This is recorded as a limitation in the final report.
"""
)

nb["cells"].extend(new_cells)
nbf.write(nb, PATH)
print(f"appended {len(new_cells)} cells to {PATH}; total now {len(nb['cells'])}")
