import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_model(name: str, model, X_test, y_test, save_dir: str = None) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_pr = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"AUC-PR : {auc_pr:.4f}")
    print(f"F1     : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Legit", "Pred Fraud"],
        yticklabels=["True Legit", "True Fraud"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/cm_{name.replace(' ', '_')}.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax, name=name)
    ax.set_title(f"Precision-Recall Curve — {name}")
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/pr_{name.replace(' ', '_')}.png", dpi=150)
    plt.show()

    return {"model": name, "AUC-PR": auc_pr, "F1": f1}
