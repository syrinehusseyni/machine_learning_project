import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# =========================
# SAVE MODEL
# =========================
def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✔ Model saved: {filepath}")


# =========================
# METRICS
# =========================
def calculate_metrics(y_true, y_pred, y_proba):

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }

    print("\n📊 METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


# =========================
# CONFUSION MATRIX
# =========================
def plot_confusion_matrix(y_true, y_pred, save_path):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✔ Saved: {save_path}")


# =========================
# ROC CURVE
# =========================
def plot_roc_curve(y_true, y_proba, save_path):

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')

    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✔ Saved: {save_path}")

    return roc_auc


# =========================
# FEATURE IMPORTANCE
# =========================
def plot_feature_importance(model, feature_names, save_path):

    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(7,5))
    plt.barh(range(len(idx)), importances[idx])
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# MODEL COMPARISON (FIXED)
# =========================
def compare_models(results):

    df = pd.DataFrame(results).T.sort_values("F1", ascending=False)

    print("\n📊 MODEL COMPARISON")
    print(df)

    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/model_comparison.csv")

    return df