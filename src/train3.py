import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


# =====================================================
# 📥 LOAD DATA
# =====================================================
def load_data():

    path = r"C:\Users\ASUS\Downloads\projet_ml_retail\data\train_test"

    X_train = pd.read_csv(os.path.join(path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(path, "X_test.csv"))

    y_train = pd.read_csv(os.path.join(path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(path, "y_test.csv")).values.ravel()

    return X_train, X_test, y_train, y_test


# =====================================================
# ⚖️ SMOTE
# =====================================================
def apply_smote(X_train, y_train):

    print("⚖️ Applying SMOTE...")

    smote = SMOTE(random_state=42, k_neighbors=5)

    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("Before SMOTE:", np.bincount(pd.factorize(y_train)[0]))
    print("After SMOTE:", np.bincount(pd.factorize(y_res)[0]))

    return X_res, y_res


# =====================================================
# 🚀 TRAIN LIGHTGBM
# =====================================================
def train_model(X_train, y_train):

    print("🌟 Training LightGBM...")

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# =====================================================
# 📊 CONFUSION MATRIX
# =====================================================
def plot_confusion_matrix(model, X_test, y_test, le):

    y_pred = model.predict(X_test)

    y_test_dec = le.inverse_transform(y_test)
    y_pred_dec = le.inverse_transform(y_pred)

    cm = confusion_matrix(y_test_dec, y_pred_dec, labels=le.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Blues")

    plt.title("LightGBM - Confusion Matrix")
    plt.show()


# =====================================================
# 📊 FEATURE IMPORTANCE
# =====================================================
def plot_feature_importance(model, X_train):

    importances = model.feature_importances_
    features = X_train.columns

    indices = np.argsort(importances)[-10:]  # top 10 features

    plt.figure(figsize=(8,5))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [features[i] for i in indices])

    plt.title("LightGBM - Top Feature Importance")
    plt.show()


# =====================================================
# 📊 EVALUATION
# =====================================================
def evaluate(model, X_test, y_test, le):

    print("📊 Predicting...")

    y_pred = model.predict(X_test)

    print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))

    print("\n📊 Classification Report:\n")

    y_test_dec = le.inverse_transform(y_test)
    y_pred_dec = le.inverse_transform(y_pred)

    print(classification_report(y_test_dec, y_pred_dec))


# =====================================================
# 💾 SAVE MODEL
# =====================================================
def save_model(model, le):

    path = r"C:\Users\ASUS\Downloads\projet_ml_retail\models"
    os.makedirs(path, exist_ok=True)

    joblib.dump(model, os.path.join(path, "lightgbm_churn.pkl"))
    joblib.dump(le, os.path.join(path, "label_encoder.pkl"))

    print("✅ Model saved!")


# =====================================================
# 🚀 MAIN
# =====================================================
if __name__ == "__main__":

    print("📥 Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("🔄 Encoding labels...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print("⚖️ Balancing...")
    X_train_res, y_train_res = apply_smote(X_train, y_train_enc)

    print("🌟 Training...")
    model = train_model(X_train_res, y_train_res)

    print("📊 Evaluation...")
    evaluate(model, X_test, y_test_enc, le)

    # =================================================
    # 📊 VISUALIZATIONS (ADDED PART)
    # =================================================
    print("📊 Plotting confusion matrix...")
    plot_confusion_matrix(model, X_test, y_test_enc, le)

    print("📊 Plotting feature importance...")
    plot_feature_importance(model, X_train)

    print("💾 Saving...")
    save_model(model, le)

    print("🎯 DONE")