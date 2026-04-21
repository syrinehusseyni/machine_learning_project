import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from imblearn.over_sampling import SMOTE


# =====================================================
# 📥 LOAD CLUSTERED DATA
# =====================================================
def load_data():

    path = r"C:\Users\ASUS\Downloads\projet_ml_retail\data\train_test"

    # ✅ NOW USING CLUSTERED FILES
    X_train = pd.read_csv(os.path.join(path, "X_train_clustered.csv"))
    X_test = pd.read_csv(os.path.join(path, "X_test_clustered.csv"))

    y_train = pd.read_csv(os.path.join(path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(path, "y_test.csv")).values.ravel()

    return X_train, X_test, y_train, y_test


# =====================================================
# ⚖️ SMOTE
# =====================================================
def apply_smote(X_train, y_train):

    print("⚖️ Applying SMOTE...")

    smote = SMOTE(
        random_state=42,
        k_neighbors=5
    )

    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("Before SMOTE:", np.bincount(pd.factorize(y_train)[0]))
    print("After SMOTE:", np.bincount(pd.factorize(y_res)[0]))

    return X_res, y_res


# =====================================================
# 🌲 TRAIN MODEL
# =====================================================
def train_model(X_train, y_train):

    print("🌲 Training Random Forest on clustered data...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# =====================================================
# 📊 EVALUATION
# =====================================================
def evaluate(model, X_test, y_test):

    print("📊 Predicting...")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n🎯 Accuracy:", acc)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))


# =====================================================
# 💾 SAVE MODEL
# =====================================================
def save_model(model):

    path = r"C:\Users\ASUS\Downloads\projet_ml_retail\models"
    os.makedirs(path, exist_ok=True)

    joblib.dump(model, os.path.join(path, "churn_model_clustered.pkl"))

    print("✅ Model saved!")


# =====================================================
# 🚀 MAIN
# =====================================================
if __name__ == "__main__":

    print("📥 Loading clustered data...")
    X_train, X_test, y_train, y_test = load_data()

    print("⚖️ Balancing dataset...")
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    print("🌲 Training model...")
    model = train_model(X_train_res, y_train_res)

    print("📊 Evaluating...")
    evaluate(model, X_test, y_test)

    print("💾 Saving model...")
    save_model(model)

    print("🎯 DONE")