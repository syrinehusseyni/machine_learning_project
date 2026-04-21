import pandas as pd
import numpy as np
import os
import joblib

import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


# =====================================================
# 📥 LOAD CLUSTERED DATA
# =====================================================
def load_data():

    path = r"C:\Users\ASUS\Downloads\projet_ml_retail\data\train_test"

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

    smote = SMOTE(random_state=42, k_neighbors=5)

    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("Before SMOTE:", np.bincount(pd.factorize(y_train)[0]))
    print("After SMOTE:", np.bincount(pd.factorize(y_res)[0]))

    return X_res, y_res


# =====================================================
# 🚀 TRAIN MODEL
# =====================================================
def train_model(X_train, y_train):

    print("🚀 Training XGBoost on clustered data...")

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)

    return model


# =====================================================
# 📊 EVALUATION
# =====================================================
def evaluate(model, X_test, y_test, label_encoder):

    print("📊 Predicting...")

    y_pred = model.predict(X_test)

    print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))

    print("\n📊 Classification Report:\n")

    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    print(classification_report(y_test_decoded, y_pred_decoded))


# =====================================================
# 💾 SAVE MODEL
# =====================================================
def save_model(model, encoder):

    path = r"C:\Users\ASUS\Downloads\projet_ml_retail\models"
    os.makedirs(path, exist_ok=True)

    joblib.dump(model, os.path.join(path, "xgboost_churn_clustered.pkl"))
    joblib.dump(encoder, os.path.join(path, "label_encoder.pkl"))

    print("✅ Model + Encoder saved!")


# =====================================================
# 🚀 MAIN
# =====================================================
if __name__ == "__main__":

    print("📥 Loading clustered data...")
    X_train, X_test, y_train, y_test = load_data()

    print("🔄 Encoding labels...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print("⚖️ Balancing dataset...")
    X_train_res, y_train_res = apply_smote(X_train, y_train_enc)

    print("🚀 Training model...")
    model = train_model(X_train_res, y_train_res)

    print("📊 Evaluating...")
    evaluate(model, X_test, y_test_enc, le)

    print("💾 Saving model...")
    save_model(model, le)

    print("🎯 DONE")