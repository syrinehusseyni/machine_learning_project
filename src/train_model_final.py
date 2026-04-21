import pandas as pd
import numpy as np
import joblib
import os

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# ---------------- DATA PATHS ----------------
DATA_PATHS = {
    'X_train': 'data/train_test/X_train.csv',
    'X_test': 'data/train_test/X_test.csv',
    'y_train': 'data/train_test/y_train.csv',
    'y_test': 'data/train_test/y_test.csv'
}

# ---------------- MAIN ----------------
def main():

    print("🚀 STARTING FIXED CHURN TRAINING PIPELINE")

    # ---------------- LOAD DATA ----------------
    X_train = pd.read_csv(DATA_PATHS['X_train'])
    X_test = pd.read_csv(DATA_PATHS['X_test'])
    y_train = pd.read_csv(DATA_PATHS['y_train']).values.ravel()
    y_test = pd.read_csv(DATA_PATHS['y_test']).values.ravel()

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # ---------------- SMOTE (BALANCE TRAIN ONLY) ----------------
    print("\n⚖️ Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("After SMOTE:", np.bincount(y_train_res))

    # ---------------- FULL PIPELINE ----------------
    # 🔥 THIS FIXES YOUR PCA MISMATCH PROBLEM
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=10)),
        ("model", RandomForestClassifier(
            n_estimators=250,
            max_depth=15,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # ---------------- TRAIN ----------------
    print("\n🔥 Training model...")
    pipeline.fit(X_train_res, y_train_res)

    # ---------------- EVALUATION ----------------
    print("\n📊 Evaluating model...")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n📌 CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))

    print("F1-score:", f1_score(y_test, y_pred))

    # ---------------- SAVE MODEL ----------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/churn_pipeline.pkl")

    print("\n✅ Model saved at: models/churn_pipeline.pkl")

    # ---------------- TEST SAMPLE ----------------
    print("\n🧪 Sample predictions:")
    for i in range(5):
        print(f"Customer {i+1} -> "
              f"Churn: {pipeline.predict(X_test.iloc[[i]])[0]} | "
              f"Prob: {pipeline.predict_proba(X_test.iloc[[i]])[0][1]:.2f}")

    print("\n🎯 TRAINING COMPLETED SUCCESSFULLY")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()