import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


def train_regression_on_raw_data():
    print("🚀 Entraînement du modèle de régression sur données brutes...")

    # ---------------- LOAD DATA ----------------
    X_train = pd.read_csv('data/regression_specific/X_train_reg.csv')
    X_test = pd.read_csv('data/regression_specific/X_test_reg.csv')
    y_train = pd.read_csv('data/regression_specific/y_train_reg.csv').values.ravel()
    y_test = pd.read_csv('data/regression_specific/y_test_reg.csv').values.ravel()

    # ---------------- LOG TRANSFORM ----------------
    y_train_log = np.log1p(np.maximum(y_train, 0))
    y_test_log = np.log1p(np.maximum(y_test, 0))

    # ---------------- MODEL ----------------
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    print("📊 Cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5, scoring='r2')

    model.fit(X_train, y_train_log)

    # ---------------- PREDICTIONS ----------------
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # ---------------- METRICS ----------------
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n📌 RESULTS:")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")

    # =====================================================
    # 📊 IMAGE 1 — ACTUAL VS PREDICTED REVENUE
    # =====================================================
    plt.figure(figsize=(7,5))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--'
    )

    plt.title("📌 IMAGE 1 — Actual vs Predicted Revenue")
    plt.xlabel("Actual Revenue (DT)")
    plt.ylabel("Predicted Revenue (DT)")

    plt.text(
        x=y_test.min(),
        y=y_test.max()*0.9,
        s="Use this image in: Revenue Prediction Chapter",
        color="red",
        fontsize=9
    )

    plt.show()   # 👈 SCREENSHOT HERE
    plt.close()

    # =====================================================
    # 📊 IMAGE 2 — FEATURE IMPORTANCE
    # =====================================================
    plt.figure(figsize=(7,5))

    importances = model.feature_importances_
    features = X_train.columns

    feat_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    sns.barplot(data=feat_df, x="Importance", y="Feature")

    plt.title("📌 IMAGE 2 — Top Features for Revenue Prediction")

    plt.text(
        x=0,
        y=-1,
        s="Use this image in Feature Importance section",
        color="red",
        fontsize=9
    )

    plt.show()   # 👈 SCREENSHOT HERE
    plt.close()

    # ---------------- SAVE MODEL ----------------
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/regression_model_raw.pkl')

    print("\n✅ Model trained successfully")


if __name__ == "__main__":
    train_regression_on_raw_data()