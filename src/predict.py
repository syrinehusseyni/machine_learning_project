import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATHS = {
    'churn_pipeline': 'models/churn_pipeline.pkl',        # ✅ full pipeline (scaler+pca+clf)
    'regressor':      'models/regression_model_raw.pkl',  # ✅ correct filename
    'kmeans':         'models/kmeans_model.pkl',
    'scaler_cluster': 'models/scaler_cluster.pkl',        # ✅ needed for kmeans
    'pca_cluster':    'models/pca_cluster.pkl'            # ✅ needed for kmeans
}

DATA_TEST_PATH   = 'data/train_test/X_test.csv'
TARGET_TEST_PATH = 'data/train_test/y_test.csv'


def align(df, cols):
    """Align dataframe to expected columns, encoding categoricals."""
    df = df.copy()
    for c in df.select_dtypes(include="object").columns:
        df[c] = pd.Categorical(df[c]).codes
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols].fillna(0)


def run_comprehensive_predictions():

    print("🚀 Prediction + Visualization Pipeline")

    # ---------------- LOAD MODELS ----------------
    models = {}
    for key, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
            print(f"✔ Loaded {key}")
        else:
            print(f"⚠ Missing: {path}")

    required = ['churn_pipeline', 'regressor', 'kmeans', 'scaler_cluster', 'pca_cluster']
    if not all(k in models for k in required):
        print("❌ Missing required models. Aborting.")
        return

    # ---------------- LOAD DATA ----------------
    X_test = pd.read_csv(DATA_TEST_PATH)
    print(f"✔ Loaded test data: {X_test.shape}")

    y_true = None
    if os.path.exists(TARGET_TEST_PATH):
        y_true = pd.read_csv(TARGET_TEST_PATH).values.ravel()

    results = pd.DataFrame(index=range(len(X_test)))

    # =====================================================
    # CHURN PREDICTION (pipeline handles scaler+pca internally)
    # =====================================================
    churn_pipeline = models['churn_pipeline']
    churn_cols = churn_pipeline.named_steps['scaler'].feature_names_in_
    X_churn = align(X_test, churn_cols)

    results['Churn'] = churn_pipeline.predict(X_churn)
    results['Churn_Prob'] = churn_pipeline.predict_proba(X_churn)[:, 1] * 100
    print(f"✔ Churn predictions done. Churn rate: {results['Churn'].mean()*100:.1f}%")

    # IMAGE 1: CHURN DISTRIBUTION
    plt.figure(figsize=(6, 4))
    sns.countplot(x=results['Churn'])
    plt.title("IMAGE 1 — Churn Distribution (0 = No Churn, 1 = Churn)")
    plt.xlabel("Churn Class")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.show()

    # IMAGE 2: CHURN PROBABILITY
    plt.figure(figsize=(7, 4))
    plt.hist(results['Churn_Prob'], bins=30, color='orange')
    plt.title("IMAGE 2 — Churn Probability Distribution (%)")
    plt.xlabel("Probability (%)")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.show()

    # =====================================================
    # REVENUE PREDICTION
    # =====================================================
    reg_cols = models['regressor'].feature_names_in_
    X_reg = align(X_test, reg_cols)

    results['Revenue'] = np.expm1(models['regressor'].predict(X_reg))
    print(f"✔ Revenue predictions done. Avg: {results['Revenue'].mean():.2f} DT")

    # IMAGE 3: REVENUE DISTRIBUTION
    plt.figure(figsize=(7, 4))
    sns.histplot(results['Revenue'], bins=30, kde=True)
    plt.title("IMAGE 3 — Predicted Revenue Distribution")
    plt.xlabel("Revenue (DT)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # IMAGE 4: REPORT-READY REVENUE
    plt.figure(figsize=(8, 5))
    sns.histplot(results['Revenue'], bins=30, kde=True, color='green')
    plt.title(
        "REVENUE PREDICTION (REGRESSION MODEL)\n"
        "📊 To be used in: Revenue Prediction Chapter",
        fontsize=12
    )
    plt.xlabel("Predicted Revenue (DT)")
    plt.ylabel("Number of Customers")
    plt.text(
        results['Revenue'].min(),
        plt.ylim()[1] * 0.9,
        "FIGURE: Revenue Prediction Output (REPORT CHAPTER)",
        fontsize=9, color="red", fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    # =====================================================
    # CLUSTERING (needs scaler_cluster → pca_cluster → kmeans)
    # =====================================================
    scaler_cluster = models['scaler_cluster']
    pca_cluster    = models['pca_cluster']
    kmeans         = models['kmeans']

    X_clust = align(X_test, scaler_cluster.feature_names_in_)
    X_clust_scaled = scaler_cluster.transform(X_clust)
    X_pca          = pca_cluster.transform(X_clust_scaled)
    results['Cluster'] = kmeans.predict(X_pca)
    print(f"✔ Clustering done. Segments: {results['Cluster'].nunique()}")

    # IMAGE 5: CLUSTER DISTRIBUTION
    plt.figure(figsize=(6, 4))
    sns.countplot(x=results['Cluster'])
    plt.title("IMAGE 5 — Customer Segments (KMeans Clusters)")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.show()

    # =====================================================
    # BUSINESS INSIGHTS
    # =====================================================

    # IMAGE 6: REVENUE VS CHURN
    plt.figure(figsize=(7, 4))
    sns.boxplot(x=results['Churn'], y=results['Revenue'])
    plt.title("IMAGE 6 — Revenue vs Churn Behavior")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Revenue (DT)")
    plt.tight_layout()
    plt.show()

    print("\n✅ All labeled images ready for report screenshots!")
    return results


if __name__ == "__main__":
    results = run_comprehensive_predictions()
    if results is not None:
        print("\n📋 Sample Results:")
        print(results.head(10))