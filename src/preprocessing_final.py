import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import joblib

warnings.filterwarnings('ignore')

# -----------------------------
# DROP HIGH CARDINALITY
# -----------------------------
def drop_high_cardinality(df, threshold=0.90):
    cols_to_drop = []
    for col in df.columns:
        if col not in ['Churn', 'MonetaryTotal']:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > threshold:
                cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop)


# -----------------------------
# OUTLIERS VISUALIZATION
# -----------------------------
def plot_outliers(X, save_path='reports/outliers.png'):

    iso = IsolationForest(contamination=0.06, random_state=42)
    labels = iso.fit_predict(X)

    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)

    plt.figure(figsize=(8,6))

    plt.scatter(X_2d[labels == 1, 0],
                X_2d[labels == 1, 1],
                c='blue', label='Inliers', alpha=0.6)

    plt.scatter(X_2d[labels == -1, 0],
                X_2d[labels == -1, 1],
                c='red', label='Outliers', alpha=0.9)

    plt.title("Outlier Detection using Isolation Forest")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f" Outliers image saved at: {save_path}")


# -----------------------------
# PCA VISUALIZATION
# -----------------------------
def plot_pca(X, y=None, save_path='reports/pca_visualization.png'):

    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)

    plt.figure(figsize=(8,6))

    if y is not None:
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(label="Churn / Class")
    else:
        plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)

    plt.title("PCA Projection of Customer Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f" PCA image saved at: {save_path}")


# -----------------------------
# CLUSTER DISTRIBUTION (🔥 NEW ADDED)
# -----------------------------
def plot_cluster_distribution(labels, save_path='reports/cluster_distribution.png'):

    counts = pd.Series(labels).value_counts().sort_index()

    plt.figure(figsize=(7,5))
    plt.bar(counts.index, counts.values)

    plt.title("Customer Cluster Distribution (KMeans)")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f" Cluster image saved at: {save_path}")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def clean_and_prepare_data(file_path):

    if not os.path.exists(file_path):
        print(f" Erreur : fichier introuvable {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f" Données initiales : {len(df)} lignes")

    # ---------------- OUTLIERS ----------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features_for_outlier = [c for c in numeric_cols if c not in ['Churn', 'CustomerID']]

    iso_forest = IsolationForest(contamination=0.06, random_state=42)
    outlier_preds = iso_forest.fit_predict(df[features_for_outlier].fillna(0))

    df = df[outlier_preds == 1].reset_index(drop=True)

    plot_outliers(df[features_for_outlier].fillna(0))

    # ---------------- DATE FEATURES ----------------
    df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
    df['RegYear'] = df['RegistrationDate'].dt.year.fillna(df['RegistrationDate'].dt.year.median())
    df['RegMonth'] = df['RegistrationDate'].dt.month.fillna(df['RegistrationDate'].dt.month.median())

    # ---------------- CLEANING ----------------
    y_reg_full = df['MonetaryTotal'].fillna(df['MonetaryTotal'].median())

    cols_to_drop = [
        'Recency', 'AccountStatus', 'RFMSegment', 'ChurnRiskCategory',
        'CustomerID', 'RegistrationDate', 'LastLoginIP', 'NewsletterSubscribed',
        'MonetaryAvg', 'TotalQuantity', 'TotalTransactions'
    ]

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = drop_high_cardinality(df)

    # ---------------- ENCODING ----------------
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Churn':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df = df.fillna(df.median())

    # ---------------- FEATURES ----------------
    X_raw = df.drop(columns=['Churn', 'MonetaryTotal'], errors='ignore')
    y_class = df['Churn']

    # ---------------- SCALING ----------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    plot_pca(X_scaled, y_class)

    # ---------------- PCA ----------------
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    X_final = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(10)])

    # ---------------- KMEANS ----------------
    print(" Entraînement KMeans...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_final)

    # 🔥 NEW: cluster visualization
    plot_cluster_distribution(kmeans.labels_)

    # ---------------- SAVE DATA ----------------
    os.makedirs('data/processed', exist_ok=True)

    df_processed_all = X_final.copy()
    df_processed_all['Churn'] = y_class.values
    df_processed_all['MonetaryTotal'] = y_reg_full.values

    df_processed_all.to_csv('data/processed/processed_data_final.csv', index=False)

    # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )

    os.makedirs('data/train_test', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    X_train.to_csv('data/train_test/X_train.csv', index=False)
    X_test.to_csv('data/train_test/X_test.csv', index=False)
    y_train.to_csv('data/train_test/y_train.csv', index=False)
    y_test.to_csv('data/train_test/y_test.csv', index=False)

    # ---------------- SAVE MODELS ----------------
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(pca, 'models/pca_model.pkl')
    joblib.dump(kmeans, 'models/kmeans_model.pkl')

    # ---------------- TEST EXPORT ----------------
    print(" Génération X_test brut...")

    df_test_brut = df.loc[X_test.index, X_raw.columns]

    df_test_brut.to_csv('data/train_test/X_test_brut_40.csv', index=False)

    print(" Pipeline preprocessing terminé ✔")


# ---------------- RUN ----------------
if __name__ == "__main__":
    clean_and_prepare_data('data/raw/retail_customers_COMPLETE_CATEGORICAL.csv')