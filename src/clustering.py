import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =====================================================
# 📥 LOAD DATA
# =====================================================
def load_data():
    path = r"data/train_test"

    X_train = pd.read_csv(os.path.join(path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(path, "X_test.csv"))

    return X_train, X_test


# =====================================================
# 📊 ELBOW METHOD (FIXED - NOT STUCK)
# =====================================================
def elbow_method(X):

    print("📊 Running Elbow Method...")

    distortions = []

    K_range = range(2, 10)

    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)

        distortions.append(model.inertia_)

    # plot
    plt.figure()
    plt.plot(K_range, distortions, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.show()


# =====================================================
# 🚀 TRAIN CLUSTERING MODEL
# =====================================================
def train_kmeans(X, k=4):

    print(f"🚀 Training KMeans with k={k}")

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    print(f"📊 Silhouette Score: {score:.4f}")

    return model, labels


# =====================================================
# 💾 SAVE CLUSTERS
# =====================================================
def save_clusters(X_train, X_test, model):

    train_labels = model.predict(X_train)
    test_labels = model.predict(X_test)

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["Cluster"] = train_labels
    X_test["Cluster"] = test_labels

    path = r"data/train_test"

    X_train.to_csv(os.path.join(path, "X_train_clustered.csv"), index=False)
    X_test.to_csv(os.path.join(path, "X_test_clustered.csv"), index=False)

    joblib.dump(model, "models/kmeans.pkl")

    print("✅ Clustering saved!")


# =====================================================
# 🚀 MAIN
# =====================================================
if __name__ == "__main__":

    print("📥 Loading data...")
    X_train, X_test = load_data()

    print("📊 Step 1: Elbow Method")
    elbow_method(X_train)

    print("🚀 Step 2: Training final model")
    model, train_labels = train_kmeans(X_train, k=4)

    print("💾 Saving clusters")
    save_clusters(X_train, X_test, model)

    print("🎯 DONE")