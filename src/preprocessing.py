import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =====================================================
# 📥 LOAD DATA
# =====================================================
def load_data(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.read_csv(path, sep=";", encoding="latin1")


# =====================================================
# 🧹 CLEAN + FEATURE ENGINEERING
# =====================================================
def clean_data(df):

    df = df.copy()
    df.columns = df.columns.str.strip()

    # -------------------------
    # TARGET DETECTION
    # -------------------------
    if "ChurnRiskCategory" in df.columns:
        target_col = "ChurnRiskCategory"
    elif "Churn" in df.columns:
        target_col = "Churn"
    else:
        raise ValueError("❌ Target column not found")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # remove constant columns
    X = X.loc[:, X.nunique() > 1]

    # -------------------------
    # DATE FEATURES
    # -------------------------
    date_cols = [c for c in X.columns if "date" in c.lower()]

    for col in date_cols:
        X[col] = pd.to_datetime(X[col], errors="coerce")

        X[col + "_year"] = X[col].dt.year
        X[col + "_month"] = X[col].dt.month
        X[col + "_day"] = X[col].dt.day
        X[col + "_weekday"] = X[col].dt.weekday

        X.drop(columns=[col], inplace=True)

    # -------------------------
    # FEATURE ENGINEERING SAFE
    # -------------------------
    if {"MonetaryTotal", "Recency"}.issubset(X.columns):
        X["MonetaryPerDay"] = X["MonetaryTotal"] / (X["Recency"] + 1)

    if {"MonetaryTotal", "Frequency"}.issubset(X.columns):
        X["AvgBasketValue"] = X["MonetaryTotal"] / (X["Frequency"] + 1)

    if {"Recency", "CustomerTenure"}.issubset(X.columns):
        X["TenureRatio"] = X["Recency"] / (X["CustomerTenure"] + 1)

    # -------------------------
    # DROP IRRELEVANT COLUMNS
    # -------------------------
    drop_cols = ["Newsletter", "LastLoginIP"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    return X, y


# =====================================================
# ✂️ SPLIT
# =====================================================
def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# =====================================================
# ⚙️ TRAIN PREPROCESSING
# =====================================================
def preprocess_train(X_train):

    X = X_train.copy()

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns

    # -------------------------
    # IMPUTATION
    # -------------------------
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    cat_modes = {}
    for col in cat_cols:
        mode = X[col].mode()[0] if not X[col].mode().empty else "missing"
        X[col] = X[col].fillna(mode)
        cat_modes[col] = mode

    # -------------------------
    # ENCODING (SAFE ONE-HOT)
    # -------------------------
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    train_columns = X.columns

    # -------------------------
    # SCALING
    # -------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------
    # PCA (SAFE MODE)
    # -------------------------
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, scaler, cat_modes, pca, train_columns


# =====================================================
# ⚙️ TEST PREPROCESSING
# =====================================================
def preprocess_test(X_test, scaler, cat_modes, pca, train_columns):

    X = X_test.copy()

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns

    # -------------------------
    # IMPUTATION
    # -------------------------
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    for col in cat_cols:
        if col in cat_modes:
            X[col] = X[col].fillna(cat_modes[col])

    # -------------------------
    # ENCODING
    # -------------------------
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # align columns
    X = X.reindex(columns=train_columns, fill_value=0)

    # -------------------------
    # SCALING + PCA
    # -------------------------
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    return X_pca


# =====================================================
# 💾 SAVE DATA
# =====================================================
def save_data(X_train, X_test, y_train, y_test):

    path = "data/train_test"
    os.makedirs(path, exist_ok=True)

    pd.DataFrame(X_train).to_csv(os.path.join(path, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(path, "X_test.csv"), index=False)

    y_train.to_csv(os.path.join(path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(path, "y_test.csv"), index=False)


# =====================================================
# 🚀 MAIN PIPELINE
# =====================================================
if __name__ == "__main__":

    print("📥 Loading data...")
    df = load_data("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")

    print("🧹 Cleaning...")
    X, y = clean_data(df)

    print("✂️ Splitting...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("⚙️ Training preprocessing...")
    X_train, scaler, cat_modes, pca, train_columns = preprocess_train(X_train)

    print("⚙️ Testing preprocessing...")
    X_test = preprocess_test(X_test, scaler, cat_modes, pca, train_columns)

    print("💾 Saving...")
    save_data(X_train, X_test, y_train, y_test)

    print("✅ PIPELINE FIXED AND STABLE")