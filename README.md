# 🛍️ Retail Intelligence System

### End-to-End Data Engineering, Machine Learning & Business Analytics

## 📌 Overview

This project presents a complete **Retail Intelligence System** designed to transform raw retail data into actionable business insights.
It integrates **data preprocessing, machine learning models, and a web application** to support decision-making in customer analytics.

The system focuses on three main tasks:

* Customer Segmentation
* Churn Prediction
* Revenue Forecasting

---

## 🎯 Objectives

* Build a robust **data preprocessing pipeline**
* Identify customer groups using **clustering techniques**
* Predict customer churn using **classification models**
* Estimate customer value using **regression models**
* Deploy an interactive **web application** for real-time insights

---

## 🧠 System Architecture

The system follows a modular pipeline:

**Data Input → Data Processing → Machine Learning → Dashboard**

* Data Input: Retail dataset (CSV)
* Processing: Cleaning, feature engineering, scaling, PCA
* ML Models: Segmentation, classification, regression
* Output: Predictions + business insights via web app

---

## 📊 Dataset Description

The dataset simulates retail transactions and includes:

* `CustomerID` – Unique customer identifier
* `InvoiceDate` – Transaction timestamp
* `Quantity` – Number of items purchased
* `UnitPrice` – Price per item
* `Country` – Customer location

### 🎯 Target Variables

* `Churn` → Customer retention indicator
* `MonetaryTotal` → Total customer spending

---

## ⚙️ Data Preprocessing

* Handling missing values (median / mode)
* Encoding categorical variables
* Feature selection & high-cardinality removal
* Outlier detection using **Isolation Forest**
* Feature scaling using **StandardScaler**

---

## 🧩 Feature Engineering

* Extraction of temporal features:

  * Year
  * Month
  * Recency
* Improves model performance and customer behavior understanding

---

## 📉 Dimensionality Reduction

* Applied **PCA (Principal Component Analysis)**
* Reduced dataset to **10 principal components**
* Benefits:

  * Noise reduction
  * Faster computation
  * Better model generalization

---

## 👥 Customer Segmentation

Using **K-Means Clustering (k = 4)**:

* 🟢 VIP Customers (high value & engagement)
* 🔵 Stable Customers
* 🟡 Occasional Customers
* 🔴 At-Risk Customers

👉 Helps in targeted marketing and retention strategies

---

## 🔮 Churn Prediction

Formulated as a **binary classification problem**

### Models used:

* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* XGBoost ✅ (Best model)

### ⚖️ Handling Imbalance:

* SMOTE (Synthetic Minority Oversampling)

### 📈 Performance:

* Accuracy: ~90%
* Best F1-score achieved by **XGBoost**

---

## 💰 Revenue Prediction

* Regression model to estimate **customer lifetime value**
* Outliers handled using percentile clipping
* Provides insights for:

  * Marketing strategies
  * Revenue optimization

---

## 🌐 Web Application

Built using **Flask**

### Features:

* Upload customer data
* Predict churn in real-time
* Estimate revenue
* Visualize customer segments
* Display business insights

---

## 🚀 Business Value

* Identify high-value customers
* Detect churn risk early
* Optimize marketing campaigns
* Improve revenue forecasting
* Enable data-driven decisions

---

## ⚠️ Limitations

* Dataset is simulated
* No real-time data streaming
* Limited explainability (no SHAP yet)

---

## 🔮 Future Work

* Add **model explainability (SHAP)**
* Integrate **real-time data pipelines**
* Use **deep learning models**
* Deploy via cloud (API + scalable backend)

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* XGBoost
* Pandas / NumPy
* Flask
* Matplotlib / Seaborn

---

## 👩‍💻 Author

**Syrine Housseini**
Engineering Student – ENIS

---

## 📄 Project Report

For more details, refer to the full report included in this repository.

---

## ⭐ Final Note

This project demonstrates how an **end-to-end machine learning pipeline** can be applied to real-world retail problems to generate actionable insights and support intelligent business decisions.
