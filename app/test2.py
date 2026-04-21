import pandas as pd
import numpy as np

np.random.seed(42)

# =========================
# CONFIG
# =========================
N_SAMPLES = 200

# =========================
# GENERATE SYNTHETIC CUSTOMER DATA
# =========================
def generate_data(n=N_SAMPLES):

    data = {}

    # --- Basic numeric behavior features ---
    data["Recency"] = np.random.randint(1, 365, n)
    data["Frequency"] = np.random.randint(1, 50, n)
    data["MonetaryTotal"] = np.random.randint(20, 5000, n)

    data["MonetaryAvg"] = data["MonetaryTotal"] / (data["Frequency"] + 1)

    data["TotalQuantity"] = np.random.randint(1, 200, n)
    data["TotalTransactions"] = np.random.randint(1, 30, n)

    # --- Time features ---
    data["RegYear"] = np.random.choice([2021, 2022, 2023, 2024], n)
    data["RegMonth"] = np.random.randint(1, 13, n)

    # --- Risk signals ---
    data["LatePayments"] = np.random.randint(0, 10, n)
    data["SupportTickets"] = np.random.randint(0, 15, n)

    # --- Categorical features ---
    data["AccountStatus"] = np.random.choice(["Active", "Inactive", "Suspended"], n)
    data["ChurnRiskCategory"] = np.random.choice(["Low", "Medium", "High"], n)

    data["NewsletterSubscribed"] = np.random.choice([0, 1], n)

    # --- IDs (will be dropped but needed for realism) ---
    data["CustomerID"] = np.arange(1000, 1000 + n)

    # =========================
    # SYNTHETIC CHURN LOGIC
    # =========================
    churn_score = (
        data["Recency"] * 0.02 +
        data["LatePayments"] * 0.3 +
        data["SupportTickets"] * 0.25 -
        data["Frequency"] * 0.05
    )

    prob = 1 / (1 + np.exp(-0.1 * (np.array(churn_score) - 5)))

    data["Churn"] = (prob > 0.5).astype(int)

    # =========================
    # SYNTHETIC REVENUE LOGIC
    # =========================
    revenue = (
        data["MonetaryTotal"] * np.random.uniform(0.8, 1.2, n)
        - data["Recency"] * 2
        + data["Frequency"] * 10
    )

    data["MonetaryTotal"] = np.maximum(revenue, 10)

    # =========================
    # CREATE DATAFRAME
    # =========================
    df = pd.DataFrame(data)

    return df


# =========================
# SAVE FILE
# =========================
if __name__ == "__main__":
    df = generate_data(300)

    df.to_csv("test_customers.csv", index=False)

    print("✅ Test dataset created: test_customers.csv")
    print(df.head())