# src/vaccine_prediction_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/state_features.csv")
df = df.fillna(0)

# Vaccine demand features
feature_cols = [
    "total_gov_hospitals", "total_sub_centres", "total_phcs", "total_chcs",
    "total_beds_all_levels", "total_doctors", "total_vaccination_doses",
    "infrastructure_score", "healthcare_readiness_score"
]

# Synthetic growth target: 7% annual growth over 4 years = ~1.31x
df["vaccine_target_2028"] = df["total_vaccination_doses"] * 1.31

X = df[feature_cols]
y = df["vaccine_target_2028"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

df["predicted_vaccine_doses_2028_29"] = model.predict(X)

# Derive cold chain units needed (1 cold chain unit per 50,000 doses)
df["cold_chain_units_needed"] = (df["predicted_vaccine_doses_2028_29"] / 50000).apply(np.ceil).astype(int)

# Vaccine budget: approx Rs 45 per dose + cold chain capex
df["vaccine_budget_2028_29"] = (
    df["predicted_vaccine_doses_2028_29"] * 45 +
    df["cold_chain_units_needed"] * 200000
) / 1e7  # in Crores

df[[
    "state", "total_vaccination_doses",
    "predicted_vaccine_doses_2028_29",
    "cold_chain_units_needed",
    "vaccine_budget_2028_29"
]].to_csv("data/processed/vaccine_predictions_2028_29.csv", index=False)

print("vaccine_predictions_2028_29.csv created successfully")
print(df[["state", "predicted_vaccine_doses_2028_29", "vaccine_budget_2028_29"]].head())
