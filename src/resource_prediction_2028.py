# src/resource_prediction_2028.py
import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/state_features.csv")
df = df.fillna(0)

# WHO norm: 3 beds per 1000 population
# Estimate population from vaccination doses as proxy (1 dose ~= 1 person coverage target)
df["estimated_population"] = df["total_vaccination_doses"] / 2  # 2 doses per person assumption

# Beds needed at WHO norm by 2028-29 (4% population growth)
df["population_2028"] = df["estimated_population"] * 1.17  # 4% over 4 years
df["beds_needed_2028_29"] = (df["population_2028"] / 1000 * 3).apply(np.ceil).astype(int)
df["bed_gap_2028_29"] = (df["beds_needed_2028_29"] - df["total_beds_all_levels"]).clip(lower=0)

# Hospitals needed: 1 CHC per 100 beds gap
df["new_hospitals_needed_2028_29"] = (df["bed_gap_2028_29"] / 100).apply(np.ceil).astype(int)

# Doctors needed: 1 doctor per 1000 population (WHO)
df["doctors_needed_2028_29"] = (df["population_2028"] / 1000).apply(np.ceil).astype(int)
df["doctor_gap_2028_29"] = (df["doctors_needed_2028_29"] - df["total_doctors"]).clip(lower=0)

# Infrastructure budget in Crores
# New hospital: Rs 10Cr each, new bed: Rs 5L each, new doctor hire: Rs 15L/yr
df["hospital_infra_budget_2028_29"] = (
    df["new_hospitals_needed_2028_29"] * 10 +
    df["bed_gap_2028_29"] * 0.05 +
    df["doctor_gap_2028_29"] * 0.15
)

df[[
    "state",
    "beds_needed_2028_29", "bed_gap_2028_29",
    "new_hospitals_needed_2028_29",
    "doctors_needed_2028_29", "doctor_gap_2028_29",
    "hospital_infra_budget_2028_29"
]].to_csv("data/processed/resource_predictions_2028_29.csv", index=False)

print("resource_predictions_2028_29.csv created successfully")
print(df[["state", "beds_needed_2028_29", "new_hospitals_needed_2028_29", "doctor_gap_2028_29"]].head())
