import pandas as pd
import numpy as np

# Load state master dataset
df = pd.read_csv("data/processed/state_master.csv")

# ----------------------------
# Fill missing values
# ----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# ----------------------------
# Feature Engineering
# ----------------------------

# Beds per government hospital
df["beds_per_gov_hospital"] = df["total_gov_beds"] / df["total_gov_hospitals"].replace(0, np.nan)

# Doctors per government hospital
df["doctors_per_gov_hospital"] = df["total_doctors"] / df["total_gov_hospitals"].replace(0, np.nan)

# Vaccination intensity relative to total beds
df["vaccination_per_bed"] = df["total_vaccination_doses"] / df["total_beds_all_levels"].replace(0, np.nan)

# PHC to CHC support ratio
df["phc_chc_ratio"] = df["phc_beds"] / df["chc_beds"].replace(0, np.nan)

# Medical college bed share
df["medical_college_bed_ratio"] = df["medical_college_beds"] / df["total_beds_all_levels"].replace(0, np.nan)

# District hospital bed share
df["district_hospital_bed_ratio"] = df["district_hospital_beds"] / df["total_beds_all_levels"].replace(0, np.nan)

# Infrastructure score (simple combined proxy)
df["infrastructure_score"] = (
    df["total_beds_all_levels"] +
    df["total_sub_centres"] +
    df["total_phcs"] +
    df["total_chcs"] +
    df["nabh_accredited_hospitals"]
)

# Healthcare readiness score
df["healthcare_readiness_score"] = (
    0.30 * df["total_doctors"] +
    0.25 * df["total_beds_all_levels"] +
    0.15 * df["nabh_accredited_hospitals"] +
    0.15 * df["total_phcs"] +
    0.15 * df["total_chcs"]
)

# Hospital adequacy score
df["hospital_adequacy_score"] = (
    0.5 * df["beds_per_gov_hospital"].fillna(0) +
    0.5 * df["doctors_per_gov_hospital"].fillna(0)
)

# Budget need proxy score
df["budget_need_score"] = (
    0.35 * (1 / (df["beds_per_gov_hospital"].replace(0, np.nan))).fillna(0) +
    0.35 * (1 / (df["doctors_per_gov_hospital"].replace(0, np.nan))).fillna(0) +
    0.30 * df["vaccination_per_bed"].fillna(0)
)

# Replace inf/nan after division
df = df.replace([np.inf, -np.inf], 0)
df = df.fillna(0)

# ----------------------------
# Save engineered dataset
# ----------------------------
df.to_csv("data/processed/state_features.csv", index=False)

print("state_features.csv created successfully")
print(df.head())
print("Shape:", df.shape)

print("\nNew engineered columns:")
engineered_cols = [
    "beds_per_gov_hospital",
    "doctors_per_gov_hospital",
    "vaccination_per_bed",
    "phc_chc_ratio",
    "medical_college_bed_ratio",
    "district_hospital_bed_ratio",
    "infrastructure_score",
    "healthcare_readiness_score",
    "hospital_adequacy_score",
    "budget_need_score"
]
print(engineered_cols)