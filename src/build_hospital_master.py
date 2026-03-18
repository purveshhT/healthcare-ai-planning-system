import pandas as pd

# Load raw files
hospital_df = pd.read_csv("data/raw/hospital_data_large.csv")
budget_df = pd.read_csv("data/raw/budget_allocation_large.csv")
directory_df = pd.read_csv("data/raw/hospital_directory.csv", low_memory=False)

# Standardize names
hospital_df["hospital_name_clean"] = hospital_df["hospital_name"].astype(str).str.strip().str.lower()
budget_df["hospital_name_clean"] = budget_df["hospital_name"].astype(str).str.strip().str.lower()
directory_df["hospital_name_clean"] = directory_df["Hospital_Name"].astype(str).str.strip().str.lower()

# Merge hospital + budget
hospital_master = pd.merge(
    hospital_df,
    budget_df,
    on="hospital_name_clean",
    how="left",
    suffixes=("_hospital", "_budget")
)

# Merge with hospital directory
hospital_master = pd.merge(
    hospital_master,
    directory_df[
        [
            "hospital_name_clean",
            "State",
            "District",
            "Specialties",
            "Number_Doctor",
            "Total_Num_Beds"
        ]
    ],
    on="hospital_name_clean",
    how="left"
)

# Rename columns
hospital_master = hospital_master.rename(columns={
    "hospital_name_hospital": "hospital_name",
    "State": "state",
    "District": "district",
    "Number_Doctor": "number_doctor",
    "Total_Num_Beds": "total_num_beds",
    "Specialties": "specialties"
})

# Derived features
hospital_master["utilization_ratio"] = hospital_master["budget_spent"] / hospital_master["budget_allocated"]
hospital_master["remaining_budget"] = hospital_master["budget_allocated"] - hospital_master["budget_spent"]

# Save
hospital_master.to_csv("data/processed/hospital_master.csv", index=False)

print("hospital_master.csv created successfully")
print(hospital_master.head())
print("Shape:", hospital_master.shape)