import pandas as pd

# Load files
hospital_df = pd.read_csv("data/raw/hospital_data_large.csv")
budget_df = pd.read_csv("data/raw/budget_allocation_large.csv")

# Standardize hospital names
hospital_df["hospital_name_clean"] = hospital_df["hospital_name"].astype(str).str.strip().str.lower()
budget_df["hospital_name_clean"] = budget_df["hospital_name"].astype(str).str.strip().str.lower()

print("Original hospital rows:", len(hospital_df))
print("Duplicate hospital names before cleaning:", hospital_df.duplicated(subset=["hospital_name_clean"]).sum())

# Remove duplicate hospital records
hospital_df = hospital_df.drop_duplicates(subset=["hospital_name_clean"])

print("Hospital rows after cleaning:", len(hospital_df))
print("Duplicate hospital names after cleaning:", hospital_df.duplicated(subset=["hospital_name_clean"]).sum())

# Keep only needed budget columns
budget_df = budget_df[
    ["hospital_name_clean", "year", "budget_allocated", "budget_spent", "remark"]
].copy()

print("Original budget rows:", len(budget_df))
print("Duplicate hospital-year rows before cleaning:",
      budget_df.duplicated(subset=["hospital_name_clean", "year"]).sum())

# Remove duplicate hospital-year records
budget_df = budget_df.drop_duplicates(subset=["hospital_name_clean", "year"])

print("Budget rows after cleaning:", len(budget_df))
print("Duplicate hospital-year rows after cleaning:",
      budget_df.duplicated(subset=["hospital_name_clean", "year"]).sum())

# Merge hospital + cleaned budget
hospital_master = pd.merge(
    hospital_df,
    budget_df,
    on="hospital_name_clean",
    how="left"
)

# Derived features
hospital_master["utilization_ratio"] = hospital_master["budget_spent"] / hospital_master["budget_allocated"]
hospital_master["remaining_budget"] = hospital_master["budget_allocated"] - hospital_master["budget_spent"]
hospital_master["budget_per_bed"] = hospital_master["budget_allocated"] / hospital_master["bed_capacity"]
hospital_master["budget_per_department"] = hospital_master["budget_allocated"] / hospital_master["department_count"]

# Save
hospital_master.to_csv("data/processed/hospital_master.csv", index=False)

print("\nhospital_master.csv created successfully")
print(hospital_master.head())
print("Shape:", hospital_master.shape)

print("\nDuplicate hospital-year rows in hospital_master:")
print(hospital_master.duplicated(subset=["hospital_name", "year"]).sum())