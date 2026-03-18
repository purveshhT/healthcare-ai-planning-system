import pandas as pd

# Load files
gov_hosp_df = pd.read_csv("data/raw/RS_Session_259_AU_595_A_i.csv")
beds_df = pd.read_csv("data/raw/RS_Session_266_AU_911_C_to_D_iii.csv")
doctors_df = pd.read_csv("data/raw/RS_Session_259_AU_573_A_to_D_i.csv")
vacc_df = pd.read_csv("data/raw/RS_Session_254_AU_1705.csv")
rural_df = pd.read_csv("data/raw/District-Wise_Rural_HealthCare_Infrastructure_1.csv")
nabh_df = pd.read_csv("data/raw/RS_Session_267_AU_288_A_to_C_i_0.csv")

# ----------------------------
# Create raw state columns
# ----------------------------
gov_hosp_df["state"] = gov_hosp_df["State/UT/Division"].astype(str).str.strip()
beds_df["state"] = beds_df["State/UT"].astype(str).str.strip()
vacc_df["state"] = vacc_df["State/UT"].astype(str).str.strip()
nabh_df["state"] = nabh_df["State/UT"].astype(str).str.strip()
rural_df["state"] = rural_df["States/Union Territory"].astype(str).str.strip()

doctors_df["state"] = (
    doctors_df["Name of State Medical Council"]
    .astype(str)
    .str.replace(" Medical Council", "", regex=False)
    .str.strip()
)

# ----------------------------
# State name normalization map
# ----------------------------
state_map = {
    "A & N Islands": "Andaman and Nicobar Islands",
    "Andaman & Nicobar Islands": "Andaman and Nicobar Islands",
    "Dadra & Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "Dadra and Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu",
    "Daman and Diu": "Dadra and Nagar Haveli and Daman and Diu",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "NCT of Delhi": "Delhi",
    "Delhi#": "Delhi",
    "Orissa": "Odisha",
    "Orissa Council of Medical Registration": "Odisha",
    "Pondicherry": "Puducherry",
    "Puducherry #": "Puducherry",
    "Uttaranchal": "Uttarakhand",
    "Uttar Pradesh*(#)": "Uttar Pradesh",
    "Kerala#": "Kerala",
    "Chattisgarh": "Chhattisgarh",
    "Travancore": "Kerala",
    "Erstwhile of India": None,
    "All India/Total": None,
    "India": None,
    "All India": None,
    "Total": None
}

def normalize_state(series):
    series = series.astype(str).str.strip()
    series = series.replace(state_map)
    return series

gov_hosp_df["state"] = normalize_state(gov_hosp_df["state"])
beds_df["state"] = normalize_state(beds_df["state"])
doctors_df["state"] = normalize_state(doctors_df["state"])
vacc_df["state"] = normalize_state(vacc_df["state"])
rural_df["state"] = normalize_state(rural_df["state"])
nabh_df["state"] = normalize_state(nabh_df["state"])

# Drop rows where state becomes None
gov_hosp_df = gov_hosp_df.dropna(subset=["state"])
beds_df = beds_df.dropna(subset=["state"])
doctors_df = doctors_df.dropna(subset=["state"])
vacc_df = vacc_df.dropna(subset=["state"])
rural_df = rural_df.dropna(subset=["state"])
nabh_df = nabh_df.dropna(subset=["state"])

# ----------------------------
# Aggregate rural district data to state level
# ----------------------------
rural_state = rural_df.groupby("state", as_index=False).agg({
    "Number of Sub Centres": "sum",
    "Number of  Primary Health Centres": "sum",
    "Number of Community Health Centres": "sum",
    "Sub Divisional Hospital": "sum",
    "District Hospital": "sum"
})

# ----------------------------
# Keep useful columns
# ----------------------------
gov_hosp_df = gov_hosp_df[[
    "state",
    "Total Hospital (Government ) - Number",
    "Total Hospital (Government ) - Beds"
]]

beds_df = beds_df[[
    "state",
    "PHC",
    "CHC",
    "SUB DISTRICT/ SUB DIVISIONAL HOSPITAL",
    "DISTRICT HOSPITAL",
    "MEDICAL COLLEGE",
    "Total No. of Beds"
]]

doctors_df = doctors_df[[
    "state",
    "Total Number of Allopathic Doctors"
]]

vacc_df["total_vaccination_doses"] = (
    vacc_df["Doses Administered - Male"] + vacc_df["Doses Administered - Female"]
)

vacc_df = vacc_df[[
    "state",
    "Doses Administered - Male",
    "Doses Administered - Female",
    "total_vaccination_doses"
]]

nabh_df = nabh_df[[
    "state",
    "No. of NABH Accredited Hospitals"
]]

# ----------------------------
# Remove duplicates by state
# ----------------------------
gov_hosp_df = gov_hosp_df.groupby("state", as_index=False).first()
beds_df = beds_df.groupby("state", as_index=False).first()
doctors_df = doctors_df.groupby("state", as_index=False).first()
vacc_df = vacc_df.groupby("state", as_index=False).first()
nabh_df = nabh_df.groupby("state", as_index=False).first()

# ----------------------------
# Merge datasets
# ----------------------------
state_master = gov_hosp_df.merge(beds_df, on="state", how="outer")
state_master = state_master.merge(doctors_df, on="state", how="outer")
state_master = state_master.merge(vacc_df, on="state", how="outer")
state_master = state_master.merge(rural_state, on="state", how="outer")
state_master = state_master.merge(nabh_df, on="state", how="outer")

# ----------------------------
# Rename columns
# ----------------------------
state_master = state_master.rename(columns={
    "Total Hospital (Government ) - Number": "total_gov_hospitals",
    "Total Hospital (Government ) - Beds": "total_gov_beds",
    "PHC": "phc_beds",
    "CHC": "chc_beds",
    "SUB DISTRICT/ SUB DIVISIONAL HOSPITAL": "sub_divisional_hospital_beds",
    "DISTRICT HOSPITAL": "district_hospital_beds",
    "MEDICAL COLLEGE": "medical_college_beds",
    "Total No. of Beds": "total_beds_all_levels",
    "Total Number of Allopathic Doctors": "total_doctors",
    "Doses Administered - Male": "doses_male",
    "Doses Administered - Female": "doses_female",
    "Number of Sub Centres": "total_sub_centres",
    "Number of  Primary Health Centres": "total_phcs",
    "Number of Community Health Centres": "total_chcs",
    "Sub Divisional Hospital": "total_sub_divisional_hospitals",
    "District Hospital": "total_district_hospitals",
    "No. of NABH Accredited Hospitals": "nabh_accredited_hospitals"
})

# Sort nicely
state_master = state_master.sort_values("state").reset_index(drop=True)

# Save
state_master.to_csv("data/processed/state_master.csv", index=False)

print("state_master.csv created successfully")
print(state_master.head())
print("Shape:", state_master.shape)

print("\nMissing values:")
print(state_master.isnull().sum())

print("\nStates in final dataset:")
print(state_master["state"].tolist())