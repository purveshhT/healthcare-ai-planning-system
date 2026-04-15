# src/recommendation_engine.py  — UPGRADED
import pandas as pd

features_df = pd.read_csv("data/processed/state_features.csv")
vaccine_df = pd.read_csv("data/processed/vaccine_predictions_2028_29.csv")
resource_df = pd.read_csv("data/processed/resource_predictions_2028_29.csv")
budget_df = pd.read_csv("data/processed/budget_breakdown_2028_29.csv")

# Select only new columns from vaccine_df (avoid duplicate total_vaccination_doses)
vaccine_cols = ["state", "predicted_vaccine_doses_2028_29", "cold_chain_units_needed", "vaccine_budget_2028_29"]
df = features_df.merge(vaccine_df[vaccine_cols], on="state", how="left")
df = df.merge(resource_df, on="state", how="left")
df = df.merge(budget_df[["state", "total_recommended_budget_2028_29", "budget_priority_tier",
                          "vaccine_budget_pct", "infra_budget_pct", "ops_budget_pct"]], on="state", how="left")
df = df.fillna(0)


recommendations = []

for _, row in df.iterrows():
    state = row["state"]
    recs = {"state": state, "vaccine_rec": "", "hospital_rec": "", "doctor_rec": "",
            "supply_chain_rec": "", "budget_rec": "", "priority": row.get("budget_priority_tier", "Medium Priority")}

    # Vaccine recommendation
    dose_gap = row["predicted_vaccine_doses_2028_29"] - row["total_vaccination_doses"]
    if dose_gap > df["predicted_vaccine_doses_2028_29"].quantile(0.75):
        recs["vaccine_rec"] = (
            f"URGENT: Scale up vaccine procurement by {dose_gap:,.0f} doses for 2028-29. "
            f"Deploy {row['cold_chain_units_needed']} additional cold chain units. "
            "Establish last-mile delivery partnerships with district health offices."
        )
    elif dose_gap > 0:
        recs["vaccine_rec"] = (
            f"Increase vaccine stock by {dose_gap:,.0f} doses. "
            f"Maintain {row['cold_chain_units_needed']} cold chain units for distribution."
        )
    else:
        recs["vaccine_rec"] = "Vaccine coverage is sufficient. Monitor seasonal demand spikes."

    # Hospital recommendation
    if row["new_hospitals_needed_2028_29"] > 5:
        recs["hospital_rec"] = (
            f"Build {row['new_hospitals_needed_2028_29']} new hospitals. "
            f"Add {row['bed_gap_2028_29']:,.0f} beds to meet WHO standard of 3 beds/1000 population."
        )
    elif row["new_hospitals_needed_2028_29"] > 0:
        recs["hospital_rec"] = (
            f"Expand capacity in {row['new_hospitals_needed_2028_29']} existing facilities. "
            f"Bridge {row['bed_gap_2028_29']:,.0f} bed gap through PPP models."
        )
    else:
        recs["hospital_rec"] = "Hospital capacity meets projected demand. Focus on quality upgrades and NABH accreditation."

    # Doctor recommendation
    if row["doctor_gap_2028_29"] > df["doctor_gap_2028_29"].quantile(0.75):
        recs["doctor_rec"] = (
            f"Critical doctor shortage projected: recruit {row['doctor_gap_2028_29']:,.0f} doctors by 2028-29. "
            "Open new medical college seats and launch rural health incentive schemes."
        )
    elif row["doctor_gap_2028_29"] > 0:
        recs["doctor_rec"] = (
            f"Recruit {row['doctor_gap_2028_29']:,.0f} additional doctors. "
            "Incentivize rural postings with additional pay and housing allowances."
        )
    else:
        recs["doctor_rec"] = "Doctor availability meets WHO norms. Invest in specialist training and skill upgrades."

    # Supply chain recommendation
    if row["vaccination_per_bed"] > df["vaccination_per_bed"].quantile(0.75):
        recs["supply_chain_rec"] = (
            "High vaccine-to-bed ratio indicates supply chain stress. "
            "Establish regional vaccine warehouses and strengthen transport fleet. "
            "Implement IoT-based cold chain monitoring."
        )
    else:
        recs["supply_chain_rec"] = (
            "Supply chain is manageable. Digitize inventory tracking with barcode/RFID system. "
            "Conduct quarterly stock audits."
        )

    # Budget recommendation
    recs["budget_rec"] = (
        f"Recommended total budget for 2028-29: Rs {row['total_recommended_budget_2028_29']:,.2f} Cr. "
        f"Breakdown — Operations: {row.get('ops_budget_pct', 0):.1f}%, "
        f"Vaccine: {row.get('vaccine_budget_pct', 0):.1f}%, "
        f"Infrastructure: {row.get('infra_budget_pct', 0):.1f}%. "
        f"Priority Tier: {row.get('budget_priority_tier', 'Medium Priority')}."
    )

    # Combined recommendation (for backward compatibility)
    recs["recommendation"] = " | ".join([
        recs["vaccine_rec"], recs["hospital_rec"],
        recs["doctor_rec"], recs["supply_chain_rec"], recs["budget_rec"]
    ])

    recommendations.append(recs)

rec_df = pd.DataFrame(recommendations)
rec_df.to_csv("data/processed/state_recommendations.csv", index=False)
print("state_recommendations.csv upgraded successfully")
print(rec_df[["state", "priority", "budget_rec"]].head(5))