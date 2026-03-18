import pandas as pd

# Load engineered dataset
df = pd.read_csv("data/processed/state_features.csv")

recommendations = []

for _, row in df.iterrows():
    state = row["state"]
    recs = []

    if row["healthcare_readiness_score"] < df["healthcare_readiness_score"].median():
        recs.append("Increase healthcare readiness through additional staffing and infrastructure investment.")

    if row["budget_need_score"] > df["budget_need_score"].median():
        recs.append("Prioritize higher budget allocation for upcoming planning cycle.")

    if row["beds_per_gov_hospital"] < df["beds_per_gov_hospital"].median():
        recs.append("Expand hospital bed capacity and strengthen district-level bed availability.")

    if row["doctors_per_gov_hospital"] < df["doctors_per_gov_hospital"].median():
        recs.append("Improve doctor availability through targeted recruitment and redistribution.")

    if row["nabh_accredited_hospitals"] < df["nabh_accredited_hospitals"].median():
        recs.append("Promote quality improvement and accreditation support for hospitals.")

    if row["vaccination_per_bed"] > df["vaccination_per_bed"].median():
        recs.append("Strengthen vaccine logistics and supply chain planning for high service demand.")

    if not recs:
        recs.append("Maintain current healthcare performance and continue monitoring resource utilization.")

    recommendations.append({
        "state": state,
        "recommendation": " ".join(recs)
    })

recommendation_df = pd.DataFrame(recommendations)
recommendation_df.to_csv("data/processed/state_recommendations.csv", index=False)

print("state_recommendations.csv created successfully")
print(recommendation_df.head(10))