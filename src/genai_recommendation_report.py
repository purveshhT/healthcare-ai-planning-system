import pandas as pd

# Load processed files
features_df = pd.read_csv("data/processed/state_features.csv")
pred_df = pd.read_csv("data/processed/state_budget_predictions.csv")
rec_df = pd.read_csv("data/processed/state_recommendations.csv")

# Merge them
df = features_df.merge(pred_df, on="state", how="left")
df = df.merge(rec_df, on="state", how="left")

reports = []

for _, row in df.iterrows():
    report = f"""
State: {row['state']}

Healthcare Readiness Score: {row['healthcare_readiness_score']:.2f}
Infrastructure Score: {row['infrastructure_score']:.2f}
Budget Need Score: {row['budget_need_score']:.2f}

Predicted Budget for 2028-29: {row['predicted_budget_2028_29']:.2f}

AI Recommendation:
{row['recommendation']}

Summary:
The healthcare system in {row['state']} is evaluated using infrastructure, doctor availability,
hospital capacity, vaccination demand, and accreditation indicators. Based on the projected
resource requirement and current readiness, the predicted budget for 2028-29 is
{row['predicted_budget_2028_29']:.2f}. Strategic improvements should focus on the
identified gaps to strengthen future healthcare planning and delivery.
""".strip()

    reports.append({
        "state": row["state"],
        "genai_report": report
    })

report_df = pd.DataFrame(reports)
report_df.to_csv("data/processed/genai_state_reports.csv", index=False)

print("genai_state_reports.csv created successfully")
print(report_df.head(5).to_string(index=False))