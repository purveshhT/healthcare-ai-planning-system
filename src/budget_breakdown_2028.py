# src/budget_breakdown_2028.py
import pandas as pd

budget_df = pd.read_csv("data/processed/state_budget_predictions.csv")
vaccine_df = pd.read_csv("data/processed/vaccine_predictions_2028_29.csv")
resource_df = pd.read_csv("data/processed/resource_predictions_2028_29.csv")

df = budget_df.merge(vaccine_df[["state", "vaccine_budget_2028_29"]], on="state", how="left")
df = df.merge(resource_df[["state", "hospital_infra_budget_2028_29"]], on="state", how="left")
df = df.fillna(0)

# Total recommended budget = model prediction + vaccine budget + infra budget
df["total_recommended_budget_2028_29"] = (
    df["predicted_budget_2028_29"] +
    df["vaccine_budget_2028_29"] +
    df["hospital_infra_budget_2028_29"]
)

# Budget split percentages
df["vaccine_budget_pct"] = (df["vaccine_budget_2028_29"] / df["total_recommended_budget_2028_29"] * 100).round(1)
df["infra_budget_pct"] = (df["hospital_infra_budget_2028_29"] / df["total_recommended_budget_2028_29"] * 100).round(1)
df["ops_budget_pct"] = (df["predicted_budget_2028_29"] / df["total_recommended_budget_2028_29"] * 100).round(1)

# Priority tier
def classify_priority(row):
    if row["total_recommended_budget_2028_29"] > df["total_recommended_budget_2028_29"].quantile(0.75):
        return "Critical Priority"
    elif row["total_recommended_budget_2028_29"] > df["total_recommended_budget_2028_29"].quantile(0.50):
        return "High Priority"
    elif row["total_recommended_budget_2028_29"] > df["total_recommended_budget_2028_29"].quantile(0.25):
        return "Medium Priority"
    else:
        return "Low Priority"

df["budget_priority_tier"] = df.apply(classify_priority, axis=1)

df.to_csv("data/processed/budget_breakdown_2028_29.csv", index=False)
print("budget_breakdown_2028_29.csv created successfully")
print(df[["state", "total_recommended_budget_2028_29", "budget_priority_tier"]].head(10))
