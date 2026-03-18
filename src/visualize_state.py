import os
import pandas as pd
import matplotlib.pyplot as plt

# Create output folder if not exists
os.makedirs("reports/figures", exist_ok=True)

# Load dataset
df = pd.read_csv("data/processed/state_features.csv")

# ----------------------------
# Top 10 Healthcare Readiness
# ----------------------------
top_readiness = df.sort_values("healthcare_readiness_score", ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.bar(top_readiness["state"], top_readiness["healthcare_readiness_score"])
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 States by Healthcare Readiness Score")
plt.xlabel("State")
plt.ylabel("Healthcare Readiness Score")
plt.tight_layout()
plt.savefig("reports/figures/top_readiness_states.png")
plt.close()

# ----------------------------
# Top 10 Budget Need
# ----------------------------
top_budget_need = df.sort_values("budget_need_score", ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.bar(top_budget_need["state"], top_budget_need["budget_need_score"])
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 States by Budget Need Score")
plt.xlabel("State")
plt.ylabel("Budget Need Score")
plt.tight_layout()
plt.savefig("reports/figures/top_budget_need_states.png")
plt.close()

# ----------------------------
# Top 10 Infrastructure
# ----------------------------
top_infra = df.sort_values("infrastructure_score", ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.bar(top_infra["state"], top_infra["infrastructure_score"])
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 States by Infrastructure Score")
plt.xlabel("State")
plt.ylabel("Infrastructure Score")
plt.tight_layout()
plt.savefig("reports/figures/top_infrastructure_states.png")
plt.close()

print("Charts created successfully in reports/figures/")