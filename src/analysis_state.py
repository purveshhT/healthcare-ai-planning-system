import pandas as pd

# Load engineered dataset
df = pd.read_csv("data/processed/state_features.csv")

# Top 10 states by healthcare readiness
top_readiness = df.sort_values("healthcare_readiness_score", ascending=False)[
    ["state", "healthcare_readiness_score"]
].head(10)

# Bottom 10 states by healthcare readiness
bottom_readiness = df.sort_values("healthcare_readiness_score", ascending=True)[
    ["state", "healthcare_readiness_score"]
].head(10)

# Top 10 states by budget need
top_budget_need = df.sort_values("budget_need_score", ascending=False)[
    ["state", "budget_need_score"]
].head(10)

# Top 10 states by infrastructure score
top_infra = df.sort_values("infrastructure_score", ascending=False)[
    ["state", "infrastructure_score"]
].head(10)

# Save analysis outputs
top_readiness.to_csv("data/processed/top_readiness_states.csv", index=False)
bottom_readiness.to_csv("data/processed/bottom_readiness_states.csv", index=False)
top_budget_need.to_csv("data/processed/top_budget_need_states.csv", index=False)
top_infra.to_csv("data/processed/top_infrastructure_states.csv", index=False)

print("Analysis files created successfully\n")

print("Top 10 States by Healthcare Readiness:")
print(top_readiness)

print("\nBottom 10 States by Healthcare Readiness:")
print(bottom_readiness)

print("\nTop 10 States by Budget Need:")
print(top_budget_need)

print("\nTop 10 States by Infrastructure Score:")
print(top_infra)