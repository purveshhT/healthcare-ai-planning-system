import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load state features dataset
df = pd.read_csv("data/processed/state_features.csv")

# Create a synthetic target for budget recommendation
# This acts as a proxy until a true state-wise historical budget dataset is added
df["recommended_budget"] = (
    0.35 * df["total_beds_all_levels"] +
    0.30 * df["total_doctors"] +
    0.20 * df["total_vaccination_doses"] / 1000 +
    0.15 * df["nabh_accredited_hospitals"]
)

# Fill missing values
df = df.fillna(0)

# Features
feature_cols = [
    "total_gov_hospitals",
    "total_gov_beds",
    "total_doctors",
    "total_vaccination_doses",
    "total_sub_centres",
    "total_phcs",
    "total_chcs",
    "nabh_accredited_hospitals",
    "infrastructure_score",
    "healthcare_readiness_score",
    "budget_need_score"
]

X = df[feature_cols]
y = df["recommended_budget"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42, n_estimators=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model trained successfully")
print("MAE:", mae)
print("R2 Score:", r2)

# Predict budgets for all states
df["predicted_budget_2028_29"] = model.predict(X)

# Save predictions
df[["state", "recommended_budget", "predicted_budget_2028_29"]].to_csv(
    "data/processed/state_budget_predictions.csv", index=False
)

print("\nstate_budget_predictions.csv created successfully")
print(df[["state", "recommended_budget", "predicted_budget_2028_29"]].head(10))