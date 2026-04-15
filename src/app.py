import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Healthcare AI Planning System", layout="wide")

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_state_data():
    state_features = pd.read_csv("data/processed/state_features.csv")
    budget_predictions = pd.read_csv("data/processed/state_budget_predictions.csv")
    recommendations = pd.read_csv("data/processed/state_recommendations.csv")
    genai_reports = pd.read_csv("data/processed/genai_state_reports.csv")

    df = state_features.merge(budget_predictions, on="state", how="left")
    df = df.merge(recommendations, on="state", how="left")
    df = df.merge(genai_reports, on="state", how="left")
    return df

@st.cache_data
def load_hospital_data():
    return pd.read_csv("data/processed/hospital_master.csv")

@st.cache_data
def load_vaccine_data():
    return pd.read_csv("data/processed/vaccine_predictions_2028_29.csv")

@st.cache_data
def load_resource_data():
    return pd.read_csv("data/processed/resource_predictions_2028_29.csv")

@st.cache_data
def load_budget_breakdown():
    return pd.read_csv("data/processed/budget_breakdown_2028_29.csv")

state_df = load_state_data()
hospital_df = load_hospital_data()
vaccine_df_pred = load_vaccine_data()
resource_df_pred = load_resource_data()
budget_breakdown_df = load_budget_breakdown()

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.title("Healthcare AI Planning System")
page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "State Dashboard",
        "Hospital Dashboard",
        "Vaccine & Supply Chain",
        "2028-29 Predictions",
        "Budget Recommendations",
        "Downloads"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Info")
st.sidebar.write(f"States covered: {state_df['state'].nunique()}")
st.sidebar.write(f"Hospitals covered: {hospital_df['hospital_name'].nunique()}")

# ----------------------------
# Page 1: Project Overview
# ----------------------------
if page == "Project Overview":
    st.title("Healthcare AI Planning System")
    st.markdown("### AI-powered resource analysis, budget prediction, and recommendation platform")

    st.info(
        "This system analyzes healthcare infrastructure, doctor availability, hospital capacity, "
        "vaccination demand, and budget requirements to support state-wise healthcare planning."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("States Covered", f"{state_df['state'].nunique()}")
    c2.metric("Hospitals Covered", f"{hospital_df['hospital_name'].nunique()}")
    c3.metric("Avg Readiness Score", f"{state_df['healthcare_readiness_score'].mean():.2f}")
    c4.metric("Avg Predicted Budget", f"{state_df['predicted_budget_2028_29'].mean():.2f}")

    st.markdown("## Project Modules")
    st.success("Healthcare Resource Analysis")
    st.success("Supply Chain Recommendation")
    st.success("Budget Allocation Model")
    st.success("Future Prediction (2028–29)")
    st.success("Generative AI Recommendation")
    st.success("Vaccine & Supply Chain Forecasting")
    st.success("Hospital & Doctor Gap Analysis 2028-29")
    st.success("3-Way Budget Breakdown (Operations / Vaccine / Infrastructure)")

    st.markdown("## About the Dashboard")
    st.write(
        "Use the sidebar to navigate between the state-level dashboard, hospital-level dashboard, "
        "vaccine & supply chain forecasts, 2028-29 resource predictions, budget recommendations, "
        "and downloadable processed outputs."
    )

# ----------------------------
# Page 2: State Dashboard
# ----------------------------
elif page == "State Dashboard":
    st.title("State Dashboard")

    selected_state = st.sidebar.selectbox(
        "Select a State",
        sorted(state_df["state"].unique())
    )

    state_data = state_df[state_df["state"] == selected_state].iloc[0]

    st.markdown(f"## State Overview: {selected_state}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Healthcare Readiness Score", f"{state_data['healthcare_readiness_score']:.2f}")
    col2.metric("Infrastructure Score", f"{state_data['infrastructure_score']:.2f}")
    col3.metric("Budget Need Score", f"{state_data['budget_need_score']:.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Predicted Budget 2028-29", f"{state_data['predicted_budget_2028_29']:.2f}")
    col5.metric("Total Doctors", f"{state_data['total_doctors']:.0f}")
    col6.metric("Total Beds", f"{state_data['total_beds_all_levels']:.0f}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Readiness Analysis",
        "Budget Analysis",
        "Recommendations",
        "GenAI Report"
    ])

    with tab1:
        st.subheader("Top 10 States by Healthcare Readiness")
        top_readiness = state_df.sort_values("healthcare_readiness_score", ascending=False).head(10)
        st.bar_chart(top_readiness.set_index("state")["healthcare_readiness_score"])

        st.subheader("Top 10 States by Infrastructure Score")
        top_infra = state_df.sort_values("infrastructure_score", ascending=False).head(10)
        st.bar_chart(top_infra.set_index("state")["infrastructure_score"])

        st.dataframe(
            top_readiness[["state", "healthcare_readiness_score", "infrastructure_score"]],
            width="stretch"
        )

    with tab2:
        st.subheader("Top 10 States by Budget Need")
        top_budget_need = state_df.sort_values("budget_need_score", ascending=False).head(10)
        st.bar_chart(top_budget_need.set_index("state")["budget_need_score"])

        st.dataframe(
            top_budget_need[["state", "budget_need_score", "predicted_budget_2028_29"]],
            width="stretch"
        )

    with tab3:
        st.subheader(f"AI Recommendation for {selected_state}")
        st.info(state_data["recommendation"])

        st.dataframe(
            state_df.sort_values("budget_need_score", ascending=False)[
                ["state", "budget_need_score", "recommendation"]
            ].head(10),
            width="stretch"
        )

    with tab4:
        st.subheader(f"Generative AI Report for {selected_state}")
        st.text(state_data["genai_report"])

# ----------------------------
# Page 3: Hospital Dashboard
# ----------------------------
elif page == "Hospital Dashboard":
    st.title("Hospital Dashboard")

    selected_hospital = st.sidebar.selectbox(
        "Select a Hospital",
        sorted(hospital_df["hospital_name"].unique())
    )

    hospital_data = hospital_df[hospital_df["hospital_name"] == selected_hospital].iloc[0]

    st.markdown(f"## Hospital Overview: {selected_hospital}")

    c1, c2, c3 = st.columns(3)
    c1.metric("City", str(hospital_data["city"]))
    c2.metric("Bed Capacity", f"{hospital_data['bed_capacity']:.0f}")
    c3.metric("Department Count", f"{hospital_data['department_count']:.0f}")

    # Hospital latest budget row
    latest_hospital_rows = hospital_df[hospital_df["hospital_name"] == selected_hospital].sort_values("year")
    latest_row = latest_hospital_rows.iloc[-1]

    c4, c5, c6 = st.columns(3)
    c4.metric("Budget Allocated", f"{latest_row['budget_allocated']:.2f}")
    c5.metric("Budget Spent", f"{latest_row['budget_spent']:.2f}")
    c6.metric("Utilization Ratio", f"{latest_row['utilization_ratio']:.2f}")

    c7, c8 = st.columns(2)
    c7.metric("Budget Per Bed", f"{latest_row['budget_per_bed']:.2f}")
    c8.metric("Budget Per Department", f"{latest_row['budget_per_department']:.2f}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Budget Utilization",
        "Top Spending Hospitals",
        "Low Capacity Hospitals",
        "Hospital Data"
    ])

    with tab1:
        st.subheader("Budget Utilization Over Years")
        util_df = latest_hospital_rows[["year", "budget_allocated", "budget_spent"]].copy()
        util_df = util_df.set_index("year")
        st.line_chart(util_df)

        st.subheader("Hospital Budget Records")
        st.dataframe(
            latest_hospital_rows[
                ["hospital_name", "year", "budget_allocated", "budget_spent", "utilization_ratio", "budget_per_bed"]
            ],
            width="stretch"
        )

    with tab2:
        st.subheader("Top 10 Hospitals by Budget Spent")
        top_spending = hospital_df.sort_values("budget_spent", ascending=False)[
            ["hospital_name", "city", "budget_spent", "budget_allocated", "utilization_ratio"]
        ].head(10)
        st.bar_chart(top_spending.set_index("hospital_name")["budget_spent"])
        st.dataframe(top_spending, width="stretch")

    with tab3:
        st.subheader("Low Capacity Hospitals")
        low_capacity = hospital_df.sort_values("bed_capacity", ascending=True)[
            ["hospital_name", "city", "bed_capacity", "department_count", "budget_allocated"]
        ].head(10)
        st.bar_chart(low_capacity.set_index("hospital_name")["bed_capacity"])
        st.dataframe(low_capacity, width="stretch")

    with tab4:
        st.subheader("Hospital Dataset Preview")
        st.dataframe(hospital_df.head(20), width="stretch")

# ----------------------------
# Page 4: Vaccine & Supply Chain
# ----------------------------
elif page == "Vaccine & Supply Chain":
    st.title("🧪 Vaccine & Supply Chain Dashboard")

    selected_state = st.sidebar.selectbox(
        "Select a State", sorted(vaccine_df_pred["state"].unique())
    )

    sv = vaccine_df_pred[vaccine_df_pred["state"] == selected_state].iloc[0]
    rec_row = state_df[state_df["state"] == selected_state]

    st.markdown(f"## {selected_state} — Vaccine Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Doses", f"{sv['total_vaccination_doses']:,.0f}")
    c2.metric("Predicted 2028-29 Doses", f"{sv['predicted_vaccine_doses_2028_29']:,.0f}")
    c3.metric("Cold Chain Units Needed", f"{sv['cold_chain_units_needed']}")
    c4.metric("Vaccine Budget 2028-29 (₹Cr)", f"{sv['vaccine_budget_2028_29']:.2f}")

    tab1, tab2, tab3 = st.tabs(["State Vaccine Forecast", "Cold Chain Needs", "Supply Chain Recommendations"])

    with tab1:
        st.subheader("Top 15 States — Predicted Vaccine Demand 2028-29")
        top_vacc = vaccine_df_pred.sort_values("predicted_vaccine_doses_2028_29", ascending=False).head(15)
        fig = px.bar(
            top_vacc,
            x="state",
            y=["total_vaccination_doses", "predicted_vaccine_doses_2028_29"],
            barmode="group",
            title="Current vs Predicted Vaccine Doses",
            labels={"value": "Doses", "variable": "Period"},
            color_discrete_map={
                "total_vaccination_doses": "steelblue",
                "predicted_vaccine_doses_2028_29": "orangered"
            }
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Cold Chain Infrastructure Required by State")
        fig2 = px.bar(
            vaccine_df_pred.sort_values("cold_chain_units_needed", ascending=False).head(20),
            x="state",
            y="cold_chain_units_needed",
            title="Cold Chain Units Needed per State",
            color="cold_chain_units_needed",
            color_continuous_scale="Reds"
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader(f"Supply Chain Recommendation — {selected_state}")
        if not rec_row.empty and "supply_chain_rec" in rec_row.columns:
            st.info(rec_row.iloc[0]["supply_chain_rec"])
        st.subheader("Vaccine Budget Allocation — All States")
        st.dataframe(
            vaccine_df_pred.sort_values("vaccine_budget_2028_29", ascending=False)[
                ["state", "predicted_vaccine_doses_2028_29", "cold_chain_units_needed", "vaccine_budget_2028_29"]
            ],
            use_container_width=True
        )

# ----------------------------
# Page 5: 2028-29 Predictions
# ----------------------------
elif page == "2028-29 Predictions":
    st.title("📊 2028-29 Resource Predictions")

    selected_state = st.sidebar.selectbox(
        "Select a State", sorted(resource_df_pred["state"].unique())
    )
    sr = resource_df_pred[resource_df_pred["state"] == selected_state].iloc[0]
    rec_row = state_df[state_df["state"] == selected_state]

    st.markdown(f"## {selected_state} — 2028-29 Forecast")

    c1, c2, c3 = st.columns(3)
    c1.metric("Beds Needed 2028-29", f"{sr['beds_needed_2028_29']:,}")
    c2.metric("Bed Gap to Bridge", f"{sr['bed_gap_2028_29']:,}")
    c3.metric("New Hospitals Needed", f"{sr['new_hospitals_needed_2028_29']}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Doctors Needed 2028-29", f"{sr['doctors_needed_2028_29']:,}")
    c5.metric("Doctor Gap", f"{sr['doctor_gap_2028_29']:,}")
    c6.metric("Infra Budget (₹Cr)", f"{sr['hospital_infra_budget_2028_29']:.2f}")

    tab1, tab2, tab3 = st.tabs(["Hospital Needs", "Doctor Needs", "State Comparison"])

    with tab1:
        st.subheader("New Hospitals Needed — Top 20 States")
        fig = px.bar(
            resource_df_pred.sort_values("new_hospitals_needed_2028_29", ascending=False).head(20),
            x="state",
            y="new_hospitals_needed_2028_29",
            title="Hospitals to be Built by 2028-29",
            color="new_hospitals_needed_2028_29",
            color_continuous_scale="Blues"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        if not rec_row.empty and "hospital_rec" in rec_row.columns:
            st.info(f"**{selected_state} Recommendation:** {rec_row.iloc[0]['hospital_rec']}")

    with tab2:
        st.subheader("Doctor Gap — Top 20 States")
        fig2 = px.bar(
            resource_df_pred.sort_values("doctor_gap_2028_29", ascending=False).head(20),
            x="state",
            y="doctor_gap_2028_29",
            title="Doctor Shortage to be Filled by 2028-29",
            color="doctor_gap_2028_29",
            color_continuous_scale="Oranges"
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

        if not rec_row.empty and "doctor_rec" in rec_row.columns:
            st.info(f"**{selected_state} Recommendation:** {rec_row.iloc[0]['doctor_rec']}")

    with tab3:
        st.subheader("Full State Resource Comparison Table — 2028-29")
        st.dataframe(
            resource_df_pred[[
                "state", "beds_needed_2028_29", "bed_gap_2028_29",
                "new_hospitals_needed_2028_29", "doctors_needed_2028_29",
                "doctor_gap_2028_29", "hospital_infra_budget_2028_29"
            ]].sort_values("hospital_infra_budget_2028_29", ascending=False),
            use_container_width=True
        )

# ----------------------------
# Page 6: Budget Recommendations
# ----------------------------
elif page == "Budget Recommendations":
    st.title("💰 Budget Recommendations 2028-29")

    selected_state = st.sidebar.selectbox(
        "Select a State", sorted(budget_breakdown_df["state"].unique())
    )
    sb = budget_breakdown_df[budget_breakdown_df["state"] == selected_state].iloc[0]
    rec_row = state_df[state_df["state"] == selected_state]

    st.markdown(f"## {selected_state} — Budget Breakdown")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Budget 2028-29 (₹Cr)", f"{sb['total_recommended_budget_2028_29']:.2f}")
    c2.metric("Vaccine Budget (₹Cr)", f"{sb['vaccine_budget_2028_29']:.2f}")
    c3.metric("Infrastructure Budget (₹Cr)", f"{sb['hospital_infra_budget_2028_29']:.2f}")
    c4.metric("Priority Tier", str(sb['budget_priority_tier']))

    tab1, tab2, tab3 = st.tabs(["Budget Pie Chart", "State Rankings", "Recommendation"])

    with tab1:
        pie_data = {
            "Category": ["Operations", "Vaccine", "Infrastructure"],
            "Budget": [
                float(sb.get("predicted_budget_2028_29", 0)),
                float(sb.get("vaccine_budget_2028_29", 0)),
                float(sb.get("hospital_infra_budget_2028_29", 0))
            ]
        }
        pie_df = pd.DataFrame(pie_data)
        fig = px.pie(pie_df, names="Category", values="Budget",
                     title=f"Budget Split — {selected_state}")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("All States — Total Budget Ranking 2028-29")
        fig2 = px.bar(
            budget_breakdown_df.sort_values("total_recommended_budget_2028_29", ascending=False),
            x="state",
            y="total_recommended_budget_2028_29",
            color="budget_priority_tier",
            title="State-wise Budget Requirements 2028-29",
            labels={"total_recommended_budget_2028_29": "Budget (₹ Crores)"}
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        if not rec_row.empty and "budget_rec" in rec_row.columns:
            st.subheader("Budget Recommendation")
            st.success(rec_row.iloc[0]["budget_rec"])
        st.subheader("State Budget Table")
        st.dataframe(
            budget_breakdown_df[[
                "state", "total_recommended_budget_2028_29", "vaccine_budget_2028_29",
                "hospital_infra_budget_2028_29", "budget_priority_tier"
            ]].sort_values("total_recommended_budget_2028_29", ascending=False),
            use_container_width=True
        )

# ----------------------------
# Page 7: Downloads
# ----------------------------
elif page == "Downloads":
    st.title("Download Processed Outputs")

    st.markdown("### State-level files")
    with open("data/processed/state_master.csv", "rb") as f:
        st.download_button("Download state_master.csv", f, file_name="state_master.csv")

    with open("data/processed/state_features.csv", "rb") as f:
        st.download_button("Download state_features.csv", f, file_name="state_features.csv")

    with open("data/processed/state_budget_predictions.csv", "rb") as f:
        st.download_button("Download state_budget_predictions.csv", f, file_name="state_budget_predictions.csv")

    with open("data/processed/state_recommendations.csv", "rb") as f:
        st.download_button("Download state_recommendations.csv", f, file_name="state_recommendations.csv")

    with open("data/processed/genai_state_reports.csv", "rb") as f:
        st.download_button("Download genai_state_reports.csv", f, file_name="genai_state_reports.csv")

    st.markdown("### Hospital-level files")
    with open("data/processed/hospital_master.csv", "rb") as f:
        st.download_button("Download hospital_master.csv", f, file_name="hospital_master.csv")

    st.markdown("### 2028-29 Prediction Files")
    with open("data/processed/vaccine_predictions_2028_29.csv", "rb") as f:
        st.download_button("Download vaccine_predictions_2028_29.csv", f, file_name="vaccine_predictions_2028_29.csv")

    with open("data/processed/resource_predictions_2028_29.csv", "rb") as f:
        st.download_button("Download resource_predictions_2028_29.csv", f, file_name="resource_predictions_2028_29.csv")

    with open("data/processed/budget_breakdown_2028_29.csv", "rb") as f:
        st.download_button("Download budget_breakdown_2028_29.csv", f, file_name="budget_breakdown_2028_29.csv")