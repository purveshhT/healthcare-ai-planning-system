import streamlit as st
import pandas as pd

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

state_df = load_state_data()
hospital_df = load_hospital_data()

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.title("Healthcare AI Planning System")
page = st.sidebar.radio(
    "Navigate",
    ["Project Overview", "State Dashboard", "Hospital Dashboard", "Downloads"]
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

    st.markdown("## About the Dashboard")
    st.write(
        "Use the sidebar to navigate between the state-level dashboard, hospital-level dashboard, "
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
# Page 4: Downloads
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