import streamlit as st
import pandas as pd

st.set_page_config(page_title="Healthcare AI Planning System", layout="wide")

@st.cache_data
def load_data():
    state_features = pd.read_csv("data/processed/state_features.csv")
    budget_predictions = pd.read_csv("data/processed/state_budget_predictions.csv")
    recommendations = pd.read_csv("data/processed/state_recommendations.csv")
    genai_reports = pd.read_csv("data/processed/genai_state_reports.csv")

    df = state_features.merge(budget_predictions, on="state", how="left")
    df = df.merge(recommendations, on="state", how="left")
    df = df.merge(genai_reports, on="state", how="left")
    return df

df = load_data()

st.title("Healthcare AI Planning System")
st.markdown("### State-wise Resource Analysis, Budget Prediction, and AI Recommendations")

st.sidebar.header("Dashboard Filters")
selected_state = st.sidebar.selectbox("Select a State", sorted(df["state"].unique()))
state_data = df[df["state"] == selected_state].iloc[0]

# Top summary cards
st.markdown("## National Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("States Covered", f"{df['state'].nunique()}")
c2.metric("Avg Readiness Score", f"{df['healthcare_readiness_score'].mean():.2f}")
c3.metric("Avg Budget Need Score", f"{df['budget_need_score'].mean():.2f}")
c4.metric("Avg Predicted Budget", f"{df['predicted_budget_2028_29'].mean():.2f}")

st.divider()

# Selected state overview
st.markdown(f"## State Overview: {selected_state}")
col1, col2, col3 = st.columns(3)
col1.metric("Healthcare Readiness Score", f"{state_data['healthcare_readiness_score']:.2f}")
col2.metric("Infrastructure Score", f"{state_data['infrastructure_score']:.2f}")
col3.metric("Budget Need Score", f"{state_data['budget_need_score']:.2f}")

col4, col5, col6 = st.columns(3)
col4.metric("Predicted Budget 2028-29", f"{state_data['predicted_budget_2028_29']:.2f}")
col5.metric("Total Doctors", f"{state_data['total_doctors']:.0f}")
col6.metric("Total Beds", f"{state_data['total_beds_all_levels']:.0f}")

st.divider()

# Charts
tab1, tab2, tab3, tab4 = st.tabs([
    "Readiness Analysis",
    "Budget Analysis",
    "Recommendations",
    "GenAI Report"
])

with tab1:
    st.subheader("Top 10 States by Healthcare Readiness")
    top_readiness = df.sort_values("healthcare_readiness_score", ascending=False).head(10)
    st.bar_chart(top_readiness.set_index("state")["healthcare_readiness_score"])

    st.subheader("Top 10 States by Infrastructure Score")
    top_infra = df.sort_values("infrastructure_score", ascending=False).head(10)
    st.bar_chart(top_infra.set_index("state")["infrastructure_score"])

    st.subheader("Readiness Table")
    st.dataframe(
        top_readiness[["state", "healthcare_readiness_score", "infrastructure_score"]],
        width="stretch"
    )

with tab2:
    st.subheader("Top 10 States by Budget Need")
    top_budget_need = df.sort_values("budget_need_score", ascending=False).head(10)
    st.bar_chart(top_budget_need.set_index("state")["budget_need_score"])

    st.subheader("Predicted Budget 2028-29 by Top Need States")
    st.dataframe(
        top_budget_need[["state", "budget_need_score", "predicted_budget_2028_29"]],
        width="stretch"
    )

with tab3:
    st.subheader(f"AI Recommendation for {selected_state}")
    st.info(state_data["recommendation"])

    st.subheader("Top Budget Need States")
    st.dataframe(
        df.sort_values("budget_need_score", ascending=False)[
            ["state", "budget_need_score", "recommendation"]
        ].head(10),
        width="stretch"
    )

with tab4:
    st.subheader(f"Generative AI Report for {selected_state}")
    st.text(state_data["genai_report"])