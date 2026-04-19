"""
chatbot.py — Google Gemini AI Chatbot Integration
==================================================
This module handles all Gemini API interactions for the Healthcare AI Planning System.
It is completely self-contained and does NOT modify any existing module.

Integration points:
  - Called from app.py only when the user selects "AI Assistant" from the sidebar.
  - Receives lightweight summaries of existing dataframes (state_df, hospital_df, etc.)
    as context strings — never the full dataset — to keep prompts short and fast.

Requirements:
  - pip install google-genai python-dotenv
  - Add GEMINI_API_KEY to the .env file in the project root.

SDK: google-genai (replaces the deprecated google-generativeai package)
"""

import os
import time
import textwrap
from google import genai                        # new SDK: google-genai
from google.genai import types                  # for GenerateContentConfig
from dotenv import load_dotenv                  # python-dotenv: loads .env into os.environ

# Real-time global health data (Disease.sh API) — added for live context enrichment
from realtime import fetch_live_health_data, format_live_data

# --- Load .env — override=True ensures .env always wins over shell env vars ---
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "gemini-2.5-flash"   # Only model with free-tier quota in this project (5 RPM / 20 RPD)

# System-level prompt — enforces structured, analytical, decision-support responses.
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Healthcare AI Planning Analyst for India, embedded in a
    data-driven decision-support system used by policymakers and health officials.

    Your role:
    - Analyze healthcare data spanning state readiness, hospital capacity,
      vaccine demand, doctor shortages, and 2028-29 budget projections.
    - Provide sharp, evidence-based insights — not generic summaries.
    - Every response MUST follow this exact structure:

      **1. Key Insight**
      (1-2 sentences: the single most important finding from the data)

      **2. Analysis**
      (3-5 bullet points: specific data-backed observations with numbers)

      **3. Recommendation**
      (2-3 actionable policy steps, prioritised by urgency)

    Rules:
    - Always cite specific states, hospitals, or metrics from the context data.
    - Never produce vague paragraphs — use bullet points and numbers.
    - If the data does not contain enough information, state what is missing
      and what additional data would be needed.
    - Use Indian healthcare policy terminology where appropriate.
""").strip()

# ---------------------------------------------------------------------------
# Gemini client — created once per process
# ---------------------------------------------------------------------------

def _get_client():
    """
    Returns a configured google.genai.Client using GEMINI_API_KEY from .env.
    Returns None if the key is missing or still a placeholder.
    """
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key or api_key == "your_api_key_here":
        return None
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Context builder — converts DataFrames to compact text summaries
# ---------------------------------------------------------------------------

def build_context(
    state_df=None,
    hospital_df=None,
    vaccine_df=None,
    resource_df=None,
    budget_df=None,
) -> str:
    """
    Builds an INSIGHT-BASED context string from available dataframes.

    Instead of raw row dumps, each section surfaces the most analytically
    useful slices: top/bottom performers, critical gaps, high-risk states.
    This gives the AI model meaningful patterns to reason over.

    Parameters
    ----------
    state_df, hospital_df, vaccine_df, resource_df, budget_df : pd.DataFrame | None
        Any of the project's main dataframes.

    Returns
    -------
    str
        A structured insight summary ready for the Gemini prompt.
    """
    sections = []
    N = 5  # rows per insight slice — keep prompt compact

    # -----------------------------------------------------------------------
    # 1. STATE INSIGHTS
    # -----------------------------------------------------------------------
    if state_df is not None and not state_df.empty:
        base_cols = [c for c in [
            "state", "healthcare_readiness_score", "infrastructure_score",
            "budget_need_score", "predicted_budget_2028_29",
            "total_doctors", "total_beds_all_levels"
        ] if c in state_df.columns]

        # Top 5 most ready states
        top5_ready = (state_df
            .sort_values("healthcare_readiness_score", ascending=False)
            .head(N)[base_cols])

        # Bottom 5 least ready (highest need)
        bot5_ready = (state_df
            .sort_values("healthcare_readiness_score", ascending=True)
            .head(N)[base_cols])

        # Top 5 highest predicted budget need
        if "predicted_budget_2028_29" in state_df.columns:
            top5_budget = (state_df
                .sort_values("predicted_budget_2028_29", ascending=False)
                .head(N)[[c for c in ["state", "predicted_budget_2028_29", "budget_need_score"] if c in state_df.columns]])
            sections.append(
                "=== STATE INSIGHTS ==="
                "\n[Top 5 states — highest predicted budget need 2028-29]\n"
                + top5_budget.to_string(index=False)
            )

        sections.append(
            "\n[Top 5 states — best healthcare readiness]\n"
            + top5_ready.to_string(index=False)
            + "\n\n[Bottom 5 states — lowest healthcare readiness (highest risk)]\n"
            + bot5_ready.to_string(index=False)
        )

    # -----------------------------------------------------------------------
    # 2. HOSPITAL INSIGHTS
    # -----------------------------------------------------------------------
    if hospital_df is not None and not hospital_df.empty:
        hosp_cols = [c for c in [
            "hospital_name", "city", "bed_capacity",
            "budget_allocated", "budget_spent", "utilization_ratio"
        ] if c in hospital_df.columns]

        # Hospitals with highest budget utilization (over-stretched)
        high_util = (hospital_df
            .sort_values("utilization_ratio", ascending=False)
            .head(N)[hosp_cols])

        # Hospitals with lowest bed capacity (infrastructure gap)
        low_beds = (hospital_df
            .sort_values("bed_capacity", ascending=True)
            .head(N)[hosp_cols])

        sections.append(
            "=== HOSPITAL INSIGHTS ==="
            "\n[Top 5 hospitals — highest budget utilization ratio (over-stretched)]\n"
            + high_util.to_string(index=False)
            + "\n\n[Top 5 hospitals — lowest bed capacity (infrastructure gap)]\n"
            + low_beds.to_string(index=False)
        )

    # -----------------------------------------------------------------------
    # 3. VACCINE & SUPPLY CHAIN INSIGHTS
    # -----------------------------------------------------------------------
    if vaccine_df is not None and not vaccine_df.empty:
        vacc_cols = [c for c in [
            "state", "total_vaccination_doses",
            "predicted_vaccine_doses_2028_29",
            "cold_chain_units_needed", "vaccine_budget_2028_29"
        ] if c in vaccine_df.columns]

        # Highest predicted vaccine demand
        top5_vacc = (vaccine_df
            .sort_values("predicted_vaccine_doses_2028_29", ascending=False)
            .head(N)[vacc_cols])

        # States needing most cold-chain units (supply chain pressure)
        top5_cold = (vaccine_df
            .sort_values("cold_chain_units_needed", ascending=False)
            .head(N)[[c for c in ["state", "cold_chain_units_needed", "vaccine_budget_2028_29"] if c in vaccine_df.columns]])

        sections.append(
            "=== VACCINE & SUPPLY CHAIN INSIGHTS ==="
            "\n[Top 5 states — highest predicted vaccine demand 2028-29]\n"
            + top5_vacc.to_string(index=False)
            + "\n\n[Top 5 states — highest cold-chain infrastructure need]\n"
            + top5_cold.to_string(index=False)
        )

    # -----------------------------------------------------------------------
    # 4. RESOURCE GAP INSIGHTS (2028-29)
    # -----------------------------------------------------------------------
    if resource_df is not None and not resource_df.empty:
        res_cols = [c for c in [
            "state", "beds_needed_2028_29", "bed_gap_2028_29",
            "new_hospitals_needed_2028_29", "doctors_needed_2028_29",
            "doctor_gap_2028_29", "hospital_infra_budget_2028_29"
        ] if c in resource_df.columns]

        # States with largest doctor shortage
        top5_doc_gap = (resource_df
            .sort_values("doctor_gap_2028_29", ascending=False)
            .head(N)[res_cols])

        # States needing most new hospitals
        top5_hosp_need = (resource_df
            .sort_values("new_hospitals_needed_2028_29", ascending=False)
            .head(N)[[c for c in ["state", "new_hospitals_needed_2028_29", "bed_gap_2028_29", "hospital_infra_budget_2028_29"] if c in resource_df.columns]])

        sections.append(
            "=== RESOURCE GAP INSIGHTS 2028-29 ==="
            "\n[Top 5 states — largest doctor shortage]\n"
            + top5_doc_gap.to_string(index=False)
            + "\n\n[Top 5 states — most new hospitals needed]\n"
            + top5_hosp_need.to_string(index=False)
        )

    # -----------------------------------------------------------------------
    # 5. BUDGET PRIORITY INSIGHTS (2028-29)
    # -----------------------------------------------------------------------
    if budget_df is not None and not budget_df.empty:
        bud_cols = [c for c in [
            "state", "total_recommended_budget_2028_29",
            "vaccine_budget_2028_29", "hospital_infra_budget_2028_29",
            "budget_priority_tier"
        ] if c in budget_df.columns]

        # Highest total budget requirement
        top5_bud = (budget_df
            .sort_values("total_recommended_budget_2028_29", ascending=False)
            .head(N)[bud_cols])

        # Lowest budget — potentially under-resourced
        bot5_bud = (budget_df
            .sort_values("total_recommended_budget_2028_29", ascending=True)
            .head(N)[bud_cols])

        # Tier 1 priority states (highest urgency)
        if "budget_priority_tier" in budget_df.columns:
            tier1 = budget_df[budget_df["budget_priority_tier"] == budget_df["budget_priority_tier"].min()][bud_cols]
            sections.append(
                "=== BUDGET PRIORITY INSIGHTS 2028-29 ==="
                "\n[Top 5 states — highest total budget requirement]\n"
                + top5_bud.to_string(index=False)
                + "\n\n[Bottom 5 states — lowest budget (risk of under-resourcing)]\n"
                + bot5_bud.to_string(index=False)
                + "\n\n[Tier-1 Priority States (most urgent intervention needed)]\n"
                + tier1.to_string(index=False)
            )
        else:
            sections.append(
                "=== BUDGET PRIORITY INSIGHTS 2028-29 ==="
                "\n[Top 5 states — highest total budget requirement]\n"
                + top5_bud.to_string(index=False)
            )

    # -------------------------------------------------------------------
    # 6. LIVE GLOBAL HEALTH DATA (Disease.sh API — top-5 COVID countries)
    # -------------------------------------------------------------------
    live_raw  = fetch_live_health_data()
    live_text = format_live_data(live_raw) if live_raw else "Live data currently unavailable."
    sections.append(
        "=== LIVE GLOBAL HEALTH DATA ==="
        "\n[Real-time COVID-19 snapshot — top 5 countries by total cases]\n"
        + live_text
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Core API call
# ---------------------------------------------------------------------------

def get_ai_response(user_input: str, context: str) -> str:
    """
    Sends user_input (plus project context) to Gemini and returns the response text.

    Parameters
    ----------
    user_input : str
        The message typed by the user.
    context : str
        A compact summary of project data (built by build_context()).

    Returns
    -------
    str
        The AI-generated response, or an error message if the call fails.
    """
    client = _get_client()
    if client is None:
        return (
            "⚠️ **Gemini API key not configured.**\n\n"
            "Please add your key to the `.env` file in the project root:\n\n"
            "```\n"
            "GEMINI_API_KEY=your-actual-api-key-here\n"
            "```\n\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    # --- Prompt is built once; shared across retry attempts (unchanged logic) ---
    full_prompt = (
        "[HEALTHCARE INSIGHT DATA — use as the factual basis for your analysis]\n"
        "This system combines two data sources:\n"
        "  1. Static healthcare dataset: state readiness, hospital capacity, vaccine demand,"
        " doctor shortages, and 2028-29 budget projections for India.\n"
        "  2. Live global health data: real-time COVID-19 statistics from the Disease.sh API"
        " (top 5 countries by total cases).\n\n"
        f"{context}\n\n"
        "[ANALYTICAL INSTRUCTIONS]\n"
        "- Base your response on the data provided above (both static and live sources).\n"
        "- Where relevant, correlate global COVID-19 trends with local Indian healthcare insights"
        " (e.g., vaccine demand pressure, resource strain, budget implications).\n"
        "- Identify patterns, outliers, and cross-dataset correlations.\n"
        "- Cite specific state names, hospital names, and numeric values.\n"
        "- Structure your response EXACTLY as:\n"
        "    1. Key Insight\n"
        "    2. Analysis\n"
        "    3. Recommendation\n\n"
        f"[USER QUESTION]\n{user_input}"
    )

    # ---------------------------------------------------------------------------
    # RETRY LOGIC — up to 2 attempts with a 2-second delay between them.
    # Handles transient 503 (model overloaded) and rate-limit (429) errors.
    # ---------------------------------------------------------------------------
    _MAX_RETRIES  = 2
    _RETRY_DELAY  = 2   # seconds
    last_exc      = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,        # Lower = more factual, less hallucination
                    max_output_tokens=1024, # Keep responses concise
                ),
            )
            return response.text   # success — return immediately

        except Exception as exc:
            last_exc = exc
            err = str(exc)

            # --- Specific, user-friendly messages for known transient errors ---
            if "503" in err or "unavailable" in err.lower() or "overloaded" in err.lower():
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                    continue   # retry
                # Final attempt also failed with 503
                return (
                    "🔄 **Gemini AI is temporarily overloaded (503).**\n\n"
                    "The model is under heavy load. Please try again in a moment.\n\n"
                    "**Based on the loaded healthcare data, here are general insights:**\n"
                    "- Several states show elevated healthcare demand relative to available infrastructure.\n"
                    "- Critical infrastructure gaps persist in low-resource regions, "
                    "requiring prioritised capital allocation.\n"
                    "- Vaccine supply-chain pressure is highest in high-population states; "
                    "cold-chain expansion should be fast-tracked.\n"
                    "- Strategic, tier-based budget allocation can optimise outcomes "
                    "across both Tier-1 and Tier-2 priority states."
                )

            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                # Rate-limit — no point retrying immediately
                return (
                    "⏳ **Gemini API quota exceeded.**\n\n"
                    "You have hit the free-tier rate limit for your API key. "
                    "Please wait a minute and try again, or check your quota at:\n\n"
                    "👉 https://ai.dev/rate-limit\n\n"
                    "**Tips to avoid this:**\n"
                    "- Wait ~1 minute between queries\n"
                    "- Enable billing on your Google AI Studio project for higher limits\n"
                    "- Use shorter / fewer queries per session"
                )

            # Unknown error — retry if attempts remain, else fall through
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)

    # ---------------------------------------------------------------------------
    # FALLBACK — all retries exhausted; return a meaningful healthcare response
    # ---------------------------------------------------------------------------
    return (
        "⚠️ **AI service is currently busy. Please try again shortly.**\n\n"
        "Based on the available healthcare planning data, here are key observations:\n"
        "- Some states show significantly higher healthcare demand than current "
        "infrastructure can support.\n"
        "- Infrastructure gaps are most acute in low-resource and rural regions, "
        "warranting urgent investment.\n"
        "- Vaccine demand forecasts for 2028-29 indicate cold-chain expansion "
        "is a critical supply-chain priority.\n"
        "- Strategic, tiered budget allocation — prioritising Tier-1 states — "
        "is recommended to maximise health outcomes per rupee spent.\n\n"
        f"_(Underlying error: {last_exc})_"
    )


# ---------------------------------------------------------------------------
# Streamlit chatbot UI  (called from app.py)
# ---------------------------------------------------------------------------

def render_chatbot_page(state_df=None, hospital_df=None,
                        vaccine_df=None, resource_df=None, budget_df=None):
    """
    Renders the full AI Assistant page inside the Streamlit app.
    Call this function from app.py when the user selects "AI Assistant".

    All dataframe parameters are optional; pass whichever are available.
    """
    import streamlit as st

    # Page header
    st.title("🤖 AI Assistant — Healthcare Planning Advisor")
    st.markdown(
        "Ask any question about state readiness, hospital budgets, vaccine forecasts, "
        "resource gaps, or 2028-29 budget recommendations. "
        "The assistant has access to a summary of all project data."
    )

    # API key status indicator (validates key is set and not a placeholder)
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    api_key_valid = bool(api_key) and api_key != "your_api_key_here"
    if api_key_valid:
        st.success("✅ Gemini API key detected — Assistant is ready.")
    else:
        st.warning(
            "⚠️ `GEMINI_API_KEY` not set or still a placeholder in `.env`. "
            "Add your real key and restart the app."
        )

    st.markdown("---")

    # ---- Session state — maintain chat history across reruns ----
    # Key: "chat_history" — list of {"role": "user"|"assistant", "content": str}
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # ---- Build context once per session (cached in session state) ----
    if "chatbot_context" not in st.session_state:
        with st.spinner("📊 Loading project data context…"):
            st.session_state["chatbot_context"] = build_context(
                state_df=state_df,
                hospital_df=hospital_df,
                vaccine_df=vaccine_df,
                resource_df=resource_df,
                budget_df=budget_df,
            )

    context = st.session_state["chatbot_context"]

    # ---- Chat display area ----
    chat_container = st.container()
    with chat_container:
        if not st.session_state["chat_history"]:
            st.info(
                "👋 Hello! I'm your Healthcare Planning AI Assistant powered by Google Gemini.\n\n"
                "Try asking:\n"
                "- *\"Which states have the highest budget need in 2028-29?\"*\n"
                "- *\"How many new hospitals are needed in Maharashtra?\"*\n"
                "- *\"What is the doctor gap in high-priority states?\"*\n"
                "- *\"Summarise vaccine demand forecasts for the top 5 states.\"*"
            )
        else:
            for message in st.session_state["chat_history"]:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(message["content"])

    # ---- Input area ----
    user_input = st.chat_input("Ask a question about healthcare data…")

    if user_input:
        # Append user message to history
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_input}
        )

        # Show a spinner while waiting for Gemini
        with st.spinner("🔄 Consulting Gemini AI…"):
            ai_reply = get_ai_response(user_input, context)

        # Append AI reply to history
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": ai_reply}
        )

        # Rerun to refresh the chat display
        st.rerun()

    # ---- Utility buttons ----
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
            st.session_state["chat_history"] = []
            # Also drop cached context so it refreshes on re-enter
            if "chatbot_context" in st.session_state:
                del st.session_state["chatbot_context"]
            st.rerun()

    st.markdown("---")
    st.caption(
        "Powered by **Google Gemini 2.5 Flash** · "
        "Context: top-8 rows per dataset · "
        "Model: `gemini-2.5-flash`"
    )
