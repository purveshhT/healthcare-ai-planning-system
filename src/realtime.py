"""
realtime.py — Real-Time Global Health Data Integration
=======================================================
Fetches live COVID-19 statistics from the Disease.sh public API and
formats them as a compact, human-readable summary for injection into
the Gemini chatbot context.

Design principles:
  - Fully self-contained: no imports from other project modules.
  - Graceful degradation: any network/parsing failure returns safe fallbacks.
  - Minimal footprint: only top-5 countries fetched and surfaced.

API used: https://disease.sh/v3/covid-19/countries
  - Free, no auth required, returns per-country COVID-19 aggregates.
"""

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_API_URL     = "https://disease.sh/v3/covid-19/countries"
_TIMEOUT_SEC = 8    # seconds before giving up on the network call
_TOP_N       = 5    # only surface the top-N countries to keep prompts compact


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_live_health_data() -> list:
    """
    Fetches the top-5 countries by total COVID-19 cases from Disease.sh.

    Returns
    -------
    list of dict
        Each dict contains at minimum: 'country', 'cases', 'deaths', 'recovered'.
        Returns an empty list if the request fails for any reason.
    """
    try:
        response = requests.get(_API_URL, timeout=_TIMEOUT_SEC)
        response.raise_for_status()          # raises HTTPError for 4xx/5xx

        data = response.json()               # list of country objects

        if not isinstance(data, list) or len(data) == 0:
            return []

        # Sort descending by total cases and keep top-N
        sorted_data = sorted(data, key=lambda x: x.get("cases", 0), reverse=True)
        return sorted_data[:_TOP_N]

    except Exception:
        # Network timeouts, DNS failures, bad JSON, HTTP errors — all silenced
        return []


def format_live_data(data: list) -> str:
    """
    Converts the raw Disease.sh JSON list into a concise, readable text block.

    Parameters
    ----------
    data : list
        Output of fetch_live_health_data().

    Returns
    -------
    str
        Human-readable summary, or a fallback message if data is empty.

    Example output
    --------------
    Top Countries by COVID Cases:
    * USA: Cases=100,000,000, Deaths=1,200,000, Recovered=95,000,000
    * India: Cases=44,000,000, Deaths=530,000, Recovered=43,000,000
    ...
    """
    if not data:
        return "Live data currently unavailable."

    lines = ["Top Countries by COVID Cases:"]
    for entry in data:
        country   = entry.get("country", "Unknown")
        cases     = entry.get("cases",     0)
        deaths    = entry.get("deaths",    0)
        recovered = entry.get("recovered", 0)

        lines.append(
            f"* {country}: "
            f"Cases={cases:,}, "
            f"Deaths={deaths:,}, "
            f"Recovered={recovered:,}"
        )

    return "\n".join(lines)
