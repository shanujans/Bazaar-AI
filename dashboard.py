"""
Bazaar-AI — Phase 2: Compute-Saver Live Dashboard
==================================================
Streamlit app that polls the FastAPI backend (`main.py`) and renders a
judge-facing "AMD ROCm / Gemma-4" themed live view: headline cost/VRAM
metrics up top, a live routing log underneath showing every inbound
message (WhatsApp + direct API) as it gets classified and routed between
the edge (gemma-4-26b-a4b-it) and heavy (gemma-4-31b-it, AMD MI300X) tiers.

Run alongside the backend:
    uvicorn main:app --port 8080
    streamlit run dashboard.py

The backend's default port is 8080 (see README_PHASE_1.md / main.py's
`__main__` block), so that's what this dashboard points at by default.
Change it in the sidebar (or via the API_BASE_URL env var) if you're
running the backend somewhere else.
"""

import os
import html
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
POLL_SECONDS_DEFAULT = 2
REQUEST_TIMEOUT = 2.5

st.set_page_config(
    page_title="Bazaar-AI | Compute-Saver Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme — dark, high-tech, AMD ROCm red / Gemma teal accents
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Inter:wght@400;600;800&display=swap');

    :root {
        --bg: #05070a;
        --panel: #0d1117;
        --panel-border: #1c2530;
        --amd-red: #ED1C24;
        --amd-red-glow: rgba(237, 28, 36, 0.35);
        --gemma-teal: #00E5A0;
        --gemma-teal-glow: rgba(0, 229, 160, 0.30);
        --text-dim: #7b8794;
        --text-bright: #F2F5F7;
    }

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(237,28,36,0.08), transparent 40%),
            radial-gradient(circle at 85% 10%, rgba(0,229,160,0.07), transparent 40%),
            var(--bg);
    }
    #MainMenu, footer, header {visibility: hidden;}

    .bazaar-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--panel-border);
        margin-bottom: 22px;
    }
    .bazaar-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 30px;
        font-weight: 800;
        color: var(--text-bright);
        letter-spacing: 0.5px;
        margin: 0;
    }
    .bazaar-title span { color: var(--amd-red); }
    .bazaar-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        color: var(--text-dim);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .status-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 20px;
        letter-spacing: 1px;
    }
    .status-live {
        background: rgba(0,229,160,0.12);
        color: var(--gemma-teal);
        border: 1px solid rgba(0,229,160,0.4);
        box-shadow: 0 0 14px var(--gemma-teal-glow);
    }
    .status-down {
        background: rgba(237,28,36,0.12);
        color: var(--amd-red);
        border: 1px solid rgba(237,28,36,0.5);
        box-shadow: 0 0 14px var(--amd-red-glow);
    }

    .metric-card {
        background: linear-gradient(180deg, var(--panel) 0%, #090c10 100%);
        border: 1px solid var(--panel-border);
        border-radius: 14px;
        padding: 22px 24px;
        height: 100%;
    }
    .metric-card.accent-teal { border-top: 3px solid var(--gemma-teal); }
    .metric-card.accent-red { border-top: 3px solid var(--amd-red); }
    .metric-card.accent-neutral { border-top: 3px solid #3a4552; }
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        color: var(--text-dim);
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 42px;
        font-weight: 800;
        color: var(--text-bright);
        line-height: 1.1;
    }
    .metric-value.teal { color: var(--gemma-teal); text-shadow: 0 0 18px var(--gemma-teal-glow); }
    .metric-value.red { color: var(--amd-red); text-shadow: 0 0 18px var(--amd-red-glow); }
    .metric-sub {
        font-size: 12.5px;
        color: var(--text-dim);
        margin-top: 8px;
    }

    .ratio-track {
        margin-top: 14px;
        height: 8px;
        border-radius: 6px;
        background: rgba(237,28,36,0.18);
        overflow: hidden;
    }
    .ratio-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--gemma-teal), #00b881);
        box-shadow: 0 0 10px var(--gemma-teal-glow);
    }

    .chip-row { display: flex; gap: 12px; margin-top: 18px; }
    .chip {
        flex: 1;
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .chip-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 20px;
        font-weight: 700;
        color: var(--text-bright);
    }
    .chip-label {
        font-size: 10.5px;
        color: var(--text-dim);
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    .section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 15px;
        font-weight: 700;
        color: var(--text-bright);
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 30px 0 12px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .pulse-dot {
        width: 9px; height: 9px; border-radius: 50%;
        background: var(--gemma-teal);
        box-shadow: 0 0 10px var(--gemma-teal-glow);
        animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; } 50% { opacity: 0.35; } 100% { opacity: 1; }
    }

    .log-row {
        display: grid;
        grid-template-columns: 92px 92px 150px 1fr 190px 1.4fr;
        gap: 14px;
        align-items: center;
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-left: 3px solid #3a4552;
        border-radius: 8px;
        padding: 11px 16px;
        margin-bottom: 7px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
    }
    .log-row.route-edge { border-left-color: var(--gemma-teal); }
    .log-row.route-heavy { border-left-color: var(--amd-red); }
    .log-time { color: var(--text-dim); }
    .log-user { color: #9fb0c0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .log-msg { color: var(--text-bright); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .log-reason { color: var(--text-dim); font-size: 11.5px; }

    .badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 5px;
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        text-align: center;
    }
    .badge-whatsapp { background: rgba(0,229,160,0.14); color: var(--gemma-teal); border: 1px solid rgba(0,229,160,0.35); }
    .badge-api { background: rgba(120,140,160,0.14); color: #9fb0c0; border: 1px solid rgba(120,140,160,0.35); }
    .badge-edge { background: rgba(0,229,160,0.14); color: var(--gemma-teal); border: 1px solid rgba(0,229,160,0.35); }
    .badge-heavy { background: rgba(237,28,36,0.14); color: var(--amd-red); border: 1px solid rgba(237,28,36,0.4); }
    .badge-fallback { background: rgba(255,180,0,0.14); color: #ffb400; border: 1px solid rgba(255,180,0,0.4); margin-left: 6px; }

    .empty-state {
        text-align: center;
        padding: 50px 20px;
        color: var(--text-dim);
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        border: 1px dashed var(--panel-border);
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Connection")
    api_base_url = st.text_input("Backend URL", value=DEFAULT_API_BASE_URL).rstrip("/")
    poll_seconds = st.slider("Refresh interval (s)", 1, 10, POLL_SECONDS_DEFAULT)
    log_limit = st.slider("Routing log rows", 5, 100, 25)
    if "paused" not in st.session_state:
        st.session_state.paused = False
    st.session_state.paused = st.toggle("⏸ Pause live updates", value=st.session_state.paused)

    st.markdown("---")
    st.markdown(
        "**Stack:** FastAPI · ComputeRouter · Gemma-4 (edge + heavy) · "
        "AMD MI300X Dev Cloud · Ollama/vLLM · $0 commercial API spend"
    )


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_json(path: str):
    try:
        resp = requests.get(f"{api_base_url}{path}", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as exc:  # noqa: BLE001 — surfaced to the UI, not swallowed
        return None, str(exc)


metrics, metrics_err = fetch_json("/api/metrics")
log_data, log_err = fetch_json(f"/api/routing-log?limit={log_limit}")
backend_ok = metrics_err is None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
status_class = "status-live" if backend_ok else "status-down"
status_text = "● LIVE" if backend_ok else "● BACKEND OFFLINE"

st.markdown(
    f"""
    <div class="bazaar-header">
        <div>
            <div class="bazaar-title">⚡ BAZAAR<span>-AI</span></div>
            <div class="bazaar-subtitle">Compute-Saver Live Dashboard · Track 3 LocalFirst</div>
        </div>
        <div class="status-pill {status_class}">{status_text}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not backend_ok:
    st.markdown(
        f"""
        <div class="empty-state">
            Can't reach the backend at <b>{html.escape(api_base_url)}</b>.<br><br>
            Start it with:  <code>uvicorn main:app --port 8080</code><br>
            Or update the "Backend URL" in the sidebar if it's running elsewhere.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    total_calls = metrics.get("total_calls", 0)
    edge_calls = metrics.get("edge_compute_calls", 0)
    heavy_calls = metrics.get("heavy_compute_calls", 0)
    fallback_calls = metrics.get("fallback_calls", 0)
    edge_ratio = metrics.get("edge_compute_ratio", 0.0)
    edge_ratio_pct = round(edge_ratio * 100, 1)

    # -----------------------------------------------------------------
    # Headline metric cards
    # -----------------------------------------------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card accent-teal">
                <div class="metric-label">Total API Cost</div>
                <div class="metric-value teal">$0.00</div>
                <div class="metric-sub">100% self-hosted Gemma-4 · zero commercial API spend</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card accent-teal">
                <div class="metric-label">Edge Compute Ratio · VRAM Saved</div>
                <div class="metric-value teal">{edge_ratio_pct}%</div>
                <div class="ratio-track"><div class="ratio-fill" style="width:{edge_ratio_pct}%;"></div></div>
                <div class="metric-sub">{edge_calls} / {total_calls} requests served on gemma-4-26b-a4b-it (edge)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card accent-red">
                <div class="metric-label">Heavy GPU Tasks · MI300X Engaged</div>
                <div class="metric-value red">{heavy_calls}</div>
                <div class="metric-sub">gemma-4-31b-it requests on the AMD MI300X Dev Cloud node</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------
    # Secondary stat chips
    # -----------------------------------------------------------------
    avg_latency = "—"
    if log_data and log_data.get("entries"):
        latencies = [e["latency_ms"] for e in log_data["entries"] if "latency_ms" in e]
        if latencies:
            avg_latency = f"{int(sum(latencies) / len(latencies))} ms"

    st.markdown(
        f"""
        <div class="chip-row">
            <div class="chip"><div class="chip-value">{total_calls}</div><div class="chip-label">Total Requests</div></div>
            <div class="chip"><div class="chip-value">{edge_calls}</div><div class="chip-label">Edge Calls</div></div>
            <div class="chip"><div class="chip-value">{heavy_calls}</div><div class="chip-label">Heavy Calls</div></div>
            <div class="chip"><div class="chip-value">{fallback_calls}</div><div class="chip-label">Simulated Fallbacks</div></div>
            <div class="chip"><div class="chip-value">{avg_latency}</div><div class="chip-label">Avg Latency (shown log)</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------
    # Live routing log feed
    # -----------------------------------------------------------------
    st.markdown(
        '<div class="section-title"><span class="pulse-dot"></span>Live Routing Log — WhatsApp &amp; API Traffic</div>',
        unsafe_allow_html=True,
    )

    entries = (log_data or {}).get("entries", [])

    if log_err:
        st.warning(f"Couldn't load routing log: {log_err}")
    elif not entries:
        st.markdown(
            """
            <div class="empty-state">
                No traffic yet. Send a WhatsApp message to your Twilio sandbox number,
                or POST to <code>/api/chat/</code>, and it'll appear here within
                the refresh interval.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for e in entries:
            ts_raw = e.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_raw).astimezone().strftime("%H:%M:%S")
            except ValueError:
                ts = ts_raw[:8]

            source = e.get("source", "api")
            source_badge = "badge-whatsapp" if source == "whatsapp" else "badge-api"
            source_label = "WhatsApp" if source == "whatsapp" else "API"

            target = e.get("target", "edge-compute")
            is_heavy = target == "heavy-compute"
            row_class = "route-heavy" if is_heavy else "route-edge"
            model_badge = "badge-heavy" if is_heavy else "badge-edge"
            model_used = html.escape(str(e.get("model_used", "")))

            fallback_html = (
                '<span class="badge badge-fallback">SIM</span>'
                if e.get("simulated_fallback")
                else ""
            )

            user_id = html.escape(str(e.get("user_id", "")))[:22]
            message = html.escape(str(e.get("message", "")))
            reason = html.escape(str(e.get("routing_reason", "")))
            latency = e.get("latency_ms", "—")

            st.markdown(
                f"""
                <div class="log-row {row_class}">
                    <span class="log-time">{ts}</span>
                    <span class="badge {source_badge}">{source_label}</span>
                    <span class="log-user">{user_id}</span>
                    <span class="log-msg" title="{message}">&ldquo;{message}&rdquo;</span>
                    <span><span class="badge {model_badge}">{model_used}</span>{fallback_html}
                        <span style="color:var(--text-dim); margin-left:6px;">{latency}ms</span></span>
                    <span class="log-reason">{reason}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Raw log table (sortable / exportable)"):
            df = pd.DataFrame(entries)
            st.dataframe(df, width='stretch', hide_index=True)
            st.download_button(
                "Download as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="bazaar_ai_routing_log.csv",
                mime="text/csv",
            )

st.markdown(
    f"<div style='text-align:center; color:#3a4552; font-family:JetBrains Mono, monospace; "
    f"font-size:11px; margin-top:24px;'>last updated {datetime.now().strftime('%H:%M:%S')} · "
    f"polling {html.escape(api_base_url)} every {poll_seconds}s</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if not st.session_state.paused:
    time.sleep(poll_seconds)
    st.rerun()
