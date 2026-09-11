"""
streamlit_app.py
===================
Minimal live dashboard polling the FastAPI backend.

Does one job: raise a visible alert the moment a new attack is detected
(red banner + toast) and list the captured alerts in a feed table.

RUN:
  cd dashboard && streamlit run streamlit_app.py
"""

import os
import time
import requests
import pandas as pd
import streamlit as st

# Override with IDS_BACKEND_URL to point at a backend on another port.
BACKEND_URL = os.environ.get("IDS_BACKEND_URL", "http://127.0.0.1:8000")

# Colour per attack class, used by both the banner and the table.
SEVERITY = {
    "DDoS": "#b3001b",
    "DoS": "#d1495b",
    "Botnet": "#7b2cbf",
    "BFA": "#7b2cbf",
    "U2R": "#5a189a",
    "Web-Attack": "#9d4edd",
    "Probe": "#e08700",
}
DEFAULT_SEVERITY = "#d1495b"

st.set_page_config(page_title="SDN Intrusion Detection Dashboard", layout="wide")
st.title("ML-Based SDN Intrusion Detection Dashboard")

refresh_sec = st.sidebar.slider("Refresh interval (sec)", 1, 10, 3)

# Survives st.rerun(), so we can tell a genuinely new alert apart from one
# we have already announced. Without this the banner would re-fire on every
# single refresh for as long as the alert stays in the feed.
if "last_seen_id" not in st.session_state:
    st.session_state.last_seen_id = None


def fetch_json(path, params=None):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=2)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def highlight_severity(row):
    colour = SEVERITY.get(row["attack_class"], DEFAULT_SEVERITY)
    return [f"background-color: {colour}22; color: {colour}"] * len(row)


def render_banner(alert):
    colour = SEVERITY.get(alert["attack_class"], DEFAULT_SEVERITY)
    when = pd.to_datetime(alert["timestamp"], unit="s").strftime("%H:%M:%S")
    st.markdown(
        f"""
        <div style="background:{colour};color:#fff;padding:14px 18px;
                    border-radius:8px;margin-bottom:12px;
                    font-size:1.05rem;line-height:1.6">
          <strong style="font-size:1.25rem">&#9888; ATTACK DETECTED &mdash;
          {alert['attack_class']}</strong><br>
          <code style="background:rgba(0,0,0,.25);color:#fff;padding:1px 6px;
                       border-radius:3px">{alert['src_ip']}</code>
          &rarr;
          <code style="background:rgba(0,0,0,.25);color:#fff;padding:1px 6px;
                       border-radius:3px">{alert['dst_ip']}</code>
          &nbsp;&middot;&nbsp; confidence {alert['confidence']:.0%}
          &nbsp;&middot;&nbsp; {alert['packet_count']:,} packets
          &nbsp;&middot;&nbsp; switch {alert['dpid']}
          &nbsp;&middot;&nbsp; {when}
        </div>
        """,
        unsafe_allow_html=True,
    )


placeholder = st.empty()

with placeholder.container():
    alerts = fetch_json("/alerts", {"limit": 200})

    if alerts is None:
        st.error("Cannot reach backend at "
                  f"{BACKEND_URL}. Is `uvicorn main:app` running in backend/?")
    else:
        # ---- new-alert detection -------------------------------------
        # /alerts comes back newest-first, so the highest id is alerts[0].
        newest_id = alerts[0]["id"] if alerts else None
        first_load = st.session_state.last_seen_id is None

        if first_load:
            fresh = []
        else:
            fresh = [a for a in alerts if a["id"] > st.session_state.last_seen_id]

        if newest_id is not None:
            st.session_state.last_seen_id = newest_id

        # Toast every new alert, newest first, capped so a sustained flood
        # cannot stack hundreds of popups on one refresh.
        for a in fresh[:3]:
            st.toast(
                f"{a['attack_class']} from {a['src_ip']} "
                f"({a['confidence']:.0%})",
                icon="🚨",
            )
        if len(fresh) > 3:
            st.toast(f"+{len(fresh) - 3} more alerts this refresh", icon="🚨")

        # ---- banner --------------------------------------------------
        if alerts:
            render_banner(alerts[0])
        else:
            st.success("No attacks detected. Monitoring...")

        # ---- live alert feed -----------------------------------------
        st.subheader("Live Alert Feed")
        if alerts:
            df = pd.DataFrame(alerts)
            df["time"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df[["time", "attack_class", "confidence", "src_ip", "dst_ip",
                     "dpid", "packet_count"]]
            st.dataframe(
                df.style.apply(highlight_severity, axis=1),
                use_container_width=True,
                height=400,
            )
        else:
            st.info("No alerts yet - run an attack simulation from attacks/ "
                     "on the attacker host.")

st.caption(f"Auto-refreshing every {refresh_sec}s. Backend: {BACKEND_URL}")
time.sleep(refresh_sec)
st.rerun()
