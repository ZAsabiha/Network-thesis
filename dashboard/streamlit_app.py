"""
streamlit_app.py
===================
Live dashboard polling the FastAPI backend.

Two jobs:
  1. Raise a visible alert the moment a new attack is detected (red banner +
     toast) and list the captured alerts in a feed table.
  2. Draw the Mininet topology you built (from mininet_topo/topology_state.json)
     and, whenever an attack is active, light up the attacker and the victim on
     the graph with a red edge between them.

The topology picture updates itself: rebuild the network at a different scale
and the graph here follows on the next refresh. Attacker/victim identification
comes straight from the IDS alert (src_ip = attacker, dst_ip = victim), so a
node only turns red once the model has actually flagged the traffic.

RUN:
  cd dashboard && streamlit run streamlit_app.py
"""

import os
import time
import requests
import pandas as pd
import streamlit as st

from topology_view import active_attacks, build_topology_dot, load_state

# Override with IDS_BACKEND_URL to point at a backend on another port.
BACKEND_URL = os.environ.get("IDS_BACKEND_URL", "http://127.0.0.1:8000")

# Colour per attack class, used by the banner and the table.
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
# ---- manual mitigation control -------------------------------------
_state = load_state()
if _state:
    st.sidebar.markdown("### 🛡️ Manual Block")
    _hosts = _state.get("hosts", [])
    _attackers = [h for h in _hosts if h["role"] == "attacker"] or _hosts
    _labels = {f'{h["name"]} ({h["ip"]})': h for h in _attackers}
    _pick = st.sidebar.selectbox("Attacker to block", list(_labels.keys()))
    _dur = st.sidebar.slider("Block for (sec)", 30, 600, 120, step=30)
    if st.sidebar.button("🚫 Block now"):
        _h = _labels[_pick]
        try:
            _r = requests.post(f"{BACKEND_URL}/manual_block",
                               json={"src_mac": _h["mac"], "src_ip": _h["ip"],
                                     "duration_sec": _dur}, timeout=2)
            if _r.ok:
                st.sidebar.success(f'Block requested: {_h["name"]} for {_dur}s '
                                   f'(applies within ~3s)')
            else:
                st.sidebar.error("Backend rejected the request.")
        except requests.exceptions.RequestException:
            st.sidebar.error("Cannot reach backend.")

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


def render_topology(state, active):
    st.subheader("Network Topology")
    if state is None:
        st.info(
            "No topology yet. Start the network with "
            "`sudo python3 mininet_topo/topology.py` and this map appears "
            "automatically once it writes topology_state.json.")
        return

    params = state.get("params", {})
    st.caption(
        "Mode: **%s** · %d switches · %d hosts · attackers: %s"
        % (state.get("mode", "?"), len(state.get("switches", [])),
           len(state.get("hosts", [])),
           ", ".join(h["name"] for h in state["hosts"]
                     if h["role"] == "attacker") or "none"))

    # Text call-out of who is attacking whom, above the graph.
    if active:
        for a in active:
            atk = a["attacker_name"] or a["attacker_ip"]
            vic = "%s (%s)" % (a["victim_name"], a["victim_role"]) \
                if a["victim_name"] else a["victim_ip"]
            srcs = " from %d sources" % a["source_count"] \
                if a["source_count"] > 1 else ""
            st.error("🔴 **%s** → **%s** — %s at %.0f%% confidence%s"
                     % (atk, vic, a["attack_class"], 100 * a["confidence"], srcs))
    else:
        st.success("No active attacks. Fabric shown in normal state.")

    st.graphviz_chart(build_topology_dot(state, active), use_container_width=True)
    st.caption("Legend: green=web · blue=dns · purple=db · grey=client/attacker · "
               "dashed link=WAN (lossy). Red node=attacker, orange=victim during "
               "an attack. Dark boxes are switches (core→aggregation→edge).")


placeholder = st.empty()

with placeholder.container():
    alerts = fetch_json("/alerts", {"limit": 200})
    mitigations = fetch_json("/mitigations", {"limit": 200}) or []
    state = load_state()

    if alerts is None:
        st.error("Cannot reach backend at "
                  f"{BACKEND_URL}. Is `uvicorn main:app` running in backend/?")
        # Still show the topology so the map is useful without a backend.
        render_topology(state, [])
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

        # ---- topology map with attacker/victim highlighting ----------
        ip_map = {h["ip"]: h for h in state["hosts"]} if state else {}
        active = active_attacks(alerts, ip_map, time.time())
        render_topology(state, active)

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
        
                # ---- mitigation / blocked sources ----------------------------
        st.subheader("🛡️ Mitigation — Blocked Attackers")
        active_blocks = [m for m in mitigations if m.get("active")]
        if active_blocks:
            c1, c2 = st.columns(2)
            c1.metric("Currently Blocked", len(active_blocks))
            c2.metric("Total Blocks (this run)", len(mitigations))
            mdf = pd.DataFrame(active_blocks)
            mdf["blocked at"] = pd.to_datetime(mdf["blocked_at"], unit="s")
            mdf["expires in (s)"] = mdf["remaining_sec"].apply(
                lambda s: "permanent" if s > 31_000_000 else int(round(s)))
            mdf = mdf[["src_mac", "src_ip", "attack_class", "dpid",
                       "blocked at", "expires in (s)", "reason"]]
            st.dataframe(mdf, use_container_width=True, hide_index=True)
            st.caption("Blocked by MAC — a spoofed flood forges the IP but not "
                       "the hardware address, so one rule stops it. Each block "
                       "is a self-expiring lease; an ongoing attack is re-blocked.")
        else:
            st.info("No attackers currently blocked. A block appears here within "
                    "one poll cycle of an attack being detected.")

st.caption(f"Auto-refreshing every {refresh_sec}s. Backend: {BACKEND_URL}")
time.sleep(refresh_sec)
st.rerun()
