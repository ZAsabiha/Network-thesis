"""
streamlit_app.py
===================
Live dashboard polling the FastAPI backend.

  1. Alerts: red banner + toasts + a live feed table.
  2. Topology map (from mininet_topo/topology_state.json): during an attack it
     lights up the attacker (red) and victim (orange) with a red arrow showing
     the direction of the attack.
  3. Mitigation: which attackers are blocked, with a live per-attacker
     countdown, plus a manual block control in the sidebar.

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

refresh_sec = st.sidebar.slider("Refresh interval (sec)", 1, 10, 1)

# ---- manual mitigation control -------------------------------------
_state = load_state()
if _state:
    st.sidebar.markdown("### 🛡️ Manual Block")
    _hosts = _state.get("hosts", [])
    _attackers = [h for h in _hosts if h["role"] == "attacker"] or _hosts
    _labels = {f'{h["name"]} ({h["ip"]})': h for h in _attackers}
    _pick = st.sidebar.selectbox("Attacker to block", list(_labels.keys()))
    _dur = st.sidebar.slider("Block for (sec)", 30, 600, 30, step=30)
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
# we have already announced.
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


def render_mitigations(mitigations, state=None):
    """Blocked-attacker panel: one row per attacker (MAC), live countdown."""
    st.subheader("🛡️ Mitigation — Blocked Attackers")
    active_blocks = [m for m in mitigations if m.get("active")]

    # Map hardware/IP back to the friendly host name (h1, h6, ...) so the
    # panel shows WHICH host is blocked, not just its MAC/IP.
    _hosts = (state or {}).get("hosts", [])
    name_by_mac = {h.get("mac"): h.get("name") for h in _hosts}
    name_by_ip = {h.get("ip"): h.get("name") for h in _hosts}

    if not active_blocks:
        st.info("No attackers currently blocked. A block appears here within "
                "one poll cycle of an attack being detected.")
        return

    # Deduplicate by MAC: a host blocked both manually and by the detector is
    # still ONE blocked attacker. Keep the longest-remaining lease, merge the
    # reasons (e.g. "DoS, Manual").
    by_mac = {}
    for m in active_blocks:
        mac = m.get("src_mac", "")
        e = by_mac.get(mac)
        if e is None:
            e = {"src_mac": mac, "src_ip": m.get("src_ip", ""),
                 "remaining_sec": 0, "duration_sec": 0, "classes": set()}
            by_mac[mac] = e
        e["classes"].add(m.get("attack_class", "") or "")
        if not e["src_ip"] and m.get("src_ip"):
            e["src_ip"] = m.get("src_ip")
        if (m.get("remaining_sec", 0) or 0) > e["remaining_sec"]:
            e["remaining_sec"] = m.get("remaining_sec", 0) or 0
            e["duration_sec"] = m.get("duration_sec", 0) or 0

    rows = list(by_mac.values())
    st.metric("Currently Blocked", len(rows))

    for e in rows:
        rem = e["remaining_sec"]
        dur = e["duration_sec"]
        permanent = rem > 31_000_000
        who = e["src_ip"] or e["src_mac"]
        host_name = name_by_mac.get(e["src_mac"]) or name_by_ip.get(e["src_ip"])
        label = f"{host_name} — {e['src_mac']}" if host_name else e["src_mac"]
        cls = ", ".join(sorted(c for c in e["classes"] if c)) or "blocked"

        if permanent:
            st.error(f"🔴 **{label}**  ({who}) — {cls} — "
                     f"**PERMANENTLY blocked** (repeat offender)")
        else:
            st.warning(f"🟠 **{label}**  ({who}) — {cls} — "
                       f"unblocks in **{int(rem)}s**")
            frac = max(0.0, min(1.0, rem / dur)) if dur else 0.0
            st.progress(frac)

    st.caption("One row per attacker (blocked by MAC — a spoofed flood forges "
               "the IP but not the hardware address). Each block is a lease that "
               "auto-expires unless it escalates to permanent for a repeat "
               "offender.")


placeholder = st.empty()

with placeholder.container():
    alerts = fetch_json("/alerts", {"limit": 200})
    mitigations = fetch_json("/mitigations", {"limit": 200}) or []
    state = load_state()

    if alerts is None:
        st.error("Cannot reach backend at "
                  f"{BACKEND_URL}. Is `uvicorn main:app` running in backend/?")
        render_topology(state, [])
    else:
        # ---- new-alert detection -------------------------------------
        newest_id = alerts[0]["id"] if alerts else None
        first_load = st.session_state.last_seen_id is None

        if first_load:
            fresh = []
        else:
            fresh = [a for a in alerts if a["id"] > st.session_state.last_seen_id]

        if newest_id is not None:
            st.session_state.last_seen_id = newest_id

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

        # ---- topology map with attack DIRECTION ----------------------
        ip_map = {h["ip"]: h for h in state["hosts"]} if state else {}
        active = active_attacks(alerts, ip_map, time.time())
        render_topology(state, active)

        # ---- mitigation panel (live countdown) -----------------------
        render_mitigations(mitigations, state)

        # ---- live alert feed -----------------------------------------
        st.subheader("Live Alert Feed")
        if alerts:
            full = pd.DataFrame(alerts)
            full["time"] = pd.to_datetime(full["timestamp"], unit="s")

            # One verdict per victim: keep the highest-confidence class seen for
            # each destination, newest first. The model emits several classes
            # for a single attack (incidental flows, boundary flips); collapsing
            # per dst_ip shows the verdict that matters instead of the spray.
            df = (full.sort_values("confidence", ascending=False)
                      .drop_duplicates("dst_ip", keep="first")
                      .sort_values("time", ascending=False))
            df = df[["time", "attack_class", "confidence", "src_ip", "dst_ip",
                     "dpid", "packet_count"]]
            st.dataframe(
                df.style.apply(highlight_severity, axis=1),
                use_container_width=True,
                height=400,
            )
            st.caption("One row per victim (its highest-confidence class). "
                       "Expand below for every raw alert.")
            with st.expander("Raw alert feed (all classes)"):
                raw = full[["time", "attack_class", "confidence", "src_ip",
                            "dst_ip", "dpid", "packet_count"]]
                st.dataframe(
                    raw.style.apply(highlight_severity, axis=1),
                    use_container_width=True,
                    height=300,
                )
        else:
            st.info("No alerts yet - run an attack simulation from attacks/ "
                     "on the attacker host.")

st.caption(f"Auto-refreshing every {refresh_sec}s. Backend: {BACKEND_URL}")
time.sleep(refresh_sec)
st.rerun()