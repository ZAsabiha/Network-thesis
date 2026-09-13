"""
topology_view.py
==================
Pure helpers behind the dashboard's topology map: read the topology state,
work out which attacks are currently active, and turn both into a Graphviz DOT
string. Kept separate from streamlit_app.py so it carries no Streamlit runtime
and can be unit-tested on its own (streamlit_app imports these).

st.graphviz_chart renders the returned DOT in the browser, so nothing here
needs a graphviz binary or the python graphviz package installed.
"""

import json
import os

STATE_PATH = os.environ.get(
    "IDS_TOPOLOGY_STATE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "mininet_topo", "topology_state.json"))

# An alert counts as an ACTIVE attack (and is highlighted) for this many
# seconds after its timestamp: long enough that a sustained attack stays lit
# (the controller re-alerts on every ~3s poll), short enough that the graph and
# banner clear within a few seconds once the attacker withdraws.
ACTIVE_WINDOW_SEC = 6

ROLE_FILL = {
    "web": "#2a9d8f",
    "dns": "#4361ee",
    "db": "#7b2cbf",
    "client": "#ced4da",
    "attacker": "#6c757d",
}
ROLE_DARK_TEXT = {"client"}          # roles needing black text on their fill

TIER_STYLE = {
    "core": dict(shape="box3d", fill="#1d3557", font="white"),
    "agg": dict(shape="box", fill="#457b9d", font="white"),
    "edge": dict(shape="box", fill="#a8dadc", font="black"),
}
ATTACKER_FILL = "#b3001b"
VICTIM_FILL = "#e08700"


def load_state(path=STATE_PATH):
    """Read topology_state.json. Returns None if the topology has not run yet."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def active_attacks(alerts, ip_map, now, window=ACTIVE_WINDOW_SEC):
    """Collapse recent alerts into one entry per (attacker, victim, class).

    Alerts arrive newest-first, so the first time a key is seen it is the most
    recent instance -- that is the one whose confidence/counts are kept.
    """
    out, seen = [], set()
    for a in alerts:
        if now - a.get("timestamp", 0) > window:
            continue
        key = (a["src_ip"], a["dst_ip"], a["attack_class"])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "attacker_ip": a["src_ip"],
            "attacker_name": ip_map.get(a["src_ip"], {}).get("name"),
            "victim_ip": a["dst_ip"],
            "victim_name": ip_map.get(a["dst_ip"], {}).get("name"),
            "victim_role": ip_map.get(a["dst_ip"], {}).get("role"),
            "attack_class": a["attack_class"],
            "confidence": a.get("confidence", 0.0),
            "source_count": a.get("source_count", 1),
        })
    return out


def _esc(text):
    return str(text).replace('"', '').replace("\n", " ")


def build_topology_dot(state, active):
    """Turn topology state + active attacks into a Graphviz DOT string."""
    active_src = {a["attacker_ip"] for a in active}
    active_dst = {a["victim_ip"] for a in active}

    lines = [
        "digraph topo {",
        '  rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.7;',
        '  node [style=filled, fontname="Helvetica", fontsize=10, color="#333333"];',
        '  edge [color="#adb5bd"];',
    ]

    # Switches, grouped per tier so they line up on the same rank.
    by_tier = {}
    for s in state.get("switches", []):
        by_tier.setdefault(s["tier"], []).append(s)
    for tier, style in TIER_STYLE.items():
        names = []
        for s in by_tier.get(tier, []):
            names.append('"%s"' % s["name"])
            lines.append(
                '  "%s" [label="%s\\n%s", shape=%s, fillcolor="%s", fontcolor="%s"];'
                % (s["name"], _esc(s["name"]), tier, style["shape"],
                   style["fill"], style["font"]))
        if names:
            lines.append("  {rank=same; %s}" % " ".join(names))

    # Hosts, coloured by role, overridden red/orange when part of an attack.
    for h in state["hosts"]:
        is_atk = h["ip"] in active_src
        is_vic = h["ip"] in active_dst
        fill = ROLE_FILL.get(h["role"], "#ced4da")
        font = "black" if h["role"] in ROLE_DARK_TEXT else "white"
        penwidth, border = 1, "#333333"
        if is_atk:
            fill, font, penwidth, border = ATTACKER_FILL, "white", 3, "#000000"
        elif is_vic:
            fill, font, penwidth, border = VICTIM_FILL, "black", 3, "#000000"
        tag = h["role"].upper() if (is_atk or is_vic) else h["role"]
        label = "%s\\n%s\\n%s" % (_esc(h["name"]), tag, _esc(h["ip"]))
        lines.append(
            '  "%s" [label="%s", shape=box, fillcolor="%s", fontcolor="%s", '
            'color="%s", penwidth=%d];'
            % (h["name"], label, fill, font, border, penwidth))

    # Fabric (switch<->switch) links.
    for a, b in state.get("fabric_links", []):
        lines.append('  "%s" -> "%s" [dir=none, penwidth=1.6, color="#6c757d"];'
                     % (a, b))

    # Host<->edge links; WAN links dashed to show the noisy paths.
    for h in state["hosts"]:
        style = "dashed" if h.get("link_profile") == "wan" else "solid"
        lines.append('  "%s" -> "%s" [dir=none, style=%s, color="#ced4da"];'
                     % (h["edge_switch"], h["name"], style))

    # Active-attack edges: attacker -> victim, red, on top. A spoofed source
    # that maps to no real host becomes an external node.
    ext_added = set()
    for a in active:
        src_id = a["attacker_name"]
        if not src_id:
            src_id = "ext_" + a["attacker_ip"].replace(".", "_")
            if src_id not in ext_added:
                ext_added.add(src_id)
                lines.append(
                    '  "%s" [label="%s\\nexternal/spoofed", shape=doubleoctagon, '
                    'fillcolor="#b3001b", fontcolor="white"];'
                    % (src_id, _esc(a["attacker_ip"])))
        dst_id = a["victim_name"]
        if not dst_id:
            continue  # unknown victim: nothing on this topology to point at
        lbl = "%s %.0f%%" % (a["attack_class"], 100 * a["confidence"])
        if a["source_count"] > 1:
            lbl += "\\n\u00d7%d srcs" % a["source_count"]
        lines.append(
            '  "%s" -> "%s" [color="#b3001b", penwidth=3, style=bold, '
            'label="%s", fontcolor="#b3001b", fontsize=11, constraint=false];'
            % (src_id, dst_id, lbl))

    lines.append("}")
    return "\n".join(lines)
