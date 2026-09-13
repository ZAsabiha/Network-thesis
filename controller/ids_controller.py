"""
ids_controller.py
=================
Ryu application implementing the detection path of the thesis architecture:

    Mininet -> OpenFlow switch -> Ryu controller -> flow feature extraction
             -> ML model -> attack classification -> dashboard / alert

The machine-learning model is the ONLY detector. There are no thresholds, no
rules and no heuristics deciding whether traffic is malicious - the single
number that gates an alert is the model's own predicted probability
(CONFIDENCE_THRESHOLD). Mitigation is a separate closed-loop step that runs
only AFTER the model has raised an alert: the attacker's MAC is blocked with a
timed DROP flow (see the MITIGATION block below).

RUN:
  ryu-manager controller/ids_controller.py

Prerequisites:
  * model/ids_model.pkl and model/model_metadata.json  (python3 model/train_model.py)
  * backend running                                    (cd backend && uvicorn main:app --port 8000)
The controller still runs if the backend is down; alerts are logged only.
"""

import json
import os
import sys
import time
from collections import deque

import joblib
import requests

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ether_types, ethernet, ipv4, packet
from ryu.ofproto import ofproto_v1_3

# ryu-manager loads this file through its own module loader, which does NOT put
# the script's directory on sys.path the way `python3 file.py` does. Both this
# directory (feature_extractor) and ../model (feature_config) must be added
# explicitly or the sibling imports fail even though the files are right here.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)
sys.path.append(os.path.join(THIS_DIR, "..", "model"))

from feature_config import (BYTE_FEATURE_NAMES, CONTEXT_FEATURE_NAMES,  # noqa: E402
                            FEATURE_NAMES, build_feature_frame)
from feature_extractor import extract_batch  # noqa: E402

MODEL_DIR = os.path.join(THIS_DIR, "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "ids_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

BACKEND_URL = os.environ.get("IDS_BACKEND_URL", "http://127.0.0.1:8000")
POLL_INTERVAL_SEC = 3

# Below this probability the flow is reported as Normal regardless of which
# class won. This is the model's own confidence, not a traffic threshold - it
# trades recall for a lower false-positive rate and is the knob to sweep for
# the ROC / FPR discussion in the thesis.
CONFIDENCE_THRESHOLD = 0.60

# Must match --flow-timeout used in train_model.py: the model was trained on
# entries with this lifetime, so changing one without the other reintroduces
# exactly the train/serve mismatch this rewrite fixed.
FLOW_HARD_TIMEOUT = 10

# A flood stays detectable on every poll, so without a cooldown one hping3 run
# writes an alert row every POLL_INTERVAL_SEC per flow and buries the feed.
ALERT_COOLDOWN_SEC = 10

NORMAL_CLASS = "Normal"

# Flow entries below this many packets carry almost no signal: ARP replies,
# a single ICMP ping, a reverse-direction TCP ACK. The model classifies them
# anyway and scatters them across Probe/Web-Attack/Botnet/BFA, burying the real
# verdict in noise. Skip them before inference. Kept deliberately low so a
# genuine flood (thousands of packets per entry) is never touched.
MIN_CLASSIFY_PACKETS = 5

# ----------------------------------------------------------------------------
# MITIGATION. On an alert the controller pushes a high-priority DROP flow that
# matches the attacker's MAC address, cutting them off in the data plane. MAC,
# not IP, because a spoofed flood forges a new source IP every packet but keeps
# one real MAC - so a single MAC rule stops the whole flood. Each block is a
# self-expiring LEASE, so a wrong block heals itself instead of blackholing a
# host forever.
# ----------------------------------------------------------------------------
MITIGATION_ENABLED = True
BLOCK_PRIORITY = 100
BLOCK_DURATION = {
    "DoS": 30, "DDoS": 30, "Probe": 30,
    "BFA": 30, "Botnet": 30, "Web-Attack": 30,
}
DEFAULT_BLOCK_SEC = 30
MAX_MACS_PER_ALERT = 50
# Escalation is PER VICTIM (see _mitigate): a victim that has never been
# attacked by this MAC blocks it with a timed BLOCK_DURATION lease first; if
# the same MAC returns to a victim that already blocked it, the block is
# permanent. A sustained attack escalates the same way, because the victim
# is recorded on the very first block.
REPEAT_OFFENCE_LIMIT = 2      # unused now; escalation is per-victim, kept for ref
PERMANENT_BLOCK = False       # first offence is timed; escalation handles repeats
_FOREVER = 10 ** 9

class IDSController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(IDSController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.last_alert_at = {}
        # Mitigation state: MAC -> lease-expiry epoch (so we don't re-block an
        # already-blocked MAC every poll). whitelist = server/victim MACs that
        # must never be blocked, read from the topology state file.
        self.blocked = {}
        self.offense_count = {}       # MAC -> how many times we've blocked it
        # MAC -> set of victim IPs this attacker has already been blocked for.
        # A victim already in this set 'knows' the attacker, so a repeat hit
        # on it is blocked permanently instead of getting another lease.
        self.attacked_victims = {}
        self.whitelist = self._load_whitelist()

        # Rolling inference-latency window, reported for the thesis metrics.
        self.latency_ms = deque(maxlen=500)
        self.flows_seen = 0
        self.alerts_raised = 0

        self.model = None
        self.metadata = {}
        # Which columns this particular model wants. The controller always
        # builds all of FEATURE_NAMES, then hands over the subset the model was
        # actually fitted on, so an ablation model (--no-context-features)
        # deploys through the same path without an edit here.
        self.model_features = list(FEATURE_NAMES)
        self._load_model()

        self.monitor_thread = hub.spawn(self._monitor)

    def _load_whitelist(self):
        """Server/victim MAC addresses that must never be blocked."""
        wl = set()
        state_path = os.path.join(THIS_DIR, "..", "mininet_topo",
                                  "topology_state.json")
        try:
            with open(state_path) as fh:
                state = json.load(fh)
            protected_ips = set(state.get("servers", {}).values())
            if state.get("victim_ip"):
                protected_ips.add(state["victim_ip"])
            for h in state.get("hosts", []):
                if h.get("ip") in protected_ips and h.get("mac"):
                    wl.add(h["mac"])
        except (OSError, ValueError):
            self.logger.info("No readable topology_state.json - whitelist empty.")
        self.logger.info("Mitigation whitelist MACs (never blocked): %s",
                         sorted(wl) or "none")
        return wl

    # ------------------------------------------------------------------
    # Model loading + the checks that would have caught the original bug
    # ------------------------------------------------------------------
    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            self.logger.error(
                "No model at %s. Run:  python3 model/train_model.py --csv data/\n"
                "Detection is disabled until a model exists.", MODEL_PATH)
            return

        self.model = joblib.load(MODEL_PATH)
        self.logger.info("Loaded model from %s", MODEL_PATH)

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as fh:
                self.metadata = json.load(fh)
        else:
            self.logger.warning(
                "model_metadata.json missing - cannot verify the feature "
                "contract this model was trained against.")

        # A bare estimator carries no preprocessing, so whatever transform
        # training applied is simply gone at inference time. That is how the
        # original 278MB pkl came to answer "Normal" to every live flow.
        if not hasattr(self.model, "steps"):
            self.logger.error(
                "Model is a bare %s, not a Pipeline. Any preprocessing used "
                "during training is missing, so predictions cannot be trusted. "
                "Retrain with model/train_model.py.",
                type(self.model).__name__)

        expected = list(self.metadata.get("features") or [])
        unknown = [f for f in expected if f not in FEATURE_NAMES]
        if unknown:
            self.logger.error(
                "FEATURE MISMATCH - model wants %s, which this controller "
                "cannot build (it produces %s). Retrain.", unknown, FEATURE_NAMES)
            self.model = None
            return
        if expected:
            self.model_features = expected
            missing = [f for f in FEATURE_NAMES if f not in expected]
            if missing:
                self.logger.info(
                    "Model uses %d of %d features; not using %s.",
                    len(expected), len(FEATURE_NAMES), missing)
                if any(f in missing for f in CONTEXT_FEATURE_NAMES):
                    self.logger.warning(
                        "Destination-context features are absent, so this model "
                        "cannot detect a spoofed-source flood - that signal "
                        "lives in the fan-in/fan-out counts, not in any single "
                        "flow. This is the --no-context-features ablation.")
                if any(f in missing for f in BYTE_FEATURE_NAMES):
                    self.logger.info(
                        "Byte-derived features were dropped at training time "
                        "because the dataset does not record byte counts for "
                        "attack traffic. Expected on InSDN.")

        uncovered = self.metadata.get("uncovered_shapes") or []
        if uncovered:
            self.logger.warning(
                "Model does not cover these traffic shapes: %s. They will pass "
                "unreported.", "; ".join(uncovered))

        n_in = getattr(self.model, "n_features_in_", None)
        if n_in is not None and n_in != len(self.model_features):
            self.logger.error(
                "Model expects %d features, metadata lists %d. Retrain.",
                n_in, len(self.model_features))
            self.model = None
            return

        if self.metadata.get("thesis_ready") is False:
            self.logger.warning(
                "This model was trained on SYNTHETIC traffic (%s). Fine for a "
                "wiring check, not for reported results.",
                self.metadata.get("dataset"))

        self.logger.info("Classes: %s", list(getattr(self.model, "classes_", [])))

    # ------------------------------------------------------------------
    # L2 learning switch, with IP-aware flow installation
    # ------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath

        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, 0, parser.OFPMatch(), actions)
        self.logger.info("Switch %s connected", datapath.id)

    def _add_flow(self, datapath, priority, match, actions, hard_timeout=0):
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                instructions=inst, hard_timeout=hard_timeout)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst, src, dpid = eth.dst, eth.src, datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            # Match on IP fields for IPv4. A purely L2 match produces flow stats
            # with no ipv4_src/ipv4_dst/ip_proto at all, which is why alerts
            # used to carry MAC addresses as "IPs" and every flow reached the
            # model with an unknown protocol.
            ip = pkt.get_protocol(ipv4.ipv4)
            if ip is not None:
                match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                        eth_src=src,
                                        ipv4_src=ip.src, ipv4_dst=ip.dst,
                                        ip_proto=ip.proto)
            else:
                match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self._add_flow(datapath, 1, match, actions,
                           hard_timeout=FLOW_HARD_TIMEOUT)

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        datapath.send_msg(parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port,
            actions=actions, data=data))

    # ------------------------------------------------------------------
    # Poll -> extract -> classify
    # ------------------------------------------------------------------
    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                dp.send_msg(dp.ofproto_parser.OFPFlowStatsRequest(dp))
            self._poll_manual_blocks()
            hub.sleep(POLL_INTERVAL_SEC)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        if self.model is None:
            return
        dpid = ev.msg.datapath.id

        # One call for the whole reply: flows_to_dst / distinct_srcs_to_dst
        # describe a destination across every entry in this batch, so they
        # cannot be computed one flow at a time.
        vectors, infos = extract_batch(ev.msg.body)

        if not vectors:
            return

        # Drop low-signal entries before inference (see MIN_CLASSIFY_PACKETS).
        # A destination under a spoofed fan-in is kept even when each forged
        # source sent only a packet or two, so DDoS detection is unaffected.
        kept = [(v, i) for v, i in zip(vectors, infos)
                if i["packet_count"] >= MIN_CLASSIFY_PACKETS
                or i["distinct_srcs_to_dst"] >= 3]
        if not kept:
            return
        vectors = [v for v, _ in kept]
        infos = [i for _, i in kept]

        self.flows_seen += len(vectors)

        # One predict_proba for the whole reply. Calling the model per flow
        # costs ~34 ms of fixed scikit-learn overhead each time; batching the
        # same flows costs ~0.3 ms each, which is what keeps classification
        # comfortably inside the 3-second poll interval.
        t0 = time.perf_counter()
        try:
            frame = build_feature_frame(vectors)[self.model_features]
            proba = self.model.predict_proba(frame)
        except Exception as exc:                      # noqa: BLE001
            self.logger.error("Inference failed on %d flows: %s", len(vectors), exc)
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_flow_ms = elapsed_ms / len(vectors)
        self.latency_ms.append(per_flow_ms)

        # Reporting only: the model has already classified every flow
        # individually, nothing here decides whether traffic is malicious.
        # First pass - keep the flows the model flagged above threshold.
        classes = self.model.classes_
        flagged = []
        for row, info in zip(proba, infos):
            # A whitelisted server (its replies/backscatter) is never the
            # attacker - don't raise alerts for traffic it originates.
            if info.get("src_mac") in self.whitelist:
                continue
            idx = int(row.argmax())
            predicted, confidence = str(classes[idx]), float(row[idx])
            if predicted == NORMAL_CLASS or confidence < CONFIDENCE_THRESHOLD:
                continue
            flagged.append((info, predicted, confidence))

        # Drop reverse-direction flows. A host that is itself under attack
        # answers back, and those replies get scored as their own (usually
        # bogus) attack - e.g. a flooded victim's return traffic reads as
        # Probe. Any flagged flow whose SOURCE is some other flow's victim is
        # that victim's own outbound traffic, so we suppress it. Caveat: if a
        # single host is both attacked and attacking (A<->B both flagged, or a
        # compromised victim attacking onward), its outbound attack is dropped
        # too; that trade favours a clean feed over that rare case.
        victims = {info["dst_ip"] for info, _, _ in flagged}
        flagged = [(info, predicted, confidence)
                   for info, predicted, confidence in flagged
                   if info["src_ip"] not in victims]

        # One verdict per victim. A spoofed flood is one attack spread over
        # thousands of forged sources, and a mixed episode produces flows the
        # model reads as several classes; alerting per (source, class) would
        # bury the feed. Collapse every flagged flow aimed at the same
        # destination into a single alert and let the highest-confidence flow
        # name the class.
        by_victim = {}
        for info, predicted, confidence in flagged:
            by_victim.setdefault(info["dst_ip"], []).append((info, predicted, confidence))

        for dst_ip, items in by_victim.items():
            # The highest-confidence flow is the representative: it names the
            # class and its source leads the alert. Counts sum across the
            # victim's flows so srcs/pkts/bytes describe the whole episode.
            info, predicted, confidence = max(items, key=lambda t: t[2])
            sources = {i["src_ip"] for i, _, _ in items}
            packets = sum(i["packet_count"] for i, _, _ in items)
            byts = sum(i["byte_count"] for i, _, _ in items)

            if not self._should_alert(dpid, info, predicted):
                continue

            self.alerts_raised += 1
            self.logger.warning(
                "[ALERT] %-10s conf=%.2f  %s -> %s  proto=%s  flows=%d srcs=%d "
                "pkts=%d bytes=%d rate=%.0f pps  (%.2f ms)",
                predicted, confidence, info["src_ip"], dst_ip, info["ip_proto"],
                len(items), len(sources), packets, byts,
                info["packet_count"] / max(info["duration_sec"], 1e-3), per_flow_ms)
            self._post_alert(predicted, confidence, info, dpid, per_flow_ms,
                             packets, byts, len(sources))
            # Closed-loop response: block the attacker, don't just report it.
            self._mitigate(predicted, dst_ip, [i for i, _, _ in items], dpid)

        if self.flows_seen and self.latency_ms:
            avg = sum(self.latency_ms) / len(self.latency_ms)
            self.logger.debug("flows=%d alerts=%d mean_inference=%.3f ms/flow",
                              self.flows_seen, self.alerts_raised, avg)

    def _should_alert(self, dpid, info, attack_class):
        """
        Suppress repeat alerts while the same attack is still in progress.

        Keyed on the VICTIM, not the source: a spoofed flood presents a fresh
        forged source on every poll, so a source-keyed cooldown never matches
        and suppresses nothing. ip_proto stays in the key because a host
        running a SYN flood and a UDP flood at the same victim is two attacks,
        and the second should not be swallowed as a duplicate of the first.
        The switch (dpid) is deliberately NOT in the key, so one attack that
        crosses several switches raises ONE alert, not one per switch.
        """
        key = (info["dst_ip"], info["ip_proto"], attack_class)
        now = time.time()
        if now - self.last_alert_at.get(key, 0) < ALERT_COOLDOWN_SEC:
            return False
        self.last_alert_at[key] = now
        return True

    # ------------------------------------------------------------------
    def _post_alert(self, predicted, confidence, info, dpid, inference_ms,
                    packet_count=None, byte_count=None, source_count=1):
        payload = {
            "timestamp": time.time(),
            "attack_class": predicted,
            "confidence": confidence,
            "src_ip": str(info["src_ip"]),
            "dst_ip": str(info["dst_ip"]),
            "dpid": dpid,
            "packet_count": int(packet_count if packet_count is not None
                                else info["packet_count"]),
            "byte_count": int(byte_count if byte_count is not None
                              else info["byte_count"]),
            # How many distinct sources this one alert stands for - 1 for a
            # single-source attack, thousands for a spoofed flood.
            "source_count": int(source_count),
            "detected_by": "ml-model",
            "ml_class": predicted,
            "ml_confidence": confidence,
            "inference_ms": round(inference_ms, 4),
        }
        try:
            requests.post(f"{BACKEND_URL}/alerts", json=payload, timeout=1)
        except requests.exceptions.RequestException:
            self.logger.debug("Backend unreachable - alert logged locally only")

    # ------------------------------------------------------------------
    # Mitigation: block the attacker's MAC for a class-dependent lease
    # ------------------------------------------------------------------
    def _mitigate(self, attack_class, dst_ip, infos, dpid):
        if not MITIGATION_ENABLED:
            return
        duration = BLOCK_DURATION.get(attack_class, DEFAULT_BLOCK_SEC)
        now = time.time()

        mac_to_ip = {}
        for info in infos:
            mac = info.get("src_mac")
            if mac:
                mac_to_ip.setdefault(mac, info.get("src_ip"))

        newly = []
        for mac in list(mac_to_ip)[:MAX_MACS_PER_ALERT]:
            if mac in self.whitelist:
                continue
            if now < self.blocked.get(mac, 0):     # still under an active lease
                continue

            # Escalation is PER VICTIM. A victim that has never been attacked by
            # this MAC gives it one timed lease. The block turns permanent the
            # moment this MAC is blocked for a victim that already blocked it
            # before - whether the attacker withdrew and later came back to that
            # victim, or never stopped and the first lease simply aged out (the
            # victim is already on record either way).
            prior_victims = self.attacked_victims.setdefault(mac, set())
            repeat = dst_ip in prior_victims
            prior_victims.add(dst_ip)
            self.offense_count[mac] = self.offense_count.get(mac, 0) + 1

            if repeat:
                hard_to, expires, rec_dur = 0, now + _FOREVER, 0
            else:
                hard_to, expires, rec_dur = duration, now + duration, duration

            self.blocked[mac] = expires
            for dp in list(self.datapaths.values()):
                self._install_block(dp, mac, hard_to)
            self._post_mitigation(mac, mac_to_ip[mac], attack_class, dst_ip,
                                  dpid, now, expires, rec_dur)
            newly.append((mac, repeat))

        for mac, repeat in newly:
            if repeat:
                self.logger.warning(
                    "[MITIGATED] %s -> %s: PERMANENTLY blocked %s "
                    "(repeat against a victim that already blocked it)",
                    attack_class, dst_ip, mac)
            else:
                self.logger.warning(
                    "[MITIGATED] %s -> %s: blocked %s for %ds (offence #1)",
                    attack_class, dst_ip, mac, duration)

    def _install_block(self, datapath, mac, hard_timeout):
        """Drop everything from this MAC. Empty instruction list == drop.
        hard_timeout=0 makes the block permanent (stays until removed)."""
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        match = parser.OFPMatch(eth_src=mac)
        mod = parser.OFPFlowMod(
            datapath=datapath, priority=BLOCK_PRIORITY, match=match,
            instructions=[], hard_timeout=int(hard_timeout),
            command=ofproto.OFPFC_ADD)
        datapath.send_msg(mod)

    def _post_mitigation(self, mac, sample_ip, attack_class, dst_ip, dpid,
                         blocked_at, expires_at, duration):
        payload = {
            "src_mac": mac,
            "src_ip": sample_ip or "",
            "attack_class": attack_class,
            "dpid": dpid,
            "blocked_at": blocked_at,
            "expires_at": expires_at,
            "duration_sec": duration,
            "reason": f"{attack_class} against {dst_ip}",
        }
        try:
            requests.post(f"{BACKEND_URL}/mitigations", json=payload, timeout=1)
        except requests.exceptions.RequestException:
            self.logger.debug("Backend unreachable - mitigation logged locally only")

    def _poll_manual_blocks(self):
        """Apply block requests made from the dashboard."""
        try:
            resp = requests.get(f"{BACKEND_URL}/manual_block/pending", timeout=1)
            pending = resp.json()
        except (requests.exceptions.RequestException, ValueError):
            return

        for req in pending:
            mac = req.get("src_mac")
            if not mac:
                self._ack_manual(req.get("id"))
                continue
            if mac in self.whitelist:
                self.logger.warning("[MANUAL-SKIP] %s is whitelisted", mac)
                self._ack_manual(req.get("id"))
                continue
            # Already under an active block (e.g. the detector already caught
            # it) - don't stack a duplicate record for the same attacker.
            if time.time() < self.blocked.get(mac, 0):
                self.logger.info("[MANUAL-SKIP] %s already blocked", mac)
                self._ack_manual(req.get("id"))
                continue
            duration = int(req.get("duration_sec") or DEFAULT_BLOCK_SEC)
            hard_to = 0 if PERMANENT_BLOCK else duration
            now = time.time()
            expires = now + (_FOREVER if PERMANENT_BLOCK else duration)
            self.blocked[mac] = expires
            for dp in list(self.datapaths.values()):
                self._install_block(dp, mac, hard_to)
            self._post_mitigation(mac, req.get("src_ip", ""), "Manual",
                                  req.get("victim") or "(operator)", 0,
                                  now, expires, duration)
            self.logger.warning("[MANUAL-MITIGATED] blocked %s for %ds (dashboard)",
                                mac, duration)
            self._ack_manual(req.get("id"))

    def _ack_manual(self, req_id):
        try:
            requests.post(f"{BACKEND_URL}/manual_block/ack",
                          json={"id": req_id}, timeout=1)
        except requests.exceptions.RequestException:
            pass
