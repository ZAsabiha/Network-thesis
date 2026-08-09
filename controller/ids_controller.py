"""
ids_controller.py
====================
Main Ryu application. Combines:
  - basic L2 learning switch (so normal traffic works)
  - IP-aware flow installation (so flow stats carry real src/dst IPs)
  - periodic flow-stats polling
  - threshold detection (detector.py) as the primary alert trigger
  - live ML inference (loads model trained by model/train_model.py) as a
    secondary signal reported alongside every alert
  - alert reporting to the FastAPI backend

RUN:
  ryu-manager controller/ids_controller.py

Make sure model/ids_model.pkl exists first (run model/train_model.py).
Make sure the FastAPI backend is running (backend/main.py) so alerts
have somewhere to POST to - if it's down, the controller still runs
and just logs a warning instead of crashing.
"""

import os
import sys
import time
import joblib
import requests

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4
from ryu.lib import hub

# ryu-manager uses a custom module loader that does NOT automatically add
# the script's own directory to sys.path (unlike running `python3 file.py`
# normally). Both this file's own directory (for detector.py and
# feature_extractor.py) and ../model (for feature_config.py) must be
# added explicitly, or sibling imports fail with ModuleNotFoundError
# even though the files are right next to this one.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)
sys.path.append(os.path.join(THIS_DIR, "..", "model"))

from feature_extractor import extract_features_from_flow_stat  # noqa: E402
from detector import ThresholdDetector  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "ids_model.pkl")
BACKEND_URL = "http://127.0.0.1:8000"  # FastAPI backend
POLL_INTERVAL_SEC = 3
CONFIDENCE_THRESHOLD = 0.6  # below this, treat as Normal even if model says otherwise

# The stats poll runs every POLL_INTERVAL_SEC, and an ongoing flood stays
# detectable across every single poll. Without a cooldown one hping3 run
# would write an alert row every 3 seconds per flow and bury the dashboard.
# Re-alerting on the same (switch, src, dst, class) is suppressed until this
# many seconds have passed.
ALERT_COOLDOWN_SEC = 10


class IDSController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(IDSController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.detector = ThresholdDetector()
        self.last_alert_at = {}

        if not os.path.exists(MODEL_PATH):
            self.logger.error(
                f"Model not found at {MODEL_PATH}. Run model/train_model.py first."
            )
            self.model = None
        else:
            self.model = joblib.load(MODEL_PATH)
            self.logger.info(f"Loaded IDS model from {MODEL_PATH}")
            if not hasattr(self.model, "steps"):
                # A bare classifier means the training-time scaler was not saved
                # with it, so raw flow counters land nowhere near the training
                # distribution and everything comes back "Normal". Threshold
                # rules carry detection until train_model.py is re-run.
                self.logger.warning(
                    "Model is a bare classifier, not a Pipeline - its training "
                    "scaler is missing, so ML predictions are unreliable. "
                    "Threshold rules in detector.py are driving detection."
                )

        self.monitor_thread = hub.spawn(self._monitor)

    # ------------------------------------------------------------------
    # Basic L2 learning switch (so normal traffic actually flows)
    # ------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.datapaths[datapath.id] = datapath

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                           ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, 0, match, actions)

    def _add_flow(self, datapath, priority, match, actions, hard_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                 instructions=inst, hard_timeout=hard_timeout)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            # Match on IP fields when the packet is IPv4. A purely L2 match
            # (in_port/eth_src/eth_dst) produces flow stats with no ipv4_src,
            # ipv4_dst or ip_proto at all - which is why alerts used to carry
            # MAC addresses as "IPs" and every flow scored protocol=3 ("other").
            ip = pkt.get_protocol(ipv4.ipv4)
            if ip is not None:
                match = parser.OFPMatch(
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=ip.src,
                    ipv4_dst=ip.dst,
                    ip_proto=ip.proto,
                )
            else:
                match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self._add_flow(datapath, 1, match, actions, hard_timeout=30)

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                   in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    # ------------------------------------------------------------------
    # Periodic flow-stats polling -> feature extraction -> detection
    # ------------------------------------------------------------------
    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                self._request_flow_stats(dp)
            hub.sleep(POLL_INTERVAL_SEC)

    def _request_flow_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        dpid = ev.msg.datapath.id

        flows = []
        for flow_stat in ev.msg.body:
            vector, match_info = extract_features_from_flow_stat(flow_stat)
            if vector is not None:
                flows.append((vector, match_info))

        if not flows:
            return

        # Port scans only show up when you compare flows against each other,
        # so the whole batch is aggregated once before per-flow rules run.
        batch = self.detector.build_batch_context(flows)

        for vector, match_info in flows:
            ml_class, ml_conf = self._predict(vector)
            verdict = self.detector.classify(vector, match_info, batch)

            # The ML path stays live as an independent trigger, so a properly
            # retrained Pipeline can raise alerts the rules do not cover.
            if verdict is None:
                if ml_class not in (None, "Normal") and ml_conf >= CONFIDENCE_THRESHOLD:
                    verdict = {
                        "attack_class": ml_class,
                        "confidence": ml_conf,
                        "rule": "ml-model",
                    }
                else:
                    continue

            if not self._should_alert(dpid, match_info, verdict["attack_class"]):
                continue

            self.logger.warning(
                f"[ALERT] {verdict['attack_class']} "
                f"(conf={verdict['confidence']:.2f}, {verdict['rule']}) "
                f"src={match_info['src_ip']} dst={match_info['dst_ip']} "
                f"pkts={match_info['packet_count']} bytes={match_info['byte_count']} "
                f"| model said {ml_class} ({ml_conf:.2f})"
            )
            self._post_alert(verdict, match_info, dpid, ml_class, ml_conf)

    def _should_alert(self, dpid, match_info, attack_class):
        """Rate-limit repeat alerts for an attack that is still in progress."""
        key = (dpid, match_info["src_ip"], match_info["dst_ip"], attack_class)
        now = time.time()
        if now - self.last_alert_at.get(key, 0) < ALERT_COOLDOWN_SEC:
            return False
        self.last_alert_at[key] = now
        return True

    def _predict(self, vector):
        if self.model is None:
            return None, 0.0
        proba = self.model.predict_proba([vector])[0]
        idx = proba.argmax()
        pred_class = self.model.classes_[idx]
        confidence = float(proba[idx])
        return pred_class, confidence

    # ------------------------------------------------------------------
    # Report to backend (non-blocking-ish, tolerant of backend being down)
    # ------------------------------------------------------------------
    def _post_alert(self, verdict, match_info, dpid, ml_class, ml_conf):
        payload = {
            "timestamp": time.time(),
            "attack_class": verdict["attack_class"],
            "confidence": verdict["confidence"],
            "src_ip": match_info["src_ip"],
            "dst_ip": match_info["dst_ip"],
            "dpid": dpid,
            "packet_count": match_info["packet_count"],
            "byte_count": match_info["byte_count"],
            "detected_by": verdict["rule"],
            # ml_class is None when no model is loaded; the backend expects a
            # string, so normalise before it reaches the wire.
            "ml_class": str(ml_class) if ml_class is not None else "",
            "ml_confidence": ml_conf,
        }
        try:
            requests.post(f"{BACKEND_URL}/alerts", json=payload, timeout=1)
        except requests.exceptions.RequestException:
            self.logger.debug("Backend not reachable - alert logged locally only")