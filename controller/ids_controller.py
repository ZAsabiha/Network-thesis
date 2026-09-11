"""
ids_controller.py
=================
Ryu application implementing the detection path of the thesis architecture:

    Mininet -> OpenFlow switch -> Ryu controller -> flow feature extraction
             -> ML model -> attack classification -> dashboard / alert

The machine-learning model is the ONLY detector. There are no thresholds, no
rules and no heuristics deciding whether traffic is malicious - the single
number that gates an alert is the model's own predicted probability
(CONFIDENCE_THRESHOLD). This file also performs no mitigation: it classifies
and reports, it never installs a blocking rule or rate-limits anything.

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


class IDSController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(IDSController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.last_alert_at = {}

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

        # Group the flagged flows by victim and class before reporting. A
        # spoofed flood is one attack spread over thousands of forged sources,
        # and each of those is its own flow entry - alerting per entry would
        # write 300 rows for one hping3 --rand-source run and bury the feed.
        # This is reporting only: the model has already classified every flow
        # individually, nothing here decides whether traffic is malicious.
        classes = self.model.classes_
        flagged = {}
        for row, info in zip(proba, infos):
            idx = int(row.argmax())
            predicted, confidence = str(classes[idx]), float(row[idx])
            if predicted == NORMAL_CLASS or confidence < CONFIDENCE_THRESHOLD:
                continue
            flagged.setdefault((info["dst_ip"], predicted), []).append((info, confidence))

        for (dst_ip, predicted), items in flagged.items():
            # Report the worst-offending flow as the representative, and say how
            # many distinct sources joined in.
            info, confidence = max(items, key=lambda t: t[1])
            sources = {i["src_ip"] for i, _ in items}
            packets = sum(i["packet_count"] for i, _ in items)
            byts = sum(i["byte_count"] for i, _ in items)

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
        """
        key = (dpid, info["dst_ip"], info["ip_proto"], attack_class)
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
