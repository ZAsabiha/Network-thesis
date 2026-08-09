"""
detector.py
=============
Threshold / heuristic attack detector.

WHY THIS EXISTS
---------------
model/ids_model.pkl was trained on StandardScaler-normalised features, but
only the classifier was pickled - the fitted scaler was never saved beside
it. Feeding raw OpenFlow counters (duration=3.0, packets=30000, bytes=1.8e6)
to a model that expects z-scores puts every live flow far outside the
training distribution, so it labels everything "Normal" and no alert can
ever fire. Verified: 299 of 315 sampled flow shapes come back "Normal", and
the handful of "DoS" hits peak at confidence 0.44 - under the 0.6 threshold.

So these rules are the PRIMARY trigger. The model still runs on every flow
and its opinion rides along on the alert as a secondary signal, so you can
show the comparison in your report. Once model/train_model.py is re-run on a
real dataset it saves a Pipeline(StandardScaler, RandomForest), which makes
live inference match training and lets the ML path fire on its own again.

Rules read the same feature vector the model consumes, plus per-source
aggregates computed across one whole FlowStatsReply batch - port scans are
invisible if you only ever look at one flow in isolation.
"""

from feature_config import FEATURE_NAMES  # noqa: F401  (kept in sync deliberately)

# Feature vector positions, mirroring FEATURE_NAMES order.
F_DURATION, F_PACKETS, F_BYTES, F_AVG_SIZE, F_PKT_RATE, F_BYTE_RATE, F_PROTO = range(7)

# --- Tuning knobs -----------------------------------------------------------
# Calibrated for the 100 Mbps Mininet links in mininet_topo/topology.py.
# hping3 --flood pushes tens of thousands of tiny packets per second; normal
# ping/iperf between h1-h4 stays orders of magnitude below these.
FLOOD_PKT_RATE = 800          # pps, sustained, to call it a flood
FLOOD_SMALL_PACKET = 120      # bytes; SYN/ICMP floods carry near-empty packets
VOLUMETRIC_BYTE_RATE = 5_000_000   # Bps (~40 Mbps), UDP flood territory
# A legitimate bulk transfer (iperf, file copy) fills the MTU - ~1448B TCP
# segments. Floods that slip past the packet-rate rule still tend to carry
# short packets. Without this ceiling a normal iperf run between h2 and h1
# saturates the 100 Mbps link at ~11 MB/s and gets flagged as DoS every time.
# Consequence, worth stating in the report: a flood built from MTU-sized
# packets is indistinguishable from legitimate bulk traffic at flow-stats
# granularity. That is corner case #1, not a bug in these thresholds.
VOLUMETRIC_MAX_AVG_SIZE = 1000
DDOS_DISTINCT_SOURCES = 3     # distinct srcs hammering one dst => DDoS not DoS
PROBE_MIN_FLOWS = 15          # flows from one src in a single batch
PROBE_MAX_PACKETS = 8         # a scan touches many targets with few packets each
SLOW_MIN_DURATION = 30.0      # sec, low-and-slow (Slowloris) needs a long tail
SLOW_MAX_PKT_RATE = 5.0       # pps
SLOW_MIN_PACKETS = 10         # ignore idle/stale flows with almost no traffic
# One long-lived quiet flow is an ssh session, not an attack. Slowloris is
# defined by holding MANY sockets open at once, so concurrency - not the
# shape of any single flow - is what separates the two.
SLOW_MIN_CONCURRENT = 10


def _confidence(observed, threshold, ceiling=0.97):
    """
    Scale confidence by how far past the threshold we are: exactly at the
    threshold is a weak 0.60, an order of magnitude past it saturates near
    the ceiling. Keeps borderline flash-crowd bursts visibly less certain
    than an unmistakable flood, which is the point of corner case #1.
    """
    if threshold <= 0:
        return 0.6
    ratio = observed / threshold
    if ratio <= 1:
        return 0.6
    return min(ceiling, 0.6 + 0.37 * min(1.0, (ratio - 1) / 9.0))


class BatchContext:
    """Per-source and per-destination aggregates for one FlowStatsReply."""

    def __init__(self, flows):
        self.flows_per_src = {}
        self.small_flows_per_src = {}
        self.slow_flows_per_src = {}
        self.srcs_per_dst = {}

        for vector, info in flows:
            src, dst = info["src_ip"], info["dst_ip"]
            self.flows_per_src[src] = self.flows_per_src.get(src, 0) + 1
            if info["packet_count"] <= PROBE_MAX_PACKETS:
                self.small_flows_per_src[src] = self.small_flows_per_src.get(src, 0) + 1
            if (vector[F_DURATION] >= SLOW_MIN_DURATION
                    and vector[F_PKT_RATE] < SLOW_MAX_PKT_RATE
                    and vector[F_PACKETS] >= SLOW_MIN_PACKETS):
                self.slow_flows_per_src[src] = self.slow_flows_per_src.get(src, 0) + 1
            self.srcs_per_dst.setdefault(dst, set()).add(src)

    def distinct_sources_to(self, dst):
        return len(self.srcs_per_dst.get(dst, ()))


class ThresholdDetector:
    """
    Stateless per-flow rules + the batch aggregates above.

    classify() returns None for traffic that looks normal, else a dict:
        {"attack_class": str, "confidence": float, "rule": str}
    """

    @staticmethod
    def build_batch_context(flows):
        return BatchContext(flows)

    def classify(self, vector, info, batch):
        duration = vector[F_DURATION]
        packets = vector[F_PACKETS]
        avg_size = vector[F_AVG_SIZE]
        pkt_rate = vector[F_PKT_RATE]
        byte_rate = vector[F_BYTE_RATE]
        src, dst = info["src_ip"], info["dst_ip"]

        # 1. High-rate flood of near-empty packets => SYN/ICMP flood.
        #    If several sources converge on the same victim, call it DDoS.
        if pkt_rate >= FLOOD_PKT_RATE and avg_size <= FLOOD_SMALL_PACKET:
            distinct = batch.distinct_sources_to(dst)
            if distinct >= DDOS_DISTINCT_SOURCES:
                return {
                    "attack_class": "DDoS",
                    "confidence": _confidence(pkt_rate, FLOOD_PKT_RATE),
                    "rule": f"packet_rate={pkt_rate:.0f}pps avg_size={avg_size:.0f}B "
                            f"from {distinct} sources",
                }
            return {
                "attack_class": "DoS",
                "confidence": _confidence(pkt_rate, FLOOD_PKT_RATE),
                "rule": f"packet_rate={pkt_rate:.0f}pps avg_size={avg_size:.0f}B",
            }

        # 2. Volumetric flood - too fast for rule 1's packet-size ceiling but
        #    still not MTU-filling, so it is not a well-behaved bulk transfer.
        if byte_rate >= VOLUMETRIC_BYTE_RATE and avg_size <= VOLUMETRIC_MAX_AVG_SIZE:
            distinct = batch.distinct_sources_to(dst)
            attack = "DDoS" if distinct >= DDOS_DISTINCT_SOURCES else "DoS"
            return {
                "attack_class": attack,
                "confidence": _confidence(byte_rate, VOLUMETRIC_BYTE_RATE),
                "rule": f"byte_rate={byte_rate / 1e6:.1f}MB/s",
            }

        # 3. One source fanning out across many short flows => port scan.
        #    Needs the batch view; a single scan flow looks like nothing.
        small = batch.small_flows_per_src.get(src, 0)
        if small >= PROBE_MIN_FLOWS and packets <= PROBE_MAX_PACKETS:
            return {
                "attack_class": "Probe",
                "confidence": _confidence(small, PROBE_MIN_FLOWS),
                "rule": f"{small} short flows from one source in a single poll",
            }

        # 4. Low-and-slow: many long-lived, barely-active flows held open by
        #    one source at once. This is corner case #2 - volumetric features
        #    alone would miss it, which is why it needs its own rule rather
        #    than a threshold on rate. The concurrency requirement is what
        #    keeps a single idle ssh session from tripping it.
        slow_concurrent = batch.slow_flows_per_src.get(src, 0)
        if (duration >= SLOW_MIN_DURATION
                and pkt_rate < SLOW_MAX_PKT_RATE
                and packets >= SLOW_MIN_PACKETS
                and slow_concurrent >= SLOW_MIN_CONCURRENT):
            return {
                "attack_class": "DoS",
                "confidence": _confidence(slow_concurrent, SLOW_MIN_CONCURRENT,
                                           ceiling=0.85),
                "rule": f"low-and-slow: {slow_concurrent} flows held open, "
                        f"{duration:.0f}s at {pkt_rate:.1f}pps",
            }

        return None
