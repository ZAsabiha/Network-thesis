"""
SHARED FEATURE DEFINITION
==========================
This file is imported by BOTH train_model.py (offline training) and
controller/feature_extractor.py (live inference in Ryu).

Why this matters: the #1 reason live SDN-IDS projects silently fail is that
the model is trained on features that don't match what the controller can
actually compute in real time from OpenFlow flow stats. By importing this
same list in both places, we guarantee the feature vector shape, order,
and meaning are identical in training and production.

Only include features that are DIRECTLY derivable from OFPFlowStatsReply:
duration, packet_count, byte_count, and simple ratios of these.
Do NOT include host/application-level features (e.g. NSL-KDD's
num_failed_logins, num_shells) - Ryu cannot compute these from flow stats.
"""

# Order matters. This exact order must be used everywhere a feature
# vector is built or consumed.
FEATURE_NAMES = [
    "duration_sec",       # flow duration in seconds
    "packet_count",       # total packets in the flow
    "byte_count",         # total bytes in the flow
    "avg_packet_size",    # byte_count / packet_count
    "packet_rate",        # packet_count / duration_sec
    "byte_rate",           # byte_count / duration_sec
    "protocol",            # numeric encoded: 0=TCP, 1=UDP, 2=ICMP, 3=other
]

# Label encoding used consistently across training and live prediction
CLASS_NAMES = ["Normal", "DoS", "DDoS", "Probe", "Botnet"]

PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2}


def protocol_to_numeric(proto):
    """Map a protocol string/number to the numeric encoding used in training."""
    if isinstance(proto, str):
        return PROTOCOL_MAP.get(proto.lower(), 3)
    # OpenFlow ip_proto numbers: 6=TCP, 17=UDP, 1=ICMP
    mapping = {6: 0, 17: 1, 1: 2}
    return mapping.get(proto, 3)


def build_feature_vector(duration_sec, packet_count, byte_count, protocol):
    """
    Build a feature vector in the EXACT order of FEATURE_NAMES.
    Used identically by train_model.py (on dataset rows) and
    feature_extractor.py (on live Ryu flow stats), so the model
    always sees the same shape/meaning of input.
    """
    duration_sec = max(duration_sec, 0.001)  # avoid div-by-zero
    avg_packet_size = byte_count / packet_count if packet_count > 0 else 0
    packet_rate = packet_count / duration_sec
    byte_rate = byte_count / duration_sec
    proto_num = protocol_to_numeric(protocol)

    return [
        duration_sec,
        packet_count,
        byte_count,
        avg_packet_size,
        packet_rate,
        byte_rate,
        proto_num,
    ]
