"""
feature_extractor.py
======================
Converts a Ryu OFPFlowStatsReply entry into the SAME feature vector
shape used during training (imports build_feature_vector from
model/feature_config.py to guarantee this).
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
from feature_config import build_feature_vector, FEATURE_NAMES  # noqa: E402


def extract_features_from_flow_stat(flow_stat):
    """
    flow_stat: a single entry from ev.msg.body (list of OFPFlowStats)
    returned by Ryu's EventOFPFlowStatsReply handler.

    Returns: (feature_vector: list[float], match_info: dict)
    match_info carries src/dst so the controller can act on detections
    without needing those fields inside the ML feature vector itself.
    """
    duration_sec = flow_stat.duration_sec + flow_stat.duration_nsec / 1e9
    packet_count = flow_stat.packet_count
    byte_count = flow_stat.byte_count

    match = flow_stat.match
    ip_proto = match.get("ip_proto", None)
    src_ip = match.get("ipv4_src", match.get("eth_src", "unknown"))
    dst_ip = match.get("ipv4_dst", match.get("eth_dst", "unknown"))
    in_port = match.get("in_port", None)

    # Skip flows with zero packets / near-zero duration (table-miss entries,
    # just-installed rules) - not enough signal yet, avoid false triggers
    if packet_count == 0 or duration_sec < 0.05:
        return None, None

    vector = build_feature_vector(duration_sec, packet_count, byte_count, ip_proto)

    match_info = {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "in_port": in_port,
        "duration_sec": duration_sec,
        "packet_count": packet_count,
        "byte_count": byte_count,
    }

    return vector, match_info


def vector_to_dict(vector):
    """Helper for logging/debugging - pairs feature names with values."""
    return dict(zip(FEATURE_NAMES, vector))
