"""
feature_extractor.py
====================
Turns one Ryu OFPFlowStatsReply entry into the feature vector the model was
trained on.

There is deliberately no feature logic in this file. The arithmetic lives in
model/feature_config.build_feature_vector(), which train_model.py calls on
every training row, so the training and inference vectors are produced by
literally the same function. This module only reads counters off the OpenFlow
message and hands them over.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from feature_config import (BASE_FEATURE_NAMES, FEATURE_NAMES,  # noqa: E402
                            build_feature_vector, vector_to_dict)

# A flow entry this young has counters too sparse to mean anything: a
# just-installed rule reports 1 packet over 2ms, which is a 500 pps "flood".
# Training clamps duration the same way, but the model still cannot tell a new
# rule from a real burst, so skip the entry and wait for the next poll.
MIN_DURATION_SEC = 0.05


def extract_batch(flow_stats):
    """
    Turn a whole OFPFlowStatsReply body into (vectors, infos).

    This is the entry point the controller uses. It exists because two of the
    nine features - flows_to_dst and distinct_srcs_to_dst - describe a
    destination across the whole reply, not one flow in isolation, so they
    cannot be computed while looking at a single OFPFlowStats entry.

    That scope is deliberate and matches training exactly: train_model.py
    computes the same two counts per emulated poll. Widening the window here
    (say, accumulating destinations across polls) would feed the model
    information it was not trained on.
    """
    infos = []
    for flow_stat in flow_stats:
        info = _read_counters(flow_stat)
        if info is not None:
            infos.append(info)

    flows_to_dst, srcs_to_dst = {}, {}
    flows_from_src, dsts_from_src = {}, {}
    for info in infos:
        src, dst = info["src_ip"], info["dst_ip"]
        flows_to_dst[dst] = flows_to_dst.get(dst, 0) + 1
        srcs_to_dst.setdefault(dst, set()).add(src)
        flows_from_src[src] = flows_from_src.get(src, 0) + 1
        dsts_from_src.setdefault(src, set()).add(dst)

    vectors = []
    for info in infos:
        src, dst = info["src_ip"], info["dst_ip"]
        info["flows_to_dst"] = flows_to_dst[dst]
        info["distinct_srcs_to_dst"] = len(srcs_to_dst[dst])
        info["flows_from_src"] = flows_from_src[src]
        info["distinct_dsts_from_src"] = len(dsts_from_src[src])
        vectors.append(build_feature_vector(
            info["duration_sec"], info["packet_count"], info["byte_count"],
            info["ip_proto"], info["flows_to_dst"], info["distinct_srcs_to_dst"],
            info["flows_from_src"], info["distinct_dsts_from_src"]))

    return vectors, infos


def _read_counters(flow_stat):
    """Pull the usable counters off one entry, or None if it carries no signal."""
    duration_sec = flow_stat.duration_sec + flow_stat.duration_nsec / 1e9
    if flow_stat.packet_count == 0 or duration_sec < MIN_DURATION_SEC:
        return None

    match = flow_stat.match
    return {
        "src_ip": match.get("ipv4_src", match.get("eth_src", "unknown")),
        "dst_ip": match.get("ipv4_dst", match.get("eth_dst", "unknown")),
        "in_port": match.get("in_port"),
        # Absent on the table-miss rule and any L2-only match; the model reads
        # a missing protocol as IANA 0 ("other").
        "ip_proto": match.get("ip_proto"),
        "duration_sec": duration_sec,
        "packet_count": flow_stat.packet_count,
        "byte_count": flow_stat.byte_count,
    }


def extract_features_from_flow_stat(flow_stat):
    """
    flow_stat: one entry from ev.msg.body (a list of OFPFlowStats) as delivered
    to Ryu's EventOFPFlowStatsReply handler.

    Returns (feature_vector, match_info), or (None, None) when the entry
    carries no usable signal.

    match_info holds src/dst/port so the controller can report *who* without
    those identifiers ever entering the feature vector. Keeping IPs out of the
    features is deliberate: a model that learns "10.0.0.5 is the attacker"
    scores beautifully on the test split and generalises to nothing.

    Single-entry convenience wrapper, used by the tests. The destination
    context defaults to "this is the only flow to that destination", so the
    controller must use extract_batch() instead - a per-entry call cannot see
    the other flows in the reply and would report a spoofed DDoS as one lonely
    two-packet flow.
    """
    info = _read_counters(flow_stat)
    if info is None:
        return None, None

    vector = build_feature_vector(info["duration_sec"], info["packet_count"],
                                  info["byte_count"], info["ip_proto"])
    for key in ("flows_to_dst", "distinct_srcs_to_dst",
                "flows_from_src", "distinct_dsts_from_src"):
        info[key] = 1
    return vector, info


__all__ = ["extract_batch", "extract_features_from_flow_stat", "vector_to_dict",
           "FEATURE_NAMES", "BASE_FEATURE_NAMES"]
