"""
SHARED FEATURE CONTRACT
=======================
Imported by BOTH model/train_model.py (offline training) and
controller/feature_extractor.py (live inference inside Ryu). Every feature
vector on either side is built by build_feature_vector() below, so training
and inference cannot drift apart by construction.

Only features DIRECTLY derivable from an OFPFlowStatsReply entry belong here:
duration, packet_count, byte_count, and ratios of those. Host/application
features (NSL-KDD's num_failed_logins, TCP flag counts, IAT statistics) are
deliberately excluded - a Ryu controller cannot compute them from flow stats,
so a model trained on them can never be deployed in this loop.

PROTOCOL ENCODING - read before changing
----------------------------------------
`protocol` is the RAW IANA protocol number: 1=ICMP, 6=TCP, 17=UDP, 0=unknown.

This is deliberate. OpenFlow's `ip_proto` match field is already an IANA
number, and InSDN's `Protocol` column is already an IANA number. Using them
as-is means there is no remapping table that can silently disagree between
the two sides. An earlier version of this file remapped 6->0, 17->1, 1->2 on
the controller side only, while the model had been trained on the raw numbers
- so every TCP flow reached the forest labelled 0 (which the trees read as
"unknown protocol") and was routed down the wrong branch at every protocol
split. Do not reintroduce a compaction here.
"""

# Order matters. This exact order is used everywhere a vector is built or read.
# Per-flow features: everything derivable from one OFPFlowStats entry alone.
BASE_FEATURE_NAMES = [
    "duration_sec",       # flow-entry lifetime, seconds
    "packet_count",       # packets matched by the flow entry
    "byte_count",         # bytes matched by the flow entry
    "avg_packet_size",    # byte_count / packet_count
    "packet_rate",        # packet_count / duration_sec  (pps)
    "byte_rate",          # byte_count / duration_sec    (Bps)
    "protocol",           # IANA protocol number: 1=ICMP, 6=TCP, 17=UDP, 0=other
]

# Destination-context features: computed across ONE FlowStatsReply, describing
# the victim's neighbourhood rather than this flow on its own.
#
# WHY THESE ARE NECESSARY, not a nicety
# -------------------------------------
# InSDN's DDoS traffic is a spoofed-source flood: 121,962 distinct
# (src, dst, proto) pairs, so each forged source produces its own flow-table
# entry carrying ~2 packets at ~0.2 pps. Measured on the aggregated dataset,
# the median DDoS entry is statistically indistinguishable from an idle benign
# flow on every one of the seven features above - because it genuinely is one.
# The attack exists only in the aggregate: one destination fielding flows from
# a hundred thousand sources. A per-flow model cannot represent that, no matter
# how it is tuned.
#
# These stay ML features, not thresholds: the model learns what counts as too
# many sources, nothing here decides "malicious" on its own. Both are readable
# straight off a single OFPFlowStatsReply, which is what keeps them deployable.
# Both directions are needed, because a switch installs an entry per direction
# and a spoofed flood looks opposite on each:
#   forward  (forged_src -> victim): 121,941 sources converge on one victim
#                                    -> fan-IN is huge
#   backward (victim -> forged_src): the victim answers all of them, one entry
#                                    each -> fan-OUT is huge, fan-in is 1
# Measuring only fan-in makes the reverse half of a flood look like ordinary
# one-to-one traffic. Fan-out also carries port scans, where a single host
# touches many destinations.
CONTEXT_FEATURE_NAMES = [
    "flows_to_dst",            # flow entries in this reply aimed at the same dst
    "distinct_srcs_to_dst",    # distinct sources aimed at the same dst  (fan-in)
    "flows_from_src",          # flow entries in this reply out of the same src
    "distinct_dsts_from_src",  # distinct destinations from that src    (fan-out)
]

# Features derived from the byte counter. Grouped because they stand or fall
# together: if a dataset does not record bytes, all three become noise at once.
#
# InSDN does not record them for attack traffic. Measured share of rows with
# byte_count == 0: DDoS 99.6%, Probe 63.3%, BFA 54.4%, Web-Attack 53.1% -
# against 13.2% for Normal. So "the exporter wrote no byte count" is very
# nearly a label, and avg_packet_size becomes the strongest feature in the
# model for a reason that has nothing to do with attack behaviour. A real SYN
# flood carries 60-byte packets and a real switch counts them, so a model that
# leans on this scores ~1.00 offline and cannot see a live flood at all.
# train_model.py detects this and drops the group; --keep-byte-features
# overrides for datasets that do record bytes properly.
BYTE_FEATURE_NAMES = ["byte_count", "byte_rate", "avg_packet_size"]

FEATURE_NAMES = BASE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES

N_FEATURES = len(FEATURE_NAMES)
N_BASE_FEATURES = len(BASE_FEATURE_NAMES)

# Populated from model_metadata.json at training time. Kept here only as the
# documented default for tooling that runs before a model exists; the metadata
# file written by train_model.py is the authoritative list.
CLASS_NAMES = ["Normal", "DoS", "DDoS", "Probe", "BFA"]

# IANA numbers, matching OpenFlow ip_proto and InSDN's Protocol column.
PROTOCOL_NUMBERS = {"icmp": 1, "tcp": 6, "udp": 17}
PROTOCOL_UNKNOWN = 0

# A flow entry younger than this has no meaningful rate; clamp instead of
# dividing by zero. Training applies the same clamp, so the floor is part of
# the contract rather than a controller-side quirk.
MIN_DURATION_SEC = 0.001


def protocol_to_numeric(proto):
    """Normalise a protocol value to the IANA number used in training."""
    if proto is None:
        return PROTOCOL_UNKNOWN
    if isinstance(proto, str):
        return PROTOCOL_NUMBERS.get(proto.strip().lower(), PROTOCOL_UNKNOWN)
    try:
        return int(proto)
    except (TypeError, ValueError):
        return PROTOCOL_UNKNOWN


def build_feature_vector(duration_sec, packet_count, byte_count, protocol,
                         flows_to_dst=1, distinct_srcs_to_dst=1,
                         flows_from_src=1, distinct_dsts_from_src=1):
    """
    Build one feature vector in the EXACT order of FEATURE_NAMES.

    Called identically by train_model.py (on aggregated dataset rows) and by
    feature_extractor.py (on live Ryu flow stats). This function is the single
    definition of what a "feature vector" means in this project.

    The two context arguments default to 1 - a flow that is the only one aimed
    at its destination - so a caller that has no batch view still produces a
    well-formed vector rather than a short one.
    """
    duration_sec = max(float(duration_sec), MIN_DURATION_SEC)
    packet_count = float(packet_count)
    byte_count = float(byte_count)

    avg_packet_size = byte_count / packet_count if packet_count > 0 else 0.0
    packet_rate = packet_count / duration_sec
    byte_rate = byte_count / duration_sec

    return [
        duration_sec,
        packet_count,
        byte_count,
        avg_packet_size,
        packet_rate,
        byte_rate,
        float(protocol_to_numeric(protocol)),
        float(flows_to_dst),
        float(distinct_srcs_to_dst),
        float(flows_from_src),
        float(distinct_dsts_from_src),
    ]


def build_feature_frame(vectors):
    """
    Wrap one or more feature vectors in a DataFrame carrying FEATURE_NAMES.

    The pipeline is fitted on a named DataFrame, so predicting on a bare list
    triggers sklearn's "X does not have valid feature names" warning and, worse,
    silently accepts a mis-ordered vector. Going through this helper makes the
    column names part of every prediction call.
    """
    import pandas as pd  # local import: keeps Ryu startup light until first use

    if vectors and not isinstance(vectors[0], (list, tuple)):
        vectors = [vectors]
    return pd.DataFrame(vectors, columns=FEATURE_NAMES)


def vector_to_dict(vector):
    """Pair feature names with values, for logging and debugging."""
    return dict(zip(FEATURE_NAMES, vector))
