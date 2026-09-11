"""
train_model.py
==============
Trains the IDS classifier and writes a deployment-ready
model/ids_model.pkl + model/model_metadata.json.

WHY THIS SCRIPT WAS REWRITTEN
-----------------------------
The shipped ids_model.pkl scored 0.93 accuracy on its own test split and still
labelled every live flow "Normal". Three separate reasons, all fixed here:

1. FLOW-AGGREGATION SEMANTICS (the real one).
   InSDN rows come from CICFlowMeter, which cuts a pcap into short
   bidirectional micro-flows: a SYN flood is ~30,000 rows of 2 packets each,
   lasting a few milliseconds. An OpenFlow flow-table entry is the opposite -
   ONE unidirectional entry per (ipv4_src, ipv4_dst, ip_proto) that accumulates
   every matching packet for its whole lifetime, so the same flood is ONE row
   with 30,000 packets over 3 seconds. Same seven column names, completely
   different distributions. Measured on the old model: 97.8% of its
   duration_sec split thresholds sit below 1.0 second, i.e. its entire decision
   structure lives in a regime the controller can never produce. Live flows
   landed in leaves the forest had almost no resolution over and came back
   "Normal" at ~0.5 confidence.
   -> Fixed by emulate_flow_table_entries() below, which re-aggregates the
      dataset into controller-shaped flow entries before any training happens.

2. PROTOCOL ENCODING. Training used the raw IANA number (6/17/1); the
   controller remapped to 0/1/2 before predicting, so TCP arrived as "0" =
   unknown. -> Fixed in feature_config.py; both sides now use IANA numbers.

3. BARE ESTIMATOR. joblib.dump(clf) saves a classifier, not an inference
   contract. -> A Pipeline is saved instead, plus model_metadata.json pinning
   the feature list, order and class labels.

RUN
---
  # Real training (download InSDN first, put the CSVs in data/)
  python3 model/train_model.py --csv data/

  # Smoke test with generated traffic - pipeline check only, NOT thesis data
  python3 model/train_model.py --synthetic
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_config import (BASE_FEATURE_NAMES, BYTE_FEATURE_NAMES,  # noqa: E402
                            FEATURE_NAMES, MIN_DURATION_SEC,
                            build_feature_frame, build_feature_vector)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "..", "data")
MODEL_PATH = os.path.join(HERE, "ids_model.pkl")
METADATA_PATH = os.path.join(HERE, "model_metadata.json")

NORMAL_CLASS = "Normal"

# InSDN / CICIDS / CIC-DDoS ship the same fields under different headers, and
# several exports carry leading spaces. Headers are stripped, then matched
# against these candidates in order.
COLUMN_MAP = {
    "src_ip":    ["Src IP", "Source IP", "src_ip"],
    "dst_ip":    ["Dst IP", "Destination IP", "dst_ip"],
    "protocol":  ["Protocol", "protocol"],
    "timestamp": ["Timestamp", "timestamp"],
    "duration":  ["Flow Duration", "flow_duration", "Duration"],
    "fwd_pkts":  ["Tot Fwd Pkts", "Total Fwd Packets", "Total Fwd Packet"],
    "bwd_pkts":  ["Tot Bwd Pkts", "Total Backward Packets", "Total Bwd packets"],
    "fwd_bytes": ["TotLen Fwd Pkts", "Total Length of Fwd Packets",
                  "Total Length of Fwd Packet"],
    "bwd_bytes": ["TotLen Bwd Pkts", "Total Length of Bwd Packets",
                  "Total Length of Bwd Packet"],
    "label":     ["Label", "label", "Attack", "attack_cat"],
}

# CICFlowMeter reports Flow Duration in MICROSECONDS; the controller measures
# seconds. Getting this wrong is the other classic silent train/serve skew.
DURATION_UNIT_DIVISOR = 1_000_000.0


# ---------------------------------------------------------------------------
# Dataset loading and label normalisation
# ---------------------------------------------------------------------------
def resolve(df, key, required=True):
    for candidate in COLUMN_MAP[key]:
        if candidate in df.columns:
            return candidate
    if required:
        raise SystemExit(
            f"Could not find a column for '{key}'. Tried {COLUMN_MAP[key]}.\n"
            f"Columns present: {list(df.columns)[:40]}\n"
            f"Add the right header to COLUMN_MAP['{key}'] in {__file__}."
        )
    return None


def load_dataset(path):
    """Load one CSV, or every CSV in a directory, into a single frame."""
    if os.path.isdir(path):
        csvs = sorted(os.path.join(path, f) for f in os.listdir(path)
                      if f.lower().endswith(".csv"))
        if not csvs:
            raise SystemExit(
                f"No CSV files in {path}.\n"
                "Download the InSDN dataset (Normal_data.csv, OVS.csv,\n"
                "metasploitable-2.csv) into that directory, or run with\n"
                "--synthetic for a pipeline smoke test."
            )
    elif os.path.isfile(path):
        csvs = [path]
    else:
        raise SystemExit(
            f"Dataset not found: {path}\n"
            "Pass --csv <file-or-directory>, or --synthetic for a smoke test."
        )

    frames = []
    for c in csvs:
        d = pd.read_csv(c, low_memory=False)
        d.columns = [col.strip() for col in d.columns]
        print(f"  {os.path.basename(c)}: {len(d):,} rows")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} rows from {len(csvs)} file(s)")
    return df


def normalise_label(raw):
    """
    Collapse dataset-specific label spellings onto one class vocabulary.

    InSDN alone ships 'DDoS' and 'DDoS ' as separate values and writes Botnet
    as 'BOTNET', which is why the notebook's value_counts showed DDoS twice.
    """
    r = str(raw).strip().lower()
    if r in ("normal", "benign"):
        return NORMAL_CLASS
    if "ddos" in r:
        return "DDoS"
    if "dos" in r:
        return "DoS"
    if "probe" in r or "scan" in r:
        return "Probe"
    if "bfa" in r or "brute" in r:
        return "BFA"
    if "web" in r or "xss" in r or "sql" in r:
        return "Web-Attack"
    if "bot" in r:
        return "Botnet"
    if "u2r" in r or "exploit" in r:
        return "U2R"
    return "Unknown"


# ---------------------------------------------------------------------------
# The core fix: turn micro-flows into controller-shaped flow-table entries
# ---------------------------------------------------------------------------
def reconstruct_arrival_times(ts):
    """
    Recover sub-minute arrival times from InSDN's truncated Timestamp column.

    THE PROBLEM
    -----------
    InSDN's exporter drops the seconds field in most of its output. Measured on
    the shipped CSVs:
        Normal_data.csv        68,424 rows ->    830 distinct timestamps (1% have seconds)
        OVS.csv               138,722 rows ->    504 distinct timestamps
        metasploitable-2.csv  136,743 rows ->     58 distinct timestamps (0% have seconds)
    So ~2,350 metasploitable rows share a single timestamp. Windowing 30-second
    flow entries on that directly would put every flow on a 60-second boundary,
    leave odd windows empty, and collapse duration to a constant.

    THE RECONSTRUCTION
    ------------------
    Rows carrying the same truncated timestamp are spread uniformly across the
    interval that timestamp represents (60s when the seconds field is absent,
    1s when it is present), in their original file order - which is the order
    CICFlowMeter emitted them, i.e. capture order.

    WHY IT DOES NOT LEAK
    --------------------
    The spreading is class-agnostic: it uses only row order and the timestamp's
    own precision, never the label. It does not invent the rate signal either -
    a flood genuinely produces more flows per minute than a browsing session,
    so denser packing inside a minute is a property of the traffic that the
    truncation hid, not one this function adds.

    LIMITATION, worth stating in the write-up: exact intra-minute arrival order
    is unrecoverable, so per-entry duration carries up to 60s of reconstruction
    error. Results are therefore sensitive to --flow-timeout; report the
    sensitivity rather than a single number.
    """
    start = ts.astype("int64") / 1e9                    # epoch seconds
    # A timestamp landing exactly on a minute boundary lost its seconds field.
    quantum = pd.Series(np.where(start % 60 == 0, 60.0, 1.0), index=start.index)
    order = start.groupby(start).cumcount()
    size = start.groupby(start).transform("size")
    return start + quantum * order / size


def emulate_flow_table_entries(df, flow_timeout, polls_per_entry):
    """
    Re-aggregate CICFlowMeter micro-flows into the flow-table entries a Ryu
    controller would actually poll, so training and inference see the same
    distribution.

    What the controller does (ids_controller.py):
      * installs ONE unidirectional entry per (ipv4_src, ipv4_dst, ip_proto)
        with hard_timeout=`flow_timeout`
      * polls every POLL_INTERVAL_SEC, reading cumulative packet_count and
        byte_count plus duration_sec = the age of the entry at that poll

    What this function does to match it:
      1. splits each bidirectional row into a forward (src->dst, Fwd counters)
         and a backward (dst->src, Bwd counters) record, because an OpenFlow
         match is unidirectional
      2. tiles time into `flow_timeout`-second windows; a (src, dst, proto)
         group inside one window is one entry's lifetime
      3. emits `polls_per_entry` samples per entry at evenly spaced poll times,
         each carrying the counters accumulated up to that poll and
         duration = poll_time - first_packet_time

    Step 3 is what produces the realistic 0-30s duration range the controller
    reports. It also acts as honest augmentation: the model sees each attack
    both early (few packets, short age) and late (fully developed), which is
    exactly the range it must classify in production.
    """
    src_c, dst_c = resolve(df, "src_ip"), resolve(df, "dst_ip")
    proto_c, ts_c = resolve(df, "protocol"), resolve(df, "timestamp")
    dur_c, lab_c = resolve(df, "duration"), resolve(df, "label")
    fp, bp = resolve(df, "fwd_pkts"), resolve(df, "bwd_pkts")
    fb, bb = resolve(df, "fwd_bytes"), resolve(df, "bwd_bytes")

    ts = pd.to_datetime(df[ts_c], errors="coerce", dayfirst=True, format="mixed")
    if ts.isna().mean() > 0.9:
        raise SystemExit(
            f"Could not parse the '{ts_c}' column as timestamps "
            f"({ts.isna().mean():.0%} failed).\n"
            "Flow-entry emulation needs wall-clock times to window on. Check "
            "the column's format and adjust the pd.to_datetime call in "
            "emulate_flow_table_entries()."
        )

    coarse = (ts.astype("int64") / 1e9 % 60 == 0).mean()
    print(f"{coarse:.0%} of timestamps are minute-truncated; reconstructing "
          f"sub-minute arrival times")
    start = reconstruct_arrival_times(ts)
    span = pd.to_numeric(df[dur_c], errors="coerce") / DURATION_UNIT_DIVISOR
    proto = pd.to_numeric(df[proto_c], errors="coerce").fillna(0).astype(int)
    label = df[lab_c].map(normalise_label)

    num = lambda col: pd.to_numeric(df[col], errors="coerce").fillna(0)
    fwd_pkts, bwd_pkts = num(fp), num(bp)

    def record(src, dst, pkts, byts):
        return pd.DataFrame({
            "src": df[src].astype(str) if isinstance(src, str) else src,
            "dst": df[dst].astype(str) if isinstance(dst, str) else dst,
            "proto": proto, "label": label, "start": start,
            "span": span.fillna(0.0).clip(lower=0.0),
            "pkts": pkts, "byts": byts,
        })

    # InSDN's Fwd/Bwd direction fields cannot be trusted. Measured on the
    # shipped CSVs: Tot Fwd Pkts is 0 for 100% of DDoS rows and 63% of Probe
    # rows, with every packet booked as "backward" - even though the row is
    # keyed Src IP -> Dst IP, so zero forward packets would mean the flow does
    # not exist. Splitting on those fields silently deletes the entire forward
    # half of the flood: every DDoS entry ends up as victim -> forged_source,
    # which turns the attack's 121,941-source fan-IN into a fan-OUT and makes
    # the victim look like the attacker.
    #
    # So when the fields look broken, fall back to attributing the flow's TOTAL
    # counters to its Src -> Dst key. Less faithful to a unidirectional
    # OpenFlow match, but it preserves who-attacked-whom, which matters more.
    fwd_zero = float((fwd_pkts == 0).mean())
    if fwd_zero > 0.25:
        print(f"Tot Fwd Pkts is zero on {fwd_zero:.0%} of rows - direction "
              f"fields unreliable; attributing total counters to Src -> Dst")
        micro = record(src_c, dst_c, fwd_pkts + bwd_pkts, num(fb) + num(bb))
    else:
        micro = pd.concat([record(src_c, dst_c, fwd_pkts, num(fb)),
                           record(dst_c, src_c, bwd_pkts, num(bb))],
                          ignore_index=True)
        print("direction fields look sound - splitting into unidirectional records")

    micro = micro[(micro["pkts"] > 0) & micro["start"].notna()
                  & (micro["label"] != "Unknown")].reset_index(drop=True)
    print(f"{len(micro):,} micro-flow records after direction handling")

    # An attack label wins over Normal inside a group: a flow entry that
    # carried flood packets IS carrying the attack, whatever else shared it.
    micro["attack"] = (micro["label"] != NORMAL_CLASS).astype(int)
    micro = micro.sort_values("attack", ascending=False, kind="stable")

    epoch = micro["start"].min()
    micro["window"] = ((micro["start"] - epoch) // flow_timeout).astype("int64")
    window_start = epoch + micro["window"] * flow_timeout

    # Emit one sample per poll offset. Rows whose first packet has not arrived
    # by the poll time contribute nothing to that sample, exactly as they would
    # contribute nothing to a real counter read at that instant.
    samples = []
    for k in range(1, polls_per_entry + 1):
        poll_at = window_start + flow_timeout * k / polls_per_entry
        chunk = micro[micro["start"] <= poll_at].copy()
        chunk["poll_at"] = poll_at[chunk.index]
        samples.append(chunk)
    polled = pd.concat(samples, ignore_index=True)

    grouped = polled.groupby(["src", "dst", "proto", "window", "poll_at"],
                             sort=False).agg(
        label=("label", "first"),
        packet_count=("pkts", "sum"),
        byte_count=("byts", "sum"),
        first_seen=("start", "min"),
    ).reset_index()

    grouped["duration_sec"] = (grouped["poll_at"] - grouped["first_seen"]).clip(
        lower=MIN_DURATION_SEC, upper=float(flow_timeout))
    grouped = grouped[grouped["packet_count"] > 0]

    # Destination context, computed over the entries visible in ONE emulated
    # poll - the exact scope the controller has when it processes one
    # FlowStatsReply. Computing it over the whole dataset instead would leak
    # information the controller can never see at runtime.
    to_dst = grouped.groupby(["dst", "window", "poll_at"], sort=False)["src"]
    grouped["flows_to_dst"] = to_dst.transform("size")
    grouped["distinct_srcs_to_dst"] = to_dst.transform("nunique")

    from_src = grouped.groupby(["src", "window", "poll_at"], sort=False)["dst"]
    grouped["flows_from_src"] = from_src.transform("size")
    grouped["distinct_dsts_from_src"] = from_src.transform("nunique")

    print(f"{len(grouped):,} emulated flow-table entries "
          f"(window={flow_timeout}s, {polls_per_entry} polls/entry)")
    if len(grouped) < 5000:
        print(
            "WARNING: aggregation collapsed the dataset to very few entries.\n"
            "  Aggregation is lossy by design - hundreds of micro-flows between\n"
            "  the same pair become one entry - but this few will not train a\n"
            "  useful forest. Lower --flow-timeout (and the matching\n"
            "  hard_timeout in ids_controller.py), or raise --polls-per-entry."
        )
    return grouped


def prepare_entries(entries, max_per_class, min_samples, verbose=True):
    """
    Downsample dominant classes and drop unlearnably-rare ones.

    Shared with evaluation/evaluate_model.py on purpose. If the evaluator
    rebuilt the split from the undownsampled frame, its "held-out" rows would
    overlap the rows the forest was fitted on and every metric would be
    inflated. Same function, same seed, same rows.
    """
    if max_per_class:
        over = entries["label"].value_counts()
        over = over[over > max_per_class]
        if len(over) and verbose:
            print(f"\nDownsampling to max-per-class={max_per_class:,}: "
                  f"{ {k: int(v) for k, v in over.items()} }")
        if len(over):
            entries = pd.concat([
                g.sample(max_per_class, random_state=42)
                if len(g) > max_per_class else g
                for _, g in entries.groupby("label", sort=False)
            ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    counts = entries["label"].value_counts()
    small = counts[counts < min_samples]
    if len(small):
        if verbose:
            print(f"\nDropping classes under min-samples={min_samples}: "
                  f"{ {k: int(v) for k, v in small.items()} }")
        entries = entries[~entries["label"].isin(small.index)]
    return entries


def to_feature_matrix(entries):
    """Run every emulated entry through the shared build_feature_vector()."""
    rows = [
        build_feature_vector(r.duration_sec, r.packet_count, r.byte_count, r.proto,
                             getattr(r, "flows_to_dst", 1),
                             getattr(r, "distinct_srcs_to_dst", 1),
                             getattr(r, "flows_from_src", 1),
                             getattr(r, "distinct_dsts_from_src", 1))
        for r in entries.itertuples(index=False)
    ]
    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = entries["label"].to_numpy()
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)
    return X[keep].reset_index(drop=True), y[keep.to_numpy()]


# ---------------------------------------------------------------------------
# Synthetic fallback - smoke test only
# ---------------------------------------------------------------------------
def synthetic_entries(n_per_class=6000, seed=42):
    """
    Generate controller-shaped flow entries when no dataset is available.

    THIS IS NOT THESIS DATA. It exists so the pipeline, the controller, the
    tests and the evaluation script can be run end to end before InSDN is
    downloaded. Any model trained from it is marked thesis_ready=false in
    model_metadata.json and its scores mean nothing beyond "the plumbing
    works" - the generator draws from the same distributions the classifier
    then learns, so accuracy is circular by construction.
    """
    rng = np.random.default_rng(seed)

    def block(label, n, dur, rate, size, protos, fan_in, fan_out):
        duration = rng.uniform(*dur, n)
        packet_rate = np.exp(rng.uniform(np.log(rate[0]), np.log(rate[1]), n))
        avg_size = rng.uniform(*size, n)
        packets = np.maximum(1, (packet_rate * duration).round())
        loguni = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n)).round()
        srcs, dsts = loguni(*fan_in), loguni(*fan_out)
        return pd.DataFrame({
            "label": label,
            "duration_sec": duration,
            "packet_count": packets,
            "byte_count": (packets * avg_size).round(),
            "proto": rng.choice(protos, n),
            "distinct_srcs_to_dst": srcs,
            "distinct_dsts_from_src": dsts,
            # at least one flow per peer, usually a few more
            "flows_to_dst": srcs * rng.uniform(1.0, 2.5, n).round(),
            "flows_from_src": dsts * rng.uniform(1.0, 2.5, n).round(),
        })

    return pd.concat([
        # normal browsing / ping / bulk transfer
        block("Normal", n_per_class, (0.05, 30), (0.1, 150), (60, 1460), [6, 6, 17, 1],
              (1, 4), (1, 4)),
        # single-source flood: very high rate, near-empty packets, one peer
        block("DoS", n_per_class, (0.5, 30), (600, 25000), (40, 130), [6, 17, 1],
              (1, 2), (1, 2)),
        # spoofed flood, forward half: thousands of forged sources on one victim
        block("DDoS", n_per_class // 2, (0.5, 30), (0.1, 20), (40, 120), [6, 17, 1],
              (200, 20000), (1, 3)),
        # spoofed flood, backward half: the victim answering all of them
        block("DDoS", n_per_class // 2, (0.5, 30), (0.1, 20), (40, 120), [6, 17, 1],
              (1, 3), (200, 20000)),
        # port scan: few packets per target, one scanner touching many hosts
        block("Probe", n_per_class, (0.05, 10), (0.2, 30), (40, 90), [6, 6, 17],
              (1, 3), (20, 3000)),
        # brute force: steady moderate-rate TCP with real payloads, one peer
        block("BFA", n_per_class, (5, 30), (5, 120), (150, 700), [6], (1, 2), (1, 2)),
    ], ignore_index=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def false_positive_rate(y_true, y_pred, labels):
    """
    Two FPR views, both wanted for the thesis:
      * overall: benign flows the IDS raised an alert on (the one that matters
        operationally - a noisy IDS gets switched off)
      * per class: one-vs-rest FPR for each attack class
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    benign = y_true == NORMAL_CLASS
    overall = float((y_pred[benign] != NORMAL_CLASS).mean()) if benign.any() else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for i, cls in enumerate(labels):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        per_class[cls] = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return overall, per_class


def measure_latency(pipe, X, repeats=300):
    """
    Per-flow inference latency, measured the way the controller actually calls
    the model. Two numbers because they differ by an order of magnitude:
      * single: one predict_proba per flow (the naive loop)
      * batched: one predict_proba for a whole FlowStatsReply
    ids_controller.py uses the batched path; both are reported so the thesis
    can justify that choice.
    """
    sample = X.iloc[:repeats]

    single = []
    for i in range(min(repeats, len(sample))):
        row = sample.iloc[i:i + 1]
        t0 = time.perf_counter()
        pipe.predict_proba(row)
        single.append((time.perf_counter() - t0) * 1000.0)

    batch_sizes, batched = [8, 32, 128], {}
    for b in batch_sizes:
        if len(sample) < b:
            continue
        t0 = time.perf_counter()
        for _ in range(20):
            pipe.predict_proba(sample.iloc[:b])
        elapsed = (time.perf_counter() - t0) * 1000.0 / 20
        batched[f"batch_{b}"] = {
            "total_ms": round(elapsed, 3),
            "per_flow_ms": round(elapsed / b, 4),
        }

    single = np.array(single)
    return {
        "single_flow_ms": {
            "mean": round(float(single.mean()), 3),
            "p50": round(float(np.percentile(single, 50)), 3),
            "p95": round(float(np.percentile(single, 95)), 3),
            "p99": round(float(np.percentile(single, 99)), 3),
        },
        "batched": batched,
        "note": "Model inference only. End-to-end detection latency also "
                "includes the flow-stats poll interval, which dominates.",
    }


def deployment_parity_check(pipe):
    """
    Refuse to ship a model that cannot see an obvious flood.

    These vectors are what controller/feature_extractor.py produces from a live
    hping3 --flood: raw counters, seconds, IANA protocol numbers, built through
    the same build_feature_vector() the controller calls. This is the check
    that the old pkl failed silently - it is not a substitute for the test set,
    it is a guard against train/serve skew that a test split cannot catch.
    """
    # (name, vector, must_not_be_normal)
    probes = [
        # Single-source floods: hping3 without --rand-source. One flow entry
        # accumulates everything, so the rate features carry the signal.
        ("SYN flood    30k pkts / 3s, 60B",
         build_feature_vector(3.0, 30000, 1_800_000, 6, 2, 1, 2, 1), True),
        ("UDP flood    20k pkts / 5s, 1KB",
         build_feature_vector(5.0, 20000, 20_000_000, 17, 2, 1, 2, 1), True),
        ("ICMP flood    9k pkts / 3s, 98B",
         build_feature_vector(3.0, 9000, 882_000, 1, 2, 1, 2, 1), True),
        # Spoofed flood: hping3 --rand-source. Each entry is tiny; only the
        # destination context separates it from an idle flow.
        ("Spoofed DDoS in  2 pkts, 8k srcs",
         build_feature_vector(10.0, 2, 120, 6, 12000, 8000, 2, 1), True),
        ("Spoofed DDoS out 2 pkts, 8k dsts",
         build_feature_vector(10.0, 2, 120, 6, 2, 1, 12000, 8000), True),
        # Port scan: one host touching hundreds of destinations.
        ("Port scan     3 pkts, 800 dsts",
         build_feature_vector(8.0, 3, 180, 6, 2, 1, 1200, 800), True),
        ("Normal web     40 pkts / 5s",
         build_feature_vector(5.0, 40, 28_000, 6, 3, 2, 3, 2), False),
        ("Normal ping    10 pkts / 10s",
         build_feature_vector(10.0, 10, 980, 1, 2, 1, 2, 1), False),
    ]
    print("\nDeployment parity check (raw counters, exactly as the controller sends them):")
    attack_verdicts, benign_verdicts, coverage = [], [], {}
    for name, vec, is_attack in probes:
        frame = build_feature_frame(vec)[list(pipe.feature_names_in_)]
        proba = pipe.predict_proba(frame)[0]
        j = int(proba.argmax())
        cls, conf = str(pipe.classes_[j]), float(proba[j])
        detected = (cls != NORMAL_CLASS)
        mark = "ok " if detected == is_attack else "MISS"
        print(f"  {mark} {name:34s} -> {cls:12s} conf={conf:.2f}")
        coverage[name.strip()] = {"predicted": cls, "confidence": round(conf, 3),
                                  "expected_attack": is_attack,
                                  "as_expected": detected == is_attack}
        (attack_verdicts if is_attack else benign_verdicts).append(cls)

    missed = [n for n, r in coverage.items()
              if r["expected_attack"] and not r["as_expected"]]
    if missed and len(missed) < len(attack_verdicts):
        print("\n  NOT COVERED by this model: " + "; ".join(missed))
        print("  These traffic shapes will pass unreported in the live testbed.")
        print("  On InSDN this is expected for the high-rate single-source floods:")
        print("  the dataset's benign bulk transfers reach 311,000 pps while its")
        print("  DoS traffic tops out at 6,555 pps, so at hping3 --flood rates")
        print("  the training data says 'Normal'. Capture your own Mininet flow")
        print("  stats and train on those to close the gap.")
    deployment_parity_check.coverage = coverage

    if all(v == NORMAL_CLASS for v in attack_verdicts):
        print(
            "\nREFUSING TO SAVE: every synthetic attack came back Normal.\n"
            "Training features do not line up with what the controller computes\n"
            "at runtime. Check --flow-timeout, DURATION_UNIT_DIVISOR, and that\n"
            "the protocol column really holds IANA numbers."
        )
        return False
    if all(v != NORMAL_CLASS for v in benign_verdicts):
        print(
            "\nREFUSING TO SAVE: both benign probes were flagged as attacks.\n"
            "A model that alerts on everything is as useless as one that alerts\n"
            "on nothing; it would just bury the dashboard instead of staying silent."
        )
        return False
    return True


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="InSDN CSV file, or a directory of them (default: data/)")
    ap.add_argument("--synthetic", action="store_true",
                    help="train on generated traffic - smoke test, not thesis data")
    ap.add_argument("--max-per-class", type=int, default=60000,
                    help="downsample any class above this many entries. InSDN's "
                         "spoofed DDoS produces one entry per forged source "
                         "(243,886 of them) against 414 DoS entries - a 590:1 "
                         "imbalance that drowns every other class. 0 disables.")
    ap.add_argument("--flow-timeout", type=float, default=10.0,
                    help="flow-entry lifetime to emulate; must match the "
                         "hard_timeout used in ids_controller.py (default 30s)")
    ap.add_argument("--polls-per-entry", type=int, default=3,
                    help="samples emitted per emulated entry, at evenly spaced "
                         "poll times across its lifetime. Keep it near "
                         "flow_timeout / POLL_INTERVAL_SEC so the emulated "
                         "poll cadence matches the controller's real one.")
    ap.add_argument("--keep-byte-features", action="store_true",
                    help="keep byte_count / byte_rate / avg_packet_size even "
                         "when the dataset does not record bytes for attack "
                         "traffic. On InSDN this leaks the label and produces "
                         "a model that cannot see a live flood.")
    ap.add_argument("--no-context-features", action="store_true",
                    help="train on the 7 per-flow features only, dropping "
                         "flows_to_dst / distinct_srcs_to_dst. Use it to "
                         "produce the ablation row for the write-up - it is "
                         "the configuration that cannot see spoofed DDoS.")
    ap.add_argument("--min-samples", type=int, default=100,
                    help="drop classes with fewer emulated entries than this; "
                         "InSDN's U2R (17 rows) and Web-Attack (192) cannot be "
                         "learned from seven volumetric features and only "
                         "corrupt the macro averages")
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--max-depth", type=int, default=20,
                    help="capped to keep the pickle under GitHub's 100MB limit; "
                         "the old unbounded depth-46 forest was 278MB")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--out", default=MODEL_PATH)
    ap.add_argument("--metadata", default=METADATA_PATH)
    ap.add_argument("--skip-parity-check", action="store_true",
                    help="save even if the deployment parity check fails (not advised)")
    args = ap.parse_args()

    # --- data ---------------------------------------------------------------
    if args.synthetic:
        print("SYNTHETIC MODE - generated traffic, results are not thesis data.\n")
        entries = synthetic_entries()
        dataset_name = "synthetic (train_model.py::synthetic_entries)"
        thesis_ready = False
    else:
        df = load_dataset(args.csv)
        entries = emulate_flow_table_entries(df, args.flow_timeout,
                                             args.polls_per_entry)
        dataset_name = os.path.abspath(args.csv)
        thesis_ready = True

    entries = prepare_entries(entries, args.max_per_class, args.min_samples)

    X, y = to_feature_matrix(entries)
    feature_names = BASE_FEATURE_NAMES if args.no_context_features else list(FEATURE_NAMES)

    # Byte-counter leakage check, run on the aggregated entries the model will
    # actually be fitted on.
    zero_share = (pd.DataFrame({"b": X["byte_count"], "y": y})
                  .groupby("y")["b"].apply(lambda s: float((s == 0).mean())))
    worst = zero_share.max()
    if worst > 0.5 and not args.keep_byte_features:
        print(f"\nByte counters are missing for {worst:.0%} of "
              f"'{zero_share.idxmax()}' entries "
              f"({ {k: f'{v:.0%}' for k, v in zero_share.items()} }).")
        print("  A zero byte count is an artifact of the capture, not attack "
              "behaviour, and it tracks the label closely enough to act as one.")
        print(f"  Dropping {BYTE_FEATURE_NAMES} - pass --keep-byte-features to "
              f"override.")
        feature_names = [f for f in feature_names if f not in BYTE_FEATURE_NAMES]
    X = X[feature_names]
    if args.no_context_features:
        print("ABLATION: destination-context features dropped; training on "
              f"{len(feature_names)} per-flow features only.")
    print("\nClass distribution (emulated flow entries):")
    for cls, n in pd.Series(y).value_counts().items():
        print(f"  {cls:14s} {n:9,d}")
    if len(set(y)) < 2:
        raise SystemExit("Need at least two classes to train.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y)

    # --- model --------------------------------------------------------------
    # The scaler is a no-op for a random forest, which is scale-invariant. It
    # is inside the Pipeline anyway so that the saved artifact is a complete
    # inference contract: swapping in an SVM or MLP for the comparison table
    # every thesis wants cannot reintroduce the "where did the scaler go" bug,
    # because there is nowhere to put preprocessing except inside the pickle.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=args.trees,
            max_depth=args.max_depth,
            class_weight="balanced",   # attack classes are heavily outnumbered
            n_jobs=-1,
            random_state=42,
        )),
    ])

    print(f"\nTraining Pipeline(StandardScaler, RandomForest) on {len(X_train):,} rows...")
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_secs = time.perf_counter() - t0
    print(f"Done in {train_secs:.1f}s")

    # --- evaluation ---------------------------------------------------------
    y_pred = pipe.predict(X_test)
    labels = sorted(set(y))
    print("\n=== Test-set performance ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    fpr_overall, fpr_per_class = false_positive_rate(y_test, y_pred, labels)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "false_positive_rate_overall": round(fpr_overall, 4),
        "false_positive_rate_per_class": {k: round(v, 4) for k, v in fpr_per_class.items()},
        "per_class": classification_report(y_test, y_pred, zero_division=0,
                                           output_dict=True),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        },
    }
    print(f"Overall false positive rate (benign flows alerted on): {fpr_overall:.4f}")

    latency = measure_latency(pipe, X_test)
    print(f"Inference latency: {latency['single_flow_ms']['mean']:.2f} ms/flow single, "
          + ", ".join(f"{k}={v['per_flow_ms']:.3f} ms/flow" for k, v in latency["batched"].items()))

    parity_ok = deployment_parity_check(pipe)
    if not parity_ok and not args.skip_parity_check:
        return 1
    coverage = getattr(deployment_parity_check, "coverage", {})

    # --- save ---------------------------------------------------------------
    joblib.dump(pipe, args.out)
    size_mb = os.path.getsize(args.out) / 1e6

    metadata = {
        "model": "Pipeline(StandardScaler, RandomForestClassifier)",
        "model_file": os.path.basename(args.out),
        "features": feature_names,
        "n_features": len(feature_names),
        "context_features_used": not args.no_context_features,
        "byte_features_used": not set(BYTE_FEATURE_NAMES) - set(feature_names),
        "classes": [str(c) for c in pipe.classes_],
        "dataset": dataset_name,
        "thesis_ready": thesis_ready,
        "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": round(train_secs, 1),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "model_size_mb": round(size_mb, 1),
        "flow_entry_emulation": {
            "flow_timeout_sec": args.flow_timeout,
            "polls_per_entry": args.polls_per_entry,
            "max_per_class": args.max_per_class,
            "min_samples": args.min_samples,
            "note": "Dataset micro-flows were re-aggregated into unidirectional "
                    "OpenFlow flow-table entries before training so that the "
                    "training distribution matches what the Ryu controller "
                    "reads from OFPFlowStatsReply.",
        },
        "hyperparameters": {
            "n_estimators": args.trees,
            "max_depth": args.max_depth,
            "class_weight": "balanced",
            "random_state": 42,
            "test_size": args.test_size,
        },
        "protocol_encoding": {
            "scheme": "IANA protocol numbers (matches OpenFlow ip_proto)",
            "icmp": 1, "tcp": 6, "udp": 17, "other": 0,
        },
        "metrics": metrics,
        "latency": latency,
        # Which live-traffic shapes this model actually detects. Offline metrics
        # cannot express this: the test split contains no hping3 --flood.
        "deployment_coverage": coverage,
        "uncovered_shapes": [n for n, r in coverage.items()
                             if r["expected_attack"] and not r["as_expected"]],
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with open(args.metadata, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nSaved model    -> {args.out} ({size_mb:.1f} MB)")
    print(f"Saved metadata -> {args.metadata}")
    if size_mb > 90:
        print("WARNING: over 90MB. GitHub rejects pushes above 100MB - keep the "
              "pkl gitignored and regenerate it, or lower --max-depth/--trees.")
    if not thesis_ready:
        print("\nREMINDER: this model was trained on synthetic traffic. "
              "Do not report its metrics. Retrain with --csv data/ on InSDN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
