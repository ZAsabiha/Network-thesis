"""
train_model.py
================
Trains the IDS classifier and writes model/ids_model.pkl.

THE BUG THIS SCRIPT EXISTS TO PREVENT
-------------------------------------
The previous ids_model.pkl was a bare RandomForestClassifier trained on
StandardScaler-normalised features, with the fitted scaler thrown away. The
controller feeds raw OpenFlow counters, so every live flow landed far
outside the training distribution and the model answered "Normal" to
everything - no alert ever fired, end to end.

Two things here stop that recurring:
  1. Scaler and classifier are saved together as a single Pipeline, so
     whatever transform training used is applied identically at inference.
  2. A live-scale sanity check runs before saving and refuses to write a
     model that calls an obvious flood "Normal".

FEATURES
--------
Built through model/feature_config.build_feature_vector - the exact function
controller/feature_extractor.py calls on live flow stats. Editing the feature
set in one place changes both.

RUN:
  python3 model/train_model.py --csv data/InSDN.csv
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_config import FEATURE_NAMES, build_feature_vector  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "..", "data", "flows.csv")
OUT_PATH = os.path.join(HERE, "ids_model.pkl")

# Header names differ between InSDN, CICIDS2017 and CIC-DDoS exports (and
# several ship with leading spaces). Headers are stripped first, then matched
# against these candidates in order, so one script handles all of them.
COLUMN_MAP = {
    "duration": ["Flow Duration", "flow_duration", "Duration"],
    "fwd_packets": ["Tot Fwd Pkts", "Total Fwd Packets", "Total Fwd Packet"],
    "bwd_packets": ["Tot Bwd Pkts", "Total Backward Packets", "Total Bwd packets"],
    "fwd_bytes": ["TotLen Fwd Pkts", "Total Length of Fwd Packets",
                   "Total Length of Fwd Packet"],
    "bwd_bytes": ["TotLen Bwd Pkts", "Total Length of Bwd Packets",
                   "Total Length of Bwd Packet"],
    "protocol": ["Protocol", "protocol"],
    "label": ["Label", "label", "Attack", "attack_cat"],
}

# CICFlowMeter reports Flow Duration in MICROSECONDS. The controller measures
# seconds. Getting this wrong is the other classic way to make training and
# inference disagree silently.
DURATION_UNIT_DIVISOR = 1_000_000.0


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


def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"Dataset not found: {csv_path}\n"
            "Download InSDN or CICIDS2017 and place the CSV there, then re-run "
            "with --csv pointing at it."
        )
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded {len(df):,} rows from {csv_path}")
    return df


def build_matrix(df):
    dur_col = resolve(df, "duration")
    fwd_p, bwd_p = resolve(df, "fwd_packets"), resolve(df, "bwd_packets")
    fwd_b, bwd_b = resolve(df, "fwd_bytes"), resolve(df, "bwd_bytes")
    proto_col = resolve(df, "protocol")
    label_col = resolve(df, "label")

    duration = pd.to_numeric(df[dur_col], errors="coerce") / DURATION_UNIT_DIVISOR
    packets = (pd.to_numeric(df[fwd_p], errors="coerce").fillna(0)
               + pd.to_numeric(df[bwd_p], errors="coerce").fillna(0))
    byts = (pd.to_numeric(df[fwd_b], errors="coerce").fillna(0)
            + pd.to_numeric(df[bwd_b], errors="coerce").fillna(0))
    proto = pd.to_numeric(df[proto_col], errors="coerce").fillna(0).astype(int)
    labels = df[label_col].astype(str).str.strip()

    frame = pd.DataFrame({
        "duration": duration, "packets": packets, "bytes": byts,
        "proto": proto, "label": labels,
    }).replace([np.inf, -np.inf], np.nan).dropna()

    # A flow with no packets carries no signal and produces div-by-zero ratios.
    frame = frame[frame["packets"] > 0]
    print(f"{len(frame):,} rows usable after cleaning")

    rows = [
        build_feature_vector(r.duration, r.packets, r.bytes, r.proto)
        for r in frame.itertuples(index=False)
    ]
    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = frame["label"].to_numpy()

    print("\nClass distribution:")
    for cls, n in frame["label"].value_counts().items():
        print(f"  {cls:20s} {n:8,d}")
    return X, y


def sanity_check(pipe):
    """
    Refuse to ship a model that cannot see an obvious flood.

    These vectors are what controller/feature_extractor.py actually produces
    from a live hping3 --flood: seconds, raw counters, no scaling applied by
    the caller. If the pipeline calls this "Normal", inference does not match
    training and the model is useless in the loop no matter how good its
    test-set accuracy looks.
    """
    probes = [
        ("SYN flood  30k pkts / 3s, 60B each", build_feature_vector(3.0, 30000, 1_800_000, 6)),
        ("UDP flood  20k pkts / 3s, 1KB each", build_feature_vector(3.0, 20000, 20_000_000, 17)),
    ]
    print("\nLive-scale sanity check (raw counters, exactly as the controller sends them):")
    verdicts = []
    for label, vec in probes:
        proba = pipe.predict_proba(pd.DataFrame([vec], columns=FEATURE_NAMES))[0]
        idx = proba.argmax()
        cls, conf = pipe.classes_[idx], proba[idx]
        verdicts.append(cls)
        print(f"  {label:38s} -> {cls:12s} conf={conf:.2f}")

    if all(str(v).lower() in ("normal", "benign") for v in verdicts):
        print(
            "\nREFUSING TO SAVE: the model labels both synthetic floods as normal "
            "traffic.\nTraining features do not line up with what the controller "
            "computes at runtime.\nCheck DURATION_UNIT_DIVISOR and COLUMN_MAP "
            "against your CSV's actual units and headers."
        )
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=24,
                    help="capped to keep the pickle small; the old 278MB model "
                          "had unbounded depth-46 trees")
    ap.add_argument("--skip-sanity-check", action="store_true",
                    help="save even if the live-scale check fails (not advised)")
    args = ap.parse_args()

    df = load_dataset(args.csv)
    X, y = build_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaler and classifier travel together from here on. Saving the Pipeline
    # rather than the bare estimator is the whole point of this rewrite.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=args.trees,
            max_depth=args.max_depth,
            class_weight="balanced",  # attack classes are heavily outnumbered
            n_jobs=-1,
            random_state=42,
        )),
    ])

    print(f"\nTraining on {len(X_train):,} rows...")
    pipe.fit(X_train, y_train)

    print("\nTest-set performance:")
    print(classification_report(y_test, pipe.predict(X_test), zero_division=0))

    if not sanity_check(pipe) and not args.skip_sanity_check:
        return 1

    joblib.dump(pipe, OUT_PATH)
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"\nSaved Pipeline(StandardScaler, RandomForest) -> {OUT_PATH} "
          f"({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())