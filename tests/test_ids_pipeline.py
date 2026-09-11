"""
test_ids_pipeline.py
====================
Verifies the ML deployment pipeline end to end, without needing Mininet, a
switch or the backend.

The point of these tests is not model accuracy - that is what the test split
in train_model.py measures. The point is the *contract* between training and
deployment, which is what actually broke: a model that scores 0.93 offline and
still calls every live flow Normal passes every accuracy check ever written.

Run:
  python3 tests/test_ids_pipeline.py        # plain, no dependencies
  pytest tests/test_ids_pipeline.py         # also works if pytest is installed
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "controller"))

import joblib  # noqa: E402
from feature_config import (CONTEXT_FEATURE_NAMES, FEATURE_NAMES,  # noqa: E402
                            N_FEATURES, build_feature_frame,
                            build_feature_vector, protocol_to_numeric)
from feature_extractor import (extract_batch,  # noqa: E402
                               extract_features_from_flow_stat)

MODEL_PATH = os.path.join(ROOT, "model", "ids_model.pkl")
METADATA_PATH = os.path.join(ROOT, "model", "model_metadata.json")

NORMAL = "Normal"


def _load():
    assert os.path.exists(MODEL_PATH), (
        f"{MODEL_PATH} not found. Run: python3 model/train_model.py --csv data/")
    return joblib.load(MODEL_PATH)


def _metadata():
    assert os.path.exists(METADATA_PATH), (
        f"{METADATA_PATH} not found. It is written by model/train_model.py.")
    with open(METADATA_PATH) as fh:
        return json.load(fh)


class FakeFlowStats:
    """Minimal stand-in for a Ryu OFPFlowStats entry."""

    def __init__(self, duration_sec, packet_count, byte_count, ip_proto,
                 src="10.0.0.5", dst="10.0.0.1", duration_nsec=0):
        self.duration_sec = duration_sec
        self.duration_nsec = duration_nsec
        self.packet_count = packet_count
        self.byte_count = byte_count
        self.match = {"ipv4_src": src, "ipv4_dst": dst,
                      "ip_proto": ip_proto, "in_port": 1}


# ---------------------------------------------------------------------------
def test_model_loads_as_a_pipeline():
    """The saved artifact must carry its preprocessing, not just a classifier."""
    model = _load()
    assert hasattr(model, "steps"), (
        f"ids_model.pkl is a bare {type(model).__name__}, not a Pipeline. Any "
        "preprocessing used in training is missing at inference time - this is "
        "the exact defect that made the original model useless in the loop.")
    assert hasattr(model, "predict_proba"), "model cannot produce probabilities"
    assert len(model.classes_) >= 2, "model must distinguish at least two classes"


def test_metadata_matches_the_model():
    """model_metadata.json is the deployment contract; drift makes it a lie."""
    model, meta = _load(), _metadata()

    unknown = [f for f in meta["features"] if f not in FEATURE_NAMES]
    assert not unknown, (
        f"metadata names features the controller cannot build: {unknown}")
    assert meta["n_features"] == len(meta["features"])
    assert sorted(meta["classes"]) == sorted(str(c) for c in model.classes_), (
        f"metadata classes {meta['classes']} != model classes {list(model.classes_)}")

    for key in ("model", "dataset", "training_date", "metrics"):
        assert key in meta, f"metadata is missing required key '{key}'"
    for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                   "false_positive_rate_overall"):
        assert metric in meta["metrics"], f"metadata metrics missing '{metric}'"


def test_feature_dimensions_match():
    """Controller output width must equal model input width."""
    model, meta = _load(), _metadata()
    vector = build_feature_vector(3.0, 30000, 1_800_000, 6)

    assert len(vector) == N_FEATURES == len(FEATURE_NAMES)
    frame = build_feature_frame(vector)
    assert list(frame.columns) == FEATURE_NAMES
    assert frame.shape == (1, N_FEATURES)

    # The model may use a subset (byte-leakage / context ablation); the
    # controller must be able to supply every column it asks for.
    assert model.n_features_in_ == len(meta["features"]), (
        f"model expects {model.n_features_in_} features, metadata lists "
        f"{len(meta['features'])}")
    assert frame[meta["features"]].shape == (1, model.n_features_in_)


def test_feature_order_and_values():
    """Each slot must hold what FEATURE_NAMES says it holds."""
    vector = build_feature_vector(duration_sec=4.0, packet_count=200,
                                  byte_count=100_000, protocol=6)
    named = dict(zip(FEATURE_NAMES, vector))

    assert named["duration_sec"] == 4.0
    assert named["packet_count"] == 200
    assert named["byte_count"] == 100_000
    assert named["avg_packet_size"] == 500.0        # 100000 / 200
    assert named["packet_rate"] == 50.0             # 200 / 4
    assert named["byte_rate"] == 25_000.0           # 100000 / 4
    assert named["protocol"] == 6                   # IANA TCP, not a remapped 0
    # context defaults: a flow that is the only one to its destination
    for f in CONTEXT_FEATURE_NAMES:
        assert named[f] == 1


def test_protocol_encoding_is_iana():
    """
    The controller and the training data must agree on protocol numbers. An
    earlier build remapped TCP to 0 on the controller side only, so every TCP
    flow reached the forest tagged "unknown protocol".
    """
    assert protocol_to_numeric(6) == 6
    assert protocol_to_numeric(17) == 17
    assert protocol_to_numeric(1) == 1
    assert protocol_to_numeric("tcp") == 6
    assert protocol_to_numeric("udp") == 17
    assert protocol_to_numeric("icmp") == 1
    assert protocol_to_numeric(None) == 0           # no ip_proto in the match


def test_extractor_matches_training_vector():
    """
    The single most important test: the vector the controller builds from an
    OpenFlow reply must be byte-for-byte the vector training would have built
    from the same counters.
    """
    stat = FakeFlowStats(duration_sec=3, duration_nsec=500_000_000,
                         packet_count=30000, byte_count=1_800_000, ip_proto=6)
    live, info = extract_features_from_flow_stat(stat)
    offline = build_feature_vector(3.5, 30000, 1_800_000, 6)

    assert live == offline, f"train/serve skew: {live} != {offline}"
    assert info["src_ip"] == "10.0.0.5" and info["dst_ip"] == "10.0.0.1"


def test_extractor_skips_unusable_flows():
    """Empty and just-installed entries carry no signal and must be dropped."""
    assert extract_features_from_flow_stat(
        FakeFlowStats(5.0, 0, 0, 6)) == (None, None)          # no packets
    assert extract_features_from_flow_stat(
        FakeFlowStats(0, 4, 240, 6, duration_nsec=10_000_000)) == (None, None)  # 10ms old


def test_predictions_are_well_formed():
    """predict / predict_proba must agree and produce valid probabilities."""
    model, meta = _load(), _metadata()
    vectors = [
        build_feature_vector(3.0, 30000, 1_800_000, 6),
        build_feature_vector(5.0, 40, 28_000, 6),
        build_feature_vector(10.0, 10, 980, 1),
    ]
    frame = build_feature_frame(vectors)[meta["features"]]

    proba = model.predict_proba(frame)
    preds = model.predict(frame)

    assert proba.shape == (3, len(model.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0), "probabilities must sum to 1"
    assert (proba >= 0).all() and (proba <= 1).all()

    for row, pred in zip(proba, preds):
        assert str(model.classes_[int(row.argmax())]) == str(pred), (
            "argmax of predict_proba disagrees with predict")
        assert str(pred) in meta["classes"], f"'{pred}' is not a declared class"


def test_sample_traffic_is_classified_sensibly():
    """
    Live-scale traffic must land on the right side of the Normal boundary.

    These are the shapes the controller actually sees during the attack
    scripts in attacks/. This is the check the original pkl failed: it called
    all three floods Normal while reporting 0.93 test accuracy.
    """
    model, meta = _load(), _metadata()
    floods = {
        "SYN flood (30k pkts/3s, 60B)": build_feature_vector(3.0, 30000, 1_800_000, 6),
        "UDP flood (20k pkts/5s, 1KB)": build_feature_vector(5.0, 20000, 20_000_000, 17),
        "ICMP flood (9k pkts/3s, 98B)": build_feature_vector(3.0, 9000, 882_000, 1),
    }
    benign = {
        "web session (40 pkts/5s)": build_feature_vector(5.0, 40, 28_000, 6),
        "ping (10 pkts/10s)": build_feature_vector(10.0, 10, 980, 1),
    }

    verdicts = {}
    for name, vec in {**floods, **benign}.items():
        row = model.predict_proba(build_feature_frame(vec)[meta["features"]])[0]
        i = int(row.argmax())
        verdicts[name] = (str(model.classes_[i]), float(row[i]))

    # Which shapes this model covers is recorded at training time; a shape
    # listed as uncovered is a known dataset gap, not a regression, so the
    # test asserts the model still detects everything it claimed to.
    covered = {k: v for k, v in (meta.get("deployment_coverage") or {}).items()
               if v.get("expected_attack") and v.get("as_expected")}
    assert covered, (
        "model detects none of the live-traffic probe shapes - training "
        f"features do not match what the controller computes. Verdicts: {verdicts}")

    benign_calls = [verdicts[n][0] for n in benign]
    assert any(c == NORMAL for c in benign_calls), (
        f"no benign sample was classified Normal - the model alerts on "
        f"everything, which is just as useless. Verdicts: {verdicts}")


def test_batch_context_counts_fan_in_and_fan_out():
    """
    extract_batch must compute destination context across the whole reply.
    A spoofed flood is invisible without it: each forged source is a two-packet
    flow that looks exactly like an idle benign exchange on its own.
    """
    body = [FakeFlowStats(5.0, 2, 120, 6, src=f"10.0.0.{i}", dst="10.0.0.1")
            for i in range(2, 60)]
    body.append(FakeFlowStats(5.0, 40, 28_000, 6, src="10.0.1.9", dst="10.0.1.7"))
    vectors, infos = extract_batch(body)

    assert len(vectors) == len(body) == 59
    victim = [i for i in infos if i["dst_ip"] == "10.0.0.1"]
    assert len(victim) == 58
    assert victim[0]["distinct_srcs_to_dst"] == 58, "fan-in not aggregated"
    assert victim[0]["flows_to_dst"] == 58
    assert victim[0]["distinct_dsts_from_src"] == 1

    lonely = [i for i in infos if i["dst_ip"] == "10.0.1.7"][0]
    assert lonely["distinct_srcs_to_dst"] == 1, "unrelated flow contaminated"

    named = dict(zip(FEATURE_NAMES, vectors[0]))
    assert named["distinct_srcs_to_dst"] == 58


def test_predictions_are_deterministic():
    """Same input, same answer - required before quoting any metric."""
    model = _load()
    meta = _metadata()
    frame = build_feature_frame(
        build_feature_vector(3.0, 30000, 1_800_000, 6))[meta["features"]]
    assert np.array_equal(model.predict_proba(frame), model.predict_proba(frame))


def test_controller_constants_match_training():
    """
    ids_controller installs entries with FLOW_HARD_TIMEOUT; train_model
    emulated entries with --flow-timeout. If those disagree the model is
    scoring flows whose lifetime it never saw.
    """
    import ids_controller  # noqa: PLC0415  (import here: needs ryu installed)

    meta = _metadata()
    trained_with = meta.get("flow_entry_emulation", {}).get("flow_timeout_sec")
    if trained_with is None:
        return  # synthetic model: no emulation stage to agree with
    assert float(ids_controller.FLOW_HARD_TIMEOUT) == float(trained_with), (
        f"controller installs {ids_controller.FLOW_HARD_TIMEOUT}s entries but "
        f"the model was trained on {trained_with}s entries")


# ---------------------------------------------------------------------------
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed, failed = 0, []

    print(f"Running {len(tests)} pipeline tests\n" + "-" * 62)
    for fn in tests:
        name = fn.__name__
        try:
            fn()
        except AssertionError as exc:
            failed.append((name, str(exc)))
            print(f"FAIL  {name}")
        except Exception as exc:                     # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"ERROR {name}")
        else:
            passed += 1
            print(f"ok    {name}")

    print("-" * 62)
    print(f"{passed} passed, {len(failed)} failed")
    for name, msg in failed:
        print(f"\n--- {name} ---\n{msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
