"""
evaluate_model.py
=================
Produces the evaluation numbers the thesis needs, from the saved pipeline:

  accuracy, precision, recall, F1 (per class, macro, weighted)
  false positive rate (overall and one-vs-rest per class)
  confusion matrix
  detection latency (single-flow and batched inference)
  confidence-threshold sweep (detection rate vs false-positive rate)

It re-creates the SAME train/test split train_model.py used - same seed, same
test_size read back from model_metadata.json - so nothing here is scored on
data the forest was fitted on.

Writes machine-readable results plus CSV tables ready to paste into the
write-up:
  evaluation/results/metrics.json
  evaluation/results/confusion_matrix.csv
  evaluation/results/per_class_metrics.csv
  evaluation/results/threshold_sweep.csv

RUN:
  python3 evaluation/evaluate_model.py --csv data/
  python3 evaluation/evaluate_model.py --synthetic     # smoke test only
"""

import argparse
import csv
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))

from feature_config import FEATURE_NAMES  # noqa: E402
from train_model import (NORMAL_CLASS, emulate_flow_table_entries,  # noqa: E402
                         false_positive_rate, load_dataset, measure_latency,
                         prepare_entries, synthetic_entries, to_feature_matrix)

MODEL_PATH = os.path.join(ROOT, "model", "ids_model.pkl")
METADATA_PATH = os.path.join(ROOT, "model", "model_metadata.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def rebuild_test_split(args, meta):
    """Reconstruct train_model.py's held-out split."""
    if args.synthetic:
        entries = synthetic_entries()
    else:
        emulation = meta.get("flow_entry_emulation", {})
        entries = emulate_flow_table_entries(
            load_dataset(args.csv),
            args.flow_timeout or emulation.get("flow_timeout_sec", 30.0),
            args.polls_per_entry or emulation.get("polls_per_entry", 3))

    # Reproduce training's downsampling EXACTLY, or the split below returns
    # rows the forest was fitted on and every metric comes out inflated.
    emulation = meta.get("flow_entry_emulation", {})
    entries = prepare_entries(entries,
                              emulation.get("max_per_class", 0),
                              emulation.get("min_samples", 0))

    known = set(str(c) for c in meta["classes"])
    dropped = set(entries["label"].unique()) - known
    if dropped:
        print(f"Ignoring classes the model was not trained on: {sorted(dropped)}")
        entries = entries[entries["label"].isin(known)]

    X, y = to_feature_matrix(entries)
    test_size = meta.get("hyperparameters", {}).get("test_size", 0.25)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    return X_test, y_test


def threshold_sweep(model, X_test, y_test, thresholds):
    """
    How CONFIDENCE_THRESHOLD in ids_controller.py trades detection for noise.

    Rows below the threshold are reported as Normal, which is exactly what the
    controller does. This is the table that justifies the value chosen rather
    than asserting it.
    """
    proba = model.predict_proba(X_test)
    classes = np.array([str(c) for c in model.classes_])
    top_idx = proba.argmax(axis=1)
    top_cls, top_conf = classes[top_idx], proba[np.arange(len(proba)), top_idx]

    y_test = np.asarray([str(v) for v in y_test])
    is_attack = y_test != NORMAL_CLASS

    rows = []
    for t in thresholds:
        pred = np.where(top_conf >= t, top_cls, NORMAL_CLASS)
        alerted = pred != NORMAL_CLASS

        tp = int((alerted & is_attack).sum())
        fp = int((alerted & ~is_attack).sum())
        fn = int((~alerted & is_attack).sum())
        tn = int((~alerted & ~is_attack).sum())

        # Correct class, not merely "something was flagged" - an IDS that calls
        # every DDoS a Probe has detected the attack but classified it wrong.
        correct_class = int(((pred == y_test) & is_attack).sum())

        rows.append({
            "threshold": round(float(t), 2),
            "detection_rate_recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "correct_class_rate": round(correct_class / is_attack.sum(), 4) if is_attack.any() else 0.0,
            "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "accuracy": round(float((pred == y_test).mean()), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return rows


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames or list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(ROOT, "data"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--flow-timeout", type=float, default=None,
                    help="override; defaults to the value in model_metadata.json")
    ap.add_argument("--polls-per-entry", type=int, default=None)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--out", default=RESULTS_DIR)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"No model at {args.model}. Run model/train_model.py first.")
    if not os.path.exists(METADATA_PATH):
        raise SystemExit(f"No metadata at {METADATA_PATH}. Run model/train_model.py first.")

    model = joblib.load(args.model)
    with open(METADATA_PATH) as fh:
        meta = json.load(fh)

    # The model may legitimately use a SUBSET of what feature_config can build
    # (the byte-leakage drop, the --no-context-features ablation). Only a
    # feature the controller cannot produce at all is contract drift.
    unknown = [f for f in meta["features"] if f not in FEATURE_NAMES]
    if unknown:
        raise SystemExit(
            f"Feature contract drift: the model wants {unknown}, which "
            f"feature_config.py cannot build (it produces {FEATURE_NAMES}). "
            "Retrain before evaluating.")
    if meta.get("thesis_ready") is False:
        print("WARNING: this model was trained on SYNTHETIC traffic. The numbers "
              "below measure the plumbing, not detection performance. Do not "
              "report them.\n")

    os.makedirs(args.out, exist_ok=True)
    X_test, y_test = rebuild_test_split(args, meta)
    print(f"Evaluating on {len(X_test):,} held-out flow entries\n")

    X_test = X_test[meta["features"]]
    y_pred = model.predict(X_test)
    labels = sorted(set(str(c) for c in meta["classes"]))
    y_test_s = np.asarray([str(v) for v in y_test])
    y_pred_s = np.asarray([str(v) for v in y_pred])

    print("=== Classification report ===")
    print(classification_report(y_test_s, y_pred_s, labels=labels, zero_division=0))

    fpr_overall, fpr_per_class = false_positive_rate(y_test_s, y_pred_s, labels)
    cm = confusion_matrix(y_test_s, y_pred_s, labels=labels)

    print("=== Confusion matrix (rows = actual, cols = predicted) ===")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    summary = {
        "accuracy": round(float(accuracy_score(y_test_s, y_pred_s)), 4),
        "precision_macro": round(float(precision_score(y_test_s, y_pred_s, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_test_s, y_pred_s, average="weighted", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test_s, y_pred_s, average="macro", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test_s, y_pred_s, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test_s, y_pred_s, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_test_s, y_pred_s, average="weighted", zero_division=0)), 4),
        "false_positive_rate_overall": round(fpr_overall, 4),
    }
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k:32s} {v}")
    print(f"  {'false_positive_rate_per_class':32s} "
          f"{ {k: round(v, 4) for k, v in fpr_per_class.items()} }")

    latency = measure_latency(model, X_test)
    print("\n=== Detection latency (model inference only) ===")
    s = latency["single_flow_ms"]
    print(f"  single flow : mean {s['mean']:.2f} ms, p95 {s['p95']:.2f} ms, p99 {s['p99']:.2f} ms")
    for k, v in latency["batched"].items():
        print(f"  {k:12s}: {v['total_ms']:.2f} ms total, {v['per_flow_ms']:.4f} ms/flow")
    print(f"  NOTE: {latency['note']}")

    sweep = threshold_sweep(model, X_test, y_test_s,
                            np.arange(0.0, 1.0, 0.05))
    print("\n=== Confidence-threshold sweep ===")
    print(f"  {'thr':>5} {'detect':>8} {'correct':>8} {'FPR':>8} {'prec':>8} {'acc':>8}")
    for r in sweep:
        print(f"  {r['threshold']:>5.2f} {r['detection_rate_recall']:>8.4f} "
              f"{r['correct_class_rate']:>8.4f} {r['false_positive_rate']:>8.4f} "
              f"{r['precision']:>8.4f} {r['accuracy']:>8.4f}")

    # --- persist ------------------------------------------------------------
    report = classification_report(y_test_s, y_pred_s, labels=labels,
                                   zero_division=0, output_dict=True)
    results = {
        "model": args.model,
        "dataset": meta.get("dataset"),
        "thesis_ready": meta.get("thesis_ready", True),
        "n_test_samples": int(len(X_test)),
        "labels": labels,
        "summary": summary,
        "false_positive_rate_per_class": {k: round(v, 4) for k, v in fpr_per_class.items()},
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "latency": latency,
        "threshold_sweep": sweep,
    }
    print("\nWrote:")
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  {os.path.join(args.out, 'metrics.json')}")

    write_csv(os.path.join(args.out, "confusion_matrix.csv"),
              [{"actual": labels[i], **{labels[j]: int(cm[i][j])
                                        for j in range(len(labels))}}
               for i in range(len(labels))],
              fieldnames=["actual"] + labels)

    write_csv(os.path.join(args.out, "per_class_metrics.csv"),
              [{"class": c,
                "precision": round(report[c]["precision"], 4),
                "recall": round(report[c]["recall"], 4),
                "f1_score": round(report[c]["f1-score"], 4),
                "false_positive_rate": round(fpr_per_class.get(c, 0.0), 4),
                "support": int(report[c]["support"])}
               for c in labels if c in report])

    write_csv(os.path.join(args.out, "threshold_sweep.csv"), sweep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
