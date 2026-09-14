# SDN Intrusion Detection & Mitigation System

A machine-learning intrusion **detection and mitigation** system for
software-defined networks. A single Random Forest, trained offline on the
**InSDN** dataset, runs as the *only* live detector inside a **Ryu** controller
that monitors a multi-switch **Mininet** testbed. When it flags an attack the
controller closes the loop and blocks the attacker in the data plane, and a live
**Streamlit** dashboard shows the topology, alerts, and active blocks in real time.

> The ML model is the sole detector — there are no hand-written threshold rules
> on the live path. The only number that gates an alert is the model's own
> predicted probability.

## Architecture

```
Mininet network  →  OpenFlow switches  →  Ryu controller
   → flow-stat feature extraction  →  ML model classifies each flow
   → alert  →  FastAPI backend (SQLite)  →  Streamlit dashboard
   → mitigation: high-priority DROP flow blocks the attacker's MAC
```

| Component | Path | Responsibility |
|---|---|---|
| Emulated network | `mininet_topo/topology.py`, `traffic.py` | Parametrized 3-tier tree (1 core, 2 aggregation, 4 edge switches; 12 hosts; role-based IPs); optional benign background traffic |
| Controller | `controller/ids_controller.py` | Ryu app: L2 learning switch + IP-aware flow install, flow-stats polling (3 s), batched ML inference, alert dispatch, **closed-loop mitigation** |
| Feature extraction | `controller/feature_extractor.py` | One `OFPFlowStatsReply` → feature vectors, incl. per-batch fan-in / fan-out |
| Shared contract | `model/feature_config.py` | Single definition of the feature vector, imported by both training and the live extractor |
| Model + training | `model/train_model.py` | Flow-entry emulation, leakage checks, `Pipeline(StandardScaler, RandomForest)`, deployment parity check |
| Alert store | `backend/main.py`, `backend/database.py` | FastAPI + SQLite; stores alerts and mitigations |
| Dashboard | `dashboard/streamlit_app.py`, `topology_view.py` | Live topology map, alert feed, mitigation panel, manual block control |
| Attacks | `attacks/` | Attack simulation scripts |
| Evaluation | `evaluation/evaluate_model.py` | Offline metrics for the thesis |
| Tests | `tests/test_ids_pipeline.py` | Train/serve contract tests (no Mininet needed) |

---

## Requirements

- **Ubuntu / WSL2** (Mininet needs Linux + root)
- **Python 3.10**, in the project virtualenv (`venv/`)
- Mininet, Open vSwitch, `hping3`, `nmap`, `ryu-manager`
- Python packages: `pip install -r requirements.txt`

Everything (Ryu, FastAPI, Streamlit, scikit-learn) runs from the one `venv`.

---

## Quick start (5 terminals)

Start each in order and leave it running. Victim is **h1 / 10.1.1.1**;
attacker hosts are **h6, h9, h12**.

**Terminal 1 — Backend** (fresh DB)
```bash
cd ~/Network-thesis && source venv/bin/activate && rm -f backend/ids_alerts.db && cd backend && uvicorn main:app --port 8000
```

**Terminal 2 — Ryu controller** (start *before* Mininet)
```bash
cd ~/Network-thesis && source venv/bin/activate && ryu-manager controller/ids_controller.py
```
Wait for `Loaded model from ...` and the `Switch ... connected` lines.

**Terminal 3 — Mininet topology**
```bash
cd ~/Network-thesis && sudo python3 mininet_topo/topology.py
```
`topology.py` auto-runs `mn -c` on startup, so no manual cleanup is needed.
At the `mininet>` prompt, verify connectivity (a few WAN-link drops are normal):
```bash
pingall
```

**Terminal 4 — Dashboard**
```bash
cd ~/Network-thesis && source venv/bin/activate && cd dashboard && streamlit run streamlit_app.py
```
Open http://localhost:8501.

**Terminal 5 — Watch alerts (optional)**
```bash
watch -n 2 "curl -s 'localhost:8000/alerts?limit=8' | python3 -m json.tool"
```

### Shutdown
```bash
pkill -f ryu-manager; pkill -f "uvicorn main:app"; pkill -f "streamlit run"; sudo mn -c
```

---

## Running attacks

Type these at the `mininet>` prompt. Stop any `--flood` with **Ctrl+C**.

**Detected as `DDoS`** (spoofed-source — the fan-in shape the model recognizes):
```bash
h6 hping3 -S --flood --rand-source -p 80 10.1.1.1
h6 hping3 --udp --flood --rand-source -p 53 10.1.1.1
h6 python3 attacks/control_plane_saturation.py --victim 10.1.1.1 --iface h6-eth0 --rate 500 --duration 30
```

**Provided but not reliably detected** (documented model limitation — single-source
floods above InSDN's benign rate ceiling read as `Normal`):
```bash
h6 bash -c "cd attacks && ./syn_flood.sh 10.1.1.1"     # single-source SYN flood
h6 bash -c "cd attacks && ./udp_flood.sh 10.1.1.1"     # single-source UDP flood
h6 bash -c "cd attacks && ./port_scan.sh 10.1.1.1"     # nmap SYN scan
h6 python3 attacks/slowloris_attack.py 10.1.1.1 --port 80 --sockets 100 --interval 10
h6 python3 attacks/arp_spoof.py --iface h6-eth0        # ARP-layer MITM
```

**Detected** = an `[ALERT]` line in the controller terminal + a new backend row +
the victim turning red on the dashboard (allow ~6 s / two poll cycles).

---

## Detection

- **Features (8 of 11 candidates):** `duration_sec`, `packet_count`, `packet_rate`,
  `protocol` (raw IANA number), plus four destination-context features
  (`flows_to_dst`, `distinct_srcs_to_dst`, `flows_from_src`, `distinct_dsts_from_src`).
- **Byte-derived features are dropped** — InSDN records no byte counts for attack
  traffic, so they leak the label and collapse on a live switch.
- **Fan-in / fan-out context features** make spoofed-source floods (invisible per
  flow) learnable — this is why `--rand-source` floods are detected as DDoS.
- **Confidence threshold 0.60** gates alerts; below it a flow is reported `Normal`.
- **Temporal smoothing** (`SMOOTH_WINDOW_SEC`) averages a flow's class probability
  over recent polls to stop DoS↔Probe↔BFA flapping without delaying detection.
- **Offline metrics:** 98.6 % accuracy, 0.988 weighted-F1, 0.681 macro-F1 (the gap
  is driven by three minority classes). Detail lives in `report.tex`.

### What is not detectable (by construction)
- Single-source floods faster than InSDN's benign ceiling → classified `Normal`.
- Control-plane saturation → prevents flow entries from forming, so flow-stats
  polling never sees it.
- U2R → too rare in InSDN to train; dropped.

---

## Mitigation (closed-loop)

When an alert fires the controller installs a high-priority **DROP flow** and posts
the block to the backend.

- **Blocks by MAC, not IP** — a spoofed flood forges a new source IP per packet but
  keeps one real MAC, so a single MAC rule stops the whole flood.
- **Self-expiring lease** — every block has a hard timeout (`BLOCK_DURATION`,
  currently **30 s**), so a wrong block heals itself instead of blackholing a host.
- **Per-victim escalation** — a victim gives an unknown attacker one timed lease;
  a *repeat* attack against a victim that already blocked it, or an attack sustained
  past the lease, escalates to a **permanent** block.
- **Whitelist** — server/victim MACs (from `topology_state.json`) are never blocked,
  preventing a self-inflicted DoS from a spoofed source.
- **Network-wide** — the drop is installed on every connected switch.
- **Manual block** — the dashboard sidebar can block *any* host for a chosen
  duration; it applies within one poll cycle.

---

## Backend API (FastAPI, port 8000)

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/alerts` | Controller posts an alert |
| `GET` | `/alerts` | Dashboard reads alerts (`?limit=`, `?attack_class=`) |
| `POST` | `/mitigations` | Controller posts a block record |
| `GET` | `/mitigations` | Dashboard reads blocks (`?active_only=`) |
| `POST` | `/manual_block` | Dashboard requests a manual block |
| `GET` | `/manual_block/pending` | Controller polls for pending manual blocks |
| `POST` | `/manual_block/ack` | Controller acknowledges a manual block |
| `GET` | `/stats` | Aggregate counts over a time window |
| `GET` | `/` | Health check |

The SQLite DB lives at `backend/ids_alerts.db`. Delete it for a clean run; the
schema is recreated on startup.

---

## Testing

```bash
cd ~/Network-thesis && source venv/bin/activate && python tests/test_ids_pipeline.py
```
12 tests verify the train/serve contract (feature order, IANA protocol encoding,
pipeline load, deployment sanity) without needing Mininet, a switch, or the backend.

---

## Retraining

```bash
python model/train_model.py --csv data/InSDN.csv
```
Saves a `Pipeline(StandardScaler, RandomForest)` plus `model_metadata.json`, and
runs a deployment parity check that refuses to ship a model which calls an obvious
flood `Normal`. Use `--no-context-features` to reproduce the failing (spoofed-DDoS-blind)
configuration for the ablation.

---

## Troubleshooting

- **Dashboard "Cannot reach backend"** — Terminal 1 (uvicorn) isn't running on 8000.
- **`pingall` fails** — the controller wasn't up/connected when Mininet started.
  Start the controller first, then the topology.
- **Controller prints `Terminated`** — an external signal killed it; relaunch, or run
  detached: `nohup ryu-manager controller/ids_controller.py > /tmp/ryu.log 2>&1 &`.
- **No node turns red during a single-source flood** — expected; use `--rand-source`
  (see Running attacks) — the model only reliably detects the fan-in shape.
- **`Address already in use`** — an old backend/dashboard is still running; kill it
  or pick another port and set `IDS_BACKEND_URL`.
